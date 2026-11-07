/** Native Gemini vision transport over Pi's google-generative-ai codec.
 * Pi's Google adapter refuses options.fetch, so the capture, reservation, deadline and retry wrapper is
 * installed as globalThis.fetch for one session and restored by status.dispose(). Tool images travel inside
 * functionResponse.parts. The frozen Gemini comparison files stay byte-identical.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { Bridge, SessionManager, createSession, serialTools, sha, utc, WireFailure, attemptDeadline, bufferedBody,
  safeResponseMetadata, waitForRetry, recoveryDelay, safeErrorFields, rateLimitClass, providerErrorClass,
  TRANSIENT_HTTP, parseSSE, writeRecord as write, appendRecord as append, mergeTransportAccounting, wireImageParts,
  SDK_NO_TIMEOUT_MS, ATTEMPT_TIMEOUT_MS, RECOVERY_CONTRACT, MAX_PROVIDER_ERROR_RETRIES, RATE_LIMIT_RETRY_FLOOR_MS,
  publishRetryStatus, investigationContext } from './vision_transport.mjs';
export { Bridge, SessionManager, createSession, serialTools, sha, utc, mergeTransportAccounting };

export const GEMINI_MODEL = 'gemini-3.8-flash';
export const GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';
export const GEMINI_ENDPOINT = GEMINI_BASE_URL + '/models/gemini-3.8-flash:streamGenerateContent?alt=sse';
export const GUARD_RATES = { input: 1.575, output: 7.875 };
// Returned identities accepted on the direct route (precedent: gemini_direct_transport.mjs:68).
export const GEMINI_RESPONSE_MODELS = ['gemini-3.8-flash', 'gemini-3.8-flash-20260902'];
// Complete non-error terminals that belong to Pi (native continuation or Pi's own stop mapping), never to transport retry.
const POLICY_FINISH = new Set(['MAX_TOKENS', 'SAFETY', 'RECITATION', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'SPII', 'IMAGE_SAFETY']);

export function modelConfig(id, _options = {}) {
  assert.equal(id, GEMINI_MODEL, 'Exact requested model required; no substitution');
  const model = { id, name: 'Gemini 3.8 Flash (Google direct native)', api: 'google-generative-ai', provider: 'google',
    baseUrl: GEMINI_BASE_URL, reasoning: true, input: ['text', 'image'], contextWindow: 1048576, maxTokens: 32768,
    cost: { input: GUARD_RATES.input, output: GUARD_RATES.output, cacheRead: GUARD_RATES.input, cacheWrite: 0 },
    thinkingLevelMap: { high: 'high' } };
  return { model, routing: null, canonical_slug: id, expected_provider: 'Google', api_key_env: 'GEMINI_API_KEY',
    inference_route: 'google-direct', endpoint: { url: GEMINI_ENDPOINT, provider_name: 'Google', model: id },
    accepted_response_models: GEMINI_RESPONSE_MODELS,
    native_context_length: 1048576, native_max_output_tokens: 65536, precision_status: 'unattested',
    pricing_source: 'https://ai.google.dev/gemini-api/docs/pricing; conservative BYOK-observed basis', pricing_checked_date: '2026-09-22',
    pricing_note: 'Guard basis 1.575/7.875 per M, output = total - prompt, no cache discount; not billed cost.',
    common_request: { temperature: 1, maxOutputTokens: 32768, thinkingConfig: { includeThoughts: true, thinkingLevel: 'HIGH' },
      mediaResolution: 'MEDIA_RESOLUTION_HIGH' },
    feature_limit: 'Images in functionResponse.parts; no synthetic user image turn; Pi executes tools serially.' };
}

export function classifyGeminiFrame(event) {
  if (!event || typeof event !== 'object' || Array.isArray(event)) return 'unclassified';
  if (event.error !== undefined) return 'error';
  if (Array.isArray(event.candidates)) return 'candidate';
  if (event.usageMetadata || typeof event.modelVersion === 'string' || typeof event.responseId === 'string') return 'usage';
  return 'unclassified';
}

/** safeWireCapture analogue for native frames: credential echo refused; truncate at the first error or unclassified frame. */
export function geminiWireCapture(bytes, complete, key) {
  assert(!key || !bytes.includes(Buffer.from(key)), 'Credential echo refused before any response-byte write');
  const latin = bytes.toString('latin1'); let retained = 0, omission = null, framingDiagnostic = null;
  for (const match of latin.matchAll(/[\s\S]*?\r?\n\r?\n/g)) {
    const end = match.index + match[0].length;
    const block = bytes.subarray(match.index, end).toString('utf8');
    const data = block.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
    const sseEvent = block.match(/^event:\s*([a-zA-Z0-9_.:-]{1,80})\s*$/m)?.[1] ?? null;
    let kind = null;
    try { kind = data ? classifyGeminiFrame(JSON.parse(data)) : 'empty'; } catch {}
    if (kind === 'error' || sseEvent === 'error') { omission = 'error_frame_and_following_bytes_omitted'; break; }
    if (kind === null || kind === 'unclassified') {
      framingDiagnostic = { json_parsed: kind !== null, frame_bytes: end - match.index, data_bytes: Buffer.byteLength(data), sse_event: sseEvent };
      omission = 'unclassified_frame_and_following_bytes_omitted'; break;
    }
    retained = end;
  }
  if (!omission && retained < bytes.length) {
    if (complete && /^\s*$/.test(latin.slice(retained))) retained = bytes.length;
    else omission = 'incomplete_or_unclassified_tail_omitted';
  }
  return { bytes: bytes.subarray(0, retained), receipt: {
    capture_class: complete && retained === bytes.length ? 'complete_raw_sse' : 'exact_complete_non_error_prefix_only',
    received_bytes: bytes.length, retained_bytes: retained, omitted_bytes: bytes.length - retained,
    omission, ...(framingDiagnostic ? { framing_diagnostic: framingDiagnostic } : {}),
    omitted_bytes_hashed: false, retained_sha256: sha(bytes.subarray(0, retained)), body_complete: complete,
  } };
}

// Intermediate Gemini frames report prompt counts only; only usage at or after the finish reason settles a charge.
function terminalUsageMetadata(events) {
  const finish = events.findIndex(e => e.candidates?.some(c => c?.finishReason !== undefined));
  return finish < 0 ? null : events.slice(finish).findLast(e => e.usageMetadata)?.usageMetadata ?? null;
}

export function rawGeminiUsage(events) {
  const native = terminalUsageMetadata(events);
  assert(native, 'Missing terminal Gemini usageMetadata');
  assert(Number.isSafeInteger(native.promptTokenCount) && Number.isSafeInteger(native.totalTokenCount), 'Unknown Gemini token counts');
  const count = field => {
    const value = native[field] ?? 0;
    assert(Number.isSafeInteger(value) && value >= 0, `Invalid Gemini ${field}`);
    return value;
  };
  const prompt = count('promptTokenCount'), total = count('totalTokenCount'), cached = count('cachedContentTokenCount');
  const candidates = count('candidatesTokenCount'), thoughts = count('thoughtsTokenCount');
  assert(total >= prompt, 'Gemini total below prompt');
  assert(cached <= prompt, 'Invalid Gemini cachedContentTokenCount');
  assert(candidates + thoughts <= total - prompt, 'Gemini output counts exceed total minus prompt');
  assert(native.promptTokensDetails === undefined || Array.isArray(native.promptTokensDetails), 'Invalid Gemini promptTokensDetails');
  const byModality = {};
  for (const item of native.promptTokensDetails ?? []) {
    const tokens = item?.tokenCount ?? 0;
    assert(typeof item?.modality === 'string' && /^[A-Z_]{1,40}$/.test(item.modality) && Number.isSafeInteger(tokens) && tokens >= 0,
      'Invalid Gemini promptTokensDetails');
    byModality[item.modality] = (byModality[item.modality] ?? 0) + tokens;
  }
  return { usage: { prompt_tokens: prompt, completion_tokens: total - prompt, total_tokens: total,
      prompt_tokens_details: { cached_tokens: cached }, completion_tokens_details: { reasoning_tokens: thoughts } },
    native_usage_metadata: { promptTokenCount: native.promptTokenCount, cachedContentTokenCount: native.cachedContentTokenCount ?? null,
      candidatesTokenCount: native.candidatesTokenCount ?? null, thoughtsTokenCount: native.thoughtsTokenCount ?? null,
      totalTokenCount: native.totalTokenCount, promptTokensDetails: native.promptTokensDetails ?? null },
    prompt_tokens_by_modality: byModality };
}

export function geminiCostEstimate(usage, config = modelConfig(GEMINI_MODEL)) {
  for (const field of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
    assert(Number.isSafeInteger(usage?.[field]) && usage[field] >= 0, `Invalid Gemini ${field}`);
  }
  assert.equal(usage.completion_tokens, usage.total_tokens - usage.prompt_tokens, 'Gemini completion alias must equal total minus prompt');
  const rates = config.model.cost;
  return { cost_estimate_usd: (usage.prompt_tokens * rates.input + (usage.total_tokens - usage.prompt_tokens) * rates.output) / 1e6,
    native_cost_usd: null, billing_cost_status: 'not_reported_by_api', estimated_or_reserved: true,
    cost_basis: 'guard basis 1.575/7.875 per M; output = total - prompt; no cache discount',
    pricing_source: config.pricing_source, pricing_checked_date: config.pricing_checked_date };
}

// Transport integrity only; scientific choices are not validated here.
export function validateCompleteGeminiStream(events) {
  assert(events.length && events.every(e => ['candidate', 'usage'].includes(classifyGeminiFrame(e))), 'Unsupported Gemini frame');
  const frames = events.filter(e => Array.isArray(e.candidates));
  assert(frames.length, 'Missing Gemini candidate');
  const finishes = [], calls = [];
  for (const event of frames) {
    assert(event.candidates.length === 1 && (event.candidates[0]?.index ?? 0) === 0, 'Unexpected Gemini candidate index');
    const candidate = event.candidates[0], parts = candidate.content?.parts ?? [];
    assert(Array.isArray(parts), 'Invalid Gemini content parts');
    assert(!finishes.length || parts.length === 0, 'Output followed Gemini finish reason');
    for (const part of parts) if (part?.functionCall !== undefined) {
      const call = part.functionCall;
      assert(call && typeof call.name === 'string' && /^[a-zA-Z0-9_-]+$/.test(call.name), 'Incomplete Gemini function call name');
      assert(call.args && typeof call.args === 'object' && !Array.isArray(call.args), 'Gemini function call arguments must be an object');
      assert(call.id === undefined || (typeof call.id === 'string' && call.id.length > 0), 'Invalid Gemini function call id');
      calls.push({ name: call.name, id: call.id ?? null });
    }
    if (candidate.finishReason !== undefined) {
      assert(typeof candidate.finishReason === 'string' && /^[A-Z_]{1,64}$/.test(candidate.finishReason), 'Invalid Gemini finish reason');
      finishes.push(candidate.finishReason);
    }
  }
  assert(frames.at(-1).candidates[0].finishReason, 'Last Gemini candidate frame lacks finishReason');
  rawGeminiUsage(events);
  const versions = [...new Set(events.map(e => e.modelVersion).filter(v => v !== undefined))];
  assert(versions.length === 1 && typeof versions[0] === 'string', 'Missing or mixed Gemini modelVersion');
  const ids = [...new Set(events.map(e => e.responseId).filter(v => v !== undefined))];
  assert(ids.length <= 1 && ids.every(id => typeof id === 'string'), 'Mixed Gemini response ids');
  return { finish_reasons: finishes, model_version: versions[0], response_id: ids[0] ?? null, function_calls: calls };
}

let installed = false;
/** All Gemini inference, including Pi summarization, crosses this exact-model hook and the session-scoped global fetch. */
export function installTransport(session, output, instructions, config, {
  fetchImpl = fetch, key = '', waitImpl = waitForRetry, now = Date.now, random = Math.random,
  clock = { setTimeout, clearTimeout }, maxRequests = Infinity,
} = {}) {
  assert.equal(config.model.provider, 'google', 'Unsupported transport provider');
  assert.equal(config.model.api, 'google-generative-ai'); assert.equal(config.model.id, GEMINI_MODEL);
  assert.equal(config.model.baseUrl, GEMINI_BASE_URL); assert.equal(config.endpoint.url, GEMINI_ENDPOINT);
  // fetchImpl (the budget guard or a test fake) must be captured before the global replacement below.
  assert(typeof fetchImpl === 'function', 'Explicit network fetch required');
  assert(!installed, 'A Gemini transport is already installed in this process');
  let index = fs.readdirSync(path.join(output, 'api')).filter(n => /^request_\d+\.json$/.test(n)).length;
  installed = true; // Nothing below can fail before the global replacement.
  const previousFetch = globalThis.fetch;
  const original = session.modelRuntime.streamSimple.bind(session.modelRuntime);
  const route = { inference_route: 'google-direct', endpoint_url: GEMINI_ENDPOINT, wire_model_id: GEMINI_MODEL };
  const status = { blocked: false, usage_unresolved: false, resourceCensored: false, known_cost_usd: 0,
    estimated_cost_usd: 0, native_billing_unreported_attempts: 0, requests: 0, accepted_outputs: 0 };
  let active = null, calling = false, disposed = false;
  session.modelRuntime.streamSimple = (model, incoming, options = {}) => {
    assert(!disposed, 'Gemini transport disposed; no uncaptured inference');
    assert.equal(model.id, config.model.id); assert.equal(model.provider, config.model.provider);
    const { normal, context } = investigationContext(incoming, instructions);
    const call = { category: options.maxTokens === 1 ? 'native_cache_warm' : normal ? 'investigator' : 'native_compaction_or_summary', instructionsMatch: normal };
    active = call;
    const { fetch: _sdkFetch, ...rest } = options; // Pi's Google adapter accepts only the global fetch.
    const stream = original(model, context, { ...rest, maxTokens: Math.min(options.maxTokens ?? 32768, 32768),
      reasoning: 'high', temperature: 1, maxRetries: 0, timeoutMs: SDK_NO_TIMEOUT_MS,
      onPayload: params => {
        assert.equal(params.model, GEMINI_MODEL, 'Native Gemini params name another model');
        // Shallow copies only: config.abortSignal must keep its identity (structuredClone would flatten it).
        return { ...params, config: { ...params.config, temperature: 1,
          maxOutputTokens: Math.min(params.config?.maxOutputTokens ?? 32768, 32768),
          thinkingConfig: { includeThoughts: true, thinkingLevel: 'HIGH' }, mediaResolution: 'MEDIA_RESOLUTION_HIGH' } };
      } });
    // A global fetch after this call's stream settles has no owning Pi call and is refused.
    void stream.result().catch(() => {}).finally(() => { if (active === call) active = null; });
    return stream;
  };
  const geminiVisionFetch = async function geminiVisionFetch(input, init = {}) {
    if (status.blocked) throw new Error('Transport sealed after permanent failure or cancellation');
    // A network fetch that re-enters the global wrapper would recurse; refuse and seal instead.
    assert(!calling, 'Recursive global fetch refused; the network fetch must be captured before installation');
    const url = typeof input === 'string' ? input : input?.url ?? String(input);
    assert.equal(url, GEMINI_ENDPOINT);
    assert(active, 'Global fetch outside an owning Pi Gemini call refused');
    const { category } = active;
    assert(key, 'Explicit Gemini credential required');
    assert.equal(new Headers(init.headers).get('x-goog-api-key'), key, 'SDK credential header differs from the explicit credential');
    const body = JSON.parse(init.body), generation = body.generationConfig ?? {};
    assert(!('model' in body), 'Native Gemini body must not name a model');
    assert(Number.isInteger(generation.maxOutputTokens) && generation.maxOutputTokens > 0 && generation.maxOutputTokens <= 32768);
    assert.equal(generation.temperature, 1);
    assert.deepEqual(generation.thinkingConfig, { includeThoughts: true, thinkingLevel: 'HIGH' });
    assert.equal(generation.mediaResolution, 'MEDIA_RESOLUTION_HIGH');
    assert(!init.body.includes(key), 'Credential present in request body');
    // Rebuild the direct-provider headers: no SDK client, user-agent or additional credential header crosses.
    // The key lives only in this in-memory header; request files persist the body alone.
    init = { ...init, headers: { 'x-goog-api-key': key, 'content-type': 'application/json' }, redirect: 'error' };
    const policyRequestId = crypto.randomUUID(), bodySha256 = sha(init.body);
    const images = wireImageParts(body);
    let unclassifiedFetchFailures = 0, providerErrorFailures = 0;
    for (let attempt = 0; ; attempt++) {
      if (status.requests >= maxRequests) {
        status.resourceCensored = true; status.blocked = true;
        throw new Error('Declared wire-request resource envelope exhausted; not a scientific stop');
      }
      if (init.signal?.aborted) { if (category !== 'native_cache_warm') status.blocked = true; throw new Error('Caller cancelled'); }
      const number = String(index++).padStart(4, '0'), accountingId = crypto.randomUUID();
      const chunks = [];
      let sent = false, known = false, receivedHttp = false, metadata = null, wireSaved = false;
      let providerFailure = null;
      const account = events => {
        const native = rawGeminiUsage(events);
        if (!known) {
          const accounting = geminiCostEstimate(native.usage, config);
          append(path.join(output, 'usage.jsonl'), { accounting_id: accountingId, number, category,
            policy_request_id: policyRequestId, transport_attempt: attempt, status: 'known', provider: 'google', ...route,
            model_version: events.find(e => e.modelVersion)?.modelVersion ?? null,
            response_id: events.find(e => e.responseId)?.responseId ?? null,
            usage: native.usage, native_usage_metadata: native.native_usage_metadata,
            prompt_tokens_by_modality: native.prompt_tokens_by_modality, ...accounting,
            usage_schema: 'gemini-usage-metadata-v1', precision_status: config.precision_status,
            provider_identity_basis: 'Direct HTTPS endpoint and returned modelVersion; no provider field is returned',
            output_accepted_at_accounting: false, acceptance_record: 'Separate response receipt; billing knowledge is not output acceptance' });
          known = true;
          status.estimated_cost_usd += accounting.cost_estimate_usd; status.native_billing_unreported_attempts++;
        }
        return native;
      };
      const checkIdentities = (events, required = true) => {
        const models = events.map(e => e.modelVersion).filter(v => v !== undefined);
        if (required) assert(models.length, 'Missing response model identity');
        assert(models.every(v => config.accepted_response_models.includes(v)), 'Returned model substitution refused');
      };
      const providerErrors = events => {
        const classes = events.filter(e => e.error !== undefined).map(e => providerErrorClass(e, 200));
        return classes.find(c => c.permanent) ?? classes.find(c => !c.retryable) ?? classes[0] ?? null;
      };
      // A complete MAX_TOKENS or content-policy terminal belongs to Pi, never to identical-wire transport recovery.
      const terminalPolicyBoundary = events => events.some(e => (e.candidates ?? []).some(c => POLICY_FINISH.has(c?.finishReason)));
      try {
        write(path.join(output, 'api', 'request_' + number + '.json'), { category, accounting_id: accountingId,
          policy_request_id: policyRequestId, transport_attempt: attempt, body_sha256: bodySha256, body,
          route, image_parts: images.image_parts, image_base64_bytes: images.image_base64_bytes,
          attempt_timeout_ms: ATTEMPT_TIMEOUT_MS, attempt_timeout_mode: 'header_connect_then_body_inactivity_reset_on_received_progress',
          sdk_timeout_ms: SDK_NO_TIMEOUT_MS, retry_owner: 'this_transport_only', fetch_scope: 'session_scoped_global_fetch' });
      } catch {
        if (category !== 'native_cache_warm') status.blocked = true;
        throw new Error('Local request capture failed before network invocation or deadline creation');
      }
      const started = now(), deadline = attemptDeadline(init.signal, clock);
      try {
        assert.equal(sha(init.body), bodySha256, 'Retry changed the original serialized request');
        sent = true; status.requests++;
        let response;
        calling = true;
        try { response = await deadline.network(() => fetchImpl(url, { ...init, signal: deadline.signal }), 'fetch'); }
        finally { calling = false; }
        deadline.progress();
        receivedHttp = true; metadata = safeResponseMetadata(response, now());
        write(path.join(output, 'api', 'response_' + number + '_metadata.json'), { category, ...metadata });
        const bytes = await bufferedBody(response, deadline, chunks);
        if (!response.ok) {
          let fields = {}, rateLimit = 'unknown', value = {};
          try { value = JSON.parse(bytes.toString('utf8')); fields = safeErrorFields(value); rateLimit = rateLimitClass(value); } catch {}
          providerFailure = providerErrorClass(value, response.status);
          const googleStatus = typeof value?.error?.status === 'string' && /^[A-Z_]{1,64}$/.test(value.error.status) ? value.error.status : null;
          write(path.join(output, 'api', 'http_error_' + number + '.json'), { category, status: response.status,
            fields, google_status: googleStatus, provider_error_class: providerFailure, rate_limit_classification: rateLimit,
            classification_basis: 'Allowlisted categories only; no raw error body, message or arbitrary headers persisted' });
          throw new WireFailure('provider_error');
        }
        const capture = geminiWireCapture(bytes, true, key);
        const captureFile = 'response_' + number + (capture.receipt.capture_class === 'complete_raw_sse' ? '.sse' : '_unaccepted_prefix.bin');
        fs.writeFileSync(path.join(output, 'api', captureFile), capture.bytes, { flag: 'wx', mode: 0o600 });
        write(path.join(output, 'api', 'capture_' + number + '_receipt.json'), { file: captureFile, ...capture.receipt });
        wireSaved = true;
        const wire = bytes.toString('utf8'), events = parseSSE(wire);
        // A valid terminal usage frame can settle billing even when the output is rejected.
        if (terminalUsageMetadata(events)) account(events);
        providerFailure = providerErrors(events);
        assert(capture.receipt.omission !== 'error_frame_and_following_bytes_omitted' || providerFailure, 'Unclassified error frame cannot be accepted');
        checkIdentities(events, providerFailure === null);
        const policyBoundary = terminalPolicyBoundary(events);
        if (providerFailure) {
          if (policyBoundary) providerFailure = { ...providerFailure, permanent: true, retryable: false };
          throw new WireFailure('provider_error');
        }
        assert(capture.receipt.omission === null, 'Unsupported or incomplete SSE protocol frame');
        const validated = validateCompleteGeminiStream(events);
        const native = account(events);
        // Caller cancellation wins even if the response completed concurrently.
        if (init.signal?.aborted) throw new WireFailure('caller_cancelled');
        write(path.join(output, 'api', 'response_' + number + '_receipt.json'), { category, accounting_id: accountingId,
          policy_request_id: policyRequestId, output_accepted: true, body_complete: true,
          route, image_parts_sent: images.image_parts,
          response_model: validated.model_version, response_id: validated.response_id, finish_reasons: validated.finish_reasons,
          function_call_count: validated.function_calls.length,
          function_call_ids_provided: validated.function_calls.filter(item => item.id !== null).length,
          prompt_tokens_by_modality: native.prompt_tokens_by_modality, usage: native.usage,
          capture: { file: captureFile, ...capture.receipt },
          raw_sse_sha256: capture.receipt.capture_class === 'complete_raw_sse' ? capture.receipt.retained_sha256 : null,
          reasoning_replay: 'Native Pi Google codec; thought signatures retained without analyst interpretation' });
        // Pi maps MALFORMED_FUNCTION_CALL, OTHER and unknown enums to an error stop; the completion record keeps the raw reason.
        status.accepted_outputs++; status.last_accepted_finish_reasons = validated.finish_reasons;
        // The original body is consumed. Pi receives only this validated buffered
        // response; no prefix or transport-owned upstream stream reaches tools.
        return new Response(bytes, { status: 200, headers: { 'content-type': 'text/event-stream' } });
      } catch (caught) {
        let error = caught, partialPolicyBoundary = false;
        // Preserve bytes actually read before a network/body interruption, without
        // promoting them to a complete response or delivering them to the model.
        if (receivedHttp && metadata?.status === 200 && chunks.length && !wireSaved) {
          try {
            const partial = Buffer.concat(chunks);
            const capture = geminiWireCapture(partial, false, key);
            const file = 'response_' + number + '_partial.bin';
            fs.writeFileSync(path.join(output, 'api', file), capture.bytes, { flag: 'wx', mode: 0o600 });
            const prefix = partial.toString('utf8').split(/(\r?\n\r?\n)/);
            const events = parseSSE(prefix.slice(0, -1).join(''));
            // Intermediate usage without a terminal frame leaves the charge unknown.
            if (terminalUsageMetadata(events)) account(events);
            checkIdentities(events, false);
            providerFailure = providerErrors(events);
            assert(capture.receipt.omission !== 'error_frame_and_following_bytes_omitted' || providerFailure, 'Unclassified partial error frame');
            assert(capture.receipt.omission !== 'unclassified_frame_and_following_bytes_omitted', 'Malformed complete partial-stream frame');
            partialPolicyBoundary = terminalPolicyBoundary(events);
            write(path.join(output, 'api', 'partial_' + number + '_receipt.json'), {
              accounting_id: accountingId, policy_request_id: policyRequestId, file,
              ...capture.receipt, output_accepted: false,
              complete_frames_inspected: events.length, usage_known: known });
          } catch (partialError) { error = partialError; } // malformed prefix/identity is not a transport excuse.
        }
        deadline.cleanup(); // No per-attempt timer survives into backoff.
        const wireKind = error instanceof WireFailure ? error.kind : null;
        // Stop after two unexplained pre-HTTP failures (comparison_transport.mjs policy).
        unclassifiedFetchFailures = wireKind === 'native_fetch_failed' ? unclassifiedFetchFailures + 1 : 0;
        const diagnosisRequired = unclassifiedFetchFailures >= 2;
        // A per-day or billing RESOURCE_EXHAUSTED is also a 429; bounded like the chat routes (vision_transport.mjs).
        if (wireKind === 'provider_error') providerErrorFailures++;
        const persistentProviderError = wireKind === 'provider_error' && providerErrorFailures > MAX_PROVIDER_ERROR_RETRIES;
        const networkRetry = ['attempt_timeout', 'connection_failure', 'body_connection_interrupted', 'native_fetch_failed'].includes(wireKind);
        const permanentHttp = metadata && metadata.status !== 200 && !TRANSIENT_HTTP.has(metadata.status);
        const retryable = category !== 'native_cache_warm' && sent && !diagnosisRequired && !persistentProviderError && !init.signal?.aborted && !status.resourceCensored && !partialPolicyBoundary &&
          !permanentHttp && !providerFailure?.permanent &&
          ((wireKind === 'provider_error' && providerFailure?.retryable) || networkRetry);
        if (category !== 'native_cache_warm') status.blocked = !retryable;
        if (sent && !known) {
          status.usage_unresolved = true;
          append(path.join(output, 'usage.jsonl'), { accounting_id: accountingId, number, category,
            policy_request_id: policyRequestId, transport_attempt: attempt, status: 'unknown', native_cost_usd: null,
            provider: 'google', ...route,
            network_request_sent: true, network_invocation_marker_not_transmission_proof: true,
            output_accepted_at_accounting: false, followup_inference_blocked: category !== 'native_cache_warm' && !retryable,
            owner_reviewed_retry_exception: retryable ? RECOVERY_CONTRACT + '; identical unaccepted request only' : null });
        }
        write(path.join(output, 'api', 'failure_' + number + '.json'), { category,
          error_type: wireKind ? 'WireFailure' : ['TypeError', 'SyntaxError', 'AssertionError', 'Error'].includes(error?.name) ? error.name : 'other_error',
          reason: wireKind ?? 'Local or response-contract failure',
          network_request_sent: sent, network_invocation_marker_not_transmission_proof: true,
          received_http_response: receivedHttp, usage_known: known, output_accepted: false,
          network_code: error instanceof WireFailure ? error.network_code : null,
          safe_cause_classes: error instanceof WireFailure ? error.cause_classes : [],
          network_boundary: error instanceof WireFailure ? error.boundary : null, provider_error_class: providerFailure,
          elapsed_ms: now() - started, request_body_bytes: Buffer.byteLength(init.body), received_body_bytes: Buffer.concat(chunks).length,
          diagnosis_required: diagnosisRequired, consecutive_unclassified_fetch_failures: unclassifiedFetchFailures,
          provider_error_failures: providerErrorFailures, persistent_provider_error: persistentProviderError,
          request_signal_aborted: Boolean(init.signal?.aborted), attempt_timed_out: deadline.timedOut,
          policy_request_id: policyRequestId, transport_attempt: attempt, scientific_negative: false, retry: retryable });
        const delay = retryable ? recoveryDelay(attempt, metadata?.retry_after, random, metadata?.status === 429 ? RATE_LIMIT_RETRY_FLOOR_MS : 0) : null;
        publishRetryStatus(output, status, { policy_request_id: policyRequestId, transport_attempt: attempt, number,
          reason: wireKind ?? 'local_or_response_contract_failure', http_status: metadata?.status ?? null,
          provider_error_class: providerFailure, provider_error_failures: providerErrorFailures,
          persistent_provider_error: persistentProviderError, diagnosis_required: diagnosisRequired,
          retry: retryable, next_delay_ms: delay?.milliseconds ?? null });
        if (!retryable) throw new Error('Recorded permanent transport/response failure or cancellation');
        const waitStarted = now();
        write(path.join(output, 'api', 'retry_' + number + '.json'), { category, policy_request_id: policyRequestId,
          failed_accounting_id: accountingId, attempt_completed: attempt, next_attempt: attempt + 1,
          identical_body_sha256: bodySha256, ...delay, recovery_contract: RECOVERY_CONTRACT,
          timeout_note: 'Per-wire header/body-inactivity timer cleaned up; next attempt gets a fresh300s signal. Body progress resets inactivity; caller cancellation stays authoritative.',
          prior_charge_status: known ? 'known' : 'unknown', unknown_charge_never_erased: true });
        try {
          await waitImpl(delay.milliseconds, init.signal);
          if (init.signal?.aborted) throw new Error('Caller cancelled during backoff');
          write(path.join(output, 'api', 'retry_wait_' + number + '.json'), { policy_request_id: policyRequestId,
            requested_delay_ms: delay.milliseconds, elapsed_ms: now() - waitStarted, completed: true });
        } catch {
          status.blocked = true;
          write(path.join(output, 'api', 'retry_aborted_' + number + '.json'), { policy_request_id: policyRequestId,
            request_signal_aborted: Boolean(init.signal?.aborted), followup_request_sent: false, scientific_negative: false });
          throw new Error('Retry pacing interrupted');
        }
      } finally { deadline.cleanup(); }
    }
  };
  // Non-enumerable so status copies and merges never carry it; idempotent per installation.
  Object.defineProperty(status, 'dispose', { enumerable: false, value: () => {
    if (disposed) return;
    disposed = true; active = null;
    if (globalThis.fetch === geminiVisionFetch) globalThis.fetch = previousFetch;
    installed = false;
  } });
  globalThis.fetch = geminiVisionFetch;
  return status;
}
