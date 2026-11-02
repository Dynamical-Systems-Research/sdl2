/** Direct Anthropic codec in the existing Pi runtime. Scientific interfaces are reused unchanged. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {Bridge, SessionManager, createSession, serialTools, investigationContext,
  parseSSE, sha, writeRecord as write, appendRecord as append, attemptDeadline,
  bufferedBody, recoveryDelay, retryAfter as parseRetryAfter, waitForRetry, TRANSIENT_HTTP, SDK_NO_TIMEOUT_MS} from './vision_transport.mjs';
export {Bridge, SessionManager, createSession, serialTools};
const HERE = path.dirname(fileURLToPath(import.meta.url));
export function modelConfig(id) {
  assert.equal(id, 'claude-opus-5-5');
  const catalog = JSON.parse(fs.readFileSync(path.join(HERE, 'node_modules/@earendil-works/pi-ai/dist/providers/data/anthropic.json')));
  const model = catalog['anthropic-messages']['chat:' + id];
  return {model, api_key_env: 'ANTHROPIC_API_KEY', inference_route: 'anthropic-direct',
    expected_provider: 'Anthropic', canonical_slug: id,
    endpoint: {url: 'https://api.anthropic.com/v1/messages', provider_name: 'Anthropic', model: id},
    pricing_note: 'Installed Pi catalog; token-derived estimate, not provider billed cost.'};
}

export function completedAnthropic(events, model) {
  const starts = events.filter(e => e.type === 'message_start');
  assert.equal(starts.length, 1, 'Expected one Anthropic message');
  assert.equal(starts[0].message.model, model.id, 'Returned model mismatch');
  assert(!events.some(e => e.type === 'error'), 'Provider stream error');
  assert.equal(events.filter(e => e.type === 'message_stop').length, 1, 'Incomplete Anthropic response');
  assert.equal(events.at(-1).type, 'message_stop', 'Unexpected trailing response data');
  const terminal = events.findLast(e => e.type === 'message_delta' && e.delta?.stop_reason);
  assert(terminal, 'Missing native stop reason');
  const opened = new Set(), closed = new Set(), toolBlocks = new Map();
  for (const e of events) {
    if (e.type === 'content_block_start') {
      assert(!opened.has(e.index)); opened.add(e.index);
      if (e.content_block?.type === 'tool_use') toolBlocks.set(e.index, {initial: e.content_block.input, json: ''});
    }
    if (e.type === 'content_block_delta' && e.delta?.type === 'input_json_delta') {
      assert(toolBlocks.has(e.index), 'Tool arguments without tool block');
      toolBlocks.get(e.index).json += e.delta.partial_json;
    }
    if (e.type === 'content_block_stop') { assert(opened.has(e.index)); closed.add(e.index); }
  }
  assert.deepEqual(opened, closed, 'Incomplete content block');
  // Pi handles length-truncated tools as errors rather than executing partial arguments.
  if (terminal.delta.stop_reason !== 'max_tokens') for (const t of toolBlocks.values()) {
    const args = t.json ? JSON.parse(t.json) : t.initial;
    assert(args && typeof args === 'object' && !Array.isArray(args), 'Incomplete tool JSON');
  }
  const usage = {...starts[0].message.usage};
  for (const e of events) if (e.type === 'message_delta' && e.usage) Object.assign(usage, e.usage);
  for (const name of ['input_tokens', 'output_tokens']) assert(Number.isSafeInteger(usage[name]) && usage[name] >= 0);
  const read = usage.cache_read_input_tokens ?? 0, created = usage.cache_creation_input_tokens ?? 0;
  for (const n of [read, created]) assert(Number.isSafeInteger(n) && n >= 0);
  const longWrite = usage.cache_creation?.ephemeral_1h_input_tokens ?? 0;
  const cost = (usage.input_tokens * model.cost.input + usage.output_tokens * model.cost.output +
    read * model.cost.cacheRead + (created - longWrite) * model.cost.cacheWrite + longWrite * model.cost.input * 2) / 1e6;
  return {usage, cost_estimate_usd: cost, response_id: starts[0].message.id, stop_reason: terminal.delta.stop_reason};
}

export function installTransport(session, output, instructions, config, {
  fetchImpl = fetch, key, now = Date.now, random = Math.random,
  waitImpl = waitForRetry, clock = {setTimeout, clearTimeout},
} = {}) {
  const original = session.modelRuntime.streamSimple.bind(session.modelRuntime);
  let index = fs.readdirSync(path.join(output, 'api')).filter(n => /^request_\d+\.json$/.test(n)).length;
  const status = {blocked: false, usage_unresolved: false, resourceCensored: false,
    known_cost_usd: 0, estimated_cost_usd: 0, native_billing_unreported_attempts: 0, requests: 0, accepted_outputs: 0};
  session.modelRuntime.streamSimple = (model, incoming, options = {}) => {
    assert.equal(model.id, config.model.id); assert.equal(model.provider, 'anthropic');
    const {normal, context} = investigationContext(incoming, instructions);
    const category = options.maxTokens === 1 ? 'native_cache_warm' : normal ? 'investigator' : 'native_compaction_or_summary';
    return original(model, context, {...options, reasoning: 'high', temperature: undefined,
      maxTokens: Math.min(options.maxTokens ?? 32768, 32768), maxRetries: 0, timeoutMs: SDK_NO_TIMEOUT_MS,
      fetch: async (input, init) => {
        const url = typeof input === 'string' ? input : input.url ?? String(input);
        assert.equal(url.split('?')[0], config.endpoint.url);
        const body = JSON.parse(init.body);
        assert.equal(body.model, model.id); assert.equal(body.output_config?.effort, 'high');
        assert.equal(body.thinking?.type, 'adaptive');
        assert(!key || !init.body.includes(key), 'Credential in request body');
        const headers = new Headers(init.headers);
        headers.set('x-api-key', key); headers.delete('authorization');
        for (let attempt = 1; ; attempt++) {
          const number = String(++index).padStart(6, '0'), accounting_id = crypto.randomUUID();
          const route = {inference_route: config.inference_route, endpoint_url: config.endpoint.url, wire_model_id: model.id};
          write(path.join(output, 'api', `request_${number}.json`), {category, accounting_id, transport_attempt: attempt,
            body_sha256: sha(init.body), body, route, retry_owner: 'transport'});
          status.requests++;
          const deadline = attemptDeadline(init.signal, clock);
          let http = null, retryAfter = null, accounted = false;
          try {
            const response = await deadline.network(() => fetchImpl(input, {...init, headers, redirect: 'error', signal: deadline.signal}), 'fetch');
            http = response.status; retryAfter = parseRetryAfter(response.headers.get('retry-after'), now()); deadline.progress();
            write(path.join(output, 'api', `response_${number}_metadata.json`), {status: http, request_id: response.headers.get('request-id')});
            const bytes = await bufferedBody(response, deadline, []);
            assert(!key || !bytes.includes(Buffer.from(key)), 'Credential echo refused');
            if (!response.ok) throw new Error('Provider HTTP response');
            const events = parseSSE(bytes.toString('utf8'));
            const result = completedAnthropic(events, model);
            fs.writeFileSync(path.join(output, 'api', `response_${number}.sse`), bytes, {flag: 'wx', mode: 0o600});
            append(path.join(output, 'usage.jsonl'), {accounting_id, category, status: 'known', ...route, ...result,
              billing_cost_status: 'not_reported_by_api', native_cost_usd: null, estimated_or_reserved: true});
            accounted = true; status.estimated_cost_usd += result.cost_estimate_usd;
            status.native_billing_unreported_attempts++; status.accepted_outputs++;
            return new Response(bytes, {status: http, headers: response.headers});
          } catch (error) {
            if (!accounted) {
              status.usage_unresolved = true;
              append(path.join(output, 'usage.jsonl'), {accounting_id, category, status: 'unknown', ...route,
                native_cost_usd: null, scientific_negative: false, http_status: http, error_type: error.constructor.name});
            }
            const retry = !init.signal?.aborted && attempt < 6 && (TRANSIENT_HTTP.has(http) ||
              (http === null && ['attempt_timeout', 'connection_failure', 'native_fetch_failed'].includes(error.kind)));
            write(path.join(output, 'api', `failure_${number}.json`), {http_status: http, retry, scientific_negative: false, error_type: error.constructor.name});
            if (!retry) { if (category !== 'native_cache_warm') status.blocked = true; throw new Error('Recorded Anthropic transport failure; inspect receipts'); }
            const delay = recoveryDelay(attempt, retryAfter, random, http === 429 ? 60000 : 0);
            append(path.join(output, 'transport_retries.jsonl'), {number, attempt, http_status: http, delay_ms: delay.milliseconds});
            deadline.cleanup();
            await waitImpl(delay.milliseconds, init.signal);
          } finally { deadline.cleanup(); }
        }
      },
    });
  };
  status.dispose = () => { session.modelRuntime.streamSimple = original; };
  return status;
}
