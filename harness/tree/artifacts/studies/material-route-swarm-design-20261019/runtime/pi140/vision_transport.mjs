/** Vision fork of comparison_transport.mjs for image-capable chat, Responses and Z.ai-direct routes.
 * comparison_transport.mjs and zai_direct_transport.mjs stay byte-identical for their retained studies.
 * OpenRouter pins and reasoning mode are study data (PIN records), not code.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { isDeepStrictEqual } from 'node:util';
import { pathToFileURL, fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export { SessionManager } from '@earendil-works/pi-coding-agent';
// Isolated Pi 1.3.0 candidate under artifacts/studies/<study>/runtime/pi130.
// Existing provider snapshots remain the compatibility baseline, not a fresh provider qualification.
export const REPO = path.resolve(HERE, '../../../../..');
export const PROVIDER_DIRECTORY = process.env.NDFEB_VISION_PROVIDER_DIRECTORY ??
  path.join(REPO, 'artifacts/studies/physical-fresh-20261012/runtime/providers');
export const COMPACTION_RESERVE_TOKENS = 65536;
export const BRIDGE_MODULE = 'ndfeb_sps.investigation.vision_bridge';
export const DIRECT_ZAI_ENDPOINT = 'https://api.z.ai/api/paas/v4/chat/completions';
export const DIRECT_ZAI_MODEL = 'glm-5.3-flash';
export const OPENROUTER_ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions';
export const VISION_MODELS = ['glm-5.3-flash', 'xiaomi/mimo-v2.6-pro', 'deepseek/deepseek-v4.1-flash', 'deepseek/deepseek-v4.1-flash-1017', 'gpt-6-astra',
  'qwen/qwen3.8-max-0902'];
export const PIN_SCHEMA = 'ndfeb-vision-provider-pin-v1';
const REASONING_REQUESTS = [{ effort: 'high' }, { enabled: true }, null];
export const BRIDGE_PYTHON = process.env.NDFEB_PYTHON || 'python3';
const PI = path.dirname(path.dirname(fileURLToPath(import.meta.resolve('@earendil-works/pi-coding-agent'))));
const { createAgentSession, DefaultResourceLoader, ModelRuntime, SettingsManager } =
  await import(pathToFileURL(path.join(PI, 'dist/index.js')));
const { createToolDefinition } = await import(pathToFileURL(path.join(PI, 'dist/core/tools/index.js')));
const { sandboxConfig, sandboxToolDefinitions } = await import(pathToFileURL(path.join(HERE, 'native_sandbox.mjs')));
const { InMemoryCredentialStore } = await import(pathToFileURL(path.join(path.dirname(fileURLToPath(import.meta.resolve('@earendil-works/pi-ai'))), 'auth/credential-store.js')));
export const PI_VERSION = JSON.parse(fs.readFileSync(path.join(PI, 'package.json'))).version;
export const utc = () => new Date().toISOString();
export const sha = value => crypto.createHash('sha256').update(value).digest('hex');
export const SDK_NO_TIMEOUT_MS = 2147483647; // Installed Pi's disabled-timeout convention, not infinity.
export const ATTEMPT_TIMEOUT_MS = 300000;
export const RECOVERY_CONTRACT = 'TRANSPORT.md';
export const CONTINUATION_NOTICE = '[Runtime output-limit notice] The preceding response reached its per-response output limit. Continue the same response from where it was truncated.';
export const writeRecord = (p, value) => fs.writeFileSync(p, JSON.stringify({ utc: utc(), ...value }, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
export const appendRecord = (p, value) => fs.appendFileSync(p, JSON.stringify({ utc: utc(), ...value }) + '\n', { mode: 0o600 });
const write = writeRecord, append = appendRecord;

/** Study PIN record for one OpenRouter model: endpoint tag, provider, snapshot files and reasoning mode. */
export function readPin(id, providerDirectory = PROVIDER_DIRECTORY) {
  const pinBytes = fs.readFileSync(path.join(providerDirectory, id.replaceAll('/', '__') + '__PIN.json'));
  const pin = JSON.parse(pinBytes);
  assert.equal(pin.schema, PIN_SCHEMA, 'Unknown provider PIN schema');
  assert.equal(pin.model_id, id, 'PIN record names another model');
  assert.equal(pin.provider_kind, 'openrouter', 'Only OpenRouter routes carry a PIN record');
  for (const field of ['provider_tag', 'provider_name', 'catalog_file', 'endpoints_file']) {
    assert(typeof pin[field] === 'string' && pin[field].length, `PIN record lacks ${field}`);
  }
  assert(!path.isAbsolute(pin.catalog_file) && !path.isAbsolute(pin.endpoints_file), 'PIN snapshot paths are repository-relative');
  assert(REASONING_REQUESTS.some(value => isDeepStrictEqual(pin.reasoning_request, value)),
    'PIN reasoning_request must be {effort:"high"}, {enabled:true} or null');
  return { pin, pinBytes };
}

export function modelConfig(id, { providerDirectory = PROVIDER_DIRECTORY } = {}) {
  if (id === 'gpt-6-astra') {
    const model = { id, name: 'GPT-6 Astra', api: 'openai-responses', provider: 'openai',
      baseUrl: 'https://api.openai.com/v1', reasoning: true, input: ['text', 'image'],
      contextWindow: 1048576, maxTokens: 32768,
      // Current OpenAI prompt-caching guide: GPT-5.6+ entries last at least
      // 30 minutes after a write/reuse. Native Pi uses this best-effort tier.
      promptCache: { short: 1800 },
      cost: { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
      thinkingLevelMap: { high: 'high' },
      compat: { supportsDeveloperRole: true, supportsMaxOutputTokens: true, supportsStrictMode: true } };
    return { model, routing: null, canonical_slug: id, expected_provider: 'OpenAI', api_key_env: 'OPENAI_API_KEY',
      inference_route: 'openai-direct',
      endpoint: { url: model.baseUrl + '/responses', provider_name: 'OpenAI', model: id },
      native_max_output_tokens: 128000, native_context_length: 1050000,
      pricing_source: 'https://developers.openai.com/api/docs/models/gpt-6-astra', pricing_checked_date: '2026-09-14',
      pricing_note: 'Native OpenAI token counts settle usage only. USD estimates use dated Standard rates, including long-context multipliers; they are not provider billed cost. Pi display rates alone omit the long-context multiplier.',
      common_request: { reasoning: { effort: 'high', summary: 'auto' }, max_output_tokens: 32768, store: false },
      feature_limit: 'Existing serial scientific tools, native Pi compaction and response ceiling retained. No model/provider fallback.' };
  }
  if (id === DIRECT_ZAI_MODEL) {
    const model = { id, name: 'GLM-5.3-Flash (Z.ai direct; precision unattested)',
      api: 'openai-completions', provider: 'zai', baseUrl: 'https://api.z.ai/api/paas/v4',
      reasoning: true, input: ['text', 'image'], contextWindow: 1048576, maxTokens: 32768,
      cost: { input: 0.15, output: 0.50, cacheRead: 0.03, cacheWrite: 0 },
      thinkingLevelMap: { high: 'high' },
      compat: { supportsDeveloperRole: false, thinkingFormat: 'zai', maxTokensField: 'max_tokens',
        supportsStore: false, supportsUsageInStreaming: true, supportsReasoningEffort: true,
        requiresReasoningContentOnAssistantMessages: true } };
    return { model, routing: null, canonical_slug: id, expected_provider: 'Z.ai Direct',
      api_key_env: 'ZAI_API_KEY', inference_route: 'zai-direct',
      endpoint: { url: DIRECT_ZAI_ENDPOINT, provider_name: 'Z.ai Direct', model: id },
      precision_status: 'unattested: Z.ai direct does not state FP8/BF16',
      native_context_length: 1048576, native_max_output_tokens: 131072,
      pricing_source: 'https://docs.z.ai/guides/overview/pricing', pricing_checked_date: '2026-09-18',
      pricing_note: 'Token-derived estimate at published direct Flash rates; no provider billed dollar amount is inferred',
      common_request: { thinking: { type: 'enabled', clear_thinking: false }, reasoning_effort: 'high', temperature: 1, max_tokens: 32768 },
      feature_limit: 'Same serial scientific tools and native Pi compaction. Direct provider identity; no provider/model fallback or response rewriting.' };
  }
  assert(VISION_MODELS.includes(id), 'Exact requested model required; no substitution');
  const { pin, pinBytes } = readPin(id, providerDirectory);
  const catalogPath = path.join(REPO, pin.catalog_file);
  const catalogBytes = fs.readFileSync(catalogPath);
  const row = JSON.parse(catalogBytes).data.find(r => r.id === id);
  assert(row, 'Model absent from frozen live catalog');
  const endpointPath = path.join(REPO, pin.endpoints_file);
  const endpointBytes = fs.readFileSync(endpointPath);
  const endpoint = JSON.parse(endpointBytes).data.endpoints.find(e => e.tag === pin.provider_tag);
  assert(endpoint && endpoint.provider_name === pin.provider_name, 'Pinned endpoint absent from frozen endpoint metadata');
  for (const feature of ['tools', 'tool_choice', 'reasoning', 'temperature', 'max_tokens']) {
    assert(endpoint.supported_parameters.includes(feature), `Endpoint lacks ${feature}`);
  }
  // 1,048,576 tokens, or less where the pinned endpoint accepts fewer prompt tokens (Qwen 3.8 Max: 983,616).
  const contextWindow = Math.min(1048576, endpoint.context_length, endpoint.max_prompt_tokens ?? Infinity);
  assert(contextWindow >= 983616, 'Pinned endpoint context below the fresh-cohort floor');
  const pricing = endpoint.pricing;
  const routing = { only: [pin.provider_tag], order: [pin.provider_tag], allow_fallbacks: false,
    require_parameters: true, data_collection: 'deny' };
  // Explicit prompt caching is study data in the PIN record (Alibaba's Qwen endpoint caches only marked prefixes).
  const cacheControl = pin.cache_control === 'anthropic' ? { cacheControlFormat: 'anthropic' } : {};
  const model = { id, name: row.name, api: 'openai-completions', provider: 'openrouter',
    baseUrl: 'https://openrouter.ai/api/v1', reasoning: true, input: ['text', 'image'],
    contextWindow, maxTokens: 32768,
    cost: { input: Number(pricing.prompt) * 1e6, output: Number(pricing.completion) * 1e6,
      cacheRead: Number(pricing.input_cache_read ?? 0) * 1e6, cacheWrite: Number(pricing.input_cache_write ?? 0) * 1e6 },
    thinkingLevelMap: { off: null, minimal: null, low: 'low', medium: null, high: 'high', xhigh: null, max: 'max' },
    compat: { supportsDeveloperRole: false, thinkingFormat: 'openrouter', maxTokensField: 'max_tokens',
      supportsStore: false, supportsUsageInStreaming: false, supportsReasoningEffort: true,
      requiresReasoningContentOnAssistantMessages: false, openRouterRouting: routing, ...cacheControl } };
  return { model, routing, canonical_slug: row.canonical_slug, endpoint: { ...endpoint, url: OPENROUTER_ENDPOINT },
    expected_provider: pin.provider_name, api_key_env: 'OPENROUTER_API_KEY', inference_route: 'openrouter',
    native_max_output_tokens: endpoint.max_completion_tokens, reasoning_request: pin.reasoning_request,
    pin_record: pin, pin_record_sha256: sha(pinBytes),
    catalog_sha256: sha(catalogBytes), endpoint_sha256: sha(endpointBytes),
    pricing_note: 'Pi display cost uses dated catalog rates only; authoritative accounting uses raw OpenRouter usage.cost including any discounts',
    common_request: { ...(pin.reasoning_request ? { reasoning: pin.reasoning_request } : {}), temperature: 1, max_tokens: 32768 },
    feature_limit: 'Neither exact direct endpoint advertises parallel_tool_calls; flag omitted. Tool mutations execute serially in Pi and a bridge queue.' };
}

export function bridgeEnvironment(parent = process.env) {
  const env = {};
  for (const k of ['PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME', 'NDFEB_SCIENCE_IMAGE', 'NDFEB_SCIENCE_PROFILE']) if (parent[k] !== undefined) env[k] = parent[k];
  env.PYTHONDONTWRITEBYTECODE = '1';
  return env;
}
export class Bridge {
  constructor() {
    this.child = spawn(BRIDGE_PYTHON, ['-B', '-m', BRIDGE_MODULE], {
      cwd: HERE, env: bridgeEnvironment(), stdio: ['pipe', 'pipe', 'pipe'] });
    this.closed = new Promise(resolve => this.child.once('close', resolve));
    this.pending = new Map(); this.sequence = 0; this.stderr = '';
    createInterface({ input: this.child.stdout }).on('line', line => {
      let item;
      try { item = JSON.parse(line); } catch { this.rejectAll(new Error('Invalid bridge JSON')); return; }
      const p = this.pending.get(item.id); if (!p) return;
      this.pending.delete(item.id);
      item.error ? p.reject(Object.assign(new Error('Bridge tool rejected'), { bridgeError: item.error })) : p.resolve(item.result);
    });
    this.child.stderr.on('data', b => { this.stderr = (this.stderr + b.toString()).slice(-32000); });
    this.child.on('exit', code => this.rejectAll(new Error(`Bridge exited ${code}`)));
    this.child.on('error', () => this.rejectAll(new Error('Bridge launch failed')));
  }
  rejectAll(e) { for (const p of this.pending.values()) p.reject(e); this.pending.clear(); }
  call(method, params = {}) {
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(JSON.stringify({ id, method, params }) + '\n');
    });
  }
  interrupt() {
    // Python handles SIGINT through finally blocks, including exact-container
    // cleanup. Do not SIGKILL a bridge while its scientific job is still owned.
    if (this.child.exitCode === null && this.child.signalCode === null) this.child.kill('SIGINT');
  }
  async close() { this.child.stdin.end(); await this.closed; }
}

// Native Pi implementations: no shell proxy, custom filesystem layer or network filter.
export const NATIVE_TOOL_NAMES = ['read', 'bash', 'edit', 'write', 'grep', 'find', 'ls'];
export function nativeToolSpecs(workspace) {
  return NATIVE_TOOL_NAMES.map(name => {
    const { description, parameters } = createToolDefinition(name, workspace);
    return { name, description, parameters };
  });
}

export async function createSession({ output, instructions, tools, manager, config, key = 'offline-test-placeholder', nativeTools = false }) {
  const settings = { defaultProjectTrust: 'never', transport: 'sse', cacheWarming: 'streaming', enableInstallTelemetry: false, enableAnalytics: false,
    retry: { enabled: false, maxRetries: 0, provider: { maxRetries: 0, timeoutMs: SDK_NO_TIMEOUT_MS } },
    // Native Pi compares context tokens with contextWindow - reserveTokens.
    // Reserve 65,536 so compaction precedes the provider prompt limit with image underestimation margin (integ F9.6).
    compaction: { enabled: true, reserveTokens: COMPACTION_RESERVE_TOKENS, keepRecentTokens: 20000 } };
  const settingsManager = SettingsManager.inMemory(settings);
  const agentDir = path.join(output, 'isolated_pi_config'); fs.mkdirSync(agentDir, { recursive: true });
  const credentials = new InMemoryCredentialStore();
  const runtime = await ModelRuntime.create({ credentials, modelsPath: null,
    modelsStorePath: path.join(agentDir, 'models-cache.json'), allowModelNetwork: false, refreshOnCreate: false });
  runtime.registerProvider(config.model.provider, { api: config.model.api, baseUrl: config.model.baseUrl, models: [config.model] });
  await runtime.setRuntimeApiKey(config.model.provider, key);
  const loader = new DefaultResourceLoader({ cwd: path.join(output, 'workspace'), agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: instructions });
  await loader.reload();
  const activeTools = [...(nativeTools ? NATIVE_TOOL_NAMES : []), ...tools.map(t => t.name)];
  // Native tools execute inside the investigation's sandbox when NDFEB_NATIVE_SANDBOX names one (native_sandbox.mjs).
  const sandboxed = nativeTools && sandboxConfig() ? sandboxToolDefinitions(path.join(output, 'workspace'), NATIVE_TOOL_NAMES) : [];
  const { session, modelFallbackMessage } = await createAgentSession({ cwd: path.join(output, 'workspace'), agentDir,
    modelRuntime: runtime, model: config.model, thinkingLevel: 'high', tools: activeTools, customTools: [...sandboxed, ...tools],
    resourceLoader: loader, settingsManager, sessionManager: manager });
  assert(!modelFallbackMessage); assert.equal(session.model.id, config.model.id);
  assert.deepEqual(session.getActiveToolNames().sort(), [...activeTools].sort());
  session.subscribe(event => append(path.join(output, 'pi_events.jsonl'), { event }));
  installOutputContinuation(session, output);
  return { session, settings, settingsManager };
}

export function installOutputContinuation(session, output) {
  // Pi's core rejects every tool in a length-truncated message and continues
  // with its own tool-error results. A text-only length needs a native queued
  // follow-up; otherwise a response exactly at maxTokens ends the native run.
  // This fixed runtime notice adds no task, evidence or scientific direction.
  return session.subscribe(event => {
    if (event.type !== 'turn_end' || event.message?.stopReason !== 'length') return;
    const calls = event.message.content.filter(item => item.type === 'toolCall');
    append(path.join(output, 'output_limit_continuations.jsonl'), {
      route: calls.length ? 'native_truncated_tool_feedback' : 'native_follow_up_queue',
      message_timestamp: event.message.timestamp, tool_call_ids: calls.map(call => call.id),
      notice: calls.length ? null : CONTINUATION_NOTICE,
      transport_retry: false, scientific_instruction: false,
    });
    if (!calls.length) session.agent.followUp({ role: 'user',
      content: [{ type: 'text', text: CONTINUATION_NOTICE }], timestamp: Date.now() });
  });
}

const SSE_EVENT_NAME = Symbol('transport_sse_event_name');
export function parseSSE(wire) {
  return wire.split(/\r?\n\r?\n/).flatMap(block => {
    const data = block.split(/\r?\n/).filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
    if (!data || data === '[DONE]') return [];
    const event = JSON.parse(data);
    const name = block.split(/\r?\n/).filter(l => l.startsWith('event:')).at(-1)?.slice(6).trim();
    if (name) Object.defineProperty(event, SSE_EVENT_NAME, { value: name });
    return [event];
  });
}
export function rawUsage(events, provider = 'openrouter') {
  if (provider === 'zai') {
    const usage = events.findLast(e => e.usage)?.usage;
    assert(usage, 'Missing native usage');
    for (const field of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
      assert(Number.isSafeInteger(usage[field]) && usage[field] >= 0, `Unknown raw ${field}`);
    }
    assert.equal(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens, 'Inconsistent native token total');
    const cached = usage.prompt_tokens_details?.cached_tokens;
    if (cached !== undefined) assert(Number.isSafeInteger(cached) && cached >= 0 && cached <= usage.prompt_tokens, 'Invalid native cached_tokens');
    const reasoning = usage.completion_tokens_details?.reasoning_tokens;
    if (reasoning !== undefined) assert(Number.isSafeInteger(reasoning) && reasoning >= 0 && reasoning <= usage.completion_tokens, 'Invalid native reasoning_tokens');
    return usage;
  }
  const native = events.findLast(e => e.usage || e.response?.usage);
  const reported = native?.usage ?? native?.response?.usage;
  const usage = provider === 'openai' && reported?.input_tokens !== undefined ? { ...reported,
    prompt_tokens: reported.input_tokens, completion_tokens: reported.output_tokens,
    prompt_tokens_details: reported.input_tokens_details, completion_tokens_details: reported.output_tokens_details } : reported;
  assert(usage, 'Missing native usage');
  for (const k of ['prompt_tokens', 'completion_tokens', 'total_tokens']) assert(Number.isInteger(usage[k]) && usage[k] >= 0, `Unknown raw ${k}`);
  if (provider === 'openai') {
    assert.equal(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens, 'Inconsistent native token total');
    for (const [details, fields, ceiling] of [
      [usage.prompt_tokens_details, ['cached_tokens', 'cache_write_tokens'], usage.prompt_tokens],
      [usage.completion_tokens_details, ['reasoning_tokens'], usage.completion_tokens],
    ]) for (const field of fields) if (details?.[field] !== undefined) {
      assert(Number.isInteger(details[field]) && details[field] >= 0 && details[field] <= ceiling, `Invalid native ${field}`);
    }
    assert((usage.prompt_tokens_details?.cached_tokens ?? 0) + (usage.prompt_tokens_details?.cache_write_tokens ?? 0) <= usage.prompt_tokens, 'Overlapping cache token counts');
  } else {
    assert.equal(provider, 'openrouter', 'Unsupported accounting provider');
    assert(typeof usage.cost === 'number' && Number.isFinite(usage.cost) && usage.cost >= 0, 'Unknown native usage.cost');
  }
  return usage;
}

export function openAICostEstimate(usage, config = modelConfig('gpt-6-astra')) {
  rawUsage([{ usage }], 'openai');
  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0;
  const writes = usage.prompt_tokens_details?.cache_write_tokens ?? 0;
  const longContext = usage.prompt_tokens > 272000;
  const rates = config.model.cost, inputMultiplier = longContext ? 2 : 1, outputMultiplier = longContext ? 1.5 : 1;
  return { cost_estimate_usd: ((usage.prompt_tokens - cached - writes) * rates.input * inputMultiplier +
      cached * rates.cacheRead * inputMultiplier + writes * rates.cacheWrite * inputMultiplier +
      usage.completion_tokens * rates.output * outputMultiplier) / 1e6,
    cost_basis: 'Estimated OpenAI Standard text-token pricing; not provider billed cost',
    native_cost_usd: null, billing_cost_status: 'not_reported_by_api', estimated_or_reserved: true,
    pricing_source: config.pricing_source, pricing_checked_date: config.pricing_checked_date,
    estimate_assumptions: { service_tier: 'default', long_context: longContext,
      input_multiplier: inputMultiplier, output_multiplier: outputMultiplier,
      unreported_cache_reads_treated_as_uncached: usage.prompt_tokens_details?.cached_tokens === undefined,
      unreported_cache_writes_not_estimated: usage.prompt_tokens_details?.cache_write_tokens === undefined,
      reasoning_tokens_already_in_completion_tokens: true } };
}

export function directZaiCostEstimate(usage, config = modelConfig(DIRECT_ZAI_MODEL)) {
  rawUsage([{ usage }], 'zai');
  const rates = config.model.cost;
  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0;
  return { cost_estimate_usd: ((usage.prompt_tokens - cached) * rates.input + cached * rates.cacheRead + usage.completion_tokens * rates.output) / 1e6,
    cost_basis: 'Direct Z.ai published Flash token estimate; cache discounts only for native reported cached_tokens; not provider billed cost',
    native_cost_usd: null, billing_cost_status: 'not_reported_by_api', estimated_or_reserved: true,
    pricing_source: config.pricing_source, pricing_checked_date: config.pricing_checked_date,
    estimate_assumptions: { input_usd_per_million: rates.input, output_usd_per_million: rates.output,
      published_cache_read_usd_per_million: rates.cacheRead, cached_tokens: cached,
      missing_cache_tokens_treated_as_uncached: usage.prompt_tokens_details?.cached_tokens === undefined,
      reasoning_tokens_already_in_completion_tokens: true } };
}

/** Image parts in a parsed wire body: chat/Responses data-URL image_url and Gemini inlineData (same walker as the budget guard). */
export function wireImageParts(body) {
  let image_parts = 0, image_base64_bytes = 0;
  const visit = value => {
    if (Array.isArray(value)) { for (const item of value) visit(item); return; }
    if (!value || typeof value !== 'object') return;
    for (const [name, item] of Object.entries(value)) {
      const url = name === 'image_url' ? (typeof item === 'string' ? item : item?.url) : undefined;
      if (typeof url === 'string' && url.startsWith('data:')) {
        image_parts++; image_base64_bytes += url.includes(',') ? url.length - url.indexOf(',') - 1 : 0;
      } else if (name === 'inlineData' && typeof item?.data === 'string') {
        image_parts++; image_base64_bytes += item.data.length;
      } else visit(item);
    }
  };
  visit(body);
  return { image_parts, image_base64_bytes };
}

export function safeErrorFields(value) {
  const result = {};
  for (const key of ['code', 'type', 'param']) {
    const item = value?.error?.[key];
    if (typeof item === 'number' && Number.isFinite(item)) result[key] = item;
    else if (typeof item === 'string' && /^[A-Za-z0-9_.\[\]-]{1,100}$/.test(item) && !item.startsWith('sk-')) result[key] = item;
  }
  return result;
}

export function rateLimitClass(value) {
  const message = typeof value?.error?.message === 'string' ? value.error.message : typeof value?.message === 'string' ? value.message : '';
  const categories = [
    ['rpm', /\brpm\b|requests? (?:per|\/) minute|requests?\/min/i],
    ['tpm', /\btpm\b|tokens? (?:per|\/) minute|tokens?\/min/i],
    ['daily', /\bdaily\b|per day|\brpd\b|\btpd\b/i],
  ].filter(([, pattern]) => pattern.test(message)).map(([name]) => name);
  return categories.length === 1 ? categories[0] : 'unknown';
}

// EPIPE and UND_ERR_CLOSED are how a server or proxy cutting a large image-bearing upload mid-body surfaces.
const NETWORK_CODES = new Set(['ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EHOSTUNREACH', 'ENETUNREACH',
  'EAI_AGAIN', 'ENOTFOUND', 'UND_ERR_SOCKET', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT',
  'EPIPE', 'UND_ERR_CLOSED']);
const PERMANENT_NETWORK_CODES = new Set(['ERR_TLS_CERT_ALTNAME_INVALID', 'CERT_HAS_EXPIRED',
  'CERT_NOT_YET_VALID', 'DEPTH_ZERO_SELF_SIGNED_CERT', 'SELF_SIGNED_CERT_IN_CHAIN',
  'UNABLE_TO_VERIFY_LEAF_SIGNATURE', 'UNABLE_TO_GET_ISSUER_CERT_LOCALLY',
  'UND_ERR_REQ_CONTENT_LENGTH_MISMATCH', 'UND_ERR_INVALID_ARG', 'UND_ERR_NOT_SUPPORTED',
  'ERR_INVALID_URL', 'ERR_INVALID_ARG_TYPE', 'ERR_INVALID_ARG_VALUE', 'ERR_UNESCAPED_CHARACTERS']);
// A corrupted TLS record on an otherwise healthy route is a connection failure, not a local client defect.
const TLS_TRANSIENT_CODES = new Set(['ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC', 'ERR_SSL_DECRYPTION_FAILED_OR_BAD_RECORD_MAC',
  'ERR_SSL_TLSV1_ALERT_BAD_RECORD_MAC']);
const BAD_RECORD_MAC = /bad[ _]record[ _]mac/i;
const CAUSE_PATTERNS = [
  ['request_content_length_mismatch', /request body length does not match content-length header/i],
  ['request_body_consumed', /body (?:has already been|is unusable|already) (?:read|consumed)|body object should not be disturbed or locked/i],
  ['request_duplex_required', /duplex option is required/i],
  ['tls_certificate_failure', /certificate has expired|self[- ]signed certificate|unable to verify the first certificate|unable to get local issuer certificate/i],
  ['socket_disconnected', /other side closed|socket hang up/i],
  ['tls_bad_record_mac', BAD_RECORD_MAC],
  ['fetch_failed', /^fetch failed$/],
];
// Cloudflare origin errors (520-524) and 529 overload are transient outer statuses.
export const TRANSIENT_HTTP = new Set([408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529]);
const TRANSIENT_TYPES = new Set(['rate_limit_exceeded', 'provider_overloaded', 'provider_unavailable', 'server', 'timeout', 'server_error',
  'zai_concurrency_limit', 'zai_frequency_limit', 'zai_overloaded', 'zai_network_error']);
// Z.ai direct reports errors as numeric-string codes. Only its documented concurrency, frequency, overload and
// network codes are transient; balance (1113), model/parameter (1211, 1214), content (1301), quota and any
// unlisted code stay permanent as 'unclassified_typed_error'. Applied on the Z.ai route only.
export const ZAI_TRANSIENT_CODES = new Map([['1302', 'zai_concurrency_limit'], ['1303', 'zai_frequency_limit'],
  ['1305', 'zai_overloaded'], ['1234', 'zai_network_error']]);
// OpenAI reports a request larger than the whole per-minute budget as a 429 'tokens' limit; replaying it cannot succeed.
const REQUEST_TOO_LARGE = /request too large|must be reduced in order to run/i;
// Each unaccepted attempt is booked at full reservation by the budget guard, so one identical body is not
// replayed without end: the transport seals after this many provider-error retries of one request.
export const MAX_PROVIDER_ERROR_RETRIES = 6;
// A rate-limit Retry-After shorter than the first fallback step is not followed verbatim.
export const RATE_LIMIT_RETRY_FLOOR_MS = 60000;
const PERMANENT_TYPES = new Set(['context_length_exceeded', 'max_tokens_exceeded', 'token_limit_exceeded', 'string_too_long',
  'authentication', 'permission_denied', 'payment_required', 'invalid_request', 'invalid_prompt', 'not_found',
  'precondition_failed', 'payload_too_large', 'unprocessable', 'content_policy_violation', 'refusal',
  'invalid_image', 'image_too_large', 'image_too_small', 'unsupported_image_format', 'image_not_found', 'image_download_failed',
  'authentication_error', 'permission_error', 'billing_error', 'invalid_request_error', 'invalid_api_key', 'insufficient_quota',
  'request_exceeds_rate_limit']);
function errorNodes(error) {
  const pending = [error], result = [], seen = new Set();
  for (let i = 0; i < pending.length && result.length < 16; i++) {
    const e = pending[i];
    if (!e || typeof e !== 'object' || seen.has(e)) continue;
    seen.add(e); result.push(e);
    if (e.cause) pending.push(e.cause);
    if (Array.isArray(e.errors)) pending.push(...e.errors.slice(0, 4));
  }
  return result;
}
export function networkCode(error) {
  const nodes = errorNodes(error);
  return nodes.map(e => e.code).find(code => NETWORK_CODES.has(code) || TLS_TRANSIENT_CODES.has(code)) ??
    (nodes.some(e => BAD_RECORD_MAC.test(typeof e.code === 'string' ? e.code : '') ||
      (typeof e.message === 'string' && BAD_RECORD_MAC.test(e.message))) ? 'TLS_BAD_RECORD_MAC' : null);
}
export function safeCauseClasses(error) {
  return errorNodes(error).map(e => ({
    type: ['Error', 'TypeError', 'AggregateError', 'DOMException', 'AbortError', 'TimeoutError'].includes(e.name) ? e.name : 'other_error',
    network_code: NETWORK_CODES.has(e.code) || PERMANENT_NETWORK_CODES.has(e.code) || TLS_TRANSIENT_CODES.has(e.code) ? e.code : null,
    message_class: CAUSE_PATTERNS.find(([, pattern]) => typeof e.message === 'string' && pattern.test(e.message))?.[0] ?? null,
    syscall: ['connect', 'read', 'write', 'getaddrinfo', 'getnameinfo'].includes(e.syscall) ? e.syscall : null,
  }));
}
export function permanentNetworkFailure(error) {
  return errorNodes(error).some(e => PERMANENT_NETWORK_CODES.has(e.code)) ||
    safeCauseClasses(error).some(e => ['request_content_length_mismatch', 'request_body_consumed',
      'request_duplex_required', 'tls_certificate_failure'].includes(e.message_class));
}
export function providerErrorClass(value, outerStatus = null, { zai = false } = {}) {
  const error = value?.error;
  let rawTypes = [error?.metadata?.error_type, error?.error_type, value?.error_type, error?.type,
    typeof error?.code === 'string' ? error.code : undefined].filter(v => v !== undefined && v !== null);
  // OpenAI rate limits name the exhausted dimension ('tokens'/'requests') as the type; the limit is transient.
  if (error?.code === 'rate_limit_exceeded' || outerStatus === 429) rawTypes = rawTypes.filter(v => v !== 'tokens' && v !== 'requests');
  if (zai) rawTypes = rawTypes.map(v => ZAI_TRANSIENT_CODES.get(v) ?? v);
  // Only the class is persisted, never the message.
  if (typeof error?.message === 'string' && REQUEST_TOO_LARGE.test(error.message)) rawTypes.push('request_exceeds_rate_limit');
  const types = [...new Set(rawTypes.map(v => TRANSIENT_TYPES.has(v) || PERMANENT_TYPES.has(v) ? v : 'unclassified_typed_error'))];
  const type = types.find(v => PERMANENT_TYPES.has(v)) ?? types.find(v => v === 'unclassified_typed_error') ?? types[0] ?? null;
  const code = Number.isInteger(error?.code) ? error.code : null;
  // Permanent inner types/codes take priority over a misleading outer 5xx/429.
  const permanentCode = code !== null && !TRANSIENT_HTTP.has(code) && code >= 400;
  const permanent = PERMANENT_TYPES.has(type) || type === 'unclassified_typed_error' || permanentCode ||
    (outerStatus !== null && outerStatus !== 200 && !TRANSIENT_HTTP.has(outerStatus));
  const retryable = !permanent && (TRANSIENT_TYPES.has(type) || TRANSIENT_HTTP.has(code) || TRANSIENT_HTTP.has(outerStatus));
  return { type, observed_types: types, code, permanent, retryable };
}
export function recoveryDelay(attempt, header, random = Math.random, floorMs = 0) {
  if (header?.milliseconds > 0 && header.milliseconds >= floorMs) return header;
  const sample = random(); assert(Number.isFinite(sample) && sample >= 0 && sample <= 1, 'Invalid injected jitter');
  // Positive bounded jitter: the first fallback is 60–72s, saturated at 600s.
  const base = Math.min(600000, 60000 * 2 ** Math.min(attempt, 4));
  return { milliseconds: Math.min(600000, Math.round(base * (1 + 0.2 * sample))),
    basis: 'capped_exponential_with_positive_bounded_jitter', jitter_sample: sample, fallback_base_ms: base,
    ...(header?.milliseconds > 0 ? { retry_after_ms_below_floor: header.milliseconds, floor_ms: floorMs } : {}) };
}
/** Live operator view of the latest transport failure; overwritten on every failed attempt. */
export function publishRetryStatus(output, status, value) {
  status.last_failure = value;
  status.provider_error_retries_max = MAX_PROVIDER_ERROR_RETRIES;
  const file = path.join(output, 'api', 'retry_status.json'), temporary = file + '.' + crypto.randomUUID();
  fs.writeFileSync(temporary, JSON.stringify({ utc: utc(), ...value }, null, 2) + '\n', { mode: 0o600 });
  fs.renameSync(temporary, file);
}
export class WireFailure extends Error {
  constructor(kind, source = null, boundary = null) {
    super(kind); this.name = 'WireFailure'; this.kind = kind; this.boundary = boundary;
    this.network_code = networkCode(source); this.cause_classes = safeCauseClasses(source);
  }
}
export function attemptDeadline(callerSignal, clock) {
  const controller = new AbortController(); let timedOut = false, timer;
  const abort = () => controller.abort(new WireFailure('caller_cancelled'));
  callerSignal?.addEventListener('abort', abort, { once: true });
  if (callerSignal?.aborted) abort();
  const progress = () => {
    clock.clearTimeout(timer);
    if (!controller.signal.aborted) timer = clock.setTimeout(() => { timedOut = true; controller.abort(new WireFailure('attempt_timeout')); }, ATTEMPT_TIMEOUT_MS);
  };
  progress();
  return { signal: controller.signal, get timedOut() { return timedOut; },
    progress,
    cleanup() { clock.clearTimeout(timer); callerSignal?.removeEventListener('abort', abort); },
    async network(operation, boundary) {
      if (controller.signal.aborted) throw controller.signal.reason;
      // A synchronous local TypeError is deliberately outside the awaited-network catch.
      const pending = operation();
      let rejectAbort;
      const cancelled = new Promise((_, reject) => { rejectAbort = () => reject(controller.signal.reason); });
      controller.signal.addEventListener('abort', rejectAbort, { once: true });
      if (controller.signal.aborted) rejectAbort();
      try { return await Promise.race([pending, cancelled]); }
      catch (error) {
        if (callerSignal?.aborted) throw new WireFailure('caller_cancelled', error, boundary);
        if (timedOut) throw new WireFailure('attempt_timeout', error, boundary);
        if (permanentNetworkFailure(error)) throw new WireFailure('permanent_network_configuration_failure', error, boundary);
        const code = networkCode(error);
        if (code) throw new WireFailure(boundary === 'body' ? 'body_connection_interrupted' : 'connection_failure', error, boundary);
        if (error instanceof TypeError && error.message === 'fetch failed') throw new WireFailure('native_fetch_failed', error, boundary);
        throw new WireFailure('unclassified_network_boundary_error', error, boundary);
      } finally { controller.signal.removeEventListener('abort', rejectAbort); }
    } };
}
export async function bufferedBody(response, attempt, chunks) {
  const reader = response.body?.getReader();
  if (!reader) return Buffer.alloc(0);
  try {
    while (true) {
      const result = await attempt.network(() => reader.read(), 'body');
      if (result.done) return Buffer.concat(chunks);
      chunks.push(Buffer.from(result.value));
      if (result.value.byteLength > 0) attempt.progress();
    }
  } catch (error) {
    void reader.cancel().catch(() => {});
    throw error;
  } finally { reader.releaseLock(); }
}
function responsesKeepalive(event, sseName) {
  if (event?.type !== 'keepalive' || sseName !== 'keepalive') return false;
  // A heartbeat is a no-op, never model output, accounting, identity or a tool.
  // Flat bounded scalar metadata is neutral; nested/action-bearing data is not.
  const reserved = new Set(['model', 'provider', 'response', 'choices', 'usage', 'error', 'error_type',
    'service_tier', 'output', 'item', 'delta', 'content', 'text', 'reasoning', 'summary',
    'encrypted_content', 'arguments', 'name', 'call_id', 'tool_calls', 'function_call', 'status']);
  return Object.entries(event).every(([name, value]) => name === 'type' ||
    (/^[a-zA-Z0-9_]{1,40}$/.test(name) && !reserved.has(name) &&
      (value === null || typeof value === 'boolean' ||
       (typeof value === 'number' && Number.isFinite(value)) ||
       (typeof value === 'string' && value.length <= 256))));
}
function safeWireCapture(bytes, complete, key, responses = false) {
  assert(!key || !bytes.includes(Buffer.from(key)), 'Credential echo refused before any response-byte write');
  const latin = bytes.toString('latin1'); let retained = 0, omission = null, framingDiagnostic = null;
  for (const match of latin.matchAll(/[\s\S]*?\r?\n\r?\n/g)) {
    const end = match.index + match[0].length;
    const block = bytes.subarray(match.index, end).toString('utf8');
    const data = block.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
    try {
      const event = !data || data === '[DONE]' ? null : JSON.parse(data);
      const sseName = block.match(/^event:\s*([a-zA-Z0-9_.:-]{1,80})\s*$/m)?.[1] ?? null;
      if (/^event:\s*error\s*$/m.test(block) || event?.error || event?.response?.error || event?.type === 'response.failed' || event?.error_type || event?.type === 'error' || event?.choices?.some(c => c.error)) {
        omission = 'error_frame_and_following_bytes_omitted'; break;
      }
      const heartbeat = responses && responsesKeepalive(event, sseName);
      if ((responses && event?.type === 'keepalive' && !heartbeat) ||
          (event && !event.model && !event.provider && !Array.isArray(event.choices) && !event.usage &&
           !(responses && typeof event.type === 'string' && event.type.startsWith('response.')) && !heartbeat)) {
        framingDiagnostic = { json_parsed: true, frame_bytes: end - match.index,
          data_bytes: Buffer.byteLength(data),
          event_type: typeof event.type === 'string' && /^[a-zA-Z0-9_.:-]{1,80}$/.test(event.type) ? event.type : null,
          sse_event: block.match(/^event:\s*([a-zA-Z0-9_.:-]{1,80})\s*$/m)?.[1] ?? null };
        omission = 'unclassified_frame_and_following_bytes_omitted'; break;
      }
    } catch {
      framingDiagnostic = { json_parsed: false, frame_bytes: end - match.index,
        data_bytes: Buffer.byteLength(data),
        sse_event: block.match(/^event:\s*([a-zA-Z0-9_.:-]{1,80})\s*$/m)?.[1] ?? null };
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
export function retryAfter(value, now = Date.now()) {
  if (typeof value !== 'string') return null;
  if (/^\d+(?:\.\d+)?$/.test(value.trim())) {
    const seconds = Number(value);
    return Number.isSafeInteger(Math.ceil(seconds * 1000)) ? { milliseconds: seconds * 1000, basis: 'numeric_retry_after' } : null;
  }
  if (/^[A-Za-z]{3}, \d{2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2} GMT$/.test(value)) {
    const timestamp = Date.parse(value);
    if (Number.isFinite(timestamp)) return { milliseconds: Math.max(0, timestamp - now), basis: 'http_date_retry_after' };
  }
  return null;
}
export function safeResponseMetadata(response, now = Date.now()) {
  const ids = {};
  for (const name of ['x-request-id', 'x-generation-id']) {
    const value = response.headers.get(name);
    if (typeof value === 'string' && /^[A-Za-z0-9_-]{1,160}$/.test(value) && !value.startsWith('sk-')) ids[name] = value;
  }
  return { status: response.status, identifiers: ids, retry_after: retryAfter(response.headers.get('retry-after'), now) };
}
export function waitForRetry(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new Error('Retry wait aborted')); return; }
    let timer, remaining = milliseconds;
    const abort = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort); reject(new Error('Retry wait aborted')); };
    const step = () => {
      // Avoid Node's >2^31ms overflow-to1ms; caller cancellation remains active.
      const chunk = Math.min(remaining, 2147483647);
      timer = setTimeout(() => {
        remaining -= chunk;
        if (remaining > 0) step();
        else { signal?.removeEventListener('abort', abort); resolve(); }
      }, chunk);
    };
    step();
    signal?.addEventListener('abort', abort, { once: true });
  });
}

// The native Chat codec tolerates partial argument JSON while streaming. Validate
// Fable's complete buffered tool response before giving any bytes to that codec.
// This checks transport integrity only; it does not validate scientific choices.
export function validateCompleteChatStream(events, wire) {
  assert(/data:\s*\[DONE\]\s*$/.test(wire), 'Missing Chat stream completion marker');
  assert.equal((wire.match(/data:\s*\[DONE\]/g) ?? []).length, 1, 'Duplicate Chat completion marker');
  const ids = events.map(e => e.id).filter(id => id !== undefined);
  assert(ids.length && ids.every(id => typeof id === 'string' && id === ids[0]), 'Mixed or missing Chat response IDs');
  const calls = new Map(); let finish = null, refusal = false;
  for (const event of events) {
    for (const choice of event.choices ?? []) {
      assert.equal(choice.index, 0, 'Unexpected Chat choice index');
      const delta = choice.delta ?? {};
      assert(!finish || (Object.keys(delta).length === 0 && !choice.finish_reason), 'Output followed Chat terminal');
      if (delta.refusal) refusal = true;
      for (const item of delta.tool_calls ?? []) {
        assert(Number.isInteger(item.index) && item.index >= 0, 'Missing tool delta index');
        let call = calls.get(item.index);
        if (!call) { call = { id: null, name: '', arguments: '' }; calls.set(item.index, call); }
        if (item.id !== undefined) {
          assert(typeof item.id === 'string' && item.id.length && (!call.id || call.id === item.id), 'Changing or empty tool identity');
          call.id = item.id;
        }
        assert(item.type === undefined || item.type === 'function', 'Unexpected tool type');
        if (item.function?.name !== undefined) {
          assert(typeof item.function.name === 'string', 'Invalid tool name delta'); call.name += item.function.name;
        }
        if (item.function?.arguments !== undefined) {
          assert(typeof item.function.arguments === 'string', 'Invalid tool argument delta'); call.arguments += item.function.arguments;
        }
      }
      if (choice.finish_reason) {
        assert(!finish, 'Duplicate Chat finish reason'); finish = choice.finish_reason;
      }
    }
  }
  assert(['stop', 'tool_calls', 'length', 'content_filter'].includes(finish), 'Unsupported Chat finish reason');
  if (finish === 'length' || finish === 'content_filter') return [finish];
  assert(!refusal || calls.size === 0, 'Refusal cannot execute a tool');
  assert.equal(finish === 'tool_calls', calls.size > 0, 'Tool set differs from terminal reason');
  assert.equal(new Set([...calls.values()].map(call => call.id)).size, calls.size, 'Duplicate tool call ID');
  for (const call of calls.values()) {
    assert(call.id && /^[a-zA-Z0-9_-]+$/.test(call.name), 'Incomplete tool identity');
    const args = JSON.parse(call.arguments);
    assert(args && typeof args === 'object' && !Array.isArray(args), 'Tool arguments must be a complete JSON object');
  }
  return [finish];
}

// Direct Z.ai variant (zai_direct_transport.mjs): tolerates reasoning_content in empty deltas and
// one repeated final usage frame. Always applied to the Z.ai route.
export function validateCompleteZaiStream(events, wire) {
  assert(/data:\s*\[DONE\]\s*$/.test(wire), 'Missing Chat stream completion marker');
  assert.equal((wire.match(/data:\s*\[DONE\]/g) ?? []).length, 1, 'Duplicate Chat completion marker');
  const ids = events.map(e => e.id).filter(id => id !== undefined);
  assert(ids.length && ids.every(id => typeof id === 'string' && id === ids[0]), 'Mixed or missing Chat response IDs');
  const calls = new Map(); let finish = null, nativeFinish = null, refusal = false, repeatedUsageTerminal = false;
  const emptyDelta = delta => Object.entries(delta).every(([key, value]) =>
    key === 'role' ? value === 'assistant' :
    key === 'content' ? value === '' || value === null :
    ['reasoning', 'reasoning_content'].includes(key) ? value === null || value === '' : false);
  for (const event of events) {
    for (const choice of event.choices ?? []) {
      assert.equal(choice.index, 0, 'Unexpected Chat choice index');
      const delta = choice.delta ?? {};
      if (finish) {
        assert(emptyDelta(delta), 'Substantive output followed Chat terminal');
        if (choice.finish_reason) {
          assert(!repeatedUsageTerminal && event === events.at(-1) && event.usage &&
            choice.finish_reason === finish && choice.native_finish_reason === nativeFinish,
          'Duplicate Chat terminal must be an identical final usage frame');
          repeatedUsageTerminal = true;
        }
        continue;
      }
      if (delta.refusal) refusal = true;
      for (const item of delta.tool_calls ?? []) {
        assert(Number.isInteger(item.index) && item.index >= 0, 'Missing tool delta index');
        let call = calls.get(item.index);
        if (!call) { call = { id: null, name: '', arguments: '' }; calls.set(item.index, call); }
        if (item.id !== undefined) {
          assert(typeof item.id === 'string' && item.id.length && (!call.id || call.id === item.id), 'Changing or empty tool identity');
          call.id = item.id;
        }
        assert(item.type === undefined || item.type === 'function', 'Unexpected tool type');
        if (item.function?.name !== undefined) {
          assert(typeof item.function.name === 'string', 'Invalid tool name delta'); call.name += item.function.name;
        }
        if (item.function?.arguments !== undefined) {
          assert(typeof item.function.arguments === 'string', 'Invalid tool argument delta'); call.arguments += item.function.arguments;
        }
      }
      if (choice.finish_reason) {
        assert(!finish, 'Duplicate Chat finish reason'); finish = choice.finish_reason; nativeFinish = choice.native_finish_reason;
      }
    }
  }
  assert(['stop', 'tool_calls', 'length', 'content_filter'].includes(finish), 'Unsupported Chat finish reason');
  if (finish === 'length' || finish === 'content_filter') return [finish];
  assert(!refusal || calls.size === 0, 'Refusal cannot execute a tool');
  assert.equal(finish === 'tool_calls', calls.size > 0, 'Tool set differs from terminal reason');
  assert.equal(new Set([...calls.values()].map(call => call.id)).size, calls.size, 'Duplicate tool call ID');
  for (const call of calls.values()) {
    assert(call.id && /^[a-zA-Z0-9_-]+$/.test(call.name), 'Incomplete tool identity');
    const args = JSON.parse(call.arguments);
    assert(args && typeof args === 'object' && !Array.isArray(args), 'Tool arguments must be a complete JSON object');
  }
  return [finish];
}

/** All inference including Pi summarization crosses this same exact-model hook. */
export function investigationContext(incoming, instructions) {
  // Pi 1.0 carries system instructions and tool declarations in the transcript.
  // Keep those declarations and all subsequent history while retaining the
  // campaign's exact supplied prompt (the former Context shorthand is also
  // accepted by ModelRuntime for direct calls such as compaction).
  if (typeof incoming.systemPrompt === 'string') {
    const normal = incoming.systemPrompt.startsWith(instructions);
    return { normal, context: normal ? { ...incoming, systemPrompt: instructions } : incoming };
  }
  const first = incoming.messages?.[0];
  const content = first?.role === 'system' ? (typeof first.content === 'string' ? first.content
    : first.content.map(part => part.text ?? '').join('')) : '';
  const normal = content.startsWith(instructions) || first?.sections?.preamble?.startsWith(instructions) === true;
  if (!normal) {
    // Preserved pre-1.0 histories can begin with a user message. Pi appends
    // the current system declaration later and the provider resolves it.
    // Recognize its effective preamble without rewriting that transcript.
    let preamble;
    for (const message of incoming.messages ?? []) if (message.role === 'system' &&
      Object.hasOwn(message.sections ?? {}, 'preamble')) preamble = message.sections.preamble;
    if (typeof preamble === 'string' && preamble.startsWith(instructions)) return { normal: true, context: incoming };
  }
  return { normal, context: normal ? { ...incoming,
    messages: [{ ...first, content: instructions, sections: undefined }, ...incoming.messages.slice(1)] } : incoming };
}

export function installTransport(session, output, instructions, config, {
  fetchImpl = fetch, key = '', waitImpl = waitForRetry, now = Date.now, random = Math.random,
  clock = { setTimeout, clearTimeout }, maxRequests = Infinity,
} = {}) {
  // ModelRuntime is shared by the native agent, summarization, and cache warmer.
  // Keeping Pi's agent stream intact preserves its native warming scheduler.
  const original = session.modelRuntime.streamSimple.bind(session.modelRuntime);
  assert(['openai', 'openrouter', 'zai'].includes(config.model.provider), 'Unsupported transport provider');
  const isOpenAI = config.model.provider === 'openai', isZai = config.model.provider === 'zai';
  if (isOpenAI) {
    assert.equal(config.model.id, 'gpt-6-astra');
    assert.equal(config.model.baseUrl, 'https://api.openai.com/v1');
  }
  if (isZai) {
    assert.equal(config.model.id, DIRECT_ZAI_MODEL);
    assert.equal(config.model.baseUrl + '/chat/completions', DIRECT_ZAI_ENDPOINT);
  }
  const route = { inference_route: config.inference_route, endpoint_url: config.endpoint.url, wire_model_id: config.model.id,
    ...(config.precision_status ? { precision_status: config.precision_status } : {}) };
  let index = fs.readdirSync(path.join(output, 'api')).filter(n => /^request_\d+\.json$/.test(n)).length;
  const status = { blocked: false, usage_unresolved: false, resourceCensored: false, known_cost_usd: 0,
    estimated_cost_usd: 0, native_billing_unreported_attempts: 0, requests: 0, accepted_outputs: 0 };
  session.modelRuntime.streamSimple = (model, incoming, options = {}) => {
    assert.equal(model.id, config.model.id); assert.equal(model.provider, config.model.provider);
    const { normal, context } = investigationContext(incoming, instructions);
    const category = options.maxTokens === 1 ? 'native_cache_warm' : normal ? 'investigator' : 'native_compaction_or_summary';
    return original(model, context, { ...options, maxTokens: Math.min(options.maxTokens ?? 32768, 32768),
      reasoning: 'high', reasoningEffort: 'high', temperature: isOpenAI || config.omit_sampling_controls ? undefined : 1,
      ...(config.cache_retention ? { cacheRetention: config.cache_retention } : {}), maxRetries: 0, timeoutMs: SDK_NO_TIMEOUT_MS,
      onPayload: payload => {
        const p = structuredClone(payload);
        p.model = config.model.id;
        if (isOpenAI) {
          p.reasoning = { effort: 'high', summary: 'auto' }; p.max_output_tokens = Math.min(p.max_output_tokens ?? 32768, 32768);
          p.store = false; p.service_tier = 'default';
          p.include = ['reasoning.encrypted_content'];
          delete p.provider; delete p.reasoning_effort; delete p.temperature; delete p.max_tokens;
          delete p.max_completion_tokens; delete p.stream_options; delete p.prompt_cache_retention;
        } else if (isZai) {
          p.thinking = { type: 'enabled', clear_thinking: false };
          p.reasoning_effort = 'high'; p.temperature = 1;
          p.max_tokens = Math.min(p.max_tokens ?? 32768, 32768);
          p.stream_options = { include_usage: true };
          delete p.provider; delete p.reasoning; delete p.max_completion_tokens;
          delete p.store; delete p.service_tier; delete p.prompt_cache_retention;
        } else {
          p.provider = structuredClone(config.routing);
          // Reasoning mode is PIN data: {effort:'high'}, {enabled:true} or omitted.
          if (config.reasoning_request) p.reasoning = structuredClone(config.reasoning_request); else delete p.reasoning;
          p.temperature = 1; p.max_tokens = Math.min(p.max_tokens ?? 32768, 32768);
          delete p.max_completion_tokens; delete p.store; delete p.stream_options;
          if (config.omit_sampling_controls) { delete p.temperature; delete p.tool_choice; }
        }
        delete p.parallel_tool_calls;
        for (const tool of p.tools ?? []) if (tool.type === 'function') {
          if (isOpenAI) tool.strict = false;
          else tool.function.strict = false;
        }
        return p;
      },
      fetch: async (input, init) => {
        if (status.blocked) throw new Error('Transport sealed after permanent failure or cancellation');
        const url = typeof input === 'string' ? input : input.url ?? String(input);
        assert.equal(url, config.model.baseUrl + (isOpenAI ? '/responses' : '/chat/completions'));
        const body = JSON.parse(init.body);
        assert.equal(body.model, config.model.id);
        if (isOpenAI) {
          assert.deepEqual(body.reasoning, { effort: 'high', summary: 'auto' }); assert.equal(body.store, false); assert.equal(body.service_tier, 'default');
          assert(Number.isInteger(body.max_output_tokens) && body.max_output_tokens > 0 && body.max_output_tokens <= 32768);
          assert(!('provider' in body) && !('reasoning_effort' in body) && !('max_tokens' in body) && !('temperature' in body));
          // Rebuild the direct-provider headers: no SDK/environment org, project,
          // proxy or additional credential header may cross this exact route.
          assert(key, 'Explicit OpenAI credential required');
          init = { ...init, headers: { authorization: 'Bearer ' + key, 'content-type': 'application/json' }, redirect: 'error' };
        } else if (isZai) {
          assert.equal(body.temperature, 1); assert.equal(body.reasoning_effort, 'high');
          assert.deepEqual(body.thinking, { type: 'enabled', clear_thinking: false });
          assert.deepEqual(body.stream_options, { include_usage: true });
          assert(Number.isInteger(body.max_tokens) && body.max_tokens > 0 && body.max_tokens <= 32768);
          assert(!('provider' in body) && !('reasoning' in body) && !('store' in body));
          assert(key, 'Explicit Z.ai credential required');
          init = { ...init, headers: { authorization: 'Bearer ' + key, 'content-type': 'application/json' }, redirect: 'error' };
        } else {
          assert.deepEqual(body.provider, config.routing);
          if (config.omit_sampling_controls) assert(!('temperature' in body) && !('tool_choice' in body));
          else assert.equal(body.temperature, 1);
          if (config.reasoning_request) assert.deepEqual(body.reasoning, config.reasoning_request);
          else assert(!('reasoning' in body), 'Reasoning field present although the PIN omits it');
          assert(!('store' in body));
        }
        assert(!('parallel_tool_calls' in body));
        assert(!key || !JSON.stringify(body).includes(key), 'Credential present in request body');
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
          let providerFailure = null, responseModel, provider;
          const account = events => {
            const usage = rawUsage(events, config.model.provider);
            if (!known) {
              responseModel = events.find(e => e.model || e.response?.model);
              responseModel = responseModel?.model ?? responseModel?.response?.model;
              provider = isOpenAI ? 'OpenAI' : isZai ? 'Z.ai Direct' : events.find(e => e.provider)?.provider;
              const accounting = isOpenAI ? openAICostEstimate(usage, config) : isZai ? directZaiCostEstimate(usage, config) : {
                cost_basis: 'OpenRouter native usage.cost', estimated_or_reserved: false };
              append(path.join(output, 'usage.jsonl'), { accounting_id: accountingId, number, category,
                policy_request_id: policyRequestId, transport_attempt: attempt, status: 'known', usage, ...route,
                response_model: responseModel, actual_provider: provider,
                ...(isOpenAI ? { usage_schema: 'Native Responses usage retained with Chat token aliases for the shared ledger' } : {}),
                ...(isZai ? { usage_schema: 'Unmodified native direct Z.ai Chat usage; status known describes token counts, not billed dollars' } : {}),
                ...(isOpenAI || isZai ? { provider_identity_basis: 'Direct HTTPS endpoint, not a provider field returned in the stream' } : {}),
                output_accepted_at_accounting: false, acceptance_record: 'Separate response receipt; billing knowledge is not output acceptance',
                ...accounting });
              known = true;
              if (isOpenAI || isZai) { status.estimated_cost_usd += accounting.cost_estimate_usd; status.native_billing_unreported_attempts++; }
              else status.known_cost_usd += usage.cost;
            }
            return usage;
          };
          const checkIdentities = (events, required = true) => {
            const models = events.map(e => e.model ?? e.response?.model).filter(v => v !== undefined);
            const providers = events.map(e => e.provider).filter(v => v !== undefined);
            if (required) assert(models.length && (isOpenAI || isZai || providers.length), 'Missing response identity');
            assert(models.every(v => [config.model.id, config.canonical_slug].includes(v)), 'Returned model substitution refused');
            if (isZai) assert(providers.every(v => ['Z.ai Direct', 'Z.AI', 'Z.ai'].includes(v)), 'Returned provider contradicts direct endpoint');
            else assert(providers.every(v => v === config.expected_provider), 'Returned provider differs from pinned contract');
            if (isOpenAI) assert(events.every(e => (e.service_tier ?? e.response?.service_tier) === undefined || (e.service_tier ?? e.response?.service_tier) === 'default'), 'Returned OpenAI service tier differs from requested Standard pricing');
          };
          const providerErrors = events => {
            // A flat Responses error event ({type:'error', code, message, param}) is classified by its code, not by 'error'.
            const errors = events.flatMap(e => [e.response?.error ? { error: e.response.error } : e.type === 'response.failed' ? { error: { type: 'invalid_request_error' } } : e.error ? e : e[SSE_EVENT_NAME] === 'error' || e.type === 'error' ? { error: { code: e.code, message: e.message, param: e.param } } : null,
              ...(e.choices ?? []).filter(c => c.error)].filter(Boolean));
            const classes = errors.map(e => providerErrorClass(e, 200, { zai: isZai }));
            return classes.find(c => c.permanent) ?? classes.find(c => !c.retryable) ?? classes[0] ?? null;
          };
          const terminalPolicyBoundary = events => {
            if (isOpenAI) return events.some(e => e.type === 'response.incomplete');
            const finishes = events.flatMap(e => (e.choices ?? []).map(c => c.finish_reason));
            // A valid complete length response belongs to native continuation,
            // never to identical-wire transport recovery. Error combinations
            // remain terminal below, and incomplete tool calls never execute.
            return finishes.includes('length') || finishes.includes('content_filter');
          };
          try {
            write(path.join(output, 'api', 'request_' + number + '.json'), { category, accounting_id: accountingId,
              policy_request_id: policyRequestId, transport_attempt: attempt, body_sha256: bodySha256, body,
              route, image_parts: images.image_parts, image_base64_bytes: images.image_base64_bytes,
              attempt_timeout_ms: ATTEMPT_TIMEOUT_MS, attempt_timeout_mode: 'header_connect_then_body_inactivity_reset_on_received_progress',
              sdk_timeout_ms: SDK_NO_TIMEOUT_MS, retry_owner: 'this_transport_only' });
          } catch {
            if (category !== 'native_cache_warm') status.blocked = true;
            throw new Error('Local request capture failed before network invocation or deadline creation');
          }
          const started = now(), deadline = attemptDeadline(init.signal, clock);
          try {
            assert.equal(sha(init.body), bodySha256, 'Retry changed the original serialized request');
            sent = true; status.requests++;
            const response = await deadline.network(() => fetchImpl(input, { ...init, signal: deadline.signal }), 'fetch');
            deadline.progress();
            receivedHttp = true; metadata = safeResponseMetadata(response, now());
            write(path.join(output, 'api', 'response_' + number + '_metadata.json'), { category, ...metadata });
            const bytes = await bufferedBody(response, deadline, chunks);
            if (!response.ok) {
              let fields = {}, rateLimit = 'unknown', value = {};
              try { value = JSON.parse(bytes.toString('utf8')); fields = safeErrorFields(value); rateLimit = rateLimitClass(value); } catch {}
              providerFailure = providerErrorClass(value, response.status, { zai: isZai });
              // Error responses may report tokens without an authoritative cost.
              // Incomplete billing must remain unknown without changing the
              // provider's retry classification. Only validation is optional;
              // local accounting persistence failures still stop the transport.
              let completeUsage = false;
              if (value.usage) {
                try { rawUsage([value], config.model.provider); completeUsage = true; } catch {}
              }
              if (completeUsage) account([value]);
              write(path.join(output, 'api', 'http_error_' + number + '.json'), { category, status: response.status,
                fields, provider_error_class: providerFailure, rate_limit_classification: rateLimit,
                classification_basis: 'Allowlisted categories only; no raw error body, message or arbitrary headers persisted' });
              throw new WireFailure('provider_error');
            }
            const capture = safeWireCapture(bytes, true, key, isOpenAI);
            const captureFile = 'response_' + number + (capture.receipt.capture_class === 'complete_raw_sse' ? '.sse' : '_unaccepted_prefix.bin');
            fs.writeFileSync(path.join(output, 'api', captureFile), capture.bytes, { flag: 'wx', mode: 0o600 });
            write(path.join(output, 'api', 'capture_' + number + '_receipt.json'), { file: captureFile, ...capture.receipt });
            wireSaved = true;
            const wire = bytes.toString('utf8'), events = parseSSE(wire);
            // A valid usage frame can settle billing even when the output is rejected.
            if (events.some(e => e.usage || e.response?.usage)) account(events);
            providerFailure = providerErrors(events);
            assert(capture.receipt.omission !== 'error_frame_and_following_bytes_omitted' || providerFailure, 'Unclassified error frame cannot be accepted');
            checkIdentities(events, providerFailure === null);
            const policyBoundary = terminalPolicyBoundary(events);
            if (providerFailure) {
              if (policyBoundary) providerFailure = { ...providerFailure, permanent: true, retryable: false };
              throw new WireFailure('provider_error');
            }
            assert(capture.receipt.omission === null, 'Unsupported or incomplete SSE protocol frame');
            const usage = account(events);
            let finishReasons;
            if (isOpenAI) {
              const terminals = events.filter(e => ['response.completed', 'response.incomplete'].includes(e.type));
              assert.equal(terminals.length, 1, 'Missing or duplicate Responses terminal');
              const terminal = terminals[0], response = terminal.response;
              assert.equal(events.at(-1), terminal, 'Events followed Responses terminal');
              assert(response && typeof response.id === 'string' && Array.isArray(response.output), 'Missing terminal Responses record');
              assert(events.filter(e => e.response).every(e => e.response.id === response.id), 'Mixed response IDs');
              if (terminal.type === 'response.completed') {
                assert.equal(response.status, 'completed');
                assert(response.output.every(item => item.status === undefined || item.status === 'completed'), 'Incomplete output item');
                const functionItems = response.output.filter(item => item.type === 'function_call');
                const doneItems = events.filter(e => e.type === 'response.output_item.done' && e.item?.type === 'function_call').map(e => e.item);
                const addedItems = events.filter(e => e.type === 'response.output_item.added' && e.item?.type === 'function_call').map(e => e.item);
                assert.equal(functionItems.length, doneItems.length, 'Terminal tool set differs from streamed tool set');
                assert.equal(new Set(functionItems.map(item => item.call_id)).size, functionItems.length, 'Duplicate terminal tool call');
                for (const added of addedItems) {
                  const item = functionItems.find(other => other.id === added.id);
                  assert(item && item.name === added.name && item.call_id === added.call_id, 'Unfinished or mismatched streamed tool');
                }
                for (const item of functionItems) {
                  assert(typeof item.id === 'string' && typeof item.call_id === 'string' && typeof item.name === 'string' && typeof item.arguments === 'string', 'Malformed completed function call');
                  const args = JSON.parse(item.arguments);
                  assert(args && typeof args === 'object' && !Array.isArray(args), 'Function arguments must be a complete JSON object');
                  const done = doneItems.find(other => other.id === item.id);
                  assert(done, 'Missing streamed tool completion');
                  for (const field of ['call_id', 'name', 'arguments']) assert.equal(done[field], item[field], 'Streamed and terminal tool differ');
                }
                finishReasons = [response.output.some(item => item.type === 'function_call') ? 'tool_calls' : 'stop'];
              } else {
                assert.equal(response.status, 'incomplete');
                assert.equal(response.incomplete_details?.reason, 'max_output_tokens', 'Unaccepted incomplete Responses output');
                finishReasons = ['length'];
              }
            } else if (isZai) {
              finishReasons = validateCompleteZaiStream(events, wire);
            } else {
              assert(events.some(e => e.choices?.some(c => c.finish_reason)), 'Missing terminal finish reason');
              assert(!events.some(e => e.choices?.some(c => c.finish_reason === 'error')), 'Unclassified terminal error');
              finishReasons = events.flatMap(e => (e.choices ?? []).map(c => c.finish_reason).filter(Boolean));
              if (config.require_complete_chat_stream) finishReasons = validateCompleteChatStream(events, wire);
            }
            // Caller cancellation wins even if the response completed concurrently.
            if (init.signal?.aborted) throw new WireFailure('caller_cancelled');
            write(path.join(output, 'api', 'response_' + number + '_receipt.json'), { category, accounting_id: accountingId,
              policy_request_id: policyRequestId, output_accepted: true, body_complete: true,
              route, image_parts_sent: images.image_parts,
              response_model: responseModel, actual_provider: provider, canonical_slug: config.canonical_slug, usage,
              finish_reasons: finishReasons,
              response_capture: { file: captureFile, ...capture.receipt },
              raw_sse_sha256: capture.receipt.capture_class === 'complete_raw_sse' ? capture.receipt.retained_sha256 : null,
              reasoning_replay: isOpenAI ? 'Native Pi Responses codec; encrypted reasoning replay retained without analyst interpretation' : 'Native Pi Chat codec; raw replay fields retained without analyst interpretation' });
            status.accepted_outputs++; status.last_accepted_finish_reasons = finishReasons;
            // The original body is consumed. Pi receives only this validated buffered
            // response; no prefix or transport-owned upstream stream reaches tools.
            return new Response(bytes, { status: response.status, headers: { 'content-type': 'text/event-stream' } });
          } catch (caught) {
            let error = caught, partialPolicyBoundary = false;
            // Preserve bytes actually read before a network/body interruption, without
            // promoting them to a complete response or delivering them to the model.
            if (receivedHttp && metadata?.status === 200 && chunks.length && !wireSaved) {
              try {
                const partial = Buffer.concat(chunks);
                const capture = safeWireCapture(partial, false, key, isOpenAI);
                const file = 'response_' + number + '_partial.bin';
                fs.writeFileSync(path.join(output, 'api', file), capture.bytes, { flag: 'wx', mode: 0o600 });
                const prefix = partial.toString('utf8').split(/(\r?\n\r?\n)/);
                const completePrefix = prefix.slice(0, -1).join('');
                const events = parseSSE(completePrefix);
                const partialUsageEvent = events.findLast(e => e.usage || e.response?.usage);
                const partialUsage = partialUsageEvent?.usage ?? partialUsageEvent?.response?.usage;
                if (partialUsage && (isZai ? partialUsage.total_tokens === undefined : !isOpenAI && partialUsage.cost == null)) {
                  // A stream can disconnect after token counts but before native
                  // billing arrives. Keep the charge unknown and inspect the
                  // remaining protocol/identity guards before deciding recovery.
                  for (const field of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
                    assert(Number.isInteger(partialUsage[field]) && partialUsage[field] >= 0, `Unknown raw ${field}`);
                  }
                } else if (partialUsage) account(events);
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
            // Repeated unknown pre-HTTP errors supplied no new diagnosis in the
            // preserved campaign. Stop after two; do not burn the run envelope
            // replaying an unexplained deterministic client failure indefinitely.
            unclassifiedFetchFailures = wireKind === 'native_fetch_failed' ? unclassifiedFetchFailures + 1 : 0;
            const diagnosisRequired = unclassifiedFetchFailures >= 2;
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
                policy_request_id: policyRequestId, transport_attempt: attempt, status: 'unknown', native_cost_usd: null, ...route,
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
      },
    });
  };
  return status;
}

export function serialTools(specs, invoke) {
  let queue = Promise.resolve();
  return specs.map(spec => ({ ...spec, label: spec.name, executionMode: 'sequential', execute: (callId, args) => {
    const job = queue.then(() => invoke(spec, callId, args));
    queue = job.catch(() => {}); return job;
  } }));
}

export function mergeTransportAccounting(target, source) {
  target.known_cost_usd += source.known_cost_usd;
  target.estimated_cost_usd = (target.estimated_cost_usd ?? 0) + (source.estimated_cost_usd ?? 0);
  target.native_billing_unreported_attempts = (target.native_billing_unreported_attempts ?? 0) + (source.native_billing_unreported_attempts ?? 0);
  target.requests += source.requests;
  target.accepted_outputs += source.accepted_outputs;
  target.usage_unresolved ||= source.usage_unresolved;
  target.blocked ||= source.blocked;
  target.resourceCensored ||= source.resourceCensored;
}
