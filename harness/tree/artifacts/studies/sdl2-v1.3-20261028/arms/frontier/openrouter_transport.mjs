/** v1.3 copy (arms/frontier/port_transport.py; diff in openrouter_transport.diff): admits z-ai/glm-5.3-flash (D14) and
 * reads v1.2's PINs and snapshots read-only.
 * One OpenRouter route for every frontier model of the SDL-2 release (Jarrod, 2026-10-26: one key, one billing, one
 * provider interface). pi140's vision_transport.mjs is unchanged and carries every request; its modelConfig admits a fixed
 * list of model ids, so this module builds the config for the release's models the same way, from a PIN record and frozen
 * OpenRouter snapshots:
 *   anthropic/claude-opus-5.5, anthropic/claude-sonnet-5.5, openai/gpt-6-astra, x-ai/grok-4.7,
 *   deepseek/deepseek-v4.1-flash-1017                                                 arms/frontier/providers/
 *   google/gemini-4-argon         engine-release-20261008/v2-build/providers-gemini4-argon/ (the SDL-2 RUN-1 Argon route)
 *   qwen/qwen3.8-max-0902         vision_transport.modelConfig unchanged (its qualified PIN)
 * Differences from vision_transport's OpenRouter branch, all PIN data: an endpoint that rejects temperature with thinking
 * (Anthropic, OpenAI) carries omit_sampling_controls, which the unchanged transport honours (no temperature, no
 * tool_choice); the context window is the endpoint's own (its max_prompt_tokens where stated), with no cohort floor. */
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import * as chat from '../../../material-route-swarm-design-20261019/runtime/pi140/vision_transport.mjs';
export * from '../../../material-route-swarm-design-20261019/runtime/pi140/vision_transport.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const V12_PROVIDERS = path.join(chat.REPO, 'artifacts/studies/sdl2-v1.2-20261026/arms/frontier/providers');   // v1.3: v1.2's PINs and frozen snapshots, read-only
export const PIN_DIRECTORIES = {
  'anthropic/claude-opus-5.5': V12_PROVIDERS,
  'anthropic/claude-sonnet-5.5': V12_PROVIDERS,
  'openai/gpt-6-astra': V12_PROVIDERS,
  'x-ai/grok-4.7': V12_PROVIDERS,
  'deepseek/deepseek-v4.1-flash-1017': V12_PROVIDERS,
  'z-ai/glm-5.3-flash': V12_PROVIDERS,   // v1.3 board (D14); PIN recorded Oct 26 (Z.AI, fp8)
  'google/gemini-4-argon': path.join(chat.REPO, 'artifacts/studies/engine-release-20261008/v2-build/providers-gemini4-argon'),
};
export const RELEASE_MODELS = [...Object.keys(PIN_DIRECTORIES), 'qwen/qwen3.8-max-0902'];

export function modelConfig(id, options = {}) {
  assert(RELEASE_MODELS.includes(id), 'Exact requested model required; no substitution');
  if (!(id in PIN_DIRECTORIES)) return chat.modelConfig(id, options);
  const { pin, pinBytes } = chat.readPin(id, PIN_DIRECTORIES[id]);
  const catalogBytes = fs.readFileSync(path.join(chat.REPO, pin.catalog_file));
  const row = JSON.parse(catalogBytes).data.find(r => r.id === id);
  assert(row, 'Model absent from frozen live catalog');
  const endpointBytes = fs.readFileSync(path.join(chat.REPO, pin.endpoints_file));
  const endpoint = JSON.parse(endpointBytes).data.endpoints.find(e => e.tag === pin.provider_tag);
  assert(endpoint && endpoint.provider_name === pin.provider_name, 'Pinned endpoint absent from frozen endpoint metadata');
  const omit = pin.omit_sampling_controls === true;
  for (const feature of ['tools', 'reasoning', 'max_tokens', ...(omit ? [] : ['tool_choice', 'temperature'])]) {
    assert(endpoint.supported_parameters.includes(feature), `Endpoint lacks ${feature}`);
  }
  const contextWindow = Math.min(1048576, endpoint.context_length, endpoint.max_prompt_tokens ?? Infinity);
  const pricing = endpoint.pricing;
  const routing = { only: [pin.provider_tag], order: [pin.provider_tag], allow_fallbacks: false,
    require_parameters: true, data_collection: 'deny' };
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
  return { model, routing, canonical_slug: row.canonical_slug, endpoint: { ...endpoint, url: chat.OPENROUTER_ENDPOINT },
    expected_provider: pin.provider_name, api_key_env: 'OPENROUTER_API_KEY', inference_route: 'openrouter',
    native_max_output_tokens: endpoint.max_completion_tokens, reasoning_request: pin.reasoning_request,
    ...(omit ? { omit_sampling_controls: true } : {}),
    pin_record: pin, pin_record_sha256: chat.sha(pinBytes),
    catalog_sha256: chat.sha(catalogBytes), endpoint_sha256: chat.sha(endpointBytes),
    pricing_note: 'Pi display cost uses dated catalog rates only; authoritative accounting uses raw OpenRouter usage.cost including any discounts',
    common_request: { ...(pin.reasoning_request ? { reasoning: pin.reasoning_request } : {}), ...(omit ? {} : { temperature: 1 }),
      max_tokens: 32768 },
    feature_limit: 'Tool mutations execute serially in Pi and a bridge queue; parallel_tool_calls omitted.' };
}
