/** Native Pi filesystem/network tools composed with Dynamical's scientific tools. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import * as chatTransport from './vision_transport.mjs';
import * as geminiTransport from './gemini_vision_transport.mjs';
import { sandboxConfig, sandboxIdentity } from './native_sandbox.mjs';

const { SessionManager, sha, utc } = chatTransport;
const HERE = path.dirname(fileURLToPath(import.meta.url));
const MATERIAL_ROUTE_TASK = fs.readFileSync(path.resolve(HERE, '../../AGENT_TASK.md'), 'utf8');
const MATERIAL_LAB_KICKOFF = fs.readFileSync(path.resolve(HERE, '../../../material-lab-20261019/shared/KICKOFF.md'), 'utf8');
const PI = path.dirname(path.dirname(fileURLToPath(import.meta.resolve('@earendil-works/pi-coding-agent'))));
const append = (file, value) => fs.appendFileSync(file, JSON.stringify({ utc: utc(), ...value }) + '\n', { mode: 0o600 });
const save = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', { flag: 'wx', mode: 0o600 });

export const RENDITION = { policy: 'vision-rendition-v1', maxWidth: 2000, maxHeight: 2000, maxBase64Bytes: 1048576 };
export const TEXT_TURN_NOTICE = '[Runtime continuation notice] Continue the investigation. Use record_decision to wait or stop when appropriate.';
export const MAX_CONSECUTIVE_TEXT_TURN_NOTICES = 3;
export const TRANSPORT_FILES = ['vision_transport.mjs', 'gemini_vision_transport.mjs', 'native_sandbox.mjs'];
const SOURCE_MEDIA_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
// Fresh restart (Oct 12, 2026): the native tools run in a per-investigation sandbox from the first launch, and the
// shared library/commons and outbox are part of the environment, so the operating text says so plainly.
export const NATIVE_TOOL_CONTEXT = `
Pi's native read, bash, edit, write, grep, find and ls tools are available alongside the scientific tools.
They run in a sandbox container with your workspace, the read-only source archive and live shared research memory. Your working directory is
that workspace; use paths relative to it (HOME is a temporary folder inside the sandbox). The workspace
is persistent and writable. Bash has network access: retrieve public papers/data and use available
command-line clients as needed; there is no built-in search provider.
Keep downloaded sources, URLs, retrieval dates, scripts and analysis outputs in your workspace.
Native files and web results are sources/computational artifacts, never released physical observations.
library/ is a read-only copy of your released records, historical sources, owner-built tools and context
files, refreshed at each release. library/commons/ holds records shared by the other investigators in this
cohort. shared/ provides immediate exchange; outbox/ is indexed at the next snapshot. These are shared records intended to be read,
used and cited like any other source.
Use the scientific tools for authoritative specimen history, model adoption, frozen forecasts and
experiment requests; their compute executor remains the reproducible, offline scientific environment.
Do not seek owner-only material, future physical returns or assessor materials. Do not edit campaign
databases, shared runtime code, credentials or other jobs. No external messages, purchases or lab execution.
These are operating instructions, not command restrictions. Choose your own scientific methods.
Objective: Dynamical's long-term target is production-grade sintered NdFeB without Dy or Tb, with a
reproducible joint gain in coercivity and energy product: Hci >= 20 kOe with (BH)max about 45-48 MGOe at
room temperature, operating temperature verified. The current RPI panel is one step, not the target.
The task in inspect_state and read_sources states the objective, RPI's feasible scope and the process
provenance. Propose experiments, within RPI's capabilities or explicitly flagged as scope expansions,
that most reduce uncertainty about how to reach the target. RPI's returns are released in the order
RPI produced them in historical records; this fresh investigation starts with all locally available releases.
Begin with START_HERE.md and CURRENT_STATE.md. Historical scope and budget text is dated context, not a
restriction on this fresh scientific search. Operator deployment instructions specify current compute authority.

Fresh exploratory campaign task (supersedes earlier scientific itineraries, not recorded evidence):
${MATERIAL_ROUTE_TASK}

Authorized collaborative launch context:
${MATERIAL_LAB_KICKOFF}
`;

export function transportFor(modelId) {
  return modelId === 'gemini-3.8-flash' ? geminiTransport : chatTransport;
}
export function transportHash() {
  return sha(Buffer.concat(TRANSPORT_FILES.map(name => fs.readFileSync(path.join(HERE, name)))));
}

export function runtimeManifest({ modelId, campaign, sessionFile, instructions, tools }) {
  return { model_id: modelId, campaign: path.resolve(campaign), session_file: sessionFile, pi_version: chatTransport.PI_VERSION,
    instructions_sha256: sha(instructions), tool_schema_sha256: sha(JSON.stringify(tools)),
    transport_sha256: transportHash(),
    launcher_sha256: sha(fs.readFileSync(fileURLToPath(import.meta.url))),
    provider_config_sha256: sha(JSON.stringify(transportFor(modelId).modelConfig(modelId))),
    dependency_lock_sha256: sha(fs.readFileSync(path.join(HERE, 'package-lock.json'))) };
}

export function restoredManifest(directory, campaign, current, maxRequests) {
  const originalBytes = fs.readFileSync(path.join(directory, 'session.json'));
  const original = JSON.parse(originalBytes);
  const receiptPath = path.join(directory, 'migration.json');
  if (!fs.existsSync(receiptPath)) return original;
  const receipt = JSON.parse(fs.readFileSync(receiptPath));
  assert.equal(receipt.schema, 'ndfeb-investigator-runtime-migration-v1', 'Unknown migration schema');
  assert.equal(receipt.predecessor_session_sha256, sha(originalBytes), 'Migration must bind the unchanged predecessor manifest');
  assert(receipt.reason && receipt.predecessor_directory, 'Migration needs an explicit reason and predecessor');
  assert.notEqual(path.resolve(receipt.predecessor_directory), directory, 'Migration requires a separate investigation copy');
  assert.equal(receipt.campaign, campaign, 'Migration campaign mismatch');
  const sessionFile = path.resolve(receipt.session_file);
  assert(sessionFile.startsWith(path.join(directory, 'pi_sessions') + path.sep), 'Migrated native history must be in the new investigation');
  assert.equal(fs.realpathSync(sessionFile), sessionFile, 'Migrated history must not alias predecessor files');
  assert.equal(receipt.runtime.session_file, sessionFile);
  // Pi 1.1.0 runtime (Oct 11, 2026): a pi_version or dependency-lock change is accepted only when the migration receipt
  // names that exact change in deliberate_runtime_changes (written by an explicit owner migration); model and provider
  // configuration stay strictly equal. Mirrors the explicit model-route migration below. Everything else unchanged.
  const deliberate = receipt.deliberate_runtime_changes ?? {};
  for (const field of ['model_id', 'pi_version', 'provider_config_sha256', 'dependency_lock_sha256']) {
    const change = deliberate[field];
    if (change && ['pi_version', 'dependency_lock_sha256'].includes(field)) {
      assert.equal(change.from, original[field], `Deliberate ${field} change must start from the original session`);
      assert.equal(change.to, receipt.runtime[field], `Deliberate ${field} change must match the migrated runtime`);
      continue;
    }
    assert.equal(receipt.runtime[field], original[field], `Migration cannot silently change ${field}`);
  }
  let restored = receipt.runtime;
  const routePath = path.join(directory, 'model-route-migration.json');
  if (fs.existsSync(routePath)) {
    const route = JSON.parse(fs.readFileSync(routePath));
    assert.equal(route.schema, 'ndfeb-explicit-model-route-migration-v1', 'Unknown model migration schema');
    assert.equal(route.predecessor_migration_sha256, sha(fs.readFileSync(receiptPath)), 'Model migration must bind its preserved runtime migration');
    assert.equal(route.predecessor_runtime_sha256, sha(JSON.stringify(restored)), 'Model migration predecessor runtime mismatch');
    assert(typeof route.user_authorization === 'string' && route.user_authorization.trim(), 'Explicit user model-change authorization required');
    assert.equal(route.from_model_id, restored.model_id, 'Model migration source mismatch');
    assert.equal(route.to_model_id, route.runtime.model_id, 'Model migration target mismatch');
    assert.equal(route.to_model_id, 'gpt-6-astra', 'This model-change authorization names Astra exactly');
    assert.notEqual(route.from_model_id, route.to_model_id, 'Model migration must name a changed model');
    assert.equal(route.reasoning_effort, 'high', 'Authorized reasoning effort is high');
    assert.equal(route.cumulative_wire_request_limit, 'unbounded', 'Authorized call envelope is unbounded');
    assert.equal(maxRequests, route.cumulative_wire_request_limit, 'Launch call envelope differs from explicit model migration');
    for (const field of ['instructions_sha256', 'tool_schema_sha256', 'dependency_lock_sha256', 'pi_version']) {
      assert.equal(restored[field], original[field], `Model-only migration changed original ${field}`);
    }
    assert.deepEqual(receipt.python_runtime, receipt.previous_python_runtime, 'Model-only migration changed scientific Python runtime');
    assert.deepEqual(Object.keys(route.runtime).sort(), Object.keys(restored).sort(), 'Model migration runtime shape changed');
    for (const [field, value] of Object.entries(restored)) {
      if (!['model_id', 'provider_config_sha256'].includes(field)) {
        assert.equal(route.runtime[field], value, `Model-only migration changed ${field}`);
      }
    }
    restored = route.runtime;
  }
  // Native-tool sandbox (Oct 11, 2026, Jarrod's instruction): the transport and launcher files change to run Pi's native
  // tools inside the investigation sandbox. Accepted only when pi/native-sandbox-migration.json binds this migration
  // receipt and names the exact from/to hashes; every other runtime field stays strictly equal.
  const sandboxPath = path.join(directory, 'native-sandbox-migration.json');
  if (fs.existsSync(sandboxPath)) {
    const change = JSON.parse(fs.readFileSync(sandboxPath));
    assert.equal(change.schema, 'ndfeb-native-sandbox-migration-v1', 'Unknown native sandbox migration schema');
    assert.equal(change.predecessor_migration_sha256, sha(fs.readFileSync(receiptPath)), 'Sandbox change must bind the runtime migration');
    assert(typeof change.user_authorization === 'string' && change.user_authorization.trim(), 'Explicit user authorization required');
    assert.deepEqual(Object.keys(change.changes).sort(), ['launcher_sha256', 'transport_sha256'], 'Sandbox change names only transport and launcher');
    for (const [field, { from, to }] of Object.entries(change.changes)) {
      assert.equal(from, restored[field], `Sandbox change must start from the restored ${field}`);
      restored = { ...restored, [field]: to };
    }
  }
  for (const [field, value] of Object.entries(current)) {
    if (field !== 'session_file') assert.equal(restored[field], value, `Migration runtime mismatch: ${field}`);
  }
  return restored;
}

function assertPrivateDirectory(directory, repo) {
  if (directory !== repo && !directory.startsWith(repo + path.sep)) return;
  const artifacts = path.join(repo, 'artifacts');
  assert(directory.startsWith(artifacts + path.sep), 'Run artifacts inside this repository must stay under private artifacts/');
  const probe = spawnSync('git', ['check-ignore', '-q', '--', path.relative(repo, directory)], { cwd: repo });
  assert.equal(probe.status, 0, 'Run artifacts must be explicitly Git-ignored');
  let existing = directory;
  while (!fs.existsSync(existing)) existing = path.dirname(existing);
  assert.equal(fs.realpathSync(existing), existing, 'Private run path must not traverse symlinks');
}

function lock(directory) {
  const file = path.join(directory, 'investigator.lock');
  if (fs.existsSync(file)) {
    const prior = JSON.parse(fs.readFileSync(file));
    assert(Number.isInteger(prior.pid) && prior.pid > 0, 'Invalid lock; owner inspection required');
    let live = true;
    try { process.kill(prior.pid, 0); } catch (error) { if (error.code === 'ESRCH') live = false; else throw error; }
    assert(!live, 'Another investigator holds the campaign lock');
    fs.unlinkSync(file); // Exact stale lock, never a campaign or data directory.
  }
  const token = crypto.randomUUID();
  save(file, { pid: process.pid, token });
  return () => {
    if (fs.existsSync(file) && JSON.parse(fs.readFileSync(file)).token === token) fs.unlinkSync(file);
  };
}

export function cumulativeAccounting(directory) {
  const file = path.join(directory, 'usage.jsonl');
  const rows = fs.existsSync(file) ? fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [];
  const requests = fs.readdirSync(path.join(directory, 'api')).filter(n => /^request_\d+\.json$/.test(n))
    .map(n => JSON.parse(fs.readFileSync(path.join(directory, 'api', n))));
  const accounted = new Set(rows.map(r => r.accounting_id));
  const interrupted = requests.filter(r => !accounted.has(r.accounting_id));
  return { attempts: requests.length, known_cost_usd: rows.filter(r => r.status === 'known' && Number.isFinite(r.usage?.cost)).reduce((s, r) => s + r.usage.cost, 0),
    estimated_cost_usd: rows.filter(r => Number.isFinite(r.cost_estimate_usd)).reduce((s, r) => s + r.cost_estimate_usd, 0),
    token_usage_known_billing_not_reported: rows.filter(r => r.status === 'known' && r.billing_cost_status === 'not_reported_by_api').length,
    unresolved_attempts: rows.filter(r => r.status !== 'known').length + interrupted.length,
    interrupted_before_usage_record: interrupted.map(r => r.accounting_id),
    unknown_cost_is_not_zero: true };
}

let helpers;
const imageHelpers = () => helpers ??= Promise.all(['image-process.js', 'image-convert.js', 'photon.js', 'exif-orientation.js']
  .map(name => import(pathToFileURL(path.join(PI, 'dist/utils', name)))))
  .then(([{ processImage }, { convertImageBytesToPng }, { loadPhoton }, { applyExifOrientation }]) =>
    ({ processImage, convertImageBytesToPng, loadPhoton, applyExifOrientation }));

/** True when Pi's resize would re-orient these bytes: its own EXIF reader applied to a 2x2 probe with four distinct pixels. */
function exifReorients(photon, applyExifOrientation, bytes) {
  const pixels = Uint8Array.from([1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255]);
  const probe = new photon.PhotonImage(pixels, 2, 2);
  let result;
  try {
    result = applyExifOrientation(photon, probe, bytes);
    return result !== probe || !Buffer.from(probe.get_raw_pixels()).equals(Buffer.from(pixels));
  } finally { if (result && result !== probe) result.free(); probe.free(); }
}

/** One uniform rendition for every route: crop on the stored pixel grid, then Pi's own resize at fixed limits.
 * EXIF orientation is never applied: a full frame that carries a re-orienting EXIF block is decoded on the stored
 * grid and re-encoded without EXIF, like a full-extent crop, so full views, crops and descriptor dims agree. */
export async function renderImage({ base64, mediaType, crop = null }) {
  const { processImage, convertImageBytesToPng, loadPhoton, applyExifOrientation } = await imageHelpers();
  if (!SOURCE_MEDIA_TYPES.includes(mediaType)) return { ok: false, reason: 'unsupported_media_type' };
  let bytes = Buffer.from(base64, 'base64'), mime = mediaType, convertedFrom = null, exifNeutralized = false;
  if (!crop && mediaType !== 'image/gif') {
    const photon = await loadPhoton();
    if (!photon) return { ok: false, reason: 'photon_unavailable' };
    if (exifReorients(photon, applyExifOrientation, bytes)) {
      let image;
      try {
        try { image = photon.PhotonImage.new_from_byteslice(bytes); } catch { return { ok: false, reason: 'decode_failed' }; }
        bytes = Buffer.from(image.get_bytes()); mime = 'image/png'; exifNeutralized = true;
        if (mediaType !== 'image/png') convertedFrom = mediaType;
      } finally { image?.free(); }
    }
  }
  if (crop) {
    const photon = await loadPhoton();
    if (!photon) return { ok: false, reason: 'photon_unavailable' };
    let image, cropped;
    try {
      try { image = photon.PhotonImage.new_from_byteslice(bytes); } catch { return { ok: false, reason: 'decode_failed' }; }
      const { x, y, width, height } = crop;
      if (![x, y, width, height].every(Number.isInteger) || x < 0 || y < 0 || width < 1 || height < 1 ||
        x + width > image.get_width() || y + height > image.get_height()) return { ok: false, reason: 'crop_outside_decoded_bounds' };
      // No EXIF rotation before the crop: coordinates address the stored grid the descriptor reported.
      cropped = photon.crop(image, x, y, x + width, y + height);
      bytes = Buffer.from(cropped.get_bytes()); mime = 'image/png';
    } finally { cropped?.free(); image?.free(); }
  } else if (!exifNeutralized && (mediaType === 'image/gif' || mediaType === 'image/webp')) {
    const png = await convertImageBytesToPng(bytes);
    if (!png) return { ok: false, reason: 'conversion_failed' };
    bytes = Buffer.from(png); mime = 'image/png'; convertedFrom = mediaType;
  }
  const result = await processImage(bytes, mime, { autoResizeImages: true,
    resizeOptions: { maxWidth: RENDITION.maxWidth, maxHeight: RENDITION.maxHeight, maxBytes: RENDITION.maxBase64Bytes } });
  if (!result.ok) return { ok: false, reason: 'resize_failed' };
  if (!['image/png', 'image/jpeg'].includes(result.mimeType)) return { ok: false, reason: 'unsupported_output_format' };
  if (!(result.data.length < RENDITION.maxBase64Bytes)) return { ok: false, reason: 'byte_ceiling_exceeded' };
  const delivered = Buffer.from(result.data, 'base64'), photon = await loadPhoton();
  if (!photon) return { ok: false, reason: 'photon_unavailable' };
  let decoded, width, height;
  try { decoded = photon.PhotonImage.new_from_byteslice(delivered); width = decoded.get_width(); height = decoded.get_height(); }
  catch { return { ok: false, reason: 'decode_failed' }; }
  finally { decoded?.free(); }
  if (width > RENDITION.maxWidth || height > RENDITION.maxHeight) return { ok: false, reason: 'dimension_ceiling_exceeded' };
  return { ok: true, data: result.data, mimeType: result.mimeType, delivered_sha256: sha(delivered),
    delivered_width: width, delivered_height: height, delivered_base64_bytes: result.data.length,
    source_media_type: mediaType, converted_from: convertedFrom, crop: crop ?? null,
    pixel_grid: 'stored', orientation_applied: false, exif_neutralized: exifNeutralized };
}

const stripImageData = value => {
  if (Array.isArray(value)) value.forEach(stripImageData);
  else if (value && typeof value === 'object') {
    delete value.image_base64;
    Object.values(value).forEach(stripImageData);
  }
  return value;
};

/** Single tool-result helper for the live tool path and durable recovery. No base64 ever enters the text part. */
export async function toolContent(result) {
  const textResult = stripImageData(structuredClone(result)), targets = [];
  if (result?.ok !== false && typeof result?.image_base64 === 'string') targets.push([result, textResult]);
  if (Array.isArray(result?.images)) result.images.forEach((image, index) => {
    if (image?.ok !== false && typeof image?.image_base64 === 'string') targets.push([image, textResult.images[index]]);
  });
  const renditions = [];
  for (const [source, text] of targets) {
    const rendition = await renderImage({ base64: source.image_base64, mediaType: source.media_type, crop: source.crop ?? null });
    const { data, ...record } = rendition;
    text.rendition = record;
    renditions.push({ ...rendition, source_sha256: source.sha256 ?? null });
  }
  return { content: [{ type: 'text', text: JSON.stringify(textResult) },
    ...renditions.filter(r => r.ok).map(r => ({ type: 'image', data: r.data, mimeType: r.mimeType }))], renditions };
}

export async function recoverToolResults(manager, bridge, directory) {
  // Reconcile only tool calls accepted into native durable history. Never invent
  // measurements or replay an owner action. Unstarted calls receive an interruption.
  const messages = manager.buildSessionContext().messages;
  const completed = new Set(messages.filter(m => m.role === 'toolResult').map(m => m.toolCallId));
  let count = 0;
  for (const message of messages.filter(m => m.role === 'assistant')) {
    for (const call of message.content.filter(c => c.type === 'toolCall')) {
      if (completed.has(call.id)) continue;
      let result, bridgeError = null;
      if (chatTransport.NATIVE_TOOL_NAMES.includes(call.name)) {
        // A native shell/file operation may already have run. Never repeat its effects on resume.
        result = { status: 'interrupted', reason: 'Native Pi tool result was not recorded before interruption. Inspect workspace files and process state before retrying; the command has not been replayed.' };
      } else {
      try {
        result = await bridge.call('recover_tool', { call_id: call.id, name: call.name, arguments: call.arguments });
      } catch (error) {
        const e = error?.bridgeError ?? {};
        // Allowlisted classes only; the bridge's two restore refusals mean the call never committed.
        bridgeError = { code: ['tool_contract_rejected', 'scientific_computation_failed'].includes(e.code) ? e.code : e.code ? 'other' : 'bridge_unavailable',
          type: ['ValueError', 'ContractError', 'TypeError', 'KeyError', 'FileNotFoundError', 'OSError', 'RuntimeError'].includes(e.type) ? e.type : e.type ? 'other' : null,
          never_committed: typeof e.message === 'string' && /^tool (?:was never started|has no durable intent or committed result)/.test(e.message) };
        result = { status: 'interrupted', reason: call.name === 'view_image' && e.code && !bridgeError.never_committed
          ? 'A view_image call was recorded but its image could not be restored. Inspect current state before retrying; no observation inferred.'
          : 'No recoverable committed tool result. Inspect current state before retrying; no observation inferred.' };
      }
      }
      const { content } = await toolContent(result);
      manager.appendMessage({ role: 'toolResult', toolCallId: call.id, toolName: call.name,
        content, isError: result.status === 'interrupted', timestamp: Date.now() });
      append(path.join(directory, 'recovery.jsonl'), { tool_call_id: call.id, name: call.name, result_status: result.status ?? 'restored',
        source: chatTransport.NATIVE_TOOL_NAMES.includes(call.name) ? 'native_workspace_inspection_required' : 'durable_investigator_ledger', physical_execution_repeated: false, image_parts: content.length - 1,
        bridge_error: bridgeError });
      completed.add(call.id); count++;
    }
  }
  return count;
}

export function investigationTerminationCause(last, transport, interrupted, pause) {
  if (interrupted) return 'operator_interruption';
  if (transport.resourceCensored) return 'resource_censored';
  if (transport.blocked || ['error', 'aborted'].includes(last?.stopReason)) return 'execution_failure';
  if (last?.stopReason === 'length') return 'resource_censored';
  return pause === 'stop' ? 'scientific_stop' : pause === 'wait' ? 'awaiting_operator' : 'natural_return';
}

export function installTextTurnContinuation(session, directory, invocation, suppression,
  maxConsecutive = MAX_CONSECUTIVE_TEXT_TURN_NOTICES) {
  // A text-only 'stop' otherwise ends the native run as natural_return. This fixed
  // runtime notice adds no task, evidence or scientific direction. The default is
  // three in a row; the physical observation route explicitly uses unbounded.
  // Any tool call resets the count. Pi drains the queue only when the agent would
  // stop, so every pause/compaction/interrupt/seal is checked before queueing.
  const notices = { queued: 0, suppressed: 0, limitReached: false, consecutive: 0, maxConsecutive };
  session.subscribe(event => {
    if (event.type !== 'turn_end' || !event.message) return;
    if (event.message.content.some(item => item.type === 'toolCall')) { notices.consecutive = 0; return; }
    if (event.message.stopReason !== 'stop') return;
    const suppressed = suppression() ?? (maxConsecutive !== 'unbounded' && notices.consecutive >= maxConsecutive ? 'consecutive_limit' : null);
    if (!suppressed) {
      notices.consecutive++; notices.queued++;
      session.agent.followUp({ role: 'user', content: [{ type: 'text', text: TEXT_TURN_NOTICE }], timestamp: Date.now() });
    } else {
      notices.suppressed++;
      if (suppressed === 'consecutive_limit') notices.limitReached = true;
    }
    append(path.join(directory, 'text_turn_continuations.jsonl'), { invocation, message_timestamp: event.message.timestamp,
      stop_reason: 'stop', consecutive: notices.consecutive, queued: !suppressed, suppressed_reason: suppressed,
      notice: suppressed ? null : TEXT_TURN_NOTICE, route: 'native_follow_up_queue', transport_retry: false, scientific_instruction: false });
  });
  return notices;
}
const continuationRecord = notices => ({ queued: notices?.queued ?? 0, suppressed: notices?.suppressed ?? 0,
  consecutive_limit_reached: notices?.limitReached ?? false, max_consecutive: notices?.maxConsecutive ?? MAX_CONSECUTIVE_TEXT_TURN_NOTICES,
  file: 'text_turn_continuations.jsonl' });

export async function runInvestigation({ campaign, directory, modelId, sources, maxRequests, key,
  fetchImpl = fetch, bridgeFactory, waitImpl, clock, random, now, compactBeforePrompt = false, compactionKeepRecentTokens = 20000,
  textTurnContinuationLimit = MAX_CONSECUTIVE_TEXT_TURN_NOTICES, nativeTools = true }) {
  assert(maxRequests === 'unbounded' || (Number.isInteger(maxRequests) && maxRequests > 0),
    'Explicit positive cumulative wire-request envelope or authorized unbounded policy required');
  const wireLimit = maxRequests === 'unbounded' ? Infinity : maxRequests;
  const { Bridge, createSession, installTransport, serialTools, modelConfig } = transportFor(modelId);
  const config = modelConfig(modelId);
  directory = path.resolve(directory); campaign = path.resolve(campaign);
  const repo = chatTransport.REPO;
  assertPrivateDirectory(directory, repo);
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  fs.mkdirSync(path.join(directory, 'workspace'), { recursive: true });
  fs.mkdirSync(path.join(directory, 'api'), { recursive: true });
  const unlock = lock(directory), invocation = crypto.randomUUID();
  const bridge = bridgeFactory ? bridgeFactory() : new Bridge();
  const images = { with_images: 0, text_only: 0, delivered: [] };
  let session, transport, notices, interrupted = null, pause = null, pendingCompaction;
  const cancel = signal => {
    interrupted ??= signal;
    bridge.interrupt?.();
    if (session) void session.abort().catch(() => {});
  };
  const sigint = () => cancel('SIGINT'), sigterm = () => cancel('SIGTERM');
  process.on('SIGINT', sigint); process.on('SIGTERM', sigterm);
  try {
    const init = await bridge.call('init', { campaign, directory: path.join(directory, 'scientific_state'), ...(sources === undefined ? {} : { sources }) });
    if (nativeTools) init.instructions += '\n' + NATIVE_TOOL_CONTEXT;
    const toolSpecs = [...(nativeTools ? chatTransport.nativeToolSpecs(path.join(directory, 'workspace')) : []), ...init.tools];
    const manifestPath = path.join(directory, 'session.json');
    let manager, resumed = false;
    if (fs.existsSync(manifestPath)) {
      const current = runtimeManifest({ modelId, campaign, instructions: init.instructions, tools: toolSpecs });
      const previous = restoredManifest(directory, campaign, current, maxRequests);
      assert.equal(previous.model_id, modelId, 'Model identity cannot change within an investigation');
      assert.equal(previous.campaign, campaign, 'Cannot restore into a different physical history');
      assert.equal(previous.instructions_sha256, sha(init.instructions), 'Instruction changes require a versioned investigation');
      assert.equal(previous.tool_schema_sha256, sha(JSON.stringify(toolSpecs)), 'Tool contract changed; versioned migration required');
      assert.equal(previous.transport_sha256, transportHash(), 'Transport changed during a preserved investigation');
      assert.equal(previous.launcher_sha256, sha(fs.readFileSync(fileURLToPath(import.meta.url))), 'Launcher changed during a preserved investigation');
      assert.equal(previous.dependency_lock_sha256, sha(fs.readFileSync(path.join(HERE, 'package-lock.json'))), 'Dependency lock changed during a preserved investigation');
      assert.equal(previous.provider_config_sha256, sha(JSON.stringify(config)), 'Provider catalog/config changed during a preserved investigation');
      manager = SessionManager.open(previous.session_file);
      await recoverToolResults(manager, bridge, directory);
      resumed = true;
    } else {
      manager = SessionManager.create(path.join(directory, 'workspace'), path.join(directory, 'pi_sessions'));
      save(manifestPath, runtimeManifest({ modelId, campaign, sessionFile: manager.getSessionFile(),
        instructions: init.instructions, tools: toolSpecs }));
    }
    const tools = serialTools(init.tools, async (spec, callId, args) => {
      if (pause || pendingCompaction !== undefined) return { content: [{ type: 'text', text: JSON.stringify({ status: 'paused', reason: 'No more effects after wait/stop or a compaction request in this turn.' }) }], isError: true, terminate: true };
      let result, isError = false;
      try { result = await bridge.call('tool', { call_id: callId, name: spec.name, arguments: args }); }
      catch (error) { result = { status: 'rejected', error: error.bridgeError ?? {
        code: 'bridge_unavailable', message: 'Bridge connection failed. No result or observation is inferred.' } }; isError = true; }
      // Native Pi marks thrown executions as tool errors; a returned isError
      // property alone is ignored by its tool runner. Keep the public structured
      // feedback intact rather than recording a rejected operation as success.
      if (isError) throw new Error(JSON.stringify(result));
      if (!isError && spec.name === 'compact_context' && result.status === 'requested') {
        pendingCompaction = args.notes;
        created.settingsManager.applyOverrides({ compaction: { enabled: false } });
      }
      if (!isError && spec.name === 'record_decision' && ['wait', 'stop'].includes(args.decision)) {
        pause = args.decision;
        created.settingsManager.applyOverrides({ compaction: { enabled: false } });
      }
      const { content, renditions } = await toolContent(result);
      if (renditions.length || spec.name === 'view_image') {
        if (content.length > 1) images.with_images++; else images.text_only++;
        for (const r of renditions.filter(item => item.ok)) images.delivered.push({ call_id: callId, source_sha256: r.source_sha256,
          crop: r.crop, delivered_sha256: r.delivered_sha256, delivered_width: r.delivered_width,
          delivered_height: r.delivered_height, delivered_media_type: r.mimeType });
      }
      return { content, details: { bridge_call_id: callId }, isError,
        terminate: Boolean(pause) || pendingCompaction !== undefined };
    });
    const created = await createSession({ output: directory, instructions: init.instructions, tools, manager, config, key, nativeTools });
    session = created.session;
    const accountingBefore = cumulativeAccounting(directory);
    transport = installTransport(session, directory, init.instructions, config, { key, fetchImpl,
      ...(waitImpl ? { waitImpl } : {}), ...(clock ? { clock } : {}), ...(random ? { random } : {}), ...(now ? { now } : {}),
      maxRequests: Math.max(0, wireLimit - accountingBefore.attempts) });
    notices = installTextTurnContinuation(session, directory, invocation, () => pause ? 'pause'
      : pendingCompaction !== undefined ? 'compaction_pending' : interrupted ? 'interrupted'
        : transport.blocked ? 'transport_blocked' : transport.resourceCensored ? 'resource_censored' : null, textTurnContinuationLimit);
    const originalFinish = session.agent.finishTurn;
    session.agent.finishTurn = async (turn, signal) => {
      const decision = await originalFinish?.(turn, signal);
      return pause || pendingCompaction !== undefined ? { action: 'end' } : decision;
    };
    save(path.join(directory, 'launch_' + invocation + '.json'), { invocation, resumed, config, settings: created.settings,
      max_requests: maxRequests, cumulative_before: accountingBefore, tool_schema: toolSpecs,
      tool_profile: nativeTools ? 'pi-native-dynamical-v1' : 'dynamical-only',
      native_workspace: nativeTools ? path.join(directory, 'workspace') : null,
      native_network: nativeTools ? (sandboxConfig() ? 'native tools in the investigation sandbox; sandbox network with no custom filter'
        : 'host networking via native Pi bash; no custom network filter') : null,
      native_sandbox: nativeTools ? await sandboxIdentity() : null,
      request_authority: 'investigator proposals only; operator acceptance/release absent',
      credentials_in_tools: nativeTools ? (sandboxConfig() ? 'native tools run in the sandbox: only the workspace is mounted and no host environment is passed'
        : 'native host tools inherit OS access; no credential isolation claim') : false,
      transport_files: TRANSPORT_FILES.map(name => ({ name, sha256: sha(fs.readFileSync(path.join(HERE, name))) })),
      rendition_policy: RENDITION, compact_before_prompt: compactBeforePrompt ? { keep_recent_tokens: compactionKeepRecentTokens,
        kind: 'operator infrastructure intervention before the opening prompt' } : false,
      continuation_policy: { notice: TEXT_TURN_NOTICE, max_consecutive: textTurnContinuationLimit, applies: 'all routes' } });
    if (interrupted) throw new Error('Cancelled before inference');
    if (compactBeforePrompt) {
      created.settingsManager.applyOverrides({ compaction: { ...created.settings.compaction, keepRecentTokens: compactionKeepRecentTokens } });
      try { await session.compact(); } // Explicit operator infrastructure action; same accounted transport.
      finally { created.settingsManager.applyOverrides({ compaction: created.settings.compaction }); }
    }
    const state = await bridge.call('state', { summary: true });
    await session.prompt(JSON.stringify({ kind: resumed ? 'investigation_resumed' : 'investigation_opened',
      execution_policy: { cumulative_wire_request_limit: maxRequests,
        note: 'Call accounting continues. Scientific stopping, specimen resources, evidence gates and transport failure stops remain in force.' },
      public_state: state, instruction: 'Continue the investigation from its preserved state. Choose the worthwhile next action.' }));
    while (pendingCompaction !== undefined && !interrupted && !pause) {
      const notes = pendingCompaction; pendingCompaction = undefined;
      created.settingsManager.applyOverrides({ compaction: created.settings.compaction });
      append(path.join(directory, 'compaction_requests.jsonl'), { invocation, notes, route: 'native Pi session.compact at completed tool boundary' });
      let compacted = true;
      try { await session.compact(notes); }
      catch (error) {
        if (!['Already compacted', 'Nothing to compact (session too small)'].includes(error.message)) throw error;
        compacted = false;
      }
      append(path.join(directory, 'compaction_requests.jsonl'), { invocation, compacted, completed: true });
      await session.prompt(JSON.stringify({ kind: 'native_compaction_returned', compacted,
        instruction: 'Continue the same investigation. Durable scientific state and prior commitments remain available through the tools.' }));
    }
    const last = session.messages.findLast(m => m.role === 'assistant');
    const cause = investigationTerminationCause(last, transport, interrupted, pause);
    const result = { invocation, status: 'returned', termination_cause: cause, resumed, model_id: modelId,
      session_file: manager.getSessionFile(), transport, cumulative_accounting: cumulativeAccounting(directory),
      public_state: await bridge.call('state'), scientific_negative: false,
      text_turn_continuations: continuationRecord(notices), rendition_policy: RENDITION, image_tool_results: images,
      cache_warming_status: session.cacheWarmingStatus,
      ...(cause === 'natural_return' && notices.limitReached ? { termination_note: 'natural_return_after_continuation_limit' } : {}) };
    save(path.join(directory, 'completion_' + invocation + '.json'), result);
    return result;
  } catch (error) {
    save(path.join(directory, 'failure_' + invocation + '.json'), { invocation, type: error.constructor.name,
      transport, termination_cause: interrupted ? 'operator_interruption' : transport?.resourceCensored ? 'resource_censored' : 'execution_failure',
      scientific_negative: false, text_turn_continuations: continuationRecord(notices), rendition_policy: RENDITION,
      image_tool_results: images });
    throw error;
  } finally {
    process.removeListener('SIGINT', sigint); process.removeListener('SIGTERM', sigterm);
    transport?.dispose?.(); // Gemini restores globalThis.fetch before the session is released.
    session?.dispose();
    try { await bridge.close(); } finally { unlock(); }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [campaign, directory, modelId, envelope] = process.argv.slice(2);
  const credentialName = transportFor(modelId).modelConfig(modelId).api_key_env;
  assert(process.env[credentialName], `${credentialName} must be supplied by the external authorized runtime`);
  runInvestigation({ campaign, directory, modelId, maxRequests: envelope === 'unbounded' ? envelope : Number(envelope),
    textTurnContinuationLimit: Infinity, key: process.env[credentialName] })
    .then(result => console.log(JSON.stringify(result)))
    .catch(error => { console.error(JSON.stringify({ status: 'failed', type: error.constructor.name, detail: 'Inspect local receipts' })); process.exitCode = 1; });
}
