// Native Pi tools inside a per-investigation sandbox (Oct 11, 2026; Jarrod: sandbox the investigators' native tools).
// Pi, the provider transport and the scientific tools stay on the host. read, bash, edit, write, grep, find and ls run
// inside a sandbox that mounts only this investigation's workspace (same absolute path), Pi's rg/fd read-only at
// /opt/pi-bin, and the bash full-output directory. No host environment variable is passed into the sandbox.
// Backends: a Docker container (`docker exec`) or Docker Sandboxes (`sbx exec`). Selected by NDFEB_NATIVE_SANDBOX
// (sandbox name) and NDFEB_NATIVE_SANDBOX_KIND (docker | sbx). Without NDFEB_NATIVE_SANDBOX nothing here is used.
// Tool names, descriptions and parameter schemas are Pi's own (createToolDefinition), so the tool schema is unchanged.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';

const PI = path.dirname(path.dirname(fileURLToPath(import.meta.resolve('@earendil-works/pi-coding-agent'))));
const { createToolDefinition } = await import(pathToFileURL(path.join(PI, 'dist/core/tools/index.js')));
const { detectSupportedImageMimeType } = await import(pathToFileURL(path.join(PI, 'dist/utils/mime.js')));

export function sandboxConfig(env = process.env) {
  const name = env.NDFEB_NATIVE_SANDBOX;
  if (!name) return null;
  const kind = env.NDFEB_NATIVE_SANDBOX_KIND || 'docker';
  assert(['docker', 'sbx'].includes(kind), `Unknown native sandbox backend ${kind}`);
  return { name, kind };
}

// Only what the docker/sbx client itself needs; nothing from this process reaches the sandbox.
const CLIENT_ENV = () => Object.fromEntries(['PATH', 'HOME', 'DOCKER_HOST'].filter(k => process.env[k]).map(k => [k, process.env[k]]));

function command(cfg, argv, interactive) {
  if (cfg.kind === 'sbx') return ['sbx', ['exec', ...(interactive ? ['-i'] : []), cfg.name, '--', ...argv]];
  return ['docker', ['exec', ...(interactive ? ['-i'] : []), cfg.name, ...argv]];
}

function run(cfg, argv, { input, onData, signal } = {}) {
  const [bin, args] = command(cfg, argv, input !== undefined);
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { env: CLIENT_ENV(), stdio: [input !== undefined ? 'pipe' : 'ignore', 'pipe', 'pipe'] });
    const out = [], err = [];
    child.stdout.on('data', d => (onData ? onData(d) : out.push(d)));
    child.stderr.on('data', d => (onData ? onData(d) : err.push(d)));
    child.on('error', reject);
    child.on('close', (code, sig) => resolve({ code: code ?? (sig ? 137 : 1), stdout: Buffer.concat(out), stderr: Buffer.concat(err).toString() }));
    if (input !== undefined) { child.stdin.on('error', () => {}); child.stdin.end(input); }
    if (signal) signal.addEventListener('abort', () => child.kill('SIGKILL'), { once: true });
  });
}

async function test(cfg, flag, p) {
  return (await run(cfg, ['test', flag, p])).code === 0;
}

async function readBytes(cfg, p) {
  const r = await run(cfg, ['cat', '--', p]);
  if (r.code !== 0) throw new Error(`ENOENT: cannot read '${p}' in the investigation sandbox: ${r.stderr.trim()}`);
  return r.stdout;
}

async function writeText(cfg, p, content) {
  const r = await run(cfg, ['sh', '-c', 'cat > "$1"', 'sh', p], { input: content });
  if (r.code !== 0) throw new Error(`EACCES: cannot write '${p}' in the investigation sandbox: ${r.stderr.trim()}`);
}

function operations(cfg) {
  const access = async p => { if (!(await test(cfg, '-r', p))) throw new Error(`ENOENT: no such readable file in the investigation sandbox, access '${p}'`); };
  const isDirectory = async p => {
    const r = await run(cfg, ['sh', '-c', 'if [ -d "$1" ]; then echo d; elif [ -e "$1" ]; then echo f; else exit 2; fi', 'sh', p]);
    if (r.code !== 0) throw new Error(`ENOENT: no such file or directory in the investigation sandbox, stat '${p}'`);
    return r.stdout.toString().trim() === 'd';
  };
  return {
    read: { readFile: p => readBytes(cfg, p), access,
      detectImageMimeType: async p => detectSupportedImageMimeType((await run(cfg, ['head', '-c', '4100', '--', p])).stdout) },
    write: { writeFile: (p, c) => writeText(cfg, p, c),
      mkdir: async d => { const r = await run(cfg, ['mkdir', '-p', '--', d]); if (r.code) throw new Error(r.stderr.trim()); } },
    edit: { readFile: p => readBytes(cfg, p), writeFile: (p, c) => writeText(cfg, p, c),
      access: async p => { if (!(await test(cfg, '-r', p)) || !(await test(cfg, '-w', p))) throw new Error(`EACCES: '${p}' is not readable and writable in the investigation sandbox`); } },
    grep: { isDirectory, readFile: async p => (await readBytes(cfg, p)).toString('utf8') },
    find: { exists: p => test(cfg, '-e', p) },
    ls: { exists: p => test(cfg, '-e', p),
      stat: async p => { const dir = await isDirectory(p); return { isDirectory: () => dir }; },
      readdir: async p => {
        const r = await run(cfg, ['find', p, '-mindepth', '1', '-maxdepth', '1', '-printf', '%f\\0']);
        if (r.code !== 0) throw new Error(`ENOENT: cannot list '${p}' in the investigation sandbox`);
        return r.stdout.toString('utf8').split('\0').filter(Boolean);
      } },
    bash: { exec: async (commandText, cwd, { onData, signal, timeout }) => {
      // setsid makes the shell a process-group leader; its pid file lets a timeout or abort kill the whole group.
      const token = crypto.randomUUID();
      const pidFile = `/tmp/pi-exec-${token}.pid`;
      const script = 'echo $$ > "$1"; cd -- "$2" || { echo "Working directory does not exist: $2" >&2; exit 1; }; bash -c "$3"; code=$?; rm -f "$1"; exit $code';
      const local = new AbortController();
      let timedOut = false, timer;
      // bash's kill (dash's rejects "--"): signal the whole process group of the setsid shell.
      const kill = async () => { await run(cfg, ['bash', '-c', 'p=$(cat "$1" 2>/dev/null) && kill -KILL -- "-$p"; rm -f "$1"', 'bash', pidFile]).catch(() => {}); local.abort(); };
      const onAbort = () => kill();
      if (signal) { if (signal.aborted) onAbort(); else signal.addEventListener('abort', onAbort, { once: true }); }
      if (timeout !== undefined) timer = setTimeout(() => { timedOut = true; kill(); }, timeout * 1000);
      try {
        const r = await run(cfg, ['setsid', '-w', 'bash', '-c', script, 'pi-exec', pidFile, cwd, commandText], { onData, signal: local.signal });
        if (signal?.aborted) throw new Error('aborted');
        if (timedOut) throw new Error(`timeout:${timeout}`);
        return { exitCode: r.code };
      } finally {
        if (timer) clearTimeout(timer);
        if (signal) signal.removeEventListener('abort', onAbort);
      }
    } },
  };
}

// Custom tool definitions with Pi's own names, descriptions and schemas, executing inside the sandbox. Registered as
// SDK custom tools they replace the built-in definitions of the same names (Pi's tool registry lets a custom
// definition override a built-in one).
export function sandboxToolDefinitions(workspace, names, cfg = sandboxConfig()) {
  assert(cfg, 'NDFEB_NATIVE_SANDBOX is not set');
  const ops = operations(cfg);
  return names.map(name => {
    const definition = createToolDefinition(name, workspace, { [name]: { operations: ops[name] }, read: { autoResizeImages: true, operations: ops.read } });
    return name === 'bash' ? withSandboxOutputFiles(definition, cfg) : definition;
  });
}

// Pi saves the full text of truncated bash output to a host temp file and names that path in the result. Copy the file
// to the same path inside the sandbox so the named path can be read with the sandboxed tools. October 23, 2026
// corrective release: agent stdout can print any "Full output: <path>]" text, so the path comes from Pi's structured
// result.details.fullOutputPath (text only for thrown abort/timeout errors, which carry no details), and only Pi's own
// output file qualifies: a regular, non-symlinked pi-bash-<16 hex>.log directly in the temp directory, owned by this user
// (Pi's utils/output-files.js naming). Any other named path is never read on the host.
const PI_BASH_OUTPUT = /^pi-bash-[0-9a-f]{16}\.log$/;

export function fullOutputSource(candidate, tmpRoot = os.tmpdir()) {
  if (typeof candidate !== 'string' || !path.isAbsolute(candidate)) return null;
  let root;
  try { root = fs.realpathSync(tmpRoot); } catch { return null; }
  const resolved = path.resolve(candidate);
  if (path.dirname(resolved) !== root && path.dirname(resolved) !== path.resolve(tmpRoot)) return null;
  if (!PI_BASH_OUTPUT.test(path.basename(resolved))) return null;
  let st;
  try { st = fs.lstatSync(resolved); } catch { return null; }
  if (!st.isFile() || (typeof process.getuid === 'function' && st.uid !== process.getuid())) return null;
  return resolved;
}

export function fullOutputCandidate(result, errorMessage) {
  const structured = result?.details?.fullOutputPath;
  if (structured) return structured;
  const footers = [...String(errorMessage ?? '').matchAll(/Full output: (\S+?)\]/g)];
  return footers.length ? footers[footers.length - 1][1] : null;
}

async function copyOutputFile(cfg, candidate) {
  const file = fullOutputSource(candidate);
  if (!file) return;
  try {
    await run(cfg, ['sh', '-c', 'mkdir -p "$(dirname "$1")" && cat > "$1"', 'sh', file], { input: fs.readFileSync(file) });
  } catch {}
}

function withSandboxOutputFiles(definition, cfg) {
  const execute = definition.execute.bind(definition);
  return { ...definition, async execute(...args) {
    try {
      const result = await execute(...args);
      await copyOutputFile(cfg, fullOutputCandidate(result));
      return result;
    } catch (error) {
      await copyOutputFile(cfg, fullOutputCandidate(null, error?.message));
      throw error;
    }
  } };
}

export async function sandboxIdentity(cfg = sandboxConfig()) {
  if (!cfg) return null;
  const r = await run(cfg, ['sh', '-c', 'echo "$(id -u):$(id -g) $(uname -srm)"; cat /etc/hostname']);
  assert.equal(r.code, 0, `Native sandbox ${cfg.name} is not running: ${r.stderr.trim()}`);
  const [user, host] = r.stdout.toString().trim().split('\n');
  return { backend: cfg.kind, name: cfg.name, user_and_kernel: user, hostname: host };
}
