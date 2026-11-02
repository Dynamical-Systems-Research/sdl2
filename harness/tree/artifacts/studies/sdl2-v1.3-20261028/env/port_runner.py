"""Build env/run_v13.mjs from v1.2's env/run_v12.mjs (not edited) with exact-text edits, each required to match the
stated number of times, and record the diff in env/run_v13.diff (A1, CYCLE_PLAN.md).

What changes:
  - the sandbox has no network (--network none; D10), a private IPC namespace and a neutral hostname;
  - the agent sees a neutral workspace path: pi/ is reached through a per-episode symbolic link /tmp/lab/<random token>
    and the container mounts the workspace at that same path, so Pi's own output/workspace handling is unchanged;
  - the agent package mounts are env/library-agent and env/models-agent (world rows removed);
  - v1.2's bridge environment and the Tinker route transport are used read-only; the frontier transport is v1.3's copy
    (arms/frontier/openrouter_transport.mjs, which admits GLM-5.3 Flash for D14 and reads v1.2's PINs); COVERAGE.md comes from env/coverage_v13.py from ../sdl2-v1.2-20261026;
  - the scripted check adds an offline and path probe in the sandbox, forced executor and request failures, and checks
    that no host path, user name, episode name, version string or v1.3 world identity reaches any visible surface;
  - the policy arm points at the v1.3 classical policies (A6).

  python3 env/port_runner.py
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "sdl2-v1.2-20261026/env/run_v12.mjs"
DST = HERE / "run_v13.mjs"

NEUTRAL = r"""const box='ndfeb-sdl2v13-ep-'+name;
// v1.3: the agent sees a neutral workspace path. pi/ is reached through a per-episode symbolic link /tmp/lab/<token> (a
// random token kept owner-side in LAB_TOKEN and reused on resume), and the container mounts the workspace at that same
// path, so Pi's own handling of output/workspace is unchanged and no host path, user or episode name reaches the agent.
const tokenFile=path.join(ep,'LAB_TOKEN');
if(!fs.existsSync(tokenFile))fs.writeFileSync(tokenFile,crypto.randomBytes(8).toString('hex')+'\n',{flag:'wx'});
const lab=path.join('/tmp/lab',fs.readFileSync(tokenFile,'utf8').trim()),lws=path.join(lab,'workspace');
fs.mkdirSync('/tmp/lab',{recursive:true});try{fs.unlinkSync(lab);}catch{}fs.symlinkSync(out,lab);"""

FORCED = r"""  pass('coverage_economy_counts',{expected:'preparation 15 + body 10 + anneal 6 + 3 density readings 6 + 2 loops 100 = 137; refused and reference requests free'});
  // v1.3: forced failures, so their text is part of the visible surfaces checked below
  const forced=[JSON.stringify(rejected),JSON.stringify(refused)];
  const broken=await call('stage_model',{source:'def forecast(payload):\n    raise RuntimeError("scripted executor failure")\n'});
  try{await call('adopt_model',{source_sha256:broken.source_sha256,rationale:'executor failure text',test_request:make});forced.push('adopted?');}
  catch(e){forced.push(e.message);}
  try{forced.push(JSON.stringify(await call('request_simulation',{request_id:'S-BAD',simulation:'magnetic',rationale:'malformed request',
   request:{...make,specimens:[{...make.specimens[0],specimen_id:'S-X',conditions:{...make.specimens[0].conditions,temperature_c:'hot'}}]}})));}
  catch(e){forced.push(e.message);}
  fs.writeFileSync(path.join(ep,'SCRIPTED_FORCED_FAILURES.json'),JSON.stringify(forced,null,2));
  pass('forced_executor_and_request_failures_returned',{texts:forced.length});"""

IDENTITY_OLD = r"""  const visible=[JSON.stringify(init),fs.readFileSync(path.join(ep,'TOOL_SCHEMAS.json'),'utf8'),fs.readFileSync(path.join(ep,'sources.json'),'utf8'),...pages,...visibleFiles].join('\n');
  const world=JSON.parse(fs.readFileSync(path.join(ep,'simulation-magnetic/reference_world.json')));
  const salt=crypto.createHash('sha256').update(JSON.stringify(world)).digest('hex');
  for(const secret of [world.base_world.world_id,String(world.base_world.world_seed),'"magnetic_path"','OWNER-CONFIRM','OWNER-MEDIAN',
   JSON.stringify(world.base_world.draw)])
   assert(!visible.includes(secret),'hidden world identity visible: '+secret.slice(0,80));
  pass('hidden_world_identity_absent_from_every_visible_surface',{surfaces:3+pages.length+visibleFiles.length});"""

IDENTITY_NEW = r"""  const visible=[JSON.stringify(init),fs.readFileSync(path.join(ep,'TOOL_SCHEMAS.json'),'utf8'),fs.readFileSync(path.join(ep,'sources.json'),'utf8'),...pages,...visibleFiles,...forced].join('\n');
  const world=JSON.parse(fs.readFileSync(path.join(ep,'simulation-magnetic/reference_world.json'))),bw=world.v12.base_world;
  for(const secret of [bw.world_id,String(bw.world_seed),'"magnetic_path"','OWNER-CONFIRM','OWNER-MEDIAN',JSON.stringify(bw.draw),
   '"split"','package-world-'])
   assert(!visible.includes(secret),'hidden world identity visible: '+secret.slice(0,80));
  pass('hidden_world_identity_absent_from_every_visible_surface',{surfaces:3+pages.length+visibleFiles.length+forced.length});
  const hostMarks=[study,v12,ep,out,pi,process.env.HOME,'/home/','jarrodbarnes','sdl2-v1.',name];
  const hostHits=hostMarks.filter(m=>visible.includes(m));
  assert.deepEqual(hostHits,[],'host identity visible: '+hostHits.join(', '));
  pass('no_host_path_user_episode_name_or_version_on_any_visible_surface',{marks:hostMarks.length});"""

REFERENCE_ONCE = r"""  pass('declared_degassing_completeness_refused_with_reason');
  // v1.3 (D5): the reference body is read once; an anneal or measure naming it is refused, free, with the reason
  const refAgain=await call('request_simulation',{request_id:'S-REF',simulation:'magnetic',rationale:'the reference body is read once',
   request:{origin:'synthetic',operation:'anneal',lot_id:LOT,input_stage:'as_sintered',output_stage:'after_first_anneal',
    specimens:[{specimen_id:'REF-B1',preparation_id:'REF-P',cycle_id:'REF-C1',conditions:{anneal_steps:[{stage:'after_first_anneal',
     temperature_c:900,hold_time_s:7200,heating_rate_c_min:10}],measurement_context:{piece_id:'P1',density_method:'geometric',
     field_set_id:'S-REF-ACQ'}},measurements:['density_g_cm3','hci_kOe']}]}});
  assert(!refAgain.delivery_id);assert.match(JSON.stringify(refAgain.native_diagnostics),/reference body, read once/);
  pass('reference_body_operation_refused_with_reason');"""

# Oct 29 (Phase B stop, VERIFICATION.md): Pi's grep and find spawn rg and fd on the host with the agent's path; only their
# path checks ran in the sandbox. A host-only marker beside the workspace's mount point must stay invisible to both, and
# find must see the mounted record (it saw an empty library through the host's empty mount point).
NATIVE_SEARCH_PROBE = r""" const probeMark=crypto.randomBytes(12).toString('hex');fs.writeFileSync(path.join(out,'SEARCH_PROBE.txt'),probeMark+'\n');
 try{
  const tool=n=>native.find(t=>t.name===n),text=r=>JSON.stringify(r?.content??r);
  const g=await tool('grep').execute('grep-probe',{pattern:probeMark,path:lab},new AbortController().signal,()=>{});
  const f=await tool('find').execute('find-probe',{pattern:'SEARCH_PROBE*',path:lab},new AbortController().signal,()=>{});
  const lib=await tool('find').execute('find-library',{pattern:'OVERVIEW.md',path:'library'},new AbortController().signal,()=>{});
  assert(!text(g).includes(probeMark)&&!text(f).includes('SEARCH_PROBE'),'native grep or find reached the host: '+text(g).slice(0,300)+' '+text(f).slice(0,300));
  assert(text(lib).includes('OVERVIEW.md'),'native find cannot see the mounted record: '+text(lib).slice(0,300));
 }finally{fs.rmSync(path.join(out,'SEARCH_PROBE.txt'),{force:true});}
 pass('native_grep_and_find_run_in_the_sandbox');"""

SEARCH_SHIMS = r"""// v1.3 (Oct 29): Pi's grep and find spawn rg and fd from Pi's agent bin directory on the host, with the agent's path;
// only their path checks ran in the sandbox, so a path outside the mounts searched the host. Pi's agent directory is set
// to this episode's isolated_pi_config (the directory Pi's session already uses) before Pi is imported, and its bin/
// holds rg and fd shims that run the same binaries inside this episode's sandbox. Checked by the boundary probe.
const piAgentDir=path.join(out,'isolated_pi_config');
fs.mkdirSync(path.join(piAgentDir,'bin'),{recursive:true});
for(const tool of ['rg','fd'])fs.writeFileSync(path.join(piAgentDir,'bin',tool),`#!/bin/sh\nexec docker exec ${box} /opt/pi-bin/${tool} "$@"\n`,{mode:0o755});
process.env.PI_CODING_AGENT_DIR=piAgentDir;
"""

EDITS = [
    ("const V=await import(pathToFileURL(path.join(pi,'vision_transport.mjs')));",
     SEARCH_SHIMS + "const V=await import(pathToFileURL(path.join(pi,'vision_transport.mjs')));", 1),
    ("run_v12.mjs", "run_v13.mjs", 5),
    ("// SDL-2 episode runner. The Oct 24 composition:",
     "// SDL-2 v1.3 episode runner (built from ../sdl2-v1.2-20261026/env/run_v12.mjs by env/port_runner.py; diff in\n"
     "// env/run_v13.diff): no sandbox network, a neutral workspace path, the agent package mounts, v1.3 worlds.\n"
     "// The Oct 24 composition:", 1),
    ("const here=path.dirname(fileURLToPath(import.meta.url)), study=path.dirname(here);",
     "const here=path.dirname(fileURLToPath(import.meta.url)), study=path.dirname(here);\n"
     "const v12=path.resolve(study,'../sdl2-v1.2-20261026');   // v1.2 bridge, coverage and route transports, read-only", 1),
    ("process.env.SDL2_V12_EPISODES", "process.env.SDL2_V13_EPISODES", 1),
    ("const box='ndfeb-sdl2-ep-'+name;", NEUTRAL, 1),
    ("const MASTERS={library:path.join(here,'library-master'),'models/supplied':path.join(here,'models-master/supplied')};",
     "const MASTERS={library:path.join(here,'library-agent'),'models/supplied':path.join(here,'models-agent/supplied')};", 1),
    ("process.env.NDFEB_PYTHON=path.join(study,'bridge-python/bin/python');",
     "process.env.NDFEB_PYTHON=path.join(v12,'bridge-python/bin/python');", 1),
    ("path.join(study,'route/ds_tinker_transport.mjs')", "path.join(v12,'route/ds_tinker_transport.mjs')", 1),
    # the frontier transport stays at path.join(study, ...): v1.3's copy (arms/frontier/port_transport.py; D14, GLM-5.3 Flash)
    ("path.join(here,'coverage.py')", "path.join(here,'coverage_v13.py')", 2),   # no-hints COVERAGE.md (port_env_text.py)
    ("tinker_transport:site('route/ds_tinker_transport.mjs'),\n  tinker_worker:site('route/ds_tinker_worker.py'),",
     "tinker_transport:path.join(v12,'route/ds_tinker_transport.mjs'),\n  tinker_worker:path.join(v12,'route/ds_tinker_worker.py'),", 1),
    ("arms/classical/policy_v12.py", "arms/classical/policy_v13.py", 2),
    ("arms/policy_state.py", "arms/policy_state_v13.py", 2),
    ("const trees={bridge_package:site('bridge-python/lib/python3.13/site-packages/ndfeb_sps'),",
     "const trees={bridge_package:path.join(v12,'bridge-python/lib/python3.13/site-packages/ndfeb_sps'),", 1),
    ("library_master:MASTERS.library,models_master:MASTERS['models/supplied']",
     "library_agent:MASTERS.library,models_agent:MASTERS['models/supplied']", 1),
    ("['-v',`${src}:${path.join(ws,rel)}:ro`]", "['-v',`${src}:${path.join(lws,rel)}:ro`]", 1),
    ("execFileSync('docker',['run','-d','--init','--name',box,",
     "execFileSync('docker',['run','-d','--init','--network','none','--ipc','private','--hostname','lab','--name',box,", 1),
    ("'-e','HOME='+ws,'-w',ws,\n  '-v',`${ws}:${ws}`,", "'-e','HOME='+lws,'-w',lws,\n  '-v',`${ws}:${lws}`,", 1),
    ("sandbox.sandboxToolDefinitions(ws,", "sandbox.sandboxToolDefinitions(lws,", 2),
    ("'touch models/.probe && rm models/.probe','echo isolated'",
     "'touch models/.probe && rm models/.probe',`test \"$(pwd)\" = '${lws}'`,\n"
     "  'test \"$(tail -n +2 /proc/net/route | wc -l)\" = 0','! (exec 3<>/dev/tcp/1.1.1.1/443) 2>/dev/null','echo isolated'", 1),
    ("pass('sandbox_sees_read_only_record_but_not_owner_world_commons_or_rankings');",
     "pass('sandbox_sees_read_only_record_but_not_owner_world_commons_or_rankings');\n"
     " pass('sandbox_offline_at_a_neutral_workspace_path');", 1),
    (" pass('sandbox_offline_at_a_neutral_workspace_path');",
     " pass('sandbox_offline_at_a_neutral_workspace_path');\n" + NATIVE_SEARCH_PROBE, 1),
    ("V.SessionManager.create(ws,", "V.SessionManager.create(lws,", 1),
    ("created=await T.createSession({output:out,", "created=await T.createSession({output:lab,", 1),
    ("session:manager.getSessionFile(),utc:V.utc()}", "session:manager.getSessionFile(),workspace_in_sandbox:lws,utc:V.utc()}", 1),
    ("  pass('coverage_economy_counts',{expected:'preparation 15 + body 10 + anneal 6 + 3 density readings 6 + 2 loops 100 = 137; refused and reference requests free'});",
     FORCED, 1),
    (IDENTITY_OLD, IDENTITY_NEW, 1),
    ("  pass('declared_degassing_completeness_refused_with_reason');", REFERENCE_ONCE, 1),
    (" if(started)execFileSync('docker',['rm','-f',box],{stdio:'pipe'});\n save(",
     " if(started)execFileSync('docker',['rm','-f',box],{stdio:'pipe'});\n try{fs.unlinkSync(lab);}catch{}\n save(", 1),
]


def main():
    old = SRC.read_text()
    text = old
    for a, b, n in EDITS:
        assert text.count(a) == n, (a[:90], text.count(a), n)
        text = text.replace(a, b)
    DST.write_text(text)
    diff = difflib.unified_diff(old.splitlines(True), text.splitlines(True), "sdl2-v1.2-20261026/env/run_v12.mjs",
                                "sdl2-v1.3-20261028/env/run_v13.mjs")
    (HERE / "run_v13.diff").write_text("".join(diff))
    print(f"{len(EDITS)} edits -> {DST.name}")


if __name__ == "__main__":
    main()
