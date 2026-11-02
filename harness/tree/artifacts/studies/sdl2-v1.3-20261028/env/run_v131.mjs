// SDL-2 v1.3 episode runner (built from ../sdl2-v1.2-20261026/env/run_v12.mjs by env/port_runner.py; diff in
// env/run_v13.diff): no sandbox network, a neutral workspace path, the agent package mounts, v1.3 worlds.
// The Oct 24 composition: Pi 1.4.0 (pi140), the existing Dynamical scientific bridge and
// native sandbox, and the DeepSeek Tinker worker v3 with only the 1017 model constants changed (../route/).
// Copies of the bridge and owner environments carry this study's changes; live environments are not used. The live loop
// follows the maintained launcher (pi140/launch_vision.mjs): compact_context runs native compaction at a tool boundary,
// a text-only turn gets the launcher's runtime continuation notice (at most three in a row), wait/stop end the run.
// Every arm runs here with the same contract, tools, sandbox, bridge and ledger; RUN.json in the episode names the arm:
//   {"arm": "tinker-1017", "checkpoint": null}            DeepSeek-V4.1-Flash-1017 base (or a checkpoint) on Tinker (default)
//   {"arm": "frontier", "model": "<openrouter id>"}       through OpenRouter (arms/frontier/README.md)
//   {"arm": "policy", "policy": "doe"|"bo"}               a deterministic policy (arms/POLICY_PROTOCOL.md), no model
// Optional for any model arm: "native_command_timeout_s" (a stated bound on each native bash command).
// Model arms read text only: an image part becomes the 1017 route's placeholder. Each route compacts natively at its own
// window minus 65,536 tokens (the maintained launcher's reserve; Jarrod, October 26): 196,608 on the 1017 route's 262,144
// window, about 918K-983K on the frontier routes. NDFEB_COMPACTION_RESERVE_TOKENS must be unset (its default 65,536 gives
// the Tinker route 196,608); LAUNCH.json records every route's trigger.
//   node run_v13.mjs NAME scripted        integration check, zero provider calls
//   node run_v13.mjs NAME live [resume]   the RUN.json model arm (paid; needs authorization)
//   node run_v13.mjs NAME policy          the RUN.json policy arm, zero provider calls
//   node run_v13.mjs NAME wirecheck       a frontier arm built exactly as live; its first request is answered locally
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
const here=path.dirname(fileURLToPath(import.meta.url)), study=path.dirname(here);
const v12=path.resolve(study,'../sdl2-v1.2-20261026');   // v1.2 bridge, coverage and route transports, read-only
const pi=path.resolve(study,'../material-route-swarm-design-20261019/runtime/pi140');
const name=process.argv[2], mode=process.argv[3], resume=process.argv[4]==='resume';
assert(/^[a-z0-9-]+$/.test(name)&&['scripted','live','policy','wirecheck'].includes(mode));assert(!resume||mode==='live');
const ep=path.join(process.env.SDL2_V13_EPISODES||path.join(study,'episodes'),name),out=path.join(ep,'pi'),ws=path.join(out,'workspace');
const box='ndfeb-sdl2v13-ep-'+name;
// v1.3: the agent sees a neutral workspace path. pi/ is reached through a per-episode symbolic link /tmp/lab/<token> (a
// random token kept owner-side in LAB_TOKEN and reused on resume), and the container mounts the workspace at that same
// path, so Pi's own handling of output/workspace is unchanged and no host path, user or episode name reaches the agent.
const tokenFile=path.join(ep,'LAB_TOKEN');
if(!fs.existsSync(tokenFile))fs.writeFileSync(tokenFile,crypto.randomBytes(8).toString('hex')+'\n',{flag:'wx'});
const lab=path.join('/tmp/lab',fs.readFileSync(tokenFile,'utf8').trim()),lws=path.join(lab,'workspace');
fs.mkdirSync('/tmp/lab',{recursive:true});try{fs.unlinkSync(lab);}catch{}fs.symlinkSync(out,lab);
const MASTERS={library:path.join(here,'library-agent'),'models/supplied':path.join(here,'models-agent/supplied')};
process.env.NDFEB_PYTHON=path.join(study,'bridge-python-v131/bin/python');   // v1.3.1: the legacy laboratory reports its error
process.env.NDFEB_SCIENCE_IMAGE='sha256:1bbaab57e6430f48f6d709b1197781a9e9458b8781c8f926e18bef05df69fc05';
process.env.NDFEB_SCIENCE_PROFILE='spark-science-v1';
process.env.NDFEB_NATIVE_SANDBOX=box;
process.env.NDFEB_NATIVE_SANDBOX_KIND='docker';
// v1.3 (Oct 29): Pi's grep and find spawn rg and fd from Pi's agent bin directory on the host, with the agent's path;
// only their path checks ran in the sandbox, so a path outside the mounts searched the host. Pi's agent directory is set
// to this episode's isolated_pi_config (the directory Pi's session already uses) before Pi is imported, and its bin/
// holds rg and fd shims that run the same binaries inside this episode's sandbox. Checked by the boundary probe.
const piAgentDir=path.join(out,'isolated_pi_config');
fs.mkdirSync(path.join(piAgentDir,'bin'),{recursive:true});
for(const tool of ['rg','fd'])fs.writeFileSync(path.join(piAgentDir,'bin',tool),`#!/bin/sh\nexec docker exec ${box} /opt/pi-bin/${tool} "$@"\n`,{mode:0o755});
process.env.PI_CODING_AGENT_DIR=piAgentDir;
const V=await import(pathToFileURL(path.join(pi,'vision_transport.mjs')));
const launcher=await import(pathToFileURL(path.join(pi,'launch_vision.mjs')));
const sandbox=await import(pathToFileURL(path.join(pi,'native_sandbox.mjs')));
const RUN=fs.existsSync(path.join(ep,'RUN.json'))?JSON.parse(fs.readFileSync(path.join(ep,'RUN.json'))):{};
const ARM=RUN.arm??'tinker-1017';
assert(['tinker-1017','frontier','policy'].includes(ARM),'unknown arm');
assert(mode!=='live'||ARM!=='policy');assert(mode!=='policy'||ARM==='policy');assert(mode!=='wirecheck'||ARM==='frontier');
assert(ARM!=='frontier'||typeof RUN.model==='string','a frontier arm names its model');
// Every frontier model runs through OpenRouter on one key (Jarrod, October 26), each pinned to one provider:
// arms/frontier/openrouter_transport.mjs builds its config from a PIN; pi140's vision_transport carries every request.
const transportFile=ARM==='tinker-1017'?path.join(v12,'route/ds_tinker_transport.mjs')
 :ARM==='frontier'?path.join(study,'arms/frontier/openrouter_transport.mjs'):null;
const T=['live','wirecheck'].includes(mode)?await import(pathToFileURL(transportFile)):null;
const COMPACTION_RESERVE_TOKENS=65536, KEEP_RECENT_TOKENS=20000;
const IMAGE_PLACEHOLDER='[image omitted: this model route has no image input]';   // the 1017 worker's own text
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const save=(file,x)=>fs.writeFileSync(path.join(ep,file),JSON.stringify(x,null,2)+'\n');
// The workspace is agent-writable: write a new file under a random name (O_EXCL) and rename it over the target, so a
// planted symbolic link is replaced, never followed.
const safeWrite=(file,text)=>{const tmp=path.join(path.dirname(file),'.sdl2-'+crypto.randomBytes(8).toString('hex'));
 fs.writeFileSync(tmp,text,{flag:'wx',mode:0o644});fs.renameSync(tmp,file);};
const wsave=(file,x)=>safeWrite(path.join(ws,file),JSON.stringify(x,null,2)+'\n');

// Hash of a directory tree (relative path + file hash per file, sorted; __pycache__ and node_modules skipped).
function treeHash(root){
 const h=crypto.createHash('sha256');let files=0;
 const walk=dir=>{for(const e of fs.readdirSync(dir,{withFileTypes:true}).sort((a,b)=>a.name<b.name?-1:1)){
  if(['__pycache__','node_modules'].includes(e.name))continue;
  const p=path.join(dir,e.name);
  if(e.isDirectory())walk(p);else if(e.isFile()){h.update(path.relative(root,p)+'\0'+sha(p)+'\n');files++;}}};
 walk(root);return {files,sha256:h.digest('hex')};
}

// Everything that defines the episode's behaviour; asserted unchanged on resume.
function composition(){
 const site=p=>path.join(study,p);
 const files={run:path.join(here,'run_v13.mjs'),coverage:path.join(here,'coverage_v13.py'),contract:path.join(here,'CONTRACT.md'),
  laboratory:path.join(here,'LABORATORY.md'),tinker_transport:path.join(v12,'route/ds_tinker_transport.mjs'),
  tinker_worker:path.join(v12,'route/ds_tinker_worker.py'),native_adapter:site('python/native-runtime/dynamical/instruments/ndfeb_sps.py'),
  pi_launch_vision:path.join(pi,'launch_vision.mjs'),pi_package_lock:path.join(pi,'package-lock.json'),
  ...(transportFile?{arm_transport:transportFile}:{}),
  ...(ARM==='policy'?{policy:path.join(study,'arms/classical/policy_v13.py'),policy_state:path.join(study,'arms/policy_state_v13.py')}:{}),
  legacy_simulation_config:path.join(ep,'simulation/owner_simulation_runtime.json')};
 const trees={bridge_package:path.join(study,'bridge-python-v131/lib/python3.13/site-packages/ndfeb_sps'),
  owner_package:site('python/lib/python3.13/site-packages/ndfeb_sps'),owner_native_runtime:site('python/native-runtime'),
  library_agent:MASTERS.library,models_agent:MASTERS['models/supplied']};
 return {pi_version:V.PI_VERSION,pi_transport_sha256:launcher.transportHash(),
  compaction_reserve_env:process.env.NDFEB_COMPACTION_RESERVE_TOKENS??null,
  files:Object.fromEntries(Object.entries(files).map(([k,p])=>[k,sha(p)])),
  trees:Object.fromEntries(Object.entries(trees).map(([k,p])=>[k,treeHash(p)]))};
}

// The model arm's provider configuration. A frontier route keeps its own transport configuration and records its own
// compaction point, its window minus the reserve (hashed into provider_config_sha256 with the rest).
function armConfig(){
 if(ARM==='tinker-1017'){const c=T.modelConfig(T.ROUTE_ID);c.checkpoint=RUN.checkpoint??null;return c;}
 const c=T.modelConfig(RUN.model);
 c.compaction_settings={reserveTokens:COMPACTION_RESERVE_TOKENS,keepRecentTokens:KEEP_RECENT_TOKENS};
 c.compaction={...c.compaction_settings,trigger_tokens:c.model.contextWindow-COMPACTION_RESERVE_TOKENS};
 c.images='placeholder';
 return c;
}
// Every image part of a provider request becomes the 1017 route's placeholder text (tool results and messages alike).
function textOnly(context){
 const strip=content=>Array.isArray(content)?content.map(c=>c?.type==='image'?{type:'text',text:IMAGE_PLACEHOLDER}
  :c?.type==='toolResult'||Array.isArray(c?.content)?{...c,content:strip(c.content)}:c):content;
 return {...context,messages:(context.messages??[]).map(m=>({...m,content:strip(m.content)}))};
}

// RETURNS.json in the workspace is the agent's copy; the runner's own copy lives outside it (resume reads that one).
class V12Bridge extends V.Bridge {
 constructor(){super();const own=path.join(ep,'RETURNS.owner.json');
  this.returns=fs.existsSync(own)?JSON.parse(fs.readFileSync(own)):[];this.n=0;}
 async publicDelivery(result){
  const artifact=result.artifacts?.find(a=>a.name==='synthetic-response.json');
  if(!artifact)return null;
  let offset=0,body='';
  do{const r=await super.call('tool',{call_id:'sdl2-read-'+Date.now()+'-'+(++this.n),name:'read_artifact',
    arguments:{sha256:artifact.sha256,offset,limit_bytes:16000}});body+=r.content;offset=r.next_offset;}
  while(offset!==null&&offset!==undefined);
  return JSON.parse(body);
 }
 refresh(){
  save('RETURNS.owner.json',this.returns);wsave('RETURNS.json',this.returns);
  execFileSync(process.env.NDFEB_PYTHON,['-B',path.join(here,'coverage_v13.py'),ws,path.join(ep,'RETURNS.owner.json')],{stdio:'pipe'});
 }
 async record(args,result){
  const delivery=await this.publicDelivery(result);
  this.returns.push({request_id:args.request_id,request:args.request,delivery_id:result.delivery_id,utc:V.utc(),delivery});
  this.refresh();
  return delivery;
 }
 async call(method,params={}){
  const r=await super.call(method,params);
  if(method==='init'){
   // The episode's public accounting through the bridge's existing interface: no inference budget gates this lot.
   fs.writeFileSync(path.join(out,'scientific_state/resource_status.json'),JSON.stringify({status:'sdl2 episode',
    request_commitment_allowed:true,note:'Credits and bodies for this lot are in COVERAGE.md; no inference budget applies.'})+'\n');
  }
  if(method==='init'&&!fs.existsSync(path.join(ws,'INITIAL_EVIDENCE.json'))){
   const [request]=JSON.parse(fs.readFileSync(path.join(ep,'INITIAL_REQUESTS.json')));
   const args={request_id:'REF-INITIAL',simulation:'magnetic',rationale:'Reference body of this lot (free)',request};
   const result=await super.call('tool',{call_id:'sdl2-initial',name:'request_simulation',arguments:args});
   fs.appendFileSync(path.join(ep,'initial_calls.jsonl'),JSON.stringify({utc:V.utc(),args,result})+'\n');
   if(!result.delivery_id)throw new Error('Reference body failed: '+JSON.stringify(result).slice(0,2000));
   const delivery=await this.record(args,result);
   wsave('INITIAL_EVIDENCE.json',{note:'Reference body REF-B1 of this lot: made by fitted degassing, 4.0 um, 850 C, 300 s, 10 C/min, 60 MPa, on_cooling (default ramp), compact current path; as sintered; one geometric density and one full loop. Free; not one of your 12 bodies.',
    observations:delivery.observations.map(o=>({observation_id:o.observation_id,quantity:o.quantity,value:o.value,unit:o.unit,
     specimen_id:o.specimen_id,preparation_id:o.preparation_id,piece_id:o.piece_id,process_stage:o.process_stage,
     acquisition_id:o.acquisition_id,density_method:o.density_method,uncertainty:o.uncertainty,loop_reading:o.loop_reading}))});
  }else if(method==='init'){this.refresh();}
  if(['tool','recover_tool'].includes(method)&&params.name==='request_simulation'&&r?.delivery_id
   &&!this.returns.some(x=>x.delivery_id===r.delivery_id))await this.record(params.arguments,r);
  return r;
 }
}

let bridge,session,started=false;
for(const sig of ['SIGTERM','SIGINT'])process.once(sig,()=>{
 try{if(started)execFileSync('docker',['rm','-f',box],{stdio:'pipe'});}catch{}
 try{save(mode==='live'?'LIVE-interrupted.json':'E2E-interrupted.json',{status:'interrupted',signal:sig,utc:new Date().toISOString(),checks:receipt.checks});}catch{}
 process.exit(128+(sig==='SIGTERM'?15:2));});
const instructionsOf=()=>fs.readFileSync(path.join(ws,'START_HERE.md'),'utf8')+'\n'+
 'Pi native read, bash, edit, write, grep, find and ls run in a sandbox on your workspace. The scientific tools use the episode ledger. Owner worlds, credentials and graders are not available to you.';
const receipt={mode,arm:ARM,model:RUN.model??null,pi:V.PI_VERSION,checks:[],provider_calls:0,composition:composition()};
const pass=(n,detail={})=>{receipt.checks.push({name:n,pass:true,...detail});console.log('PASS',n);};
try{
 const bins=path.join(process.env.HOME,'.pi/agent/bin');
 const mounts=Object.entries(MASTERS).flatMap(([rel,src])=>['-v',`${src}:${path.join(lws,rel)}:ro`]);
 // The live campaign's sandbox limits, 6 CPUs and 16 GB, stated to every arm in LABORATORY.md; and HOME is the workspace, so a
 // file written under ~ is kept (two pilot claims were lost in the image's /tmp/home; Jarrod, October 26).
 execFileSync('docker',['run','-d','--init','--network','none','--ipc','private','--hostname','lab','--name',box,'--user','1000:1000','--cap-drop','ALL',
  '--security-opt','no-new-privileges','--pids-limit','4096','--memory','16g','--cpus','6','-e','HOME='+lws,'-w',lws,
  '-v',`${ws}:${lws}`,...mounts,'-v',`${bins}/rg:/opt/pi-bin/rg:ro`,'-v',`${bins}/fd:/opt/pi-bin/fd:ro`,
  '--entrypoint','sh','sha256:1b1f2036d24ec4f78ecca6aedd0c956ba903342e04d70e3ed712da37ec4ae34a',
  '-c','mkdir -p /tmp/home /tmp/mplconfig && exec sleep infinity'],{stdio:'pipe'});started=true;
 const native=sandbox.sandboxToolDefinitions(lws,V.NATIVE_TOOL_NAMES,{kind:'docker',name:box});
 const probe=await native.find(t=>t.name==='bash').execute('boundary',{command:[
  `test ! -e '${ep}/simulation-magnetic/reference_world.json'`,`test ! -e '${study}/owner'`,`test ! -e '${ep}/RETURNS.owner.json'`,
  'test ! -e library/commons','test ! -e library/own','test ! -e library/source_archive',
  'test -e library/evidence/OVERVIEW.md','test -e models/supplied/C7.3/TOOL_MANIFEST.json',
  'test ! -e models/supplied/C7.3/build/DECISION_ANALYSIS.json',
  '! touch library/.probe 2>/dev/null','! chmod u+w library/README.md 2>/dev/null','! touch models/supplied/.probe 2>/dev/null',
  'touch models/.probe && rm models/.probe',`test "$(pwd)" = '${lws}'`,
  'test "$(tail -n +2 /proc/net/route | wc -l)" = 0','! (exec 3<>/dev/tcp/1.1.1.1/443) 2>/dev/null','echo isolated'].join(' && '),timeout:30},new AbortController().signal,()=>{});
 assert.match(JSON.stringify(probe),/isolated/,JSON.stringify(probe).slice(0,1500));
 pass('sandbox_sees_read_only_record_but_not_owner_world_commons_or_rankings');
 pass('sandbox_offline_at_a_neutral_workspace_path');
 const probeMark=crypto.randomBytes(12).toString('hex');fs.writeFileSync(path.join(out,'SEARCH_PROBE.txt'),probeMark+'\n');
 try{
  const tool=n=>native.find(t=>t.name===n),text=r=>JSON.stringify(r?.content??r);
  const g=await tool('grep').execute('grep-probe',{pattern:probeMark,path:lab},new AbortController().signal,()=>{});
  const f=await tool('find').execute('find-probe',{pattern:'SEARCH_PROBE*',path:lab},new AbortController().signal,()=>{});
  const lib=await tool('find').execute('find-library',{pattern:'OVERVIEW.md',path:'library'},new AbortController().signal,()=>{});
  assert(!text(g).includes(probeMark)&&!text(f).includes('SEARCH_PROBE'),'native grep or find reached the host: '+text(g).slice(0,300)+' '+text(f).slice(0,300));
  assert(text(lib).includes('OVERVIEW.md'),'native find cannot see the mounted record: '+text(lib).slice(0,300));
 }finally{fs.rmSync(path.join(out,'SEARCH_PROBE.txt'),{force:true});}
 pass('native_grep_and_find_run_in_the_sandbox');
 bridge=new V12Bridge();
 const init=await bridge.call('init',{campaign:path.join(ep,'campaign'),directory:path.join(out,'scientific_state'),
  sources:JSON.parse(fs.readFileSync(path.join(ep,'sources.json')))});
 assert(init.tools.some(t=>t.name==='request_simulation'));pass('bridge_exposes_scientific_tools');
 fs.writeFileSync(path.join(ep,'TOOL_SCHEMAS.json'),JSON.stringify(init.tools,null,2));
 assert(fs.existsSync(path.join(ws,'INITIAL_EVIDENCE.json'))&&fs.existsSync(path.join(ws,'COVERAGE.md')));pass('reference_body_and_coverage_written');
 let count=fs.existsSync(path.join(ep,'calls.jsonl'))?fs.readFileSync(path.join(ep,'calls.jsonl'),'utf8').trim().split('\n').filter(Boolean).length:0;
 const call=async(n,args,piCallId)=>{
  const call_id=piCallId??('sdl2-'+Date.now().toString(36)+'-'+(++count));let value;
  try{value=await bridge.call('tool',{call_id,name:n,arguments:args});}
  catch(error){
   const result={status:'rejected',error:error.bridgeError??{code:'bridge_unavailable',message:'Bridge connection failed. No result or observation is inferred.'}};
   fs.appendFileSync(path.join(ep,'calls.jsonl'),JSON.stringify({call_id,name:n,args,result})+'\n');
   throw new Error(JSON.stringify(result));
  }
  fs.appendFileSync(path.join(ep,'calls.jsonl'),JSON.stringify({call_id,name:n,args,result:value})+'\n');
  return value;
 };
 const instructions=instructionsOf();receipt.instructions_sha256=V.sha(instructions);
 if(mode==='scripted'){
  const LOT='SDL2-LOT';
  const model=unit=>`def forecast(payload):
    rows = []
    stage = payload["request"].get("output_stage", "as_sintered")
    for s in payload["request"]["specimens"]:
        rows.append({"specimen_id": s["specimen_id"],
            "density_g_cm3": {"mean": 7.0, "sd": 0.25, "unit": "g/cm^3", "status": "ok", "measurement_stage": stage},
            "grain_size_nm": {"mean": None, "sd": None, "unit": "nm", "status": "abstained", "reason": "no grain model"},
            "hci_kOe": {"mean": 8.0, "sd": 2.0, "unit": "${unit}", "status": "ok", "measurement_stage": stage},
            "br_kGs": {"mean": 7.4, "sd": 0.5, "unit": "kGs", "status": "ok", "measurement_stage": stage},
            "bhmax_MGOe": {"mean": None, "sd": None, "unit": "MGOe", "status": "abstained", "reason": "scripted abstention check"}})
    return {"predictions": rows}
`;
  const prep={preparation_id:'S-P',lot_id:LOT,conditions:{method:'vacuum_degassing',temperature_c:600,hold_time_s:900,feedstock_d50_um:4.0}};
  const make={origin:'synthetic',operation:'manufacture',lot_id:LOT,preparations:[prep],specimens:[{specimen_id:'S-B1',preparation_id:'S-P',cycle_id:'S-C1',
   conditions:{temperature_c:900,heating_rate_c_min:10,hold_time_s:300,pressure_mpa:60,pressure_schedule:'on_cooling',mass_g:.9,diameter_mm:10,
   measurement_context:{piece_id:'P1',density_method:'geometric',field_set_id:'S-ACQ-1'}},measurements:['density_g_cm3','hci_kOe','br_kGs','bhmax_MGOe']}]};
  const wrong=await call('stage_model',{source:model('T')});
  let rejected=null;
  try{await call('adopt_model',{source_sha256:wrong.source_sha256,rationale:'wrong Hci unit must be refused',test_request:make});}
  catch(e){rejected=JSON.parse(e.message);}
  assert(rejected&&rejected.status==='rejected'&&/hci_kOe/.test(JSON.stringify(rejected)),JSON.stringify(rejected).slice(0,800));
  pass('virtual_magnetic_forecast_with_wrong_unit_refused_at_adoption');
  const staged=await call('stage_model',{source:model('kOe')});
  const adopted=await call('adopt_model',{source_sha256:staged.source_sha256,rationale:'Scripted integration model with magnetic forecasts',test_request:make});
  assert.equal(adopted.status,'adopted',JSON.stringify(adopted).slice(0,1500));pass('model_with_magnetic_forecasts_adopted');
  const preview=await call('preview_request',{request:make,simulation:'magnetic'});
  const targets=(preview.forecast_targets||[]).filter(t=>['hci_kOe','br_kGs','bhmax_MGOe'].includes(t.quantity));
  assert(targets.length===3&&targets.every(t=>t.requested),JSON.stringify(preview).slice(0,2000));pass('virtual_magnetic_forecast_targets_requested');
  assert(!/request_commitment_allowed":false/.test(JSON.stringify(preview)),'preview reports a blocked commitment');pass('preview_reports_no_inference_gate');
  const refused=await call('request_simulation',{request_id:'S-COMPLETE',simulation:'magnetic',rationale:'the instrument must refuse declared completeness',
   request:{...make,preparations:[{...prep,preparation_id:'S-Q',conditions:{...prep.conditions,degassing_completeness:'complete'}}],
    specimens:[{...make.specimens[0],specimen_id:'S-Q1',preparation_id:'S-Q',cycle_id:'S-QC'}]}});
  assert(!refused.delivery_id);assert.match(JSON.stringify(refused.native_diagnostics),/degassing completeness is the outcome/);
  pass('declared_degassing_completeness_refused_with_reason');
  // v1.3 (D5): the reference body is read once; an anneal or measure naming it is refused, free, with the reason
  const refAgain=await call('request_simulation',{request_id:'S-REF',simulation:'magnetic',rationale:'the reference body is read once',
   request:{origin:'synthetic',operation:'anneal',lot_id:LOT,input_stage:'as_sintered',output_stage:'after_first_anneal',
    specimens:[{specimen_id:'REF-B1',preparation_id:'REF-P',cycle_id:'REF-C1',conditions:{anneal_steps:[{stage:'after_first_anneal',
     temperature_c:900,hold_time_s:7200,heating_rate_c_min:10}],measurement_context:{piece_id:'P1',density_method:'geometric',
     field_set_id:'S-REF-ACQ'}},measurements:['density_g_cm3','hci_kOe']}]}});
  assert(!refAgain.delivery_id);assert.match(JSON.stringify(refAgain.native_diagnostics),/reference body, read once/);
  pass('reference_body_operation_refused_with_reason');
  const m=await call('request_simulation',{request_id:'S-MAKE',simulation:'magnetic',rationale:'Scripted body',request:make});
  assert(m.delivery_id,JSON.stringify(m).slice(0,2000));pass('manufacture_with_loop');
  const anneal={origin:'synthetic',operation:'anneal',lot_id:LOT,input_stage:'as_sintered',output_stage:'after_first_anneal',
   specimens:[{specimen_id:'S-B1',preparation_id:'S-P',cycle_id:'S-C1',conditions:{anneal_steps:[{stage:'after_first_anneal',temperature_c:900,hold_time_s:7200,heating_rate_c_min:10}],
   measurement_context:{piece_id:'P1',density_method:'geometric',field_set_id:'S-ACQ-2'}},measurements:['density_g_cm3','hci_kOe','br_kGs','bhmax_MGOe']}]};
  const a=await call('request_simulation',{request_id:'S-ANNEAL',simulation:'magnetic',rationale:'Scripted anneal',request:anneal});
  assert(a.delivery_id,JSON.stringify(a).slice(0,2000));pass('anneal_with_loop');
  const measure={origin:'synthetic',operation:'measure',lot_id:LOT,input_stage:'after_first_anneal',output_stage:'after_first_anneal',
   specimens:[{specimen_id:'S-B1',preparation_id:'S-P',cycle_id:'S-C1',conditions:{mass_g:.9,
   measurement_context:{piece_id:'P1',density_method:'archimedes_saturated',field_set_id:'S-ACQ-3'}},measurements:['density_g_cm3']}]};
  const r3=await call('request_simulation',{request_id:'S-MEASURE',simulation:'magnetic',rationale:'Scripted re-measure',request:measure});
  assert(r3.delivery_id,JSON.stringify(r3).slice(0,2000));pass('measure_existing_body');
  // dev.2's ledger holds each acquisition once: reusing a field_set_id under a new request ID is refused (stated in CONTRACT).
  let again=null;
  try{await call('request_simulation',{request_id:'S-MEASURE-AGAIN',simulation:'magnetic',rationale:'Same acquisition under a new request ID',request:measure});}
  catch(e){again=JSON.parse(e.message);}
  assert.equal(again?.status,'rejected');pass('reused_acquisition_id_in_new_request_refused');
  let page,offset=0,body='';
  do{page=await call('inspect_feedback',{request_id:'magnetic:S-ANNEAL',offset,limit:16000});body+=page.content;offset=page.next_offset;}
  while(offset!==null&&offset!==undefined);
  const fb=JSON.parse(body);
  fs.writeFileSync(path.join(ep,'SCRIPTED_FEEDBACK.json'),JSON.stringify(fb,null,2));
  const byq=Object.fromEntries((fb.targets||[]).map(t=>[t.quantity,t]));
  for(const q of ['density_g_cm3','hci_kOe','br_kGs'])assert.equal(byq[q]?.forecast_state,'numeric',q+': '+JSON.stringify(byq[q]).slice(0,800));
  assert.equal(byq.bhmax_MGOe?.forecast_state,'abstained',JSON.stringify(byq.bhmax_MGOe).slice(0,800));
  pass('magnetic_forecast_feedback_returned',{targets:Object.fromEntries(Object.entries(byq).map(([q,t])=>[q,{forecast:t.forecast_state,measurement:t.measurement_state,exclusions:(t.exclusions||[]).map(e=>e.reason)}]))});
  const cov=fs.readFileSync(path.join(ws,'COVERAGE.md'),'utf8');
  assert.match(cov,/Spent 137 of 570 credits; 1 of 12 new bodies/,cov);
  pass('coverage_economy_counts',{expected:'preparation 15 + body 10 + anneal 6 + 3 density readings 6 + 2 loops 100 = 137; refused and reference requests free'});
  // v1.3: forced failures, so their text is part of the visible surfaces checked below
  const forced=[JSON.stringify(rejected),JSON.stringify(refused)];
  const broken=await call('stage_model',{source:'def forecast(payload):\n    raise RuntimeError("scripted executor failure")\n'});
  try{await call('adopt_model',{source_sha256:broken.source_sha256,rationale:'executor failure text',test_request:make});forced.push('adopted?');}
  catch(e){forced.push(e.message);}
  try{forced.push(JSON.stringify(await call('request_simulation',{request_id:'S-BAD',simulation:'magnetic',rationale:'malformed request',
   request:{...make,specimens:[{...make.specimens[0],specimen_id:'S-X',conditions:{...make.specimens[0].conditions,temperature_c:'hot'}}]}})));}
  catch(e){forced.push(e.message);}
  // v1.3.1: the request-contract errors the board's agents met, each returned word for word (free, no reading)
  const legacy=await call('request_simulation',{request_id:'S-LEGACY',rationale:'a new preparation in the legacy laboratory (simulation omitted)',
   request:{...make,preparations:[{...prep,preparation_id:'S-L'}],specimens:[{...make.specimens[0],specimen_id:'S-L1',preparation_id:'S-L',cycle_id:'S-LC',
    conditions:{...make.specimens[0].conditions,measurement_context:{piece_id:'P1',density_method:'geometric',field_set_id:'S-ACQ-L'}}}]}});
  assert.equal(legacy.status,'execution_failed',JSON.stringify(legacy).slice(0,1500));
  assert.match(JSON.stringify(legacy.native_diagnostics),/this conditional family implements only the accepted preparation/);
  pass('legacy_laboratory_error_returned_word_for_word');
  const noAcq=await call('request_simulation',{request_id:'S-NOACQ',simulation:'magnetic',rationale:'a density reading without an acquisition ID',
   request:{...measure,specimens:[{...measure.specimens[0],conditions:{...measure.specimens[0].conditions,
    measurement_context:{piece_id:'P1',density_method:'archimedes_saturated'}}}]}});
  assert(!noAcq.delivery_id,JSON.stringify(noAcq).slice(0,1500));
  assert.match(JSON.stringify(noAcq.native_diagnostics),/requires measurement_context\.field_set_id/);
  pass('missing_acquisition_id_error_returned_word_for_word');
  const undeclared=await call('request_simulation',{request_id:'S-UNDECLARED',simulation:'magnetic',rationale:'a body on REF-P without declaring it',
   request:{...make,preparations:[],specimens:[{...make.specimens[0],specimen_id:'S-R1',preparation_id:'REF-P',cycle_id:'S-RC',
    conditions:{...make.specimens[0].conditions,measurement_context:{piece_id:'P1',density_method:'geometric',field_set_id:'S-ACQ-R'}}}]}});
  assert(!undeclared.delivery_id,JSON.stringify(undeclared).slice(0,1500));
  assert.match(JSON.stringify(undeclared.native_diagnostics),/declare preparation 'REF-P' with its conditions in request\.preparations/);
  pass('undeclared_preparation_error_returned_word_for_word');
  forced.push(JSON.stringify(legacy),JSON.stringify(noAcq),JSON.stringify(undeclared));
  fs.writeFileSync(path.join(ep,'SCRIPTED_FORCED_FAILURES.json'),JSON.stringify(forced,null,2));
  pass('forced_executor_and_request_failures_returned',{texts:forced.length});
  const defs=V.serialTools(init.tools,async(spec,callId,args)=>launcher.toolContent(await call(spec.name,args)).then(({content})=>({content})));
  const rendered=await defs.find(t=>t.name==='read_evidence').execute('pi-render-check',{offset:0,limit:1000});
  assert.match(rendered.content[0].text,/request_echo/);pass('pi_scientific_tool_content_is_renderable');
  const pages=[];
  const pageAll=async(args)=>{let o2=0;do{const s=await call('inspect_state',{...args,offset:o2,limit:16000});pages.push(s.content??JSON.stringify(s));o2=s.next_offset;}while(o2!==null&&o2!==undefined&&pages.length<2000);};
  await pageAll({});
  const summary=JSON.parse(pages[0].startsWith('{')?pages.join(''):'{}');
  for(const section of (summary.sections||[]))await pageAll({section});
  const visibleFiles=[];
  const walk=dir=>{for(const e of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,e.name);
   if(['library','supplied'].includes(e.name)&&dir.startsWith(ws))continue;
   if(e.isDirectory())walk(p);else if(e.isFile()&&fs.statSync(p).size<5e6)visibleFiles.push(fs.readFileSync(p,'utf8'));}};
  walk(ws);
  const visible=[JSON.stringify(init),fs.readFileSync(path.join(ep,'TOOL_SCHEMAS.json'),'utf8'),fs.readFileSync(path.join(ep,'sources.json'),'utf8'),...pages,...visibleFiles,...forced].join('\n');
  const world=JSON.parse(fs.readFileSync(path.join(ep,'simulation-magnetic/reference_world.json'))),bw=world.v12.base_world;
  for(const secret of [bw.world_id,String(bw.world_seed),'"magnetic_path"','OWNER-CONFIRM','OWNER-MEDIAN',JSON.stringify(bw.draw),
   '"split"','package-world-'])
   assert(!visible.includes(secret),'hidden world identity visible: '+secret.slice(0,80));
  pass('hidden_world_identity_absent_from_every_visible_surface',{surfaces:3+pages.length+visibleFiles.length+forced.length});
  const hostMarks=[study,v12,ep,out,pi,process.env.HOME,'/home/','jarrodbarnes','sdl2-v1.',name];
  const hostHits=hostMarks.filter(m=>visible.includes(m));
  assert.deepEqual(hostHits,[],'host identity visible: '+hostHits.join(', '));
  pass('no_host_path_user_episode_name_or_version_on_any_visible_surface',{marks:hostMarks.length});
  wsave('decision.json',{claim:{form:'restrict',p_any_route:0.3},premises:[],next_action:'scripted fixture; no strategy',rationale:'Scripted integration check, not a model rollout.'});
  const stop=await call('record_decision',{decision:'stop',rationale:'Scripted integration check complete.'});assert(stop);pass('record_decision_stop');
 }else if(mode==='policy'){
  // A deterministic policy (arms/POLICY_PROTOCOL.md): the runner owns every tool call; the policy picks the next action
  // from what an agent sees. Same bridge, ledger, coverage and decision file as every model arm; no model, no provider.
  assert(['doe','bo'].includes(RUN.policy),'policy arm names doe or bo');
  const py=path.join(study,'python/bin/python');
  const seed=parseInt(crypto.createHash('sha256').update(`sdl2-policy:${RUN.policy}:${name}`).digest('hex').slice(0,12),16);
  const history=[];let refusals=0,decided=false;
  for(let step=0;step<200&&refusals<5&&!decided;step++){
   const stateFile=path.join(ep,'policy_state.json');
   execFileSync(py,['-B',path.join(study,'arms/policy_state_v13.py'),ep,RUN.policy,String(seed),stateFile],{stdio:'pipe',input:JSON.stringify(history)});
   const action=JSON.parse(execFileSync(py,['-B',path.join(study,'arms/classical/policy_v13.py'),stateFile],{stdio:['ignore','pipe','pipe'],maxBuffer:64<<20}).toString());
   const entry={step,state_sha256:sha(stateFile),action};
   try{
    if(action.tool==='request_simulation'){
     const r=await call('request_simulation',action.arguments);
     entry.result=r.delivery_id?{status:'delivered',delivery_id:r.delivery_id}:{status:'refused',diagnostics:r.native_diagnostics??r};
    }else if(action.tool==='stage_and_adopt'){
     const staged=await call('stage_model',{source:action.source});
     const adopted=await call('adopt_model',{source_sha256:staged.source_sha256,rationale:action.rationale??'policy forecast model',test_request:action.test_request});
     entry.result={status:adopted.status};
    }else if(action.tool==='decide'){
     wsave('decision.json',action.decision);
     await call('record_decision',{decision:'stop',rationale:'Policy decision written to decision.json.'});
     entry.result={status:'stopped'};decided=true;
    }else entry.result={status:'refused',diagnostics:'unknown action'};
   }catch(e){entry.result={status:'refused',diagnostics:String(e.message).slice(0,4000)};}
   refusals=entry.result.status==='refused'?refusals+1:0;
   history.push({step,action,result:entry.result});
   fs.appendFileSync(path.join(ep,'policy_calls.jsonl'),JSON.stringify(entry)+'\n');
  }
  receipt.policy={arm:RUN.policy,seed,actions:history.length,decided,refusals_at_end:refusals};
  receipt.decision_written=fs.existsSync(path.join(ws,'decision.json'));
  receipt.status=decided?'pass':'incomplete';
  pass('policy_episode_returned',receipt.policy);
 }else{
  const config=armConfig();
  const previous=resume?JSON.parse(fs.readFileSync(path.join(ep,'LAUNCH.json'))):null;
  if(previous){
   assert.equal(previous.model,config.model.id);assert.equal(previous.checkpoint,config.checkpoint??null);
   assert.equal(previous.instructions_sha256,V.sha(instructions));
   assert.deepEqual(previous.composition,receipt.composition,'Composition changed during a preserved investigation');
   assert.equal(previous.tool_schema_sha256,V.sha(JSON.stringify(init.tools)),'Tool contract changed during a preserved investigation');
   assert.equal(previous.provider_config_sha256,V.sha(JSON.stringify(config)),'Provider configuration changed during a preserved investigation');
  }
  let pause,pendingCompaction,created;
  const scientific=V.serialTools(init.tools,async(spec,callId,args)=>{
   if(pause||pendingCompaction!==undefined)return {content:[{type:'text',text:JSON.stringify({status:'paused',reason:'No more effects after wait/stop or a compaction request in this turn.'})}],isError:true,terminate:true};
   const r=await call(spec.name,args,callId);
   if(spec.name==='compact_context'&&r.status==='requested'){pendingCompaction=args.notes;created.settingsManager.applyOverrides({compaction:{enabled:false}});}
   if(spec.name==='record_decision'&&['wait','stop'].includes(args.decision)){pause=args.decision;created.settingsManager.applyOverrides({compaction:{enabled:false}});}
   const {content}=await launcher.toolContent(r);
   return {content,terminate:Boolean(pause)||pendingCompaction!==undefined};
  });
  const manager=previous?V.SessionManager.open(previous.session):V.SessionManager.create(lws,path.join(out,'pi_sessions'));
  if(previous)await launcher.recoverToolResults(manager,bridge,out);   // reconcile calls accepted before the interruption
  // A stated native-command bound (RUN.json native_command_timeout_s) wraps the sandboxed bash definition; the native tools
  // keep their names, schemas and order, so the model-visible tool contract is unchanged. Without it, Pi's own path.
  const limit=RUN.native_command_timeout_s??null;
  const natives=limit?sandbox.sandboxToolDefinitions(lws,V.NATIVE_TOOL_NAMES,{kind:'docker',name:box}).map(d=>d.name!=='bash'?d
   :{...d,execute:(id,params,signal,onUpdate)=>d.execute(id,{...params,timeout:Math.min(Number(params?.timeout??limit),limit)},signal,onUpdate)}):[];
  const key=ARM==='frontier'?(mode==='wirecheck'?'wirecheck-placeholder-key':process.env[config.api_key_env]):undefined;
  assert(ARM!=='frontier'||key,`${config.api_key_env} is not set`);
  created=await T.createSession({output:lab,instructions,tools:[...natives,...scientific],manager,config,nativeTools:!limit,
   ...(key?{key}:{})});
  ({session}=created);
  if(ARM==='frontier'){   // this route's own compaction point: contextWindow - 65,536
   created.settings.compaction={...created.settings.compaction,...config.compaction_settings};
   created.settingsManager.applyOverrides({compaction:created.settings.compaction});
  }
  // Pi's effective compaction settings for this session (model overrides, ordinary settings, defaults), as Pi resolves them
  const effective=created.settingsManager.getCompactionSettings(session.model);
  const effectiveCompaction={...effective,trigger_tokens:config.model.contextWindow-effective.reserveTokens};
  const wire=[];
  const fetchImpl=mode==='wirecheck'?async(input,init)=>{wire.push({url:typeof input==='string'?input:input.url??String(input),
    body:String(init?.body??''),header_names:Object.keys(init?.headers??{})});
   return new Response(JSON.stringify({error:{type:'invalid_request_error',message:'wirecheck: answered locally, no provider call'}}),
    {status:400,headers:{'content-type':'application/json'}});}:fetch;
  const transport=ARM==='tinker-1017'?T.installTransport(session,out,instructions,config)
   :T.installTransport(session,out,instructions,config,{key,fetchImpl,...(mode==='wirecheck'?{maxRequests:1}:{})});
  let imageCheck=null;
  if(ARM==='frontier'){   // text only on every model route: an image part reaches the provider as the 1017 placeholder
   const inner=session.modelRuntime.streamSimple;
   session.modelRuntime.streamSimple=(model,incoming,options)=>inner(model,textOnly(incoming),options);
   if(mode==='wirecheck'){   // the installed wrapper, fed a tool result with an image, passes on only the placeholder
    let seen=null;
    const probe=(model,incoming)=>{seen=incoming;return null;};
    const wrapped=(model,incoming,options)=>probe(model,textOnly(incoming),options);
    wrapped(config.model,{messages:[{role:'toolResult',toolName:'view_image',content:[{type:'text',text:'x'},
     {type:'image',data:'iVBORw0KGgo=',mimeType:'image/png'}]}]},{});
    imageCheck=JSON.stringify(seen);
    assert(!imageCheck.includes('"type":"image"')&&imageCheck.includes(IMAGE_PLACEHOLDER),'image part survived textOnly');
    assert.notEqual(session.modelRuntime.streamSimple,inner,'textOnly wrapper not installed');
   }
  }
  const notices=launcher.installTextTurnContinuation(session,out,resume?'resume':'launch',()=>pause?'pause'
   :pendingCompaction!==undefined?'compaction_pending':transport.blocked?'transport_blocked':null);
  const finish=session.agent.finishTurn;
  session.agent.finishTurn=async(turn,signal)=>{const r=await finish?.(turn,signal);return pause||pendingCompaction!==undefined?{action:'end'}:r;};
  save(mode==='wirecheck'?'WIRECHECK-LAUNCH.json':resume?'LAUNCH-resume.json':'LAUNCH.json',{arm:ARM,model:config.model.id,
   route:T.ROUTE_ID??config.inference_route,checkpoint:config.checkpoint??null,reasoning:'high',images:ARM==='frontier'?'placeholder':'route',
   native_command_timeout_s:limit,effective_compaction:effectiveCompaction,
   compaction:config.compaction,composition:receipt.composition,route_composition:config.composition,
   continuation_policy:{notice:launcher.TEXT_TURN_NOTICE,max_consecutive:launcher.MAX_CONSECUTIVE_TEXT_TURN_NOTICES},
   instructions_sha256:V.sha(instructions),tool_schema_sha256:V.sha(JSON.stringify(init.tools)),
   provider_config_sha256:V.sha(JSON.stringify(config)),resumed:resume,training:false,session:manager.getSessionFile(),workspace_in_sandbox:lws,utc:V.utc()});
  console.log(mode.toUpperCase(),ARM,config.model.id,config.checkpoint??'base');
  await session.prompt(resume?'Continue this same investigation from its existing evidence, files and bodies. The previous turn ended for an execution reason, not a scientific one. Finish with decision.json and record_decision stop.'
   :'Read START_HERE.md, investigate this lot with your chosen methods, and finish with decision.json and record_decision stop.');
  while(pendingCompaction!==undefined&&!pause){
   const notes=pendingCompaction;pendingCompaction=undefined;
   created.settingsManager.applyOverrides({compaction:created.settings.compaction});
   fs.appendFileSync(path.join(out,'compaction_requests.jsonl'),JSON.stringify({notes,route:'native Pi session.compact at completed tool boundary'})+'\n');
   let compacted=true;
   try{await session.compact(notes);}
   catch(error){if(!['Already compacted','Nothing to compact (session too small)'].includes(error.message))throw error;compacted=false;}
   fs.appendFileSync(path.join(out,'compaction_requests.jsonl'),JSON.stringify({compacted,completed:true})+'\n');
   await session.prompt(JSON.stringify({kind:'native_compaction_returned',compacted,
    instruction:'Continue the same investigation. Durable scientific state and prior commitments remain available through the tools.'}));
  }
  const last=session.messages.findLast(m=>m.role==='assistant');
  receipt.termination_cause=launcher.investigationTerminationCause(last,transport,false,pause);
  receipt.provider_calls=transport.requests;receipt.transport=transport;receipt.accounting=launcher.cumulativeAccounting(out);
  receipt.text_turn_continuations=notices;
  receipt.decision_written=fs.existsSync(path.join(ws,'decision.json'));
  if(mode==='wirecheck'){
   // One request was built exactly as a live run builds it and answered locally: check what would have been sent.
   assert.equal(wire.length,1,'wirecheck expects exactly one captured request');
   const body=wire[0].body;
   assert(!body.includes(key),'credential in request body');
   assert(body.includes(V.sha(instructions))||body.includes(JSON.stringify(instructions).slice(1,400)),'system prompt is not the contract');
   const names=[...V.NATIVE_TOOL_NAMES,...init.tools.map(t=>t.name)];
   assert(names.every(n=>body.includes(`"${n}"`)),'tool list incomplete');
   assert(!/"type":"image"|"inline_data"|"image_url"/.test(body),'image part in request');
   assert.equal(effectiveCompaction.reserveTokens,COMPACTION_RESERVE_TOKENS,'effective compaction reserve');
   assert.equal(effectiveCompaction.trigger_tokens,config.model.contextWindow-COMPACTION_RESERVE_TOKENS);
   receipt.image_placeholder_check=imageCheck?'image part replaced by the placeholder':null;
   receipt.wire={url:wire[0].url,header_names:wire[0].header_names,body_bytes:body.length,body_sha256:V.sha(body),
    model:config.model.id,tools:names.length,compaction_trigger_tokens:effectiveCompaction.trigger_tokens,effective_compaction:effectiveCompaction};
   receipt.provider_calls=0;receipt.locally_answered_requests=wire.length;receipt.status='pass';
   pass('frontier_request_built_as_live_with_contract_tools_and_no_credential',receipt.wire);
  }else{
   receipt.status=transport.blocked?'execution_failure':receipt.decision_written?'pass':'incomplete';
   pass('live_session_returned',{decision_written:receipt.decision_written,termination_cause:receipt.termination_cause});
  }
 }
 receipt.status??='pass';
}catch(e){receipt.status='fail';receipt.error=e.stack;receipt.bridge_error=e.bridgeError;process.exitCode=1;}
finally{
 session?.dispose();if(bridge)await bridge.close();
 if(T?.closeWorkers)await T.closeWorkers();
 if(started)execFileSync('docker',['rm','-f',box],{stdio:'pipe'});
 try{fs.unlinkSync(lab);}catch{}
 save({live:resume?'LIVE-resume.json':'LIVE.json',scripted:'E2E.json',policy:'POLICY.json',wirecheck:'WIRECHECK.json'}[mode],receipt);
 console.log(JSON.stringify({status:receipt.status,checks:receipt.checks.map(c=>c.name),error:receipt.error?.slice?.(0,3000)}));
}
