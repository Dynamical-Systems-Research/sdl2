/** v1.2 composition (2026-10-26): a copy of deepseek-tinker-route-v22/ds_tinker_transport.mjs (sha 4961fad8) with only
 * the model constants changed to DeepSeek-V4.1-Flash-1017 and the Pi imports pointed directly at the Pi 1.4.0 runtime
 * (material-route-swarm-design-20261019/runtime/pi140), which the Oct 24 diagnostic reached through an import hook.
 * The composition list drops ds_pi_launch.mjs and resources.mjs (the Pi 1.1.0 launcher, never used by this route). */
/** Small native SamplingClient boundary for DeepSeek-V4.1-Flash on Tinker (2026-10-10); adapted from the Qwen route's
 * tinker_transport.mjs (qwen38-learning-execution-20261003/integration). Imports the qualified V5 tools/session unchanged.
 * Differences: model and route id, the probe venv (cookbook 70767c7 has the deepseekv4_1 renderer), ds_tinker_worker.py,
 * DeepSeek prices, text-only input (images become a placeholder in the worker), offline tokenizer.
 * v22 copy (phase 2, 2026-10-12): Pi 1.1.0 from agents-pi110-v22; Pi's compaction reserve is set per launch by
 * NDFEB_COMPACTION_RESERVE_TOKENS (default 65,536: compaction when context exceeds 262,144 - 65,536 = 196,608 tokens, as v2.1;
 * keepRecentTokens 20,000), applied by the createSession wrapper below and recorded in modelConfig (hashed into
 * provider_config_sha256). Sampling, rendering, files and accounting are unchanged. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';
import {AssistantMessageEventStream, normalizeContext, resolveTranscript, getCurrentTools, getCurrentSystemPrompt} from '../../material-route-swarm-design-20261019/runtime/pi140/node_modules/@earendil-works/pi-ai/dist/index.js';
import * as V from '../../material-route-swarm-design-20261019/runtime/pi140/vision_transport.mjs';
export * from '../../material-route-swarm-design-20261019/runtime/pi140/vision_transport.mjs';
const HERE=path.dirname(fileURLToPath(import.meta.url));
export const MODEL='deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144';
export const ROUTE_ID='deepseek-v4.1-flash-1017-tinker';
export const PYTHON=path.resolve(HERE,'../../engine-release-20261008/backbone-probe-20261010/.venv/bin/python');
const sha=V.sha;
export const COMPACTION_RESERVE_TOKENS=Number(process.env.NDFEB_COMPACTION_RESERVE_TOKENS||65536);
export const COMPACTION_KEEP_RECENT_TOKENS=20000;
if(!(COMPACTION_RESERVE_TOKENS>=65536&&COMPACTION_RESERVE_TOKENS<262144))throw new Error('NDFEB_COMPACTION_RESERVE_TOKENS out of range');
// The loop takes createSession from this module (transportFor); it re-applies created.settings.compaction after operator
// compactions, so both the settings object and the manager carry the reserve.
export async function createSession(args){
  const created=await V.createSession(args);
  created.settings.compaction={...created.settings.compaction,reserveTokens:COMPACTION_RESERVE_TOKENS,keepRecentTokens:COMPACTION_KEEP_RECENT_TOKENS};
  created.settingsManager.applyOverrides({compaction:created.settings.compaction});
  return created;
}
export function modelConfig(id) {
  if(id!==ROUTE_ID) throw new Error('Unexpected Tinker investigator route');
  return {model:{id:MODEL,name:'DeepSeek-V4.1-Flash-1017 Tinker',api:'openai-completions',provider:'tinker',
    baseUrl:'https://tinker.thinkingmachines.dev/services',reasoning:true,input:['text'],
    contextWindow:262144,maxTokens:32768,cost:{input:.60,output:3.60,cacheRead:.12,cacheWrite:0}},
    route_id:ROUTE_ID,canonical_slug:MODEL,expected_provider:'Tinker',api_key_env:null,inference_route:'tinker-native',
    checkpoint:process.env.TINKER_MODEL_PATH||null,
    compaction:{reserveTokens:COMPACTION_RESERVE_TOKENS,keepRecentTokens:COMPACTION_KEEP_RECENT_TOKENS,trigger_tokens:262144-COMPACTION_RESERVE_TOKENS},
    endpoint:{url:'native SamplingClient',provider_name:'Tinker',model:MODEL},
    composition:{sources:['ds_tinker_transport.mjs','ds_tinker_worker.py'].map(name=>({path:path.join(HERE,name),sha256:sha(fs.readFileSync(path.join(HERE,name)))}))}};
}
export function nativeRequest(incoming) {
  const context=resolveTranscript(normalizeContext(incoming),false);
  const tools=getCurrentTools(context.messages).map(t=>({type:'function',function:{name:t.name,description:t.description,parameters:t.parameters,strict:false}}));
  const messages=[];
  const system=getCurrentSystemPrompt(context.messages);
  if(system) messages.push({role:'system',content:system});
  for(const m of context.messages) {
    if(m.role==='system') continue;
    const content=typeof m.content==='string'?[{type:'text',text:m.content}]:m.content;
    const parts=content.filter(c=>c.type!=='toolCall').map(c=>{
      if(c.type==='image') return {type:'image',image:`data:${c.mimeType};base64,${c.data}`};
      if(c.type==='thinking') return {type:'thinking',thinking:c.thinking};
      if(c.type==='text') return {type:'text',text:c.text};
      throw new Error('Unsupported native content '+c.type);
    });
    const row={role:m.role==='toolResult'?'tool':m.role,content:parts};
    if(m.role==='toolResult'){row.tool_call_id=m.toolCallId;row.name=m.toolName;}
    if(m.role==='assistant'){
      const calls=content.filter(c=>c.type==='toolCall').map(c=>({id:c.id,type:'function',function:{name:c.name,arguments:JSON.stringify(c.arguments)}}));
      if(calls.length)row.tool_calls=calls;
    }
    messages.push(row);
  }
  return {messages,tools};
}
export class Worker {
  constructor(){
    this.child=spawn(PYTHON,[path.join(HERE,'ds_tinker_worker.py')],{stdio:['pipe','pipe','pipe'],env:{...process.env,TOKENIZERS_PARALLELISM:'false',HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1'}});
    this.pending=new Map();this.id=0;this.stderr='';
    createInterface({input:this.child.stdout}).on('line',line=>{let value;try{value=JSON.parse(line);}catch{return;}
      const p=this.pending.get(value.id);if(!p)return;this.pending.delete(value.id);value.error?p.reject(Object.assign(new Error(value.error.message),{workerError:value.error})):p.resolve(value.result);});
    this.child.stderr.on('data',b=>{this.stderr=(this.stderr+b).slice(-8000);});
    this.child.on('exit',()=>{for(const p of this.pending.values())p.reject(new Error('Tinker worker exited'));this.pending.clear();});
  }
  call(request){return new Promise((resolve,reject)=>{const id=++this.id;this.pending.set(id,{resolve,reject});this.child.stdin.write(JSON.stringify({...request,id})+'\n');});}
  async close(){if(this.child.exitCode!==null||this.child.signalCode!==null)return;this.child.stdin.end();await new Promise(r=>this.child.once('exit',r));}
}
const workers=new Set();
export async function closeWorkers(){await Promise.all([...workers].map(w=>w.close()));workers.clear();}
export function installTransport(session,output,instructions,config,{worker,now=Date.now}={}) {
  worker??=new Worker();workers.add(worker);
  const api=path.join(output,'api');fs.mkdirSync(api,{recursive:true});
  let index=fs.readdirSync(api).filter(n=>/^request_\d+\.json$/.test(n)).length;
  const status={requests:0,accepted_outputs:0,blocked:false,usage_unresolved:false,resourceCensored:false,estimated_cost_usd:0};
  const write=(file,value)=>fs.writeFileSync(file,JSON.stringify(value)+'\n',{mode:0o600,flag:'wx'});
  session.modelRuntime.streamSimple=(model,incoming,options={})=>{
    const stream=new AssistantMessageEventStream();
    void(async()=>{
      const accountingId=crypto.randomUUID(),number=String(index++).padStart(4,'0');let accounted=false;
      const {normal,context}=V.investigationContext(incoming,instructions);
      const category=options.maxTokens===1?'native_cache_warm':normal?'investigator':'native_compaction_or_summary';
      const request={...nativeRequest(context),method:'sample',max_tokens:Math.min(options.maxTokens??32768,32768),
        checkpoint:config.checkpoint,prefix_path:path.join(api,'prefix_'+number+'.json'),
        ...(process.env.TINKER_SAMPLE_SEED?{seed:Number(process.env.TINKER_SAMPLE_SEED)}:{})};
      const message={role:'assistant',api:model.api,provider:model.provider,model:model.id,providerThinkingLevel:'high',
        content:[],usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},stopReason:'stop',timestamp:now()};
      try{
        if(options.signal?.aborted)throw new Error('Request aborted before sampling');
        write(path.join(api,'request_'+number+'.json'),{accounting_id:accountingId,category,body:request,native_context:context});
        status.requests++;stream.push({type:'start',partial:message});
        const result=await worker.call(request);write(path.join(api,'response_'+number+'.json'),result);
        const s=result.sequences[0];
        const cache=result.cached_tokens??0,usage={prompt_tokens:result.prompt_tokens,completion_tokens:result.generated_tokens,
          total_tokens:result.prompt_tokens+result.generated_tokens,prompt_tokens_details:{cached_tokens:cache}};
        V.appendRecord(path.join(output,'usage.jsonl'),{accounting_id:accountingId,number,category,status:'known',usage,
          billing_cost_status:'not_reported_by_api',cost_estimate_usd:result.cost_estimate_usd,wall_seconds:result.wall_seconds,inference_route:'tinker-native'});
        accounted=true;status.estimated_cost_usd+=result.cost_estimate_usd;
        message.usage={input:result.prompt_tokens-cache,output:result.generated_tokens,cacheRead:cache,cacheWrite:0,totalTokens:usage.total_tokens,
          cost:{input:(result.prompt_tokens-cache)*.60/1e6,output:result.generated_tokens*3.60/1e6,cacheRead:cache*.12/1e6,cacheWrite:0,total:result.cost_estimate_usd}};
        for(const c of s.message.content)if(c.type==='text'||c.type==='thinking')message.content.push(c);
        for(const c of s.message.tool_calls??[])message.content.push({type:'toolCall',id:c.id||crypto.randomUUID(),name:c.function.name,arguments:JSON.parse(c.function.arguments)});
        message.stopReason=s.stop_reason==='length'?'length':message.content.some(c=>c.type==='toolCall')?'toolUse':'stop';
        message.rawStopReason=s.stop_reason;
        if(s.parse_termination==='malformed'&&s.stop_reason!=='length')throw new Error('Malformed renderer response; no tool execution');
        if(options.signal?.aborted){message.stopReason='aborted';message.errorMessage='Cancelled after accounted sampling';stream.push({type:'error',reason:'aborted',error:message});}
        else{status.accepted_outputs++;stream.push({type:'done',reason:message.stopReason,message});}
        stream.end();
      }catch(e){
        status.blocked=true;status.usage_unresolved||=!accounted;message.stopReason='error';message.errorMessage=String(e.message).slice(0,500);
        write(path.join(api,'failure_'+number+'.json'),{accounting_id:accountingId,error:message.errorMessage,scientific_negative:false});
        stream.push({type:'error',reason:'error',error:message});stream.end();
      }
    })();return stream;
  };
  return status;
}
