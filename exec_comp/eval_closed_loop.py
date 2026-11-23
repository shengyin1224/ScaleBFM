"""Unified chunk-interface validation on the held-out unifolm clips."""
import argparse
from contextlib import ExitStack
import time
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument("--method",choices=("raw","A1","A2","B","B_exec"),required=True)
p.add_argument("--iterations",type=int,default=5)
p.add_argument("--limit",type=int,default=300)
p.add_argument("--compile-body",action="store_true")
p.add_argument("--output-root",default="runs/closed_loop")
p.add_argument("--label",default=None)
p.add_argument("--checkpoint",default=None)
p.add_argument("--detailed",action="store_true")
p.add_argument("--save-telemetry",action="store_true")
p.add_argument("--residual-lowpass",action="store_true")
p.add_argument("--randomization",action="store_true")
p.add_argument("--max-steps",type=int,default=0)
p.add_argument('--execution-gate',default=None)
p.add_argument('--execution-init',choices=('zero','A1'),default='zero')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();app=AppLauncher(args).app

import json
import math
import numpy as np
import torch
from isaaclab.sim import SimulationContext
from exec_comp.common import ROOT,TRACK,CHUNK,HISTORY,KP_NAMES,log,save_json,np_summary,import_path
from exec_comp.policy import FrozenPolicy
from exec_comp.env import make_env,Replay
from exec_comp.geometry import to_anchor,from_anchor,heading_anchor
from exec_comp.diff.chain import sample_pose
from exec_comp.runtime import KeypointCompensator,correct_encoded
from exec_comp.diff.optimizer import CommandOptimizer
from exec_comp.jitter import clip_jitter,aggregate_jitter,ResidualLowPass
lifecycle=import_path("exec_comp_eval_lifecycle",TRACK/"scripts/precision_teacher_student/student_lifecycle.py")


def main(resources):
    torch.manual_seed(42);torch.set_num_threads(4)
    limit=args.limit if args.method!="B" else 1
    policy=FrozenPolicy().to(args.device)
    env=make_env(limit,"validation",args.device,randomized=args.randomization);resources.callback(env.close)
    sensor=None;delay=None;dr_audit=None
    if args.randomization:
        from exec_comp.randomization import PolicyNoise,StateNoise,ActionDelay,audit
        sensor=StateNoise(policy,args.device)
        policy.from_obs=PolicyNoise(policy,args.device,sensor=sensor)
        delay=ActionDelay(limit,args.device);dr_audit=audit(env,delay)
    replay=Replay(env,policy,sanitize_prefix=True)
    correction=None
    if args.method in ("A1","A2"):correction=KeypointCompensator(args.method,checkpoint=args.checkpoint,device=args.device)
    elif args.method=="B":
        correction=CommandOptimizer(iterations=args.iterations,device=args.device)
        if args.compile_body:correction.body.net=torch.compile(correction.body.net,mode="reduce-overhead")
    elif args.method=='B_exec':
        from exec_comp.execution_controller import ExecutionController
        if not args.checkpoint or not args.execution_gate:raise ValueError('B_exec requires checkpoint and matching gate')
        correction=ExecutionController(args.checkpoint,args.execution_gate,args.execution_init,args.iterations,args.device)
    suffix=args.method+("_N"+str(args.iterations) if args.method=="B" else "")+("_compiled" if args.compile_body else "")
    suffix=args.label or suffix
    output=ROOT/args.output_root/suffix;output.mkdir(parents=True,exist_ok=False)
    all_clips=[];latency=[];latency_one=[];pooled_errors=[];jitter_clips=[]
    groups=[np.arange(args.limit)] if args.method!="B" else [np.array([i]) for i in range(args.limit)]
    for group in groups:
        n=len(group);ids=torch.arange(n,device=args.device)
        state=replay.reset(ids,group)
        observed=sensor(state) if sensor else state
        histories={k:observed[k][:,None].repeat(1,HISTORY,1,1) if observed[k].ndim==3 else observed[k][:,None].repeat(1,HISTORY,1)
                   for k in ("q","dq","kp","root_p","root_q","s","privileged")}
        histories["prev_action"]=torch.zeros(n,HISTORY,29,device=args.device)
        active=np.ones(n,dtype=bool);traces=[{"ref":[],"actual":[],"cmd":[],"delta":[],"time":[],"fall":False,"tracking_failure":False} for _ in range(n)]
        previous_replan=-1;chunk=None;chunk_at=0
        lowpass=ResidualLowPass() if args.residual_lowpass else None
        for tr in traces:tr.update(action=[],dq=[],seams=[])
        max_steps=int(replay.command.time_totals_train[group].max())
        if args.max_steps:max_steps=min(max_steps,args.max_steps)
        for step in range(max_steps):
            replan=int(step*.02*3)
            if replan!=previous_replan:
                old_command=sample_pose(chunk,torch.tensor([(step-chunk_at)*.6],device=args.device))[:,0] if chunk is not None and args.detailed else None
                raw=replay.reference((np.arange(CHUNK)*50/30).tolist())
                torch.cuda.synchronize();start=time.perf_counter()
                if args.method=="raw":chunk=raw
                elif args.method in ("A1","A2"):
                    with torch.no_grad():
                        origin,quat=heading_anchor(state["root_p"],state["root_q"])
                        kp=to_anchor(histories["kp"],origin,quat)
                        encoded=torch.cat((kp.flatten(2),histories["q"],histories["dq"]),-1).flatten(1)
                        chunk=from_anchor(correct_encoded(correction.model,encoded,to_anchor(raw,origin,quat)),origin,quat)
                elif args.method=='B_exec':
                    chunk=correction.correct(histories,raw)
                else:
                    history={k:histories[k][0] for k in ("q","dq","kp","s","privileged","prev_action")}
                    history.update(root_pos=histories["root_p"][0],root_quat=histories["root_q"][0])
                    chunk=correction.optimize_cmd(history,raw[0])[None]
                torch.cuda.synchronize();latency.append((time.perf_counter()-start)*1000)
                if args.method in ('A1','A2') and len(latency_one)<20:
                    history={k:histories[k][0] for k in ("q","dq","kp")}
                    history.update(root_pos=histories["root_p"][0],root_quat=histories["root_q"][0])
                    torch.cuda.synchronize();start=time.perf_counter();correction.correct_kp(history,raw[0]);torch.cuda.synchronize()
                    latency_one.append((time.perf_counter()-start)*1000)
                chunk_at=step;previous_replan=replan
                if old_command is not None:
                    seam=(chunk[:,0,2:4,:3]-old_command[:,2:4,:3]).detach().cpu().numpy()
                    if lowpass is not None:seam*=.5  # Both previews start from the same previously sent residual.
                    for i in np.flatnonzero(active):traces[i]["seams"].append(seam[i].copy())
            with torch.no_grad():
                command=sample_pose(chunk,(torch.arange(6,device=args.device)+step-chunk_at)*.6)
                if lowpass is not None:command=lowpass(command,replay.reference())
                requested_action=replay.action(state,command)
                action=requested_action
                if delay is not None:action=delay(action)
                env.step(action);state=replay.capture();replay.append(state,requested_action)
                observed=sensor(state) if sensor else state
                raw_next=replay.reference()[:,0]
                error=(state["kp"][...,:3]-raw_next[...,:3]).norm(dim=-1)
                fall=((state["s"][:,-1]>-.25)&(state["root_p"][:,2]<.45))
                tracking=error[:,0]>.5
                end=(replay.command.time_steps>=replay.command.time_totals[replay.command.motion_ids]-1).to(args.device)
                done=fall|tracking|end
                a=state["kp"].cpu().numpy();r=raw_next.cpu().numpy();c=command[:,0].cpu().numpy()
                actions=requested_action.cpu().numpy() if args.detailed else None
                velocities=state["dq"].cpu().numpy() if args.detailed else None
                d=done.cpu().numpy();f=fall.cpu().numpy();failed=tracking.cpu().numpy();t=replay.command.time_steps.numpy()
                for i in np.flatnonzero(active):
                    tr=traces[i];tr["ref"].append(r[i].copy());tr["actual"].append(a[i].copy());tr["cmd"].append(c[i].copy());tr["time"].append(int(t[i]))
                    if args.detailed:tr["action"].append(actions[i].copy());tr["dq"].append(velocities[i].copy())
                    if d[i]:tr["fall"]=bool(f[i]);tr["tracking_failure"]=bool(failed[i]);active[i]=False
                for key in ("q","dq","kp","root_p","root_q","s","privileged"):
                    histories[key]=torch.cat((histories[key][:,1:],observed[key][:,None]),1)
                histories["prev_action"]=torch.cat((histories["prev_action"][:,1:],requested_action[:,None]),1)
                if not active.any():break
            if step%500==0:log("closed_loop_progress",method=suffix,step=step,active=int(active.sum()))
        for i,tr in enumerate(traces):
            a=np.asarray(tr["actual"]);r=np.asarray(tr["ref"]);c=np.asarray(tr["cmd"])
            error=np.linalg.norm(a[...,:3]-r[...,:3],axis=-1)*100
            name=replay.command.motion_names_train[group[i]]
            item={"motion_id":int(group[i]),"name":name,"class":replay.prefix[name]["class"],
                  "frames":len(a),"fall":tr["fall"],"tracking_failure":tr["tracking_failure"],
                  "errors_cm":{k:np_summary(error[:,j]) for j,k in enumerate(KP_NAMES)},
                  "command_frame_difference_cm":float(np.linalg.norm(np.diff(c[...,:3],axis=0),axis=-1).mean()*100) if len(c)>1 else None}
            all_clips.append(item)
            if args.detailed:
                diagnostics=clip_jitter(c,a,tr["action"],tr["dq"],tr["seams"])
                item["jitter"]=diagnostics;jitter_clips.append(diagnostics)
                if args.save_telemetry:
                    telemetry=output/"telemetry";telemetry.mkdir(exist_ok=True)
                    np.savez_compressed(telemetry/f"clip_{int(group[i]):03d}.npz",command=c,actual=a,reference=r,action=np.asarray(tr["action"]),dq=np.asarray(tr["dq"]),seams=np.asarray(tr["seams"]),time=np.asarray(tr["time"]))
            pooled_errors.append(error)
            if int(group[i]) in (0,1,2):np.savez_compressed(output/f"trace_{int(group[i])}.npz",reference=r,actual=a,command=c,time=np.asarray(tr["time"]))
    metrics={}
    for category in ("all","standing","walking"):
        clips=[c for c in all_clips if category=="all" or c["class"]==category]
        if not clips:continue
        count=sum(c["frames"] for c in clips)
        metrics[category]={"clips":len(clips),"fall_rate":float(np.mean([c["fall"] for c in clips])),
            "tracking_failure_rate":float(np.mean([c["tracking_failure"] for c in clips])),
            "keypoint_mean_cm":{n:sum(c["frames"]*c["errors_cm"][n]["mean"] for c in clips)/count for n in KP_NAMES},
            "command_frame_difference_cm":float(np.mean([c["command_frame_difference_cm"] for c in clips if c['command_frame_difference_cm'] is not None])) if any(c['command_frame_difference_cm'] is not None for c in clips) else None}
        errors=np.concatenate([e for c,e in zip(all_clips,pooled_errors) if category=="all" or c["class"]==category])
        metrics[category]["keypoint_errors_cm"]={n:np_summary(errors[:,j]) for j,n in enumerate(KP_NAMES)}
    result={"method":suffix,"metrics":metrics,"per_clip":all_clips,"chunk_latency_ms":np_summary(latency),
            "batch1_latency_ms":np_summary(latency_one) if latency_one else (np_summary(latency) if args.method=="B" else None),
            "simulation_privileged":args.method in ("B","B_exec"),"startup_prefix":"validated","control_hz":50,"chunk_fps":30,"chunk_length":100,"replan_hz":3,
            "response_target_offset_control_steps":1 if args.method!="raw" else 0,
            "startup_guard_frames":3,
            "jitter":aggregate_jitter(jitter_clips) if jitter_clips else None,
            "residual_lowpass":args.residual_lowpass,
            "domain_randomization":dr_audit,
            'policy_action_history':'immediately sent normalized action; actuation delay applies only to physics',
            "truncated_smoke":bool(args.max_steps),
            'execution_checkpoint':args.checkpoint if args.method=='B_exec' else None,
            'execution_gate':args.execution_gate,'execution_initial':args.execution_init if args.method=='B_exec' else None,
            "fingertip_error":"N/A: source corpus contains no fingertip positions; API passthrough is tested"}
    save_json(output/"metrics.json",result);log("closed_loop_complete",method=suffix,metrics=metrics,latency=result["batch1_latency_ms"])


if __name__=="__main__":
    resources=ExitStack();lifecycle.run_with_cleanup(lambda:main(resources),app,resources,SimulationContext.instance)
