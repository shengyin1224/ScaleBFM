"""Closed-loop evaluation under the real deployment chunk-switching logic.

Differences from the round-2 evaluation, all of them demanded by A1-5:

* chunks are 100 frames at the measured deployed 15 Hz, arriving on the
  measured replan cadence with the measured inference age, instead of a
  100@30Hz chunk swapped on a fixed 3 Hz tick;
* when a chunk is consumed, and from which frame, is decided by the deployed
  PACE horizon, close boundary, continuity matcher, RTC and overlap blend;
* the anti-stall lookahead is the deployed one and can be switched off for the
  A/B the task asks for;
* A1 runs after all of that and before the cerebellum, in one of its two call
  modes, and never feeds anything back into the switching decisions.
"""
import argparse
from contextlib import ExitStack
import json
import time
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument("--method",choices=("raw","A1","B1"),required=True)
parser.add_argument("--mode",choices=("stepwise","switch_event"),default="stepwise")
parser.add_argument("--anti-stall",choices=("on","off"),default="on")
parser.add_argument("--limit",type=int,default=300)
parser.add_argument("--label",default=None)
parser.add_argument("--checkpoint",default=None)
parser.add_argument("--output-root",default="runs/round3/closed_loop")
parser.add_argument("--randomization",action="store_true")
parser.add_argument("--no-residual-lowpass",action="store_true")
parser.add_argument("--cap-guard",choices=("residual","total","limited","off"),default="residual")
parser.add_argument("--max-steps",type=int,default=0)
parser.add_argument("--save-telemetry",action="store_true")
parser.add_argument("--contract",default=None)
parser.add_argument('--collect-transitions',type=int,default=0)
parser.add_argument('--data-label',default=None)
parser.add_argument('--data-split',choices=('train','validation'),default='train')
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--worker',type=int,default=0)
parser.add_argument('--action-noise',type=float,default=.05)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app

import numpy as np
import torch
from isaaclab.sim import SimulationContext
from exec_comp.common import ROOT,TRACK,CHUNK,CHUNK_FPS,HISTORY,KP_NAMES,log,save_json,np_summary,import_path
from exec_comp.env import make_env,Replay
from exec_comp.jitter import clip_jitter,aggregate_jitter
from exec_comp.policy import FrozenPolicy
from exec_comp.round3.a1_executor import A1Executor,ExecutorLimits
from exec_comp.round3.metrics import canonical,distribution,to_links
from exec_comp.round3.switching import DeployedSwitching,make_chunk,provenance,quiet_deployment_logging
lifecycle=import_path("exec_comp_round3_lifecycle",TRACK/"scripts/precision_teacher_student/student_lifecycle.py")

CONTROL_DT=.02
STEP_NS=20_000_000
A1_GRID=np.arange(CHUNK)/CHUNK_FPS


def contract():
    path=args.contract or ROOT/"configs/online_execution_contract.json"
    return json.loads(open(path).read())




def main(resources):
    torch.manual_seed(args.seed);torch.set_num_threads(4)
    quiet_deployment_logging()
    device=args.device
    pinned=contract()
    config={k:v["value"] for k,v in pinned["switching_parameters"].items()}
    fps=float(pinned["chunk_fps"]);frames=int(pinned["chunk_frame_count"])
    replan_s=float(pinned["replan_interval_s"]["p50"])
    source_age_s=float(pinned["source_age_s"]["p50"])
    replan_steps=max(1,int(round(replan_s/CONTROL_DT)))
    delay_steps=int(round(source_age_s/CONTROL_DT))
    # One chunk frame maps onto this many 50 Hz reference frames.
    chunk_offsets=(np.arange(frames)*(1./fps)/CONTROL_DT).tolist()

    label=args.label or (f"{args.method}_{args.mode}_stall-{args.anti_stall}"
                         +("" if args.cap_guard=="total" else f"_capguard-{args.cap_guard}")
                         +("_dr" if args.randomization else ""))
    output=ROOT/args.output_root/label;output.mkdir(parents=True,exist_ok=False)
    count=args.limit
    policy=FrozenPolicy().to(device)
    env=make_env(count,"train" if args.collect_transitions else "validation",device,randomized=args.randomization,seed=args.seed);resources.callback(env.close)
    sensor=delay=dr_audit=None
    if args.randomization:
        from exec_comp.randomization import PolicyNoise,ActionDelay,audit
        from exec_comp.round3.sensors import SharedStateNoise
        sensor=SharedStateNoise(policy,device)
        policy.from_obs=PolicyNoise(policy,device,sensor=sensor)
        delay=ActionDelay(count,device);dr_audit=audit(env,delay)
    replay=Replay(env,policy,sanitize_prefix=True)

    if args.collect_transitions and (args.method!='B1' or not args.data_label or args.randomization):
        raise ValueError('Online G collection requires B1, unique data-label and nominal TRAIN clips')
    predictor=None
    if args.method=='B1':
        from exec_comp.round3.b1_controller import B1Predictor
        if args.checkpoint is None:raise ValueError('B1 needs a trained checkpoint')
        predictor=B1Predictor(args.checkpoint,device)
        if predictor.metadata['call_mode']!=args.mode:raise ValueError('G deployment call mode differs from training')
    executor=A1Executor(enable=args.method!="raw",batch=count,device=device,mode=args.mode,predictor=predictor,
                        checkpoint=args.checkpoint,
                        limits=ExecutorLimits(lowpass=not args.no_residual_lowpass,
                                              cap_guard=args.cap_guard),
                        wrist_compensation=config.get("wrist_compensation","none"))
    switchers=[DeployedSwitching(config=config,future_offsets_s=np.arange(6)*CONTROL_DT,
                                 dt=CONTROL_DT,anti_stall=args.anti_stall=="on")
               for _ in range(count)]

    ids=torch.arange(count,device=device)
    group=(np.random.default_rng(args.seed).integers(replay.command.num_motion_train,size=count)
           if args.collect_transitions else np.arange(count))
    state=replay.reset(ids,group)
    observed=sensor(state) if sensor else state
    history={k:observed[k][:,None].repeat(1,HISTORY,1,1) if observed[k].ndim==3
             else observed[k][:,None].repeat(1,HISTORY,1) for k in ("q","dq","kp","root_p","root_q","s")}
    active=np.ones(count,dtype=bool)
    traces=[{"ref":[],"actual":[],"cmd":[],"action":[],"dq":[],"time":[],
             "switch_steps":[],"grasp_steps":[],"plan_seam_cm":[],
             "fall":False,"tracking_failure":False} for _ in range(count)]
    max_steps=int(replay.command.time_totals_train[group].max())
    if args.max_steps:max_steps=min(max_steps,args.max_steps)
    writer=None;episode=np.arange(count,dtype=np.int64);next_episode=count
    if args.collect_transitions:
        from exec_comp.round3.transitions import TransitionWriter
        import math
        max_steps=math.ceil(args.collect_transitions/count)
        writer=TransitionWriter(ROOT/'runs/data'/args.data_label,args.data_split,count,
            env.unwrapped.scene.env_origins,{'checkpoint':args.checkpoint,'mode':args.mode,
                'action_sigma':args.action_noise,'source_split':'TRAIN','seed':args.seed,
                'startup_guard_frames':3,'executor_revision':'v4_original_full_accept_retained_packets'},worker=args.worker)
    pending=[];sequence=0;model_seconds=0.;switch_seconds=0.;latencies=[]
    started=time.perf_counter();lookahead_applied_steps=0

    for step in range(max_steps):
        now_ns=step*STEP_NS
        indices=np.flatnonzero(active)
        clock=time.perf_counter()
        # Under randomization the switcher sees the same noisy measurement the
        # policy does; it must not be given a privileged clean pose.
        measured_positions,measured_quats=to_links(observed["kp"])
        measured_positions=measured_positions.detach().cpu().numpy()
        measured_quats=measured_quats.detach().cpu().numpy()
        measured=observed["kp"].detach().cpu().numpy()
        for i in indices:
            switchers[i].publish_state(now_ns,measured_positions[i],measured_quats[i])

        # The producer runs on the measured replan cadence and its chunk is
        # delivered one measured inference age later, exactly as a HAT chunk
        # whose frame zero belongs to its own source observation.
        if step%replan_steps==0:
            window=replay.reference(chunk_offsets)
            positions,quaternions=to_links(window)
            # The deployed hold never delays the first chunk, so the opening
            # one is delivered without the inference age.
            pending.append({"deliver":step+(delay_steps if sequence else 0),"sequence":sequence,
                            "positions":positions.detach().cpu().numpy(),
                            "quaternions":quaternions.detach().cpu().numpy(),
                            "source":{i:switchers[i].env.state_sequence_id-1 for i in indices}})
            sequence+=1
        switched=np.zeros(count,dtype=bool)
        old_views={}
        before=np.zeros((count,4,3),dtype=np.float32);have_before=np.zeros(count,dtype=bool)
        # Snapshot what the outgoing plan says about *now*, so the seam can
        # be measured against the incoming plan at the same instant rather
        # than against the previous control step.
        for i in indices:
            if writer is not None:continue
            fresh=pending and pending[0]['deliver']<=step
            retained=(switchers[i].env._pending_raw_chunk is not None or switchers[i].env.chunk_subscriber.latest is not None)
            if not fresh and not (retained and not switchers[i].hold_active(now_ns)):continue
            if switchers[i].env.chunk is not None:
                before[i]=canonical(switchers[i].sample(now_ns))[0,:,:3];have_before[i]=True
                if executor.enabled:old_views[i]=switchers[i].fork()
        while pending and pending[0]["deliver"]<=step:
            ready=pending.pop(0)
            for i in indices:
                if i not in ready["source"]:continue
                chunk=make_chunk(ready["sequence"],ready["source"][i],now_ns,fps,
                                 (ready["positions"][i],ready["quaternions"][i]))
                switched[i]=bool(switchers[i].offer(chunk,now_ns)["accepted"])
        for i in indices:
            switched[i]=bool(switchers[i].poll(now_ns)['accepted']) or switched[i]
        switch_seconds+=time.perf_counter()-clock

        clock=time.perf_counter()
        deployed=np.zeros((count,6,4,9),dtype=np.float32)
        pure=np.zeros((count,6,4,9),dtype=np.float32)
        window=np.zeros((count,CHUNK,4,9),dtype=np.float32)
        grasp=np.zeros(count,dtype=bool);padded=np.zeros(count,dtype=bool)
        for i in indices:
            switcher=switchers[i]
            if switcher.env.chunk is None:
                hold=np.repeat(measured[i][None],6,axis=0)
                deployed[i]=hold;pure[i]=hold;window[i]=np.repeat(measured[i][None],CHUNK,axis=0)
                continue
            sample=switcher.sample(now_ns)
            deployed[i]=canonical(sample)
            raw_sample=switcher.sample_raw(now_ns,deployed=sample)
            pure[i]=deployed[i] if raw_sample is sample else canonical(raw_sample)
            grasp[i]=switcher.grasp_phase(now_ns)
            needs_window=executor.enabled and not grasp[i] and (
                executor.mode=='stepwise' or switched[i] or not bool(executor.have_delta[i]) or bool(executor.frozen[i]))
            if needs_window:
                window[i]=canonical(switcher.sample_window(now_ns,A1_GRID))
                # A1 asks for 3.33 s of future; past the chunk's own end the
                # deployed sampler repeats its last frame, which is the padding
                # A1-5 asks to be reported rather than hidden.
                padded[i]=switcher.cursor_frame(now_ns)+A1_GRID[-1]*fps>frames-1
        switch_seconds+=time.perf_counter()-clock

        deployed_pose=torch.as_tensor(deployed,device=device)
        raw_pose=torch.as_tensor(pure,device=device)
        lookahead=deployed_pose[...,:3]-raw_pose[...,:3]
        lookahead_applied_steps+=int((lookahead[:,0,2:4].norm(dim=-1).max(-1).values[torch.as_tensor(indices,device=device)]>1e-7).sum())
        state_hist={"q":history["q"],"dq":history["dq"],"kp":history["kp"],
                    "root_pos":history["root_p"][:,-1],"root_quat":history["root_q"][:,-1],'s':history['s']}
        outgoing=before.copy()
        # Compare two complete host-stage outputs at the SAME physical instant.
        # Cloned buffers keep the counterfactual old-plan call out of the live
        # LP/freeze/fallback state and out of its counters.
        audit_mask=switched&have_before
        if executor.enabled and audit_mask.any():
            od=np.zeros_like(deployed);op=np.zeros_like(pure);ow=np.zeros_like(window);og=np.zeros_like(grasp)
            for i in np.flatnonzero(audit_mask):
                view=old_views[i];sample=view.sample(now_ns);od[i]=canonical(sample)
                op[i]=canonical(view.sample_raw(now_ns,deployed=sample))
                ow[i]=canonical(view.sample_window(now_ns,A1_GRID));og[i]=view.grasp_phase(now_ns)
            old_executor=executor.fork()
            old_command,_=old_executor(deployed_pose=torch.as_tensor(od,device=device),
                raw_pose=torch.as_tensor(op,device=device),lookahead_offset=torch.as_tensor(od[...,:3]-op[...,:3],device=device),
                state_hist=state_hist,window=torch.as_tensor(ow,device=device),
                switched=torch.zeros(count,device=device,dtype=torch.bool),grasp_phase=torch.as_tensor(og,device=device),
                active_mask=torch.as_tensor(audit_mask,device=device))
            outgoing=old_command[:,0,:,:3].cpu().numpy()
        clock=time.perf_counter()
        if executor.enabled:torch.cuda.synchronize()
        command,telemetry=executor(
            deployed_pose=deployed_pose,raw_pose=raw_pose,lookahead_offset=lookahead,
            state_hist=state_hist,
            window=torch.as_tensor(window,device=device),
            switched=torch.as_tensor(switched,device=device),
            grasp_phase=torch.as_tensor(grasp,device=device),
            window_padded=torch.as_tensor(padded,device=device),active_mask=torch.as_tensor(active,device=device))
        if executor.enabled:
            torch.cuda.synchronize();elapsed=(time.perf_counter()-clock)*1000.
            model_seconds+=elapsed/1000.
            if len(latencies)<200:latencies.append(elapsed)
        # Plan-vs-plan seam at the switch instant, the same quantity round 2
        # reported: outgoing K* plus the residual then in force, against the
        # incoming K* plus the refreshed residual.
        incoming=command[:,0,:,:3].detach().cpu().numpy()
        plan_seam=np.linalg.norm(outgoing[:,2:4]-incoming[:,2:4],axis=-1).max(-1)*100.

        with torch.no_grad():
            policy_action=replay.action(state,command)
            requested=policy_action+(args.action_noise*torch.randn_like(policy_action) if writer is not None else 0.)
            action=delay(requested) if delay is not None else requested
            if writer is not None:
                writer.before(state,replay,command,replay.reference()[:,0],action,requested,policy_action,episode,switched,raw_target_next=raw_pose[:,1])
            env.step(action);state=replay.capture();replay.append(state,requested)
            observed=sensor(state) if sensor else state
            target=replay.reference()[:,0]
            error=(state["kp"][...,:3]-target[...,:3]).norm(dim=-1)
            fall=(state["s"][:,-1]>-.25)&(state["root_p"][:,2]<.45)
            failure=error[:,0]>.5
            finished=(replay.command.time_steps>=replay.command.time_totals[replay.command.motion_ids]-1).to(device)
            done=(fall|failure|finished).cpu().numpy()
            fallen=fall.cpu().numpy();failed=failure.cpu().numpy()
            reference=target.cpu().numpy();achieved=state["kp"].cpu().numpy()
            sent=command[:,0].detach().cpu().numpy()
            actions=requested.cpu().numpy();velocities=state["dq"].cpu().numpy()
            times=replay.command.time_steps.numpy()
            for i in (() if writer is not None else indices):
                trace=traces[i]
                trace["ref"].append(reference[i].copy());trace["actual"].append(achieved[i].copy())
                trace["cmd"].append(sent[i].copy());trace["action"].append(actions[i].copy())
                trace["dq"].append(velocities[i].copy());trace["time"].append(int(times[i]))
                if switched[i]:
                    trace["switch_steps"].append(len(trace["cmd"])-1)
                    if have_before[i]:trace["plan_seam_cm"].append(float(plan_seam[i]))
                if grasp[i]:trace["grasp_steps"].append(len(trace["cmd"])-1)
                if done[i]:
                    trace["fall"]=bool(fallen[i]);trace["tracking_failure"]=bool(failed[i])
                    active[i]=False
            for key,source in (("q","q"),("dq","dq"),("kp","kp"),("root_p","root_p"),("root_q","root_q"),('s','s')):
                history[key]=torch.cat((history[key][:,1:],observed[source][:,None]),1)
            if writer is not None:
                writer.after(state,target,torch.as_tensor(done,device=device),fall)
                reset_ids=torch.as_tensor(np.flatnonzero(done),device=device)
                if len(reset_ids):
                    reset_np=reset_ids.cpu().numpy()
                    new_motions=torch.randint(replay.command.num_motion_train,(len(reset_ids),)).numpy()
                    state=replay.reset(reset_ids,new_motions)
                    executor.reset(reset_ids)
                    for i in reset_np:
                        switchers[i].reset()
                        for packet in pending:packet['source'].pop(i,None)
                    episode[reset_np]=np.arange(next_episode,next_episode+len(reset_np));next_episode+=len(reset_np)
                    for key in history:history[key][reset_ids]=state[key][reset_ids,None]
                    observed=state
            elif not active.any():break
        if step%500==0:
            log("round3_progress",label=label,step=step,active=int(active.sum()),
                elapsed_s=round(time.perf_counter()-started,1))

    if writer is not None:
        writer.metadata['switching']=provenance()
        writer.close()
        save_json(output/'collection.json',{'transitions':writer.total,'data':str(writer.folder),
            'executor':executor.report(),'elapsed_s':time.perf_counter()-started})
        log('round3_collection_complete',label=label,transitions=writer.total)
        return
    result=summarize(label,traces,group,replay,switchers,executor,pinned,config,
                     latencies,dr_audit,output)
    result['lookahead_applied_env_steps']=lookahead_applied_steps
    save_json(output/"metrics.json",result)
    log("round3_complete",label=label,wrist_mean_cm=result["summary"]["wrist_mean_cm"],
        seam_p95_cm=result["summary"]["seam_step_p95_cm"])




def summarize(label,traces,group,replay,switchers,executor,pinned,config,
              latencies,dr_audit,output):
    clips=[];pooled=[];jitters=[];seam=[];nonseam=[];plan_seam=[]
    for index,trace in enumerate(traces):
        if not trace["actual"]:continue
        actual=np.asarray(trace["actual"]);reference=np.asarray(trace["ref"])
        commands=np.asarray(trace["cmd"])
        error=np.linalg.norm(actual[...,:3]-reference[...,:3],axis=-1)*100
        switches=np.asarray(trace["switch_steps"],dtype=int)
        steps=np.linalg.norm(np.diff(commands[:,2:4,:3],axis=0),axis=-1).max(-1)*100
        at_switch=switches[(switches>0)&(switches<len(steps)+1)]-1
        mask=np.zeros(len(steps),dtype=bool)
        if at_switch.size:mask[at_switch]=True
        seam.extend(steps[mask].tolist());nonseam.extend(steps[~mask].tolist())
        plan_seam.extend(trace["plan_seam_cm"])
        name=replay.command.motion_names_train[group[index]]
        diagnostics=clip_jitter(commands,actual,trace["action"],trace["dq"],[])
        diagnostics["seam_step_cm"]=steps[mask].tolist()
        diagnostics["plan_seam_cm"]=list(trace["plan_seam_cm"])
        jitters.append(diagnostics)
        clips.append({"motion_id":int(group[index]),"name":name,
                      "class":replay.prefix[name]["class"],"frames":len(actual),
                      "fall":trace["fall"],"tracking_failure":trace["tracking_failure"],
                      "switches":int(len(switches)),"grasp_steps":int(len(trace["grasp_steps"])),
                      "errors_cm":{k:np_summary(error[:,j]) for j,k in enumerate(KP_NAMES)},
                      "stall_events":switchers[index].stall_events,
                      "stall_max_lookahead_m":switchers[index].stall_max_m})
        pooled.append(error)
        if int(group[index]) in (0,1,2):
            np.savez_compressed(output/f"trace_{int(group[index])}.npz",
                                reference=reference,actual=actual,command=commands,
                                time=np.asarray(trace["time"]),switch_steps=switches)
    errors=np.concatenate(pooled) if pooled else np.zeros((0,4))
    total=sum(c["frames"] for c in clips)
    metrics={}
    for category in ("all","standing","walking"):
        selected=[c for c in clips if category=="all" or c["class"]==category]
        if not selected:continue
        frames=sum(c["frames"] for c in selected)
        subset=np.concatenate([e for c,e in zip(clips,pooled) if category=="all" or c["class"]==category])
        metrics[category]={"clips":len(selected),
            "fall_rate":float(np.mean([c["fall"] for c in selected])),
            "tracking_failure_rate":float(np.mean([c["tracking_failure"] for c in selected])),
            "keypoint_mean_cm":{n:sum(c["frames"]*c["errors_cm"][n]["mean"] for c in selected)/frames
                                for n in KP_NAMES},
            "keypoint_errors_cm":{n:np_summary(subset[:,j]) for j,n in enumerate(KP_NAMES)}}
    wrists=[metrics["all"]["keypoint_mean_cm"][n] for n in ("left_wrist","right_wrist")] if metrics else [float("nan")]
    jitter=aggregate_jitter(jitters) if jitters else None
    stall_events=sum(sum(s.stall_events.values()) for s in switchers)
    summary={"wrist_mean_cm":float(np.mean(wrists)),
             "wrist_median_cm":float(np.median(np.concatenate([errors[:,2],errors[:,3]]))) if errors.size else None,
             "seam_step_p95_cm":(distribution(seam) or {}).get("p95"),
             "plan_seam_p95_cm":(distribution(plan_seam) or {}).get("p95"),
             "stall_events":int(stall_events)}
    return {"label":label,"method":args.method,
            'executor_revision':'v4_original_full_accept_retained_packets',
            "mode":executor.mode,"anti_stall":args.anti_stall,
            "metrics":metrics,"per_clip":clips,"jitter":jitter,"summary":summary,
            "plan_seam_cm":distribution(plan_seam),
            "seam_step_cm":distribution(seam),"within_chunk_step_cm":distribution(nonseam),
            "seam_definition":{
                "plan_seam_cm":("at each accepted switch, wrist max distance between the "
                    "old-plan counterfactual full executor output and the incoming full output "
                    "at identical state/time. Controller buffers are cloned, including LP/freeze/fallback."),
                "seam_step_cm":("diagnostic: command change between consecutive control steps "
                    "at switch steps; it also contains ordinary 20 ms motion."),
                "within_chunk_step_cm":"the same diagnostic away from switch steps, as its baseline"},
            "stall_events_per_side":{side:int(sum(s.stall_events[side] for s in switchers))
                                     for side in ("left","right")},
            "executor":executor.report(),
            "a1_batch300_step_ms":np_summary(latencies) if latencies else None,
            "a1_batch300_step_ms_note":("wall time of one batched 300-robot executor call; "
                "deployment latency is measured separately on batch 1 in round3/benchmark.py"),
            "switching":provenance(),
            "chunk_contract":{"fps":pinned["chunk_fps"],"frame_count":pinned["chunk_frame_count"],
                "replan_interval_s":float(pinned["replan_interval_s"]["p50"]),
                "source_age_s":float(pinned["source_age_s"]["p50"]),
                "measured_from":pinned["runs"],"since":pinned["since"]},
            "switching_parameters":config,
            "control_hz":50,"startup_guard_frames":3,
            "domain_randomization":dr_audit,
            "residual_lowpass":not args.no_residual_lowpass,
            "oracle_producer":("the motion library is replayed as a perfect, non-re-anchored "
                "predictor, so the forward continuity search has no re-anchoring to undo and "
                "does not reproduce the +13 frame shift seen online; the brain-output "
                "evaluation does"),
            "fingertips":"N/A: the motion library has no fingertip channel, so no close event "
                "can occur here and the grasp freeze is exercised on the brain outputs and in tests",
            "truncated_smoke":bool(args.max_steps)}


if __name__=="__main__":
    resources=ExitStack()
    lifecycle.run_with_cleanup(lambda:main(resources),app,resources,SimulationContext.instance)
