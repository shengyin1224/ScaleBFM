"""Replay held-out HAT brain chunks through the real switcher and A1.

This is the evaluation that actually exercises the grasp path. The brain's
predictions carry fingertips, so the deployed close detector finds real close
events, the near cadence engages, the horizon is capped at the event, and A1's
grasp freeze has something to freeze on. Predictions use the deployed fixed alignment: one shared initial placement,
then no per-inference measurement re-anchoring. Continuity search still sees
independent HAT predictions rather than the oracle motion-library windows.

Only the robot episodes contribute tracking numbers; the human ones are kept
for the reachability diagnosis and never averaged in.
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
parser.add_argument("--predictions",default="runs/round3/hat_predictions")
parser.add_argument("--checkpoint",default=None)
parser.add_argument("--label",default=None)
parser.add_argument("--output-root",default="runs/round3/brain_evaluation")
parser.add_argument("--cap-guard",choices=("residual","total","limited","off"),default="residual")
parser.add_argument("--no-residual-lowpass",action="store_true")
parser.add_argument("--max-steps",type=int,default=0)
parser.add_argument("--contract",default=None)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app

import numpy as np
import torch
from isaaclab.sim import SimulationContext
from exec_comp.common import ROOT,TRACK,CHUNK,CHUNK_FPS,HISTORY,KP_NAMES,metadata,log,save_json,np_summary,import_path
from exec_comp.env import make_env,Replay
from exec_comp.geometry import pose9,six_to_q
from exec_comp.hat_playback import quaternion_slerp
from exec_comp.jitter import clip_jitter,aggregate_jitter
from exec_comp.policy import FrozenPolicy
from exec_comp.round3.a1_executor import A1Executor,ExecutorLimits
from exec_comp.round3.metrics import canonical,distribution,to_links
from exec_comp.round3.switching import DeployedSwitching,make_chunk,provenance,quiet_deployment_logging
lifecycle=import_path("exec_comp_round3_brain_lifecycle",TRACK/"scripts/precision_teacher_student/student_lifecycle.py")

CONTROL_DT=.02
STEP_NS=20_000_000
A1_GRID=np.arange(CHUNK)/CHUNK_FPS
WARMUP_STEPS=100


def place_on_ground(env,replay,count,device,ids):
    """Put every robot in the same standing pose before the shared warm-up."""
    joints=torch.tensor(metadata()["default_dof_pos"],device=device)[None].repeat(count,1)
    root=env.unwrapped.scene.env_origins.clone();root[:,2]+=.793
    quat=torch.zeros(count,4,device=device);quat[:,0]=1.
    robot=env.unwrapped.scene["robot"]
    robot.write_joint_state_to_sim(joints,torch.zeros_like(joints),env_ids=ids)
    robot.write_root_state_to_sim(torch.cat((root,quat,torch.zeros(count,6,device=device)),-1),env_ids=ids)
    env.unwrapped.scene.write_data_to_sim();env.unwrapped.sim.forward()
    env.unwrapped.scene.update(dt=CONTROL_DT)
    return replay.capture()


def main(resources):
    torch.manual_seed(42);torch.set_num_threads(4)
    quiet_deployment_logging()
    device=args.device
    pinned=json.loads(open(args.contract or ROOT/"configs/online_execution_contract.json").read())
    config={k:v["value"] for k,v in pinned["switching_parameters"].items()}
    fps=float(pinned["chunk_fps"])
    source_age_s=float(pinned["source_age_s"]["p50"])
    manifest=json.loads((ROOT/args.predictions/"manifest.json").read_text())
    if manifest["smoke"]:raise ValueError("A smoke prediction set is not a 25-episode evaluation")
    if float(manifest["chunk_fps"])!=fps:
        raise ValueError(f"Predictions were exported at {manifest['chunk_fps']} Hz but the "
                         f"deployment contract is {fps} Hz; re-export before evaluating")
    episodes=[e for e in manifest['episodes'] if e['type']=='robot'];count=len(episodes)
    if count!=10:raise ValueError('Tracking evaluation requires the10 held-out robot episodes')
    label=args.label or (f"{args.method}_{args.mode}_stall-{args.anti_stall}"
                         +("" if args.cap_guard=="total" else f"_capguard-{args.cap_guard}"))
    output=ROOT/args.output_root/label;output.mkdir(parents=True,exist_ok=False)

    policy=FrozenPolicy().to(device)
    env=make_env(count,"validation",device);resources.callback(env.close)
    replay=Replay(env,policy,sanitize_prefix=True)
    ids=torch.arange(count,device=device)
    replay.reset(ids,np.arange(count))
    state=place_on_ground(env,replay,count,device,ids)
    start_pose=state["kp"].clone();origin=start_pose[:,1,:3]
    replay.hist={k:state[source][:,None].repeat(1,3,1).clone()
                 for k,source in (("q","q"),("dq","dq"),("root_q","root_q"),("w","w"))}
    replay.hist["action"]=torch.zeros(count,3,29,device=device)

    queries=max(e["queries"] for e in episodes)
    chunks=[];fingertips=[];times=[];initial=[];durations=[]
    for episode in episodes:
        with np.load(episode["file"]) as data:
            keypoints=torch.tensor(data["keypoints"],device=device)
            keypoints=torch.cat((keypoints,keypoints[-1:].repeat(queries-len(keypoints),1,1,1)),0)
            chunks.append(keypoints)
            tips=torch.tensor(data["fingertip_local"],device=device)
            fingertips.append(torch.cat((tips,tips[-1:].repeat(queries-len(tips),1,1,1,1)),0))
            stamps=torch.tensor(data["query_time_s"],device=device,dtype=torch.float32)
            times.append(torch.cat((stamps,stamps[-1:].repeat(queries-len(stamps)))))
            initial.append(torch.tensor(data["initial_keypoints"],device=device))
        durations.append(episode["duration_s"])
    chunks=torch.stack(chunks);chunks[...,:3]+=origin[:,None,None,None]
    fingertips=torch.stack(fingertips);times=torch.stack(times)
    initial=torch.stack(initial);initial[...,:3]+=origin[:,None]

    warmup_failure=torch.zeros(count,dtype=torch.bool,device=device)
    history={k:state[k][:,None].repeat(1,HISTORY,1,1) if state[k].ndim==3
             else state[k][:,None].repeat(1,HISTORY,1) for k in ('q','dq','kp','root_p','root_q','s')}
    with torch.no_grad():
        for step in range(WARMUP_STEPS):
            phase=min(step*CONTROL_DT/.75,1.);alpha=phase*phase*(3-2*phase)
            position=start_pose[...,:3]*(1-alpha)+initial[...,:3]*alpha
            rotation=quaternion_slerp(six_to_q(start_pose[...,3:]),six_to_q(initial[...,3:]),
                                      start_pose.new_full((count,1,1),alpha))
            target=pose9(position,rotation)[:,None].repeat(1,6,1,1)
            action=replay.action(state,target);env.step(action)
            state=replay.capture();replay.append(state,action)
            for key in history:history[key]=torch.cat((history[key][:,1:],state[key][:,None]),1)
            warmup_failure|=(state["s"][:,-1]>-.25)&(state["root_p"][:,2]<.45)

    predictor=None
    if args.method=='B1':
        from exec_comp.round3.b1_controller import B1Predictor
        if args.checkpoint is None:raise ValueError('B1 needs a checkpoint')
        predictor=B1Predictor(args.checkpoint,device)
        if predictor.metadata['call_mode']!=args.mode:raise ValueError('G call mode differs from training')
    executor=A1Executor(enable=args.method!="raw",batch=count,device=device,mode=args.mode,predictor=predictor,
                        checkpoint=args.checkpoint,
                        limits=ExecutorLimits(lowpass=not args.no_residual_lowpass,
                                              cap_guard=args.cap_guard),
                        wrist_compensation=config.get("wrist_compensation","none"))
    switchers=[DeployedSwitching(config=config,future_offsets_s=np.arange(6)*CONTROL_DT,
                                 dt=CONTROL_DT,anti_stall=args.anti_stall=="on")
               for _ in range(count)]
    traces=[{"actual":[],"target":[],"cmd":[],"action":[],"dq":[],
             "switch_steps":[],"grasp_steps":[],"plan_seam_cm":[],"near_steps":0,
             "fall":False,"tracking_failure":False} for _ in range(count)]
    duration=np.asarray(durations)
    max_steps=int(np.ceil(duration.max()*50))
    if args.max_steps:max_steps=min(max_steps,args.max_steps)
    active=(~warmup_failure).cpu().numpy()
    chunk_positions=chunks[...,:3].cpu().numpy()
    chunk_quats=six_to_q(chunks[...,3:]).cpu().numpy()
    chunk_tips=fingertips.cpu().numpy();query_times=times.cpu().numpy()
    delivered=np.zeros(count,dtype=int);latencies=[]
    started=time.perf_counter()

    for step in range(max_steps):
        if not active.any():break
        now_ns=step*STEP_NS;seconds=step*CONTROL_DT
        indices=np.flatnonzero(active)
        measured_positions,measured_quats=to_links(state["kp"])
        measured_positions=measured_positions.detach().cpu().numpy()
        measured_quats=measured_quats.detach().cpu().numpy()
        measured=state["kp"].detach().cpu().numpy()
        source={}
        for i in indices:
            source[i]=switchers[i].publish_state(now_ns,measured_positions[i],measured_quats[i])

        switched=np.zeros(count,dtype=bool)
        old_views={}
        before=np.zeros((count,4,3),dtype=np.float32);have_before=np.zeros(count,dtype=bool)
        pending=[i for i in indices
                 if delivered[i]<episodes[i]["queries"]
                 and seconds>=query_times[i,delivered[i]]+(source_age_s if delivered[i] else 0.)]
        for i in indices:
            retained=(switchers[i].env._pending_raw_chunk is not None or switchers[i].env.chunk_subscriber.latest is not None)
            if i not in pending and not (retained and not switchers[i].hold_active(now_ns)):continue
            if switchers[i].env.chunk is not None:
                before[i]=canonical(switchers[i].sample(now_ns))[0,:,:3];have_before[i]=True
                if executor.enabled:old_views[i]=switchers[i].fork()
        for i in pending:
            index=delivered[i];delivered[i]+=1
            # The source observation is the state published when the brain was
            # queried, so the deployed stale-prefix estimate is the real one.
            age_steps=min(step,int(round(source_age_s/CONTROL_DT)) if index else 0)
            source_sequence=max(0,source[i]-age_steps)
            chunk=make_chunk(index,source_sequence,now_ns,fps,
                             (chunk_positions[i,index],chunk_quats[i,index]),
                             fingertips=chunk_tips[i,index])
            switched[i]=bool(switchers[i].offer(chunk,now_ns)["accepted"])
        for i in indices:switched[i]=bool(switchers[i].poll(now_ns)['accepted']) or switched[i]

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
            if executor.enabled and not grasp[i] and (executor.mode=='stepwise' or switched[i] or not bool(executor.have_delta[i]) or bool(executor.frozen[i])):
                window[i]=canonical(switcher.sample_window(now_ns,A1_GRID))
                padded[i]=switcher.cursor_frame(now_ns)+A1_GRID[-1]*fps>int(pinned["chunk_frame_count"])-1

        deployed_pose=torch.as_tensor(deployed,device=device)
        raw_pose=torch.as_tensor(pure,device=device)
        lookahead=deployed_pose[...,:3]-raw_pose[...,:3]
        state_hist={"q":history["q"],"dq":history["dq"],"kp":history["kp"],
                    "root_pos":history["root_p"][:,-1],"root_quat":history["root_q"][:,-1],'s':history['s']}
        outgoing=before.copy();audit_mask=switched&have_before
        if executor.enabled and audit_mask.any():
            od=np.zeros_like(deployed);op=np.zeros_like(pure);ow=np.zeros_like(window);og=np.zeros_like(grasp)
            for i in np.flatnonzero(audit_mask):
                view=old_views[i];sample=view.sample(now_ns);od[i]=canonical(sample)
                op[i]=canonical(view.sample_raw(now_ns,deployed=sample))
                ow[i]=canonical(view.sample_window(now_ns,A1_GRID));og[i]=view.grasp_phase(now_ns)
            old_command,_=executor.fork()(deployed_pose=torch.as_tensor(od,device=device),raw_pose=torch.as_tensor(op,device=device),
                lookahead_offset=torch.as_tensor(od[...,:3]-op[...,:3],device=device),state_hist=state_hist,
                window=torch.as_tensor(ow,device=device),switched=torch.zeros(count,device=device,dtype=torch.bool),
                grasp_phase=torch.as_tensor(og,device=device),active_mask=torch.as_tensor(audit_mask,device=device))
            outgoing=old_command[:,0,:,:3].cpu().numpy()
        clock=time.perf_counter()
        command,telemetry=executor(
            deployed_pose=deployed_pose,raw_pose=raw_pose,lookahead_offset=lookahead,
            state_hist=state_hist,
            window=torch.as_tensor(window,device=device),
            switched=torch.as_tensor(switched,device=device),
            grasp_phase=torch.as_tensor(grasp,device=device),
            window_padded=torch.as_tensor(padded,device=device),active_mask=torch.as_tensor(active,device=device))
        if executor.enabled and len(latencies)<200:
            torch.cuda.synchronize();latencies.append((time.perf_counter()-clock)*1000.)
        incoming=command[:,0,:,:3].detach().cpu().numpy()
        plan_seam=np.linalg.norm(outgoing[:,2:4]-incoming[:,2:4],axis=-1).max(-1)*100.

        with torch.no_grad():
            action=replay.action(state,command)
            env.step(action);state=replay.capture();replay.append(state,action)
            # The response is compared with the K* the switcher commanded for
            # this instant, 20 ms ahead, which is the quantity A1 is built to
            # close and is identical for both methods.
            target=torch.as_tensor(pure,device=device)[:,1]
            error=(state["kp"][...,:3]-target[...,:3]).norm(dim=-1)
            fall=((state["s"][:,-1]>-.25)&(state["root_p"][:,2]<.45)).cpu().numpy()
            failure=(error[:,0]>.5).cpu().numpy()
            values={"actual":state["kp"].cpu().numpy(),"target":target.cpu().numpy(),
                    "cmd":command[:,0].detach().cpu().numpy(),"action":action.cpu().numpy(),
                    "dq":state["dq"].cpu().numpy()}
            for i in indices:
                trace=traces[i]
                for key,value in values.items():trace[key].append(value[i].copy())
                if switched[i]:
                    trace["switch_steps"].append(len(trace["cmd"])-1)
                    if have_before[i]:trace["plan_seam_cm"].append(float(plan_seam[i]))
                if grasp[i]:trace["grasp_steps"].append(len(trace["cmd"])-1)
                if fall[i] or failure[i] or seconds+CONTROL_DT>=duration[i]:
                    trace["fall"]=bool(fall[i]);trace["tracking_failure"]=bool(failure[i])
                    active[i]=False
            for key in history:history[key]=torch.cat((history[key][:,1:],state[key][:,None]),1)
            if not active.any():break
        if step%500==0:
            log("round3_brain_progress",label=label,step=step,active=int(active.sum()),
                elapsed_s=round(time.perf_counter()-started,1))

    save_json(output/"metrics.json",
              summarize(label,traces,episodes,switchers,executor,warmup_failure,
                        pinned,config,latencies,manifest,output))
    log("round3_brain_complete",label=label)


def summarize(label,traces,episodes,switchers,executor,warmup_failure,pinned,config,
              latencies,manifest,output):
    clips=[];jitters=[];grasp_errors=[];grouped={"all":[],"robot":[],"human":[]};plan_seam=[]
    for index,(episode,trace) in enumerate(zip(episodes,traces)):
        item={"episode":episode['episode'],"source":episode["source"],"type":episode["type"],
              "initialization_failed":bool(warmup_failure[index]),
              "frames":len(trace["actual"]),"fall":bool(trace["fall"] or warmup_failure[index]),
              "tracking_failure":trace["tracking_failure"],
              "switches":len(trace["switch_steps"]),"grasp_frames":len(trace["grasp_steps"]),
              "stall_events":switchers[index].stall_events}
        if trace["actual"]:
            actual=np.asarray(trace["actual"]);target=np.asarray(trace["target"])
            commands=np.asarray(trace["cmd"])
            error=np.linalg.norm(actual[...,:3]-target[...,:3],axis=-1)*100
            if trace['grasp_steps']:grasp_errors.extend(error[np.asarray(trace['grasp_steps']),2:4].reshape(-1).tolist())
            item["errors_cm"]={k:np_summary(error[:,j]) for j,k in enumerate(KP_NAMES)}
            grouped["all"].append(error);grouped[episode["type"]].append(error)
            diagnostics=clip_jitter(commands,actual,trace["action"],trace["dq"],[])
            diagnostics["plan_seam_cm"]=list(trace["plan_seam_cm"])
            if episode["type"]=="robot":jitters.append(diagnostics)
            plan_seam.extend(trace["plan_seam_cm"] if episode["type"]=="robot" else [])
            np.savez_compressed(output/f"episode_{index:02d}.npz",
                                actual=actual,target=target,command=commands,
                                switch_steps=np.asarray(trace["switch_steps"]),
                                grasp_steps=np.asarray(trace["grasp_steps"]))
        else:item["errors_cm"]=None
        clips.append(item)
    groups={}
    for kind,values in grouped.items():
        if not values:continue
        pooled=np.concatenate(values)
        groups[kind]={k:np_summary(pooled[:,j]) for j,k in enumerate(KP_NAMES)}
    robot=[c for c in clips if c["type"]=="robot"]
    wrist=[groups["robot"][n]["mean"] for n in ("left_wrist","right_wrist")] if "robot" in groups else []
    jitter=aggregate_jitter(jitters) if jitters else None
    return {"label":label,"method":args.method,"mode":executor.mode,
            "anti_stall":args.anti_stall,"groups":groups,"per_clip":clips,"jitter":jitter,
            "summary":{"wrist_mean_cm":float(np.mean(wrist)) if wrist else None,
                       "wrist_median_cm":float(np.median(np.concatenate(grouped['robot'])[:,2:4])) if grouped['robot'] else None,
                       "grasp_window_wrist_mean_cm":float(np.mean(grasp_errors)) if grasp_errors else None,
                       "fall_rate":float(np.mean([c['fall'] for c in robot])),
                       "initialization_failure_rate":float(np.mean([c['initialization_failed'] for c in robot])),
                       "plan_seam_p95_cm":(distribution(plan_seam) or {}).get("p95")},
            "plan_seam_cm":distribution(plan_seam),
            "robot_episodes":len(robot),
            "grasp_frames_robot":int(sum(c["grasp_frames"] for c in robot)),
            "grasp_detection":("deployed closing_boundary on the brain's own predicted "
                "fingertips; no hand feedback exists offline, so a FOCUS handoff window "
                "cannot open and only the close boundary is live"),
            "grasp_success_rate":"N/A: no bottle in simulation and no grasp labels in the source HDF5",
            "executor":executor.report(),
            "a1_batch_step_ms":np_summary(latencies) if latencies else None,
            "switching":provenance(),"switching_parameters":config,
            "chunk_contract":{"fps":pinned["chunk_fps"],"frame_count":pinned["chunk_frame_count"],
                "replan_interval_s":manifest.get("replan_interval_s"),
                "source_age_s":float(pinned["source_age_s"]["p50"])},
            "target_definition":("achieved keypoints against the raw K* the deployed switcher "
                "commanded for the same instant, 20 ms ahead; human episodes are excluded "
                "from every averaged number"),
            "human_note":"human source targets exceed the G1 chain geometry; see round3_reachability.json",
            "predictions":str(ROOT/args.predictions),
            "checkpoint_sha256":manifest["checkpoint_sha256"],
            "truncated_smoke":bool(args.max_steps)}


if __name__=="__main__":
    resources=ExitStack()
    lifecycle.run_with_cleanup(lambda:main(resources),app,resources,SimulationContext.instance)
