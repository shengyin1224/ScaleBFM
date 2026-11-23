"""Simulate the frozen release; baseline diagnosis or perturbed shared transitions."""
import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import time
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument("--mode",choices=("baseline","collect"),required=True)
p.add_argument("--split",choices=("train","validation"),default="train")
p.add_argument("--num-envs",type=int,default=1024)
p.add_argument("--transitions",type=int,default=1667072)
p.add_argument("--worker",type=int,default=0)
p.add_argument("--max-steps",type=int,default=0)
p.add_argument("--action-noise",type=float,default=.05)
p.add_argument("--label",default="")
p.add_argument("--startup-prefix",choices=("raw","validated"),default="raw")
p.add_argument('--randomization',action='store_true')
p.add_argument('--execution-checkpoint',default=None)
p.add_argument('--execution-gate',default=None)
p.add_argument('--execution-init',choices=('zero','A1','mixed'),default='mixed')
p.add_argument('--execution-iterations',type=int,default=1)
# Legacy fixed3Hz diagnostic only. Formal round3 B1 collection uses
# exec_comp.round3.eval_switching --collect-transitions with the original switcher.
p.add_argument('--b1-checkpoint',default=None)
p.add_argument('--data-split',choices=('train','validation'),default=None)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();app=AppLauncher(args).app

from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import torch
from exec_comp.common import ROOT,metadata,log,save_json,np_summary,KP_NAMES
from exec_comp.env import make_env,Replay
from exec_comp.policy import FrozenPolicy
from exec_comp.geometry import qapply,qconj
from exec_comp.common import import_path,TRACK
from isaaclab.sim import SimulationContext
lifecycle=import_path("exec_comp_readonly_lifecycle",TRACK/"scripts/precision_teacher_student/student_lifecycle.py")


def lag_frames(reference,actual):
    if len(reference)<50 or np.linalg.norm(np.std(reference,axis=0))<.01:return None
    scores=[]
    for lag in range(-25,26):
        a=reference[max(0,-lag):len(reference)-max(0,lag)]
        b=actual[max(0,lag):len(actual)-max(0,-lag)]
        a=a-a.mean(0);b=b-b.mean(0)
        scores.append(float((a*b).sum()/max(1e-12,np.sqrt((a*a).sum()*(b*b).sum()))))
    return int(np.argmax(scores)-25)


def main(resources):
    torch.manual_seed(42+args.worker)
    policy=FrozenPolicy().to(args.device)
    env=make_env(args.num_envs,args.split,args.device,randomized=args.randomization,seed=42+args.worker);resources.callback(env.close)
    replay=Replay(env,policy,args.startup_prefix=="validated");command=replay.command
    all_ids=torch.arange(args.num_envs,device=args.device)
    motion_ids=(np.arange(args.num_envs)+args.worker*args.num_envs)%command.num_motion_train
    state=replay.reset(all_ids,motion_ids)
    log("isaac_contract",**replay.consistency_check(state),num_envs=args.num_envs,
        domain_randomization=args.randomization,proprio_noise=False,reference_noise=False,
        history=3,future_offsets=list(range(6)),focus=[0,2],split=args.split)
    sensor=None;delay=None;randomization_audit=None
    if args.randomization:
        from exec_comp.randomization import StateNoise,PolicyNoise,ActionDelay,audit
        sensor=StateNoise(policy,args.device,seed=202610252+100*args.worker)
        policy.from_obs=PolicyNoise(policy,args.device,sensor=sensor,seed=202610251+100*args.worker)
        delay=ActionDelay(args.num_envs,args.device,seed=202610253+100*args.worker);randomization_audit=audit(env,delay)
        randomization_audit.update(sensor_seed=sensor.seed,policy_noise_seed=202610251+100*args.worker)
    observed=sensor(state) if sensor else state
    baseline=args.mode=="baseline"
    controller=None;execution_history=None;execution_chunk=None;chunk_at=0;last_replan=-1
    reset_pending=torch.zeros(args.num_envs,device=args.device,dtype=torch.bool)
    if args.b1_checkpoint:
        if args.execution_checkpoint:raise ValueError('Choose either the B optimizer or the B1 network')
        from exec_comp.execution_controller import History
        from exec_comp.round3.b1_controller import B1Controller
        controller=B1Controller(args.b1_checkpoint,args.device)
        execution_history=History(state)
    elif args.execution_checkpoint:
        if baseline or args.randomization or args.split!='train':raise ValueError('B online collection is nominal TRAIN only')
        from exec_comp.execution_controller import ExecutionController,History
        controller=ExecutionController(args.execution_checkpoint,args.execution_gate,args.execution_init,args.execution_iterations,args.device)
        execution_history=History(state)
    output=ROOT/"runs"/("baseline" if baseline else "data")
    if args.label:output=output/args.label
    output.mkdir(parents=True,exist_ok=True)
    amplitude=torch.zeros(args.num_envs,1,1,device=args.device)
    tau=torch.ones(args.num_envs,1,1,device=args.device)
    ou=torch.zeros(args.num_envs,4,3,device=args.device)
    perturb=torch.zeros(args.num_envs,device=args.device,dtype=torch.bool)
    episode=np.arange(args.num_envs,dtype=np.int64)+args.worker*100000000
    next_episode=int(episode.max())+1
    def new_perturb(ids):
        perturb[ids]=torch.rand(len(ids),device=args.device)>.3
        amplitude[ids]=torch.rand(len(ids),1,1,device=args.device)*.05*perturb[ids,None,None]
        tau[ids]=.3+.7*torch.rand(len(ids),1,1,device=args.device);ou[ids]=0
    new_perturb(all_ids)
    pool=ThreadPoolExecutor(max_workers=1);resources.callback(pool.shutdown)
    pending=[];buffer={};shards=[];statistics={}
    active=np.ones(args.num_envs,dtype=bool)
    traces=[{"actual":[],"ref":[],"time":[],"fall":False} for _ in range(args.num_envs)] if baseline else None
    steps=math.ceil(args.transitions/args.num_envs) if not baseline else int(command.time_totals_train.max())+1
    if args.max_steps:steps=min(steps,args.max_steps)
    start=time.monotonic()
    def flush():
        nonlocal buffer
        if not buffer:return
        arrays={k:np.stack(v) for k,v in buffer.items()}
        destination=output/f"{args.data_split or args.split}_worker{args.worker:02d}_shard{len(shards):04d}.npz"
        if destination.exists():raise FileExistsError(destination)
        shards.append(str(destination))
        for k,a in arrays.items():
            if a.dtype.kind=="f":
                flat=a.reshape(-1,a.shape[-1]).astype(np.float64)
                v=statistics.setdefault(k,{"n":0,"sum":np.zeros(flat.shape[-1]),"sumsq":np.zeros(flat.shape[-1])})
                v["n"]+=len(flat);v["sum"]+=flat.sum(0);v["sumsq"]+=(flat*flat).sum(0)
        pending.append(pool.submit(np.savez_compressed,destination,**arrays));buffer={}
        if len(pending)>1:pending.pop(0).result()
    with torch.no_grad():
        for step in range(steps):
            before=state
            reference=replay.reference()
            cmd=reference.clone()
            if controller is not None:
                from exec_comp.diff.chain import sample_pose
                replan=int(step*.02*3)
                if replan!=last_replan:
                    full=replay.reference((np.arange(100)*50/30).tolist())
                    execution_chunk=controller.correct(execution_history,full);chunk_at=step;last_replan=replan
                    reset_pending[:]=False
                cmd=sample_pose(execution_chunk,(torch.arange(6,device=args.device)+step-chunk_at)*.6)
                cmd[reset_pending]=reference[reset_pending]
            elif not baseline:
                alpha=torch.exp(-.02/tau)
                ou=alpha*ou+(1-alpha.square()).sqrt()*amplitude*.5*torch.randn_like(ou)
                ou=ou*torch.minimum(torch.ones_like(ou[...,:1]),.05/ou.norm(dim=-1,keepdim=True).clamp_min(1e-8))
                cmd[...,:3]+=ou[:,None]
            base_action=replay.action(before,cmd)
            action=base_action if baseline else base_action+args.action_noise*torch.randn_like(base_action)
            requested_action=action
            if delay is not None:action=delay(action)
            # Preserve tensors before stepping: simulator buffers can update in place.
            if not baseline:
                row={"q":observed["q"].cpu().numpy().copy(),"dq":observed["dq"].cpu().numpy().copy(),
                     "ang_vel":before["w"].cpu().numpy().copy(),"proj_grav":before["s"][:,-3:].cpu().numpy().copy(),
                     "s":before["s"].cpu().numpy().copy(),"kp":observed["kp"].cpu().numpy().copy(),
                     "root_pos":before["root_p"].cpu().numpy().copy()-env.unwrapped.scene.env_origins.cpu().numpy(),
                     "root_quat":before["root_q"].cpu().numpy().copy(),
                     "privileged_root":before["privileged"].cpu().numpy().copy(),
                     "root_h":before["privileged"][:,:1].cpu().numpy().copy(),
                     "root_linvel_b":before["privileged"][:,1:].cpu().numpy().copy(),
                     "prev_action":replay.hist["action"][:,-1].cpu().numpy().copy(),
                     "action":action.cpu().numpy().copy(),"policy_action":base_action.cpu().numpy().copy(),
                     'requested_action':requested_action.cpu().numpy().copy(),
                     "cmd_kp":cmd.cpu().numpy().copy(),"ref_kp":reference[:,0].cpu().numpy().copy(),
                     "motion_id":command.motion_ids.numpy().astype(np.int32).copy(),
                     "t":command.time_steps.numpy().astype(np.int32).copy(),"episode":episode.copy(),
                     "command_perturbed":perturb.cpu().numpy().copy()}
                origin=env.unwrapped.scene.env_origins.cpu().numpy()
                for key in ("kp","cmd_kp","ref_kp"):row[key][...,:3]-=origin.reshape((args.num_envs,)+(1,)*(row[key].ndim-2)+(3,))
            _,_,_,_=env.step(action)
            if step<3:
                torch.testing.assert_close(env.unwrapped.action_manager.action,action)
            state=replay.capture()
            replay.append(state,requested_action)
            if execution_history is not None:execution_history.append(state,requested_action)
            observed=sensor(state) if sensor else state
            target_next=replay.reference()[:,0]
            error=(state["kp"][...,:3]-target_next[...,:3]).norm(dim=-1)
            # Physical-fall proxy separated from ordinary tracking failure.
            fall=((state["s"][:,-1]>-.25)&(state["root_p"][:,2]-env.unwrapped.scene.env_origins[:,2]<.45))
            catastrophic=error[:,0]>.5
            end=(command.time_steps>=command.time_totals[command.motion_ids]-1).to(args.device)
            done=fall|catastrophic|end
            if baseline:
                actual=state["kp"][...,:3].cpu().numpy();desired=target_next[...,:3].cpu().numpy()
                now=command.time_steps.numpy()
                done_np=done.cpu().numpy();fall_np=fall.cpu().numpy();cat_np=catastrophic.cpu().numpy()
                for i in np.flatnonzero(active):
                    traces[i]["actual"].append(actual[i].copy());traces[i]["ref"].append(desired[i].copy());traces[i]["time"].append(int(now[i]))
                    if done_np[i]:
                        traces[i]["fall"]=bool(fall_np[i]);traces[i]["tracking_failure"]=bool(cat_np[i]);active[i]=False
                if not active.any():break
            else:
                origin=env.unwrapped.scene.env_origins.cpu().numpy()
                row.update({"next_s":state["s"].cpu().numpy().copy(),"ach_kp":state["kp"].cpu().numpy().copy(),
                    "next_root_pos":state["root_p"].cpu().numpy().copy()-origin,
                    "next_root_quat":state["root_q"].cpu().numpy().copy(),
                    "next_privileged_root":state["privileged"].cpu().numpy().copy(),
                    "done":done.cpu().numpy().copy(),"fall":fall.cpu().numpy().copy()})
                row["ach_kp"][...,:3]-=origin[:,None]
                for k,v in row.items():buffer.setdefault(k,[]).append(v)
                if (step+1)%64==0:flush()
                ids=done.nonzero().flatten()
                if len(ids):
                    new_ids=torch.randint(command.num_motion_train,(len(ids),)).numpy()
                    state=replay.reset(ids,new_ids);new_perturb(ids)
                    if execution_history is not None:
                        execution_history.reset(ids,state)
                        # Fresh episode uses its own raw target until the next
                        # global3Hz replan; never reuse another motion's chunk.
                        reset_pending[ids]=True
                    if delay is not None:delay.reset(ids)
                    observed=sensor(state) if sensor else state
                    episode[ids.cpu().numpy()]=np.arange(next_episode,next_episode+len(ids));next_episode+=len(ids)
            if (step+1)%100==0:log("collect_progress",mode=args.mode,worker=args.worker,step=step+1,
                transitions=(step+1)*args.num_envs,elapsed_s=time.monotonic()-start)
    if baseline:
        groups={"standing":[],"walking":[]};per_clip=[]
        for i,trace in enumerate(traces):
            a=np.asarray(trace["actual"]);r=np.asarray(trace["ref"])
            if not len(a):continue
            info=replay.prefix.get(command.motion_names_train[motion_ids[i]])
            if info:group=info["class"]
            else:
                root=r[:,0,:2]
                group="walking" if np.max(np.linalg.norm(root-root[:1],axis=-1))>.2 else "standing"
            errors=np.linalg.norm(a-r,axis=-1)*100
            item={"motion_id":int(motion_ids[i]),"name":command.motion_names_train[motion_ids[i]],
                  "class":group,"fall":trace["fall"],"tracking_failure":trace.get("tracking_failure",False),
                  "errors_cm":{n:np_summary(errors[:,j]) for j,n in enumerate(KP_NAMES)},
                  "wrist_lag_frames":[lag_frames(r[:,j],a[:,j]) for j in (2,3)],"frames":len(a)}
            per_clip.append(item);groups[group].append(errors)
            if i<3:np.savez_compressed(output/f"trace_{i}.npz",reference=r,actual=a,motion_id=motion_ids[i])
        summary={g:{n:np_summary(np.concatenate(v)[:,j]) for j,n in enumerate(KP_NAMES)} for g,v in groups.items() if v}
        all_errors=np.concatenate([np.concatenate(v) for v in groups.values() if v])
        median=np.median(all_errors[:,2:4],axis=0)
        lags=[v for c in per_clip for v in c["wrist_lag_frames"] if v is not None]
        stop=bool(np.all(median<1) and (not lags or abs(np.median(lags))<2))
        result={"clips":len(per_clip),"groups":summary,"per_clip":per_clip,
                "fall_rate":float(np.mean([x["fall"] for x in per_clip])),
                "tracking_failure_rate":float(np.mean([x["tracking_failure"] for x in per_clip])),
                "wrist_median_cm":median.tolist(),"median_lag_frames":float(np.median(lags)) if lags else None,
                "stop_gate":stop,"elapsed_s":time.monotonic()-start,"partial":bool(args.max_steps)}
        result["startup_prefix"]=args.startup_prefix
        save_json(output/"metrics.json",result);log("baseline_complete",**{k:v for k,v in result.items() if k not in ("per_clip","groups")})
    else:
        flush()
        for future in pending:future.result()
        stats={k:{"count":v["n"],"mean":(v["sum"]/v["n"]).tolist(),
                  "std":np.sqrt(np.maximum(v["sumsq"]/v["n"]-(v["sum"]/v["n"])**2,1e-12)).tolist()} for k,v in statistics.items()}
        save_json(output/f"{args.data_split or args.split}_worker{args.worker:02d}_manifest.json",{"shards":shards,"transitions":steps*args.num_envs,
            "worker":args.worker,"num_envs":args.num_envs,"steps":steps,"statistics":stats,"state_privileged":False,
            "privileged_fields":["privileged_root","next_privileged_root","root_h","root_linvel_b"],"alignment_only":["root_pos","root_quat","next_root_pos","next_root_quat"],
            "motion_names":list(command.motion_names_train),"source_manifest":str(ROOT/"configs"/(args.split+".yaml")),
            "startup_prefix":args.startup_prefix,"domain_randomization":args.randomization,"randomization_audit":randomization_audit,"control_dt":.02,
            'a1_q_dq_kp_sensor_noisy':args.randomization,'body_s_next_s_ground_truth':True,
            'policy_history_action':'previous immediately sent action, before actuator delay; requested_action stored separately from physical action',
            "env_spacing":env.unwrapped.cfg.scene.env_spacing,"collision_filtering":True,
            'b1_controller':None if not args.b1_checkpoint else {'checkpoint':args.b1_checkpoint,'amortized':True,'optimizer_iterations':0},
            'execution_controller':None if controller is None or args.b1_checkpoint else {'checkpoint':args.execution_checkpoint,'gate':args.execution_gate,'initial':args.execution_init,'N':args.execution_iterations,'H_exec':17,'privileged':True,'replan_hz':3,'after_reset':'raw for at most16control steps until next3Hz boundary'},
            "perturbations":{"command_max_norm_m":.05,"command_tau_s":None if controller else [.3,1.],"clean_episode_probability":None if controller else .3,"action_sigma":args.action_noise},
            "kp_names":KP_NAMES,"cmd_kp_format":"6x4x9: world position + x/z rotation axes; world origin excludes env grid",
            "ach_kp_timestamp":"state after action, t+1; kp is state before action, t","no_cross_reset_sequences":True})
        log("collection_complete",split=args.split,worker=args.worker,transitions=steps*args.num_envs,shards=len(shards),elapsed_s=time.monotonic()-start)


if __name__=="__main__":
    resources=ExitStack()
    lifecycle.run_with_cleanup(lambda:main(resources),app,resources,SimulationContext.instance)
