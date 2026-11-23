"""New MuJoCo trajectories, unifolm only; no real-data imports or reads."""
import argparse,json,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from exec_comp.common import ROOT,log,digest,metadata,SOURCE_CHECKPOINT_HASH
from exec_comp.round6.env import World,XML
from exec_comp.round6.library import Clip,paths,canonical
from exec_comp.round6.online import Stream
from exec_comp.round6.policy_driver import Driver
from exec_comp.round6.protocol import settings,guard_training


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','validation'),required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--envs',type=int,default=8);p.add_argument('--transitions',type=int,default=131072);p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);p.add_argument('--controller',choices=('raw','b1'),default='raw');p.add_argument('--checkpoint');a=p.parse_args()
    guard_training();torch.set_num_threads(2);torch.manual_seed(20261027+a.worker);rng=np.random.default_rng(20261027+100*a.worker+(a.split=='validation')*10000)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);prefix=f'{a.split}_worker{a.worker:02d}'
    if list(output.glob(prefix+'*')):raise FileExistsError(prefix)
    n=a.envs;driver=Driver(a.device);worlds=[World() for _ in range(n)];pool=ThreadPoolExecutor(n);files=paths(a.split)
    from exec_comp.round6.runtime import Executor
    controllers=[Executor(a.controller,a.checkpoint,a.device) for _ in range(n)]
    states=[];observed=[];clips=[];streams=[];ages=np.zeros(n,int);starts=np.zeros(n,int);ep=np.arange(n,dtype=np.int64)+a.worker*10000000
    def reset(i):
        c=Clip(files[int(rng.integers(len(files)))]);start=int(rng.integers(10,max(11,min(c.frames-450,300))))
        truth=c.reset(worlds[i],start=start,rng=np.random.default_rng(int(rng.integers(1,2**31))))
        stream=Stream(c,np.random.default_rng(int(rng.integers(1,2**31))),True)
        return c,stream,truth,worlds[i].observe(truth),start
    for i in range(n):
        c,st,s,o,start=reset(i);clips.append(c);streams.append(st);states.append(s);observed.append(o);starts[i]=start
    driver.reset(observed);plans=[];plan_ids=np.full(n,-1,np.int64);buffer={};shards=[];falls=0;started=time.monotonic();episode_names={int(ep[i]):clips[i].path.stem for i in range(n)}
    steps=int(np.ceil(a.transitions/n))
    def flush():
        if not buffer:return
        path=output/f'{prefix}_shard{len(shards):04d}.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in buffer.items()});shards.append(str(path));buffer.clear()
    for k in range(steps):
        times=(starts+ages)/50.;deployed=[];issued=[];raw=[];events=[];grasp_flags=[];comp=[];old_coeff=[];old_ages=[];cap_age=[]
        for i in range(n):
            ref,pure,window,event,stamp=streams[i].update(ages[i]*.02,times[i],observed[i])
            if window is not None:plan_ids[i]=len(plans);plans.append(window)
            c=controllers[i];engine=c.engine;c.observe(observed[i]);grasp=streams[i].switch.grasp_phase(stamp);grasp_flags.append(grasp)
            comp.append(np.zeros((6,4,3)) if engine.pos is None else engine.pos.copy())
            curve=engine.curve;old_coeff.append(np.zeros((2,4,3)) if curve is None else curve.coefficients[...,:3].copy())
            old_ages.append(0. if curve is None else (stamp-curve.stamp_ns)*1e-9)
            cap_age.append(0. if engine.cap_since is None else (stamp-engine.cap_since)*1e-9)
            if window is not None:c.event(window,stamp,int(event['sequence_id']))
            corrected,_=c.correct(stamp,ref,pure,grasp)
            issued.append(corrected)
            deployed.append(ref);raw.append(pure);events.append(bool(event['accepted']))
        base_refs=np.stack(deployed);refs=np.stack(issued);raw=np.stack(raw);policy_action=driver.action(observed,refs[...,:3],refs[...,3:])
        action=policy_action+rng.normal(0,.05,policy_action.shape).astype(np.float32);targets=driver.targets(action)
        domain=np.array([np.r_[w.domain['hand_load_scale'],w.domain['pd_kp_scale'],w.domain['pd_kd_scale'],w.delay_s,w.domain['friction']] for w in worlds],np.float32)
        row=dict(s=np.stack([s['s'] for s in observed]),kp=np.stack([s['kp'] for s in observed]),root_pos=np.stack([s['p'] for s in observed]),root_quat=np.stack([s['quat'] for s in observed]),
            truth_s=np.stack([np.r_[s['s'],s['privileged']] for s in states]),true_root_pos=np.stack([s['p'] for s in states]),true_root_quat=np.stack([s['quat'] for s in states]),
            prev_action=driver.hist['action'][:,-1].cpu().numpy().copy(),action=action,policy_action=policy_action,
            cmd_kp=canonical(refs[:,0,:,:3],refs[:,0,:,3:]),ref_kp=canonical(raw[:,0,:,:3],raw[:,0,:,3:]),raw_tokens=canonical(raw[...,:3],raw[...,3:]),
            deployed_tokens=canonical(base_refs[...,:3],base_refs[...,3:]),grasp_phase=np.array(grasp_flags),comp_pos=np.stack(comp),comp_curve=np.stack(old_coeff),comp_age=np.array(old_ages),comp_cap_age=np.array(cap_age),
            domain=domain,hand=np.stack([np.r_[w.data.qpos[w.hq],w.data.qvel[w.hv]] for w in worlds]),plan_id=plan_ids.copy(),switch_event=np.array(events),episode=ep.copy(),
            focus=np.array([clips[i].focus(times[i]) for i in range(n)]),clip_frame=starts+ages)
        states=list(pool.map(lambda pair:pair[0].step(pair[1],pair[2]),[(w,t,clips[i].focus(times[i])) for i,(w,t) in enumerate(zip(worlds,targets))]))
        observed=[w.observe(s) for w,s in zip(worlds,states)];driver.append(observed,action)
        done=np.array([w.fallen() or starts[i]+ages[i]+12>=clips[i].frames for i,w in enumerate(worlds)])
        falls+=sum(w.fallen() for w in worlds)
        row.update(next_truth_s=np.stack([np.r_[s['s'],s['privileged']] for s in states]),ach_kp=np.stack([s['kp'] for s in states]),next_true_root_pos=np.stack([s['p'] for s in states]),next_true_root_quat=np.stack([s['quat'] for s in states]),done=done)
        for key,v in row.items():buffer.setdefault(key,[]).append(np.asarray(v,dtype=np.float32) if np.asarray(v).dtype.kind=='f' else v)
        ages+=1
        for i in np.flatnonzero(done):
            c,st,s,o,start=reset(i);clips[i]=c;streams[i]=st;states[i]=s;observed[i]=o;starts[i]=start;ages[i]=0;ep[i]+=1000;episode_names[int(ep[i])]=c.path.stem
            for key in ('q','dq','quat','w'):
                value=torch.tensor(o[key],device=a.device,dtype=torch.float32);driver.hist[key][i]=value[None].expand(3,-1)
            driver.hist['action'][i]=0
            controllers[i].engine.reset();controllers[i].rows=[]
        if (k+1)%256==0:flush()
        if k%250==0:log('r6_collect_progress',worker=a.worker,split=a.split,steps=k,transitions=k*n,falls=int(falls),accepted=[s.accepted for s in streams],elapsed_s=time.monotonic()-started)
    flush();pool.shutdown();np.savez_compressed(output/(prefix+'_plans.npz'),windows=np.stack(plans))
    manifest=dict(settings=settings(),xml=str(XML),xml_sha256=digest(XML),source_checkpoint_sha256=SOURCE_CHECKPOINT_HASH,shards=shards,plans=str(output/(prefix+'_plans.npz')),transitions=steps*n,envs=n,episode_clips=episode_names,split=a.split,worker=a.worker,falls=int(falls),elapsed_s=time.monotonic()-started,controller=a.controller,controller_sha256=digest(a.checkpoint) if a.checkpoint else None)
    (output/(prefix+'_manifest.json')).write_text(json.dumps(manifest,indent=2));log('r6_collect_complete',worker=a.worker,split=a.split,transitions=steps*n,falls=int(falls),elapsed_s=manifest['elapsed_s'])


if __name__=='__main__':main()
