"""Replay held-out HAT prediction chunks through the frozen body controller only."""
import argparse
from contextlib import ExitStack
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--method',choices=('raw','A1'),required=True)
p.add_argument('--predictions',default='runs/round2/hat_predictions');p.add_argument('--checkpoint',default=None)
p.add_argument('--label',default=None);p.add_argument('--residual-lowpass',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import json
import numpy as np
import torch
from isaaclab.sim import SimulationContext
from exec_comp.common import ROOT,TRACK,log,save_json,np_summary,KP_NAMES,metadata,import_path
from exec_comp.env import make_env,Replay
from exec_comp.policy import FrozenPolicy
from exec_comp.runtime import KeypointCompensator,correct_encoded
from exec_comp.geometry import from_anchor,to_anchor,heading_anchor
from exec_comp.hat_playback import sample,quaternion_slerp
from exec_comp.jitter import ResidualLowPass,clip_jitter,aggregate_jitter
lifecycle=import_path('exec_comp_hat_lifecycle',TRACK/'scripts/precision_teacher_student/student_lifecycle.py')


def main(resources):
    torch.manual_seed(42);torch.set_num_threads(4);device=args.device
    manifest=json.loads((ROOT/args.predictions/'manifest.json').read_text())
    if manifest['smoke']:raise ValueError('Smoke prediction set cannot be treated as full25-episode evaluation')
    episodes=manifest['episodes'];n=len(episodes);fps=float(manifest['chunk_fps'])
    if fps!=30.:raise ValueError('This documented A1 validation uses100@30Hz; resampling is required for another playback rate')
    label=args.label or args.method;output=ROOT/'runs/round2/hat_evaluation'/label;output.mkdir(parents=True,exist_ok=False)
    policy=FrozenPolicy().to(device);env=make_env(n,'validation',device);resources.callback(env.close)
    replay=Replay(env,policy,True);robot=env.unwrapped.scene['robot'];ids=torch.arange(n,device=device)
    replay.reset(ids,np.arange(n))
    q=torch.tensor(metadata()['default_dof_pos'],device=device)[None].repeat(n,1)
    root=env.unwrapped.scene.env_origins.clone();root[:,2]+=.793
    quat=torch.zeros(n,4,device=device);quat[:,0]=1
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
    robot.write_root_state_to_sim(torch.cat((root,quat,torch.zeros(n,6,device=device)),-1),env_ids=ids)
    env.unwrapped.scene.write_data_to_sim();env.unwrapped.sim.forward();env.unwrapped.scene.update(dt=.02)
    state=replay.capture();start_pose=state['kp'].clone();origin=start_pose[:,1,:3]
    replay.hist={k:state[key][:,None].repeat(1,3,1).clone() for k,key in (('q','q'),('dq','dq'),('root_q','root_q'),('w','w'))}
    replay.hist['action']=torch.zeros(n,3,29,device=device)
    count=max(e['queries'] for e in episodes);all_chunks=[];initial=[];durations=[];query_times=[]
    for episode in episodes:
        with np.load(episode['file']) as f:
            x=torch.tensor(f['keypoints'],device=device)
            x=torch.cat((x,x[-1:].repeat(count-len(x),1,1,1)),0)
            all_chunks.append(x);initial.append(torch.tensor(f['initial_keypoints'],device=device))
            times=torch.tensor(f['query_time_s'],device=device,dtype=torch.float32)
            query_times.append(torch.cat((times,times[-1:].repeat(count-len(times)))))
        durations.append(episode['duration_s'])
    chunks=torch.stack(all_chunks);chunks[...,:3]+=origin[:,None,None,None]
    query_times=torch.stack(query_times)
    initial=torch.stack(initial);initial[...,:3]+=origin[:,None]
    history={k:state[k][:,None].repeat(1,10,1,1) if state[k].ndim==3 else state[k][:,None].repeat(1,10,1) for k in ('q','dq','kp')}
    warmup_failure=torch.zeros(n,dtype=torch.bool,device=device)
    # Common raw-policy preparation, identical for both evaluation methods.
    # Smoothly bring the standing body toward the first measured source pose.
    with torch.no_grad():
        from exec_comp.geometry import six_to_q,pose9
        for step in range(100):
            t=min(step*.02/.75,1.);alpha=t*t*(3-2*t)
            pos=start_pose[...,:3]*(1-alpha)+initial[...,:3]*alpha
            rotation=quaternion_slerp(six_to_q(start_pose[...,3:]),six_to_q(initial[...,3:]),start_pose.new_full((n,1,1),alpha))
            target=pose9(pos,rotation)[:,None].repeat(1,6,1,1)
            action=replay.action(state,target);env.step(action);state=replay.capture();replay.append(state,action)
            for key in history:history[key]=torch.cat((history[key][:,1:],state[key][:,None]),1)
            warmup_failure|=((state['s'][:,-1]>-.25)&(state['root_p'][:,2]<.45))
    initialization={k:state[k].cpu().tolist() for k in ('q','dq','root_p','root_q')}
    correction=KeypointCompensator('A1',checkpoint=args.checkpoint,device=device) if args.method=='A1' else None
    lowpass=ResidualLowPass() if args.residual_lowpass else None
    traces=[{'reference':[],'actual':[],'command':[],'action':[],'dq':[],'seams':[],'residual':[],'model_residual':[],'fall':False,'failure':False} for _ in episodes]
    active=~warmup_failure.cpu().numpy();previous_replan=-1;corrected=None;at=0
    duration=np.asarray(durations);max_steps=int(np.ceil(duration.max()*50))
    with torch.no_grad():
        for step in range(max_steps):
            t=step*.02;replan=int(t*3)
            index=torch.tensor([min(replan,e['queries']-1) for e in episodes],device=device)
            raw_chunk=chunks[ids,index]
            if replan!=previous_replan:
                if corrected is not None:
                    old=sample(corrected,torch.full((n,1),(step-at)*.02*fps,device=device))[:,0]
                if correction is None:corrected=raw_chunk
                else:
                    anchor,heading=heading_anchor(state['root_p'],state['root_q'])
                    encoded=torch.cat((to_anchor(history['kp'],anchor,heading).flatten(2),history['q'],history['dq']),-1).flatten(1)
                    corrected=from_anchor(correct_encoded(correction.model,encoded,to_anchor(raw_chunk,anchor,heading)),anchor,heading)
                if previous_replan>=0:
                    seam=(corrected[:,0,2:4,:3]-old[:,2:4,:3]).cpu().numpy()
                    if lowpass:seam*=.5
                    for i in np.flatnonzero(active):traces[i]['seams'].append(seam[i])
                at=step;previous_replan=replan
            age=(t-query_times[ids,index])*fps
            frames=age[:,None]+torch.arange(6,device=device)[None]*.02*fps
            raw=sample(raw_chunk,frames)
            # The corrected chunk is based at this replan timestamp, exactly as raw.
            command=sample(corrected,frames)
            model_delta=(command[:,0,2:4,:3]-raw[:,0,2:4,:3]).norm(dim=-1)*100
            if lowpass:command=lowpass(command,raw)
            residual=(command[:,0,2:4,:3]-raw[:,0,2:4,:3]).norm(dim=-1)*100
            action=replay.action(state,command);env.step(action);state=replay.capture();replay.append(state,action)
            # Response t+20ms is compared with the same causally available plan,
            # before accepting a new prediction on the next control tick.
            target=sample(raw_chunk,(age+.02*fps)[:,None])[:,0]
            error=(state['kp'][...,:3]-target[...,:3]).norm(dim=-1)
            fall=((state['s'][:,-1]>-.25)&(state['root_p'][:,2]<.45)).cpu().numpy()
            failure=(error[:,0]>.5).cpu().numpy()
            values={'reference':target.cpu().numpy(),'actual':state['kp'].cpu().numpy(),'command':command[:,0].cpu().numpy(),
                    'action':action.cpu().numpy(),'dq':state['dq'].cpu().numpy(),'residual':residual.cpu().numpy(),'model_residual':model_delta.cpu().numpy()}
            for i in np.flatnonzero(active):
                for key,value in values.items():traces[i][key].append(value[i].copy())
                if fall[i] or failure[i] or t+.02>=duration[i]:
                    traces[i]['fall']=bool(fall[i]);traces[i]['failure']=bool(failure[i]);active[i]=False
            for key in history:history[key]=torch.cat((history[key][:,1:],state[key][:,None]),1)
            if not active.any():break
            if step%500==0:log('hat_closed_loop_progress',method=args.method,step=step,active=int(active.sum()))
    clips=[];errors=[];jitters=[];residuals=[]
    for i,(episode,tr) in enumerate(zip(episodes,traces)):
        item={'episode':i,'source':episode['source'],'type':episode['type'],'initialization_failed':bool(warmup_failure[i]),'frames':len(tr['actual']),
              'fall':bool(tr['fall'] or warmup_failure[i]),'tracking_failure':tr['failure'],'grasp_window_error_cm':None}
        if tr['actual']:
            a=np.asarray(tr['actual']);r=np.asarray(tr['reference']);c=np.asarray(tr['command'])
            e=np.linalg.norm(a[...,:3]-r[...,:3],axis=-1)*100;errors.append((episode['type'],e))
            item['errors_cm']={key:np_summary(e[:,j]) for j,key in enumerate(KP_NAMES)}
            item['residual_cm']={key:np_summary(np.asarray(tr['residual'])[:,j]) for j,key in enumerate(('left_wrist','right_wrist'))}
            item['applied_cap_fraction']=(np.asarray(tr['residual'])>=4.99).mean(0).tolist()
            item['model_cap_fraction']=(np.asarray(tr['model_residual'])>=4.99).mean(0).tolist()
            jitters.append(clip_jitter(c,a,tr['action'],tr['dq'],tr['seams']))
            residuals.append((episode['type'],np.asarray(tr['residual']),np.asarray(tr['model_residual'])))
            np.savez_compressed(output/f'episode_{i:02d}.npz',**{k:np.asarray(v) for k,v in tr.items() if isinstance(v,list)})
        else:item['errors_cm']=None
        clips.append(item)
    groups={};residual_groups={}
    for kind in ('all','human','robot'):
        selected=[e for k,e in errors if kind=='all' or k==kind]
        if selected:
            pooled=np.concatenate(selected);groups[kind]={key:np_summary(pooled[:,j]) for j,key in enumerate(KP_NAMES)}
            residual_groups[kind]={}
            for index,which in ((1,'applied'),(2,'model_before_filter')):
                mag=np.concatenate([v[index] for v in residuals if kind=='all' or v[0]==kind])
                residual_groups[kind][which]={key:{**np_summary(mag[:,j]),'p99':float(np.percentile(mag[:,j],99)),
                    'max_cm':float(mag[:,j].max()),'cap_fraction':float((mag[:,j]>=4.99).mean())} for j,key in enumerate(('left_wrist','right_wrist'))}
    result={'method':label,'groups':groups,'per_clip':clips,'jitter':aggregate_jitter(jitters) if jitters else None,
            'clips':n,'fall_rate':float(np.mean([c['fall'] for c in clips])),'tracking_failure_rate':float(np.mean([c['tracking_failure'] for c in clips])),
            'initialization_failure_rate':float(warmup_failure.float().mean()),'initialization_state':initialization,
            'residual_groups_cm':residual_groups,
            'warmup':'same raw frozen policy for2s,0.75s smooth transition toward measured initial pose; fixed opening-head world translation',
            'bottle_simulated':False,'grasp_success_rate':None,'grasp_window_error':'N/A: no provided focus/grasp labels',
            'chunk_fps':fps,'replan_hz':3,'response_alignment_ms':20,'checkpoint':args.checkpoint,'residual_lowpass':args.residual_lowpass}
    save_json(output/'metrics.json',result);log('hat_closed_loop_complete',method=label,groups=groups)


if __name__=='__main__':
    resources=ExitStack();lifecycle.run_with_cleanup(lambda:main(resources),app,resources,SimulationContext.instance)
