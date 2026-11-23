"""Paired MuJoCo evaluation, only library heldout/previously exported brain outputs."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy.signal import butter,sosfiltfilt
from exec_comp.common import ROOT,metadata,log
from exec_comp.round6.env import World
from exec_comp.round6.library import Clip,paths
from exec_comp.round6.online import Stream
from exec_comp.round6.runtime import Executor
from exec_comp.round6.policy_driver import Driver
from exec_comp.round6.switching import make_chunk
from exec_comp.geometry import six_to_q
from exec_comp.round5.engine import qblend
from exec_comp.round5.reference import canonical7,canonical9


class BrainStream(Stream):
    def __init__(self,path,origin,rng):
        super().__init__(None,rng,False)
        with np.load(path) as f:self.data={k:f[k] for k in f.files}
        self.origin=origin;self.next_query=0
    def update(self,elapsed,source_time,state):
        now=int((1000+elapsed)*1e9);q=six_to_q(torch.tensor(state['kp'][:,3:],dtype=torch.float32)).numpy();seq=self.switch.publish_state(now,state['kp'][:,:3],q)
        event={'accepted':False}
        while self.next_query<len(self.data['query_time_s']) and self.data['query_time_s'][self.next_query]+(.105 if self.next_query else 0.)<=elapsed+1e-8:
            idx=self.next_query;self.next_query+=1;kp=self.data['keypoints'][idx].copy();kp[...,:3]+=self.origin
            poseq=six_to_q(torch.tensor(kp[...,3:],dtype=torch.float32)).numpy();source=max(0,seq-5)
            chunk=make_chunk(idx,source,now,15,(kp[...,:3],poseq),self.data['fingertip_local'][idx]);result=self.switch.offer(chunk,now)
            if result['accepted']:event=result
        if not event['accepted']:event=self.switch.poll(now)
        self.accepted+=int(event['accepted']);sample=self.switch.sample(now);raw=self.switch.sample_raw(now,sample);window=None
        if event['accepted']:
            import copy
            from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
            proxy=copy.copy(self.switch.env);proxy.previous_chunk=None;proxy.blend_started_ns=None;proxy.future_offsets_s=np.arange(100)/15
            window=canonical9(HAT4OnlineMotionTrackingEnv._sample_reference(proxy,now,apply_stall_assist=False))
        return canonical7(sample),canonical7(raw),window,event,now


def highpass(values):
    if len(values)<60:return [None,None]
    filtered=sosfiltfilt(butter(4,5,fs=50,btype='highpass',output='sos'),values,axis=0)[10:-10]
    return (np.sqrt(np.mean(np.sum(filtered*filtered,axis=-1),axis=0))*1000).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=('raw','a1','b1'),default='raw');p.add_argument('--checkpoint');p.add_argument('--population',choices=('library','brain'),default='library');p.add_argument('--clips',type=int,default=20);p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch',type=int,default=4);p.add_argument('--randomized',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--xml');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();torch.set_num_threads(2)
    entries=paths('validation')[:a.clips];brain_manifest=None
    if a.population=='brain':
        folder=ROOT/'runs/round3/hat_predictions';brain_manifest=json.loads((folder/'manifest.json').read_text());entries=[dict(e,file=str(folder/Path(e['file']).name)) for e in brain_manifest['episodes'] if e['type']=='robot'][:a.clips]
    if not 0<=a.shard_index<a.shards:raise ValueError('Invalid evaluation shard')
    entries=entries[a.shard_index::a.shards]
    results=[];all_errors=[];all_actions=[];all_hf=[];started=time.monotonic()
    for begin in range(0,len(entries),a.batch):
        group=entries[begin:begin+a.batch];n=len(group);worlds=[World(xml=a.xml) for _ in group];driver=Driver(a.device);controllers=[Executor(a.method,a.checkpoint,a.device) for _ in group];states=[];streams=[];limits=[];names=[]
        for i,(entry,w) in enumerate(zip(group,worlds)):
            rng=np.random.default_rng(20261027+begin+i) if a.randomized else None
            if a.population=='library':
                clip=Clip(entry);s=clip.reset(w,rng=rng);stream=Stream(clip,np.random.default_rng(42+begin+i));limits.append(min(a.steps,clip.frames-11));names.append(entry.stem)
            else:
                s=w.reset(np.array(metadata()['default_dof_pos']),np.array([0.,0.,.793]),np.array([1.,0,0,0]),rng=rng);stream=BrainStream(entry['file'],s['kp'][1,:3],np.random.default_rng(42+begin+i));limits.append(min(a.steps,int(entry['duration_s']*50)));names.append(Path(entry['file']).stem)
            states.append(w.observe(s));streams.append(stream)
        driver.reset(states);warm_fail=np.zeros(n,bool)
        if a.population=='brain':
            start_p=np.stack([s['kp'][:,:3] for s in states]);start_q=np.stack([six_to_q(torch.tensor(s['kp'][:,3:]).float()).numpy() for s in states]);target=np.stack([st.data['initial_keypoints'] for st in streams]);target[...,:3]+=np.stack([st.origin for st in streams])[:,None];target_q=six_to_q(torch.tensor(target[...,3:]).float()).numpy()
            for k in range(100):
                alpha=min(k*.02/.75,1.);alpha=alpha*alpha*(3-2*alpha);p=start_p*(1-alpha)+target[...,:3]*alpha;q=qblend(start_q,target_q,alpha)
                action=driver.action(states,np.repeat(p[:,None],6,1),np.repeat(q[:,None],6,1));targets=driver.targets(action);states=[w.observe(w.step(t)) for w,t in zip(worlds,targets)];driver.append(states,action);warm_fail|=np.array([w.fallen() for w in worlds])
        trace=[dict(error=[],actual=[],target=[],actions=[],focus=[],dq=[],fallen=False) for _ in group];done=warm_fail.copy();pool=ThreadPoolExecutor(n)
        for k in range(max(limits)):
            refs=[];focus_flags=[];targets_next=[];observed=states
            for i,(w,st,c,s) in enumerate(zip(worlds,streams,controllers,states)):
                if done[i] or k>=limits[i]:
                    done[i]=True;kp=s['kp'];q=six_to_q(torch.tensor(kp[:,3:]).float()).numpy();refs.append(np.repeat(np.concatenate((kp[:,:3],q),-1)[None],6,0));focus_flags.append(False);targets_next.append(kp[2:,:3]);continue
                elapsed=k*.02;source_time=.2+elapsed;c.observe(s);deployed,raw,window,event,stamp=st.update(elapsed,source_time,s)
                grasp=st.switch.grasp_phase(stamp)
                focus_flags.append(st.clip.focus(source_time) if a.population=='library' else grasp)
                if window is not None:c.event(window,stamp,int(event['sequence_id']))
                corrected,info=c.correct(stamp,deployed,raw,grasp)
                if not np.array_equal(corrected[:,:2],deployed[:,:2]):raise AssertionError('head/root changed')
                refs.append(corrected)
                targets_next.append(raw[1,2:,:3]);trace[i]['focus'].append(grasp)
            if done.all():break
            refs=np.stack(refs);action=driver.action(states,refs[...,:3],refs[...,3:]);targets=driver.targets(action)
            physical=list(pool.map(lambda args:args[0].capture() if args[2] else args[0].step(args[1],args[3]),[(w,t,done[i],focus_flags[i]) for i,(w,t) in enumerate(zip(worlds,targets))]))
            states=[w.observe(s) for w,s in zip(worlds,physical)];driver.append(states,action)
            for i in range(n):
                if not done[i]:
                    t=trace[i];actual=physical[i];target=targets_next[i]
                    t['error'].append(np.linalg.norm(target-actual['kp'][2:,:3],axis=-1)*100);t['actual'].append(actual['kp'][2:,:3]);t['target'].append(target);t['dq'].append(actual['dq'])
                    t['actions'].append(action[i]);t['fallen']|=worlds[i].fallen();done[i]|=t['fallen']
        pool.shutdown()
        for i,t in enumerate(trace):
            errors=np.asarray(t['error']);actual=np.asarray(t['actual']);act=np.asarray(t['actions']);mask=np.asarray(t['focus'],bool)
            hf=highpass(actual) if len(actual) else [None,None]
            item=dict(population=a.population,method=a.method,clip=names[i],steps=len(errors),fallen=bool(t['fallen']),warmup_failed=bool(warm_fail[i]),
                wrist_mean_cm=errors.mean(0).tolist() if len(errors) else None,wrist_p95_cm=np.percentile(errors,95,axis=0).tolist() if len(errors) else None,
                grasp_wrist_cm=errors[mask].mean(0).tolist() if mask.any() else None,grasp_steps=int(mask.sum()),actual_above5Hz_vector_rms_mm=hf,
                action_step_rms=float(np.sqrt(np.mean(np.diff(act,axis=0)**2))) if len(act)>1 else None,
                joint_velocity_rms=float(np.sqrt(np.mean(np.asarray(t['dq'])**2))) if t['dq'] else None,
                accepted_switches=streams[i].accepted,network_calls=controllers[i].calls,fallback_events=controllers[i].engine.counters)
            results.append(item);log('r6_evaluation_clip',**item)
            if len(errors):all_errors.extend(errors)
            if all(v is not None for v in hf):all_hf.append(hf)
            if len(act)>1:all_actions.append(item['action_step_rms'])
    summary=dict(population=a.population,method=a.method,randomized=a.randomized,clips=len(results),falls=sum(r['fallen'] or r['warmup_failed'] for r in results),
        wrist_mean_cm=np.mean(all_errors,axis=0).tolist() if all_errors else None,actual_above5Hz_vector_rms_mm=np.mean(all_hf,axis=0).tolist() if all_hf else None,
        action_step_rms=float(np.mean(all_actions)) if all_actions else None,elapsed_s=time.monotonic()-started,
        definition='Error compares pure original token1 K*(t+20ms) with fresh physical XML FK after the20ms step, matching Isaac evaluation;>5Hz RMS is sqrt(mean squared3D highpass displacement), separately each wrist in world, mm; no robot, no object-grasp success claim.')
    log('r6_evaluation_complete',**summary)


if __name__=='__main__':main()
