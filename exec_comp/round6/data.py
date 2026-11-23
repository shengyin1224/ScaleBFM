"""Explicit R6 data roots only. Whole-clip split, reset-safe temporal windows."""
import json
from pathlib import Path
import numpy as np
import torch
from exec_comp.geometry import heading_anchor,to_anchor,pose_normalize


class Dataset:
    def __init__(self,folder,split,device='cuda:0',kind=None,extra_folders=(),balance_roots=False,min_transitions=0):
        self.device=device;self.rng=np.random.default_rng(20261027);self.streams=[];self.total=0;self.event_cache={}
        keys=None
        if kind=='a1':keys={'s','kp','root_pos','root_quat','ach_kp','cmd_kp','episode','done'}
        elif kind=='body':keys={'truth_s','next_truth_s','action','prev_action','hand','domain','episode','done'}
        elif kind=='b1':keys={'s','kp','root_pos','root_quat','true_root_pos','true_root_quat','truth_s','next_truth_s','hand','domain','prev_action','plan_id','raw_tokens','deployed_tokens','switch_event','episode','done','grasp_phase','comp_pos','comp_curve','comp_age','comp_cap_age'}
        roots=[Path(folder),*map(Path,extra_folders)]
        manifests=[p for root in roots for p in sorted(root.glob(split+'_worker*_manifest.json'))]
        if not manifests:raise FileNotFoundError('No R6 completed manifests: '+split)
        self.provenance=[]
        for file in manifests:
            m=json.loads(file.read_text());parts={}
            self.provenance.append(dict(path=str(file),transitions=m['transitions'],controller=m.get('controller','raw'),schema=m['settings'].get('schema',1)))
            for path in m['shards']:
                with np.load(file.parent/Path(path).name,allow_pickle=False) as a:
                    for k in a.files:
                        if keys is None or k in keys:parts.setdefault(k,[]).append(a[k])
            s={k:np.swapaxes(np.concatenate(v),0,1) for k,v in parts.items()}
            if kind not in ('a1','body'):s['windows']=np.load(file.parent/Path(m['plans']).name)['windows']
            self.streams.append(s);self.total+=s['episode'].size
        if self.total<min_transitions:raise ValueError(f'Insufficient {split} data: {self.total} < {min_transitions}')
        self.weights=np.array([s['episode'].size for s in self.streams],float);self.weights/=self.weights.sum()
        if balance_roots:
            for root in roots:
                ids=[i for i,p in enumerate(manifests) if p.parent==root]
                if not ids:raise ValueError('Missing data root '+str(root))
                size=sum(self.streams[i]['episode'].size for i in ids)
                for i in ids:self.weights[i]=self.streams[i]['episode'].size/size/len(roots)
    def tensor(self,x):return torch.as_tensor(np.ascontiguousarray(x),device=self.device,dtype=torch.float32)
    def choose(self,batch,future,events=False):
        index=int(self.rng.choice(len(self.streams),p=self.weights));s=self.streams[index];n,t=s['episode'].shape;out_e=[];out_t=[];count=0
        if t<=future+10:raise ValueError('Not enough time history')
        if events:
            key=(index,future)
            if key not in self.event_cache:
                grid=np.arange(9,t-future-1)
                good=(s['episode'][:,grid-9]==s['episode'][:,grid+future])&(~s['done'][:,grid+future])&s['switch_event'][:,grid]
                env,time=np.nonzero(good);time=grid[time]
                prefix=np.cumsum(s['switch_event'],axis=1)
                pairs=np.flatnonzero(prefix[env,time+future-1]-prefix[env,time]>0)
                self.event_cache[key]=(env,time,pairs)
                from exec_comp.common import log
                log('r6_event_window_coverage',stream=index,horizon=future,valid_events=len(time),real_switch_pairs=len(pairs),sampling='half actual switch-pair windows when available; no fabricated switch events')
            env,time,pairs=self.event_cache[key]
            if not len(time):raise ValueError('No reset-safe event windows')
            ids=self.rng.integers(len(time),size=batch)
            if len(pairs):ids[:batch//2]=pairs[self.rng.integers(len(pairs),size=batch//2)]
            self.rng.shuffle(ids)
            return s,env[ids],time[ids]
        for _ in range(200):
            e=self.rng.integers(n,size=batch*4);i=self.rng.integers(9,t-future-1,size=batch*4)
            good=(s['episode'][e,i-9]==s['episode'][e,i+future])&(~s['done'][e,i+future])
            if events:good &= s['switch_event'][e,i]
            out_e.append(e[good]);out_t.append(i[good]);count+=good.sum()
            if count>=batch:return s,np.concatenate(out_e)[:batch],np.concatenate(out_t)[:batch]
        raise ValueError('Not enough reset-safe samples, event='+str(events))
    def inverse(self,batch):
        offsets=np.arange(100)*50/15;lo=offsets.astype(int);hi=np.ceil(offsets).astype(int);s,e,t=self.choose(batch,int(hi[-1]));hist=t[:,None]+np.arange(-9,1)
        origin,heading=heading_anchor(self.tensor(s['root_pos'][e,t]),self.tensor(s['root_quat'][e,t]));state=self.tensor(s['s'][e[:,None],hist]);kp=to_anchor(self.tensor(s['kp'][e[:,None],hist]),origin,heading)
        weight=self.tensor(offsets-lo)[None,:,None,None]
        def future(k):
            a=self.tensor(s[k][e[:,None],t[:,None]+lo]);b=self.tensor(s[k][e[:,None],t[:,None]+hi]);return to_anchor(pose_normalize(a+(b-a)*weight),origin,heading)
        return state,kp,future('ach_kp'),future('cmd_kp')
    def body(self,batch,horizon):
        s,e,t=self.choose(batch,horizon);ids=t[:,None]+np.arange(horizon)
        truth=self.tensor(s['truth_s'][e[:,None],ids]);action=self.tensor(s['action'][e[:,None],ids]);ns=self.tensor(s['next_truth_s'][e[:,None],ids]);h=self.tensor(s['hand'][e[:,None],ids]);domain=self.tensor(s['domain'][e[:,None],ids])
        past=self.tensor(s['prev_action'][e[:,None,None],ids[:,:,None]+np.arange(-2,1)[None,None]]).flatten(2)
        return truth,action,torch.cat((past,h,domain),-1),ns
    def event(self,batch,horizon):
        s,e,t=self.choose(batch,horizon,True);hist=t[:,None]+np.arange(-9,1);ids=t[:,None]+np.arange(horizon)
        return dict(s=self.tensor(s['s'][e[:,None],hist]),kp=self.tensor(s['kp'][e[:,None],hist]),
            origin=self.tensor(s['root_pos'][e,t]),quat=self.tensor(s['root_quat'][e,t]),
            root_pos=self.tensor(s['true_root_pos'][e,t]),root_quat=self.tensor(s['true_root_quat'][e,t]),
            root_quat_history=self.tensor(s['root_quat'][e[:,None],hist]),
            truth=self.tensor(s['truth_s'][e,t]),hand=self.tensor(s['hand'][e,t]),domain=self.tensor(s['domain'][e,t]),
            actions=self.tensor(s['prev_action'][e[:,None],t[:,None]+np.arange(-2,1)]),
            window=self.tensor(s['windows'][s['plan_id'][e,t]]),raw=self.tensor(s['raw_tokens'][e[:,None],ids]),
            next_truth=self.tensor(s['next_truth_s'][e[:,None],ids]),
            events=torch.as_tensor(s['switch_event'][e[:,None],ids],device=self.device,dtype=torch.bool),
            windows=self.tensor(s['windows'][s['plan_id'][e[:,None],ids]]),
            deployed=self.tensor(s.get('deployed_tokens',s['raw_tokens'])[e[:,None],ids]),
            grasp=torch.as_tensor(s['grasp_phase'][e[:,None],ids] if 'grasp_phase' in s else np.zeros((batch,horizon),bool),device=self.device,dtype=torch.bool),
            initial_residual=self.tensor(s['comp_pos'][e,t] if 'comp_pos' in s else np.zeros((batch,6,4,3),np.float32)),
            initial_coefficients=self.tensor(s['comp_curve'][e,t] if 'comp_curve' in s else np.zeros((batch,2,4,3),np.float32)),
            initial_age=self.tensor(s['comp_age'][e,t] if 'comp_age' in s else np.zeros(batch,np.float32)),
            initial_cap_age=self.tensor(s['comp_cap_age'][e,t] if 'comp_cap_age' in s else np.zeros(batch,np.float32)))
