"""R6 simulation executor:15Hz curve,50Hz expansion; both methods wrist-only."""
from dataclasses import dataclass
import numpy as np
import torch
from exec_comp.round5.engine import CurveExecutor,Curve,qlog,qmul,qinv
from exec_comp.geometry import heading_anchor,to_anchor,from_anchor,six_to_q
from exec_comp.round6.train import load


@dataclass
class SampledCurve:
    samples:np.ndarray
    stamp_ns:int
    sequence:int
    epoch:int=0
    span_s:float=6.6
    def sample(self,now_ns,offsets):
        f=np.clip(((now_ns-self.stamp_ns)*1e-9+np.asarray(offsets))*15,0,99);lo=f.astype(int);hi=np.minimum(lo+1,99);a=(f-lo)[:,None,None]
        return self.samples[lo]+(self.samples[hi]-self.samples[lo])*a


class Executor:
    def __init__(self,method='raw',checkpoint=None,device='cpu'):
        self.method=method;self.device=device;self.engine=CurveExecutor(enable=method!='raw');self.rows=[];self.calls=0
        self.model=None
        if method!='raw':
            self.model,self.saved=load(checkpoint,device);self.model.requires_grad_(False)
            if self.saved['kind']!=method:raise ValueError('Checkpoint method mismatch')
    def observe(self,state):
        self.rows.append(state)
        if len(self.rows)>10:self.rows.pop(0)
    def event(self,window,now,sequence):
        if self.model is None:return
        rows=[self.rows[0]]*(10-len(self.rows))+self.rows
        tensor=lambda x:torch.as_tensor(np.stack(x),device=self.device,dtype=torch.float32)
        h=tensor([r['s'] for r in rows])[None];kp=tensor([r['kp'] for r in rows])[None]
        p=tensor([rows[-1]['p']]);q=tensor([rows[-1]['quat']]);goal=torch.as_tensor(window,device=self.device,dtype=torch.float32)[None]
        origin,heading=heading_anchor(p,q);anchored=to_anchor(goal,origin,heading);encoded=self.model.features(h,to_anchor(kp,origin,heading))
        with torch.no_grad():
            if self.method=='b1':
                c=self.model.coefficients(encoded,anchored);world=torch.zeros(2,4,6,device=self.device)
                from exec_comp.geometry import qapply
                world[:,2:,:3]=qapply(heading[:,None,None].expand(1,2,2,4),c)[0]
                curve=Curve(world.cpu().numpy(),now,sequence,span_s=6.6);self.engine.install(curve,now)
            else:
                out=from_anchor(self.model(encoded,anchored),origin,heading)[0]
                original=goal[0];rotation=qlog(qmul(six_to_q(out[...,3:]).cpu().numpy(),qinv(six_to_q(original[...,3:]).cpu().numpy())))
                samples=np.concatenate(((out[...,:3]-original[...,:3]).cpu().numpy(),rotation),-1);samples[:,:2]=0
                if not np.isfinite(samples).all():return
                curve=SampledCurve(samples,now,sequence)
                self.engine.initial_curve_pending=self.engine.curve is None;self.engine.previous_curve=self.engine.curve;self.engine.curve=curve;self.engine.transition_start=now
        self.calls+=1
    def correct(self,now,deployed,raw,grasp=False):
        if self.method=='raw':return deployed,{}
        result,info=self.engine.step(now,deployed,raw,offsets=np.arange(6)*.02,grasp=grasp)
        result[:,:2]=deployed[:,:2] # exact head/root passthrough, including quaternion
        return result,info
