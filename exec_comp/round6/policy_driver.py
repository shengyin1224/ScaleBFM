from collections import deque
import numpy as np
import torch
from exec_comp.policy import FrozenPolicy
from exec_comp.geometry import qapply,qconj,qmul
from exec_comp.common import ACTIVE


class Driver:
    def __init__(self,device='cpu'):
        if str(device).startswith('cuda'):torch.cuda.set_device(device)
        self.device=device;self.policy=FrozenPolicy().to(device);self.graph=None
    def reset(self,states):
        self.hist={}
        for k in ('q','dq','quat','w'):
            a=torch.as_tensor(np.stack([s[k] for s in states]),device=self.device,dtype=torch.float32);self.hist[k]=a[:,None].repeat(1,3,1)
        self.hist['action']=torch.zeros(len(states),3,29,device=self.device)
    def action(self,states,positions,quaternions,phase=None):
        p=torch.as_tensor(positions,device=self.device,dtype=torch.float32).clone();q=torch.as_tensor(quaternions,device=self.device,dtype=torch.float32)
        n=len(states);p[:,:,1]-=qapply(q[:,:,1],p.new_tensor([0.,0.,.4]).expand(n,6,3))
        fullp=torch.zeros(n,6,14,3,device=self.device);fullq=torch.zeros(n,6,14,4,device=self.device);fullq[...,0]=1
        fullp[:,:,ACTIVE]=p;fullq[:,:,ACTIVE]=q
        root=torch.as_tensor(np.stack([s['p'] for s in states]),device=self.device,dtype=torch.float32)
        quat=torch.as_tensor(np.stack([s['quat'] for s in states]),device=self.device,dtype=torch.float32);inverse=qconj(quat)[:,None,None].expand(n,6,14,4)
        p=qapply(inverse,fullp-root[:,None,None]);q=qmul(inverse,fullq)
        h=self.hist
        inputs=(h['q'],h['dq'],h['quat'],h['w'],h['action'],p,q)
        def forward(values):
            from exec_comp.diff.obs import build_obs
            return self.policy.from_obs(*build_obs(self.policy.fk,*values,phase,default_override=self.policy.default))
        with torch.no_grad():
            if str(self.device).startswith('cuda') and phase is None:
                if self.graph is None:
                    expected=forward(inputs)
                    from exec_comp.diff.graph_policy import prepare
                    prepare(self.policy)
                    self.static=[v.clone() for v in inputs]
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):forward(self.static)
                    torch.cuda.current_stream().wait_stream(stream)
                    self.graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph):self.graph_output=forward(self.static)
                    self.graph.replay();torch.testing.assert_close(self.graph_output,expected,atol=1e-5,rtol=1e-5)
                else:
                    for target,value in zip(self.static,inputs):target.copy_(value)
                    self.graph.replay()
                a=self.graph_output
            else:a=forward(inputs)
        return a.cpu().numpy()
    def targets(self,action):return np.asarray(self.policy.default.cpu())+np.asarray(self.policy.action_scale.cpu())*action
    def append(self,states,action):
        for k in ('q','dq','quat','w','action'):
            a=action if k=='action' else np.stack([s[k] for s in states]);x=torch.as_tensor(a,device=self.device,dtype=torch.float32)
            self.hist[k]=torch.cat((self.hist[k][:,1:],x[:,None]),1)
