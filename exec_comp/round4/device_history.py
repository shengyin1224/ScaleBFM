"""GPU-resident measured history; no NumPy round trip or privileged state."""
import torch
from exec_comp.diff.fk import FK
from exec_comp.geometry import from_anchor


class DeviceHistory:
    def __init__(self,device):
        self.device=device;self.fk=FK().to(device);self.count=0
        self.buffers={key:torch.zeros(10,*shape,device=device) for key,shape in
            [('q',(29,)),('dq',(29,)),('kp',(4,9)),('root_pos',(3,)),('root_quat',(4,)),('s',(64,))]}

    @torch.no_grad()
    def observe(self,q,dq,p,r,w,g):
        q,dq,p,r,w,g=[torch.as_tensor(v,device=self.device,dtype=torch.float32).reshape(-1) for v in (q,dq,p,r,w,g)]
        kp=from_anchor(self.fk.keypoints(q[None])[0],p,r)
        self.append(q,dq,p,r,w,g,kp)

    @torch.no_grad()
    def append(self,q,dq,p,r,w,g,kp):
        row=dict(q=q,dq=dq,kp=kp,root_pos=p,root_quat=r,s=torch.cat((q,dq,w,g)))
        for key,value in row.items():
            buffer=self.buffers[key]
            if self.count==0:buffer.copy_(value[None].expand_as(buffer))
            else:buffer.copy_(torch.cat((buffer[1:],value[None]),0))
        self.count+=1

    def batched(self):
        result={k:self.buffers[k][None] for k in ('q','dq','kp','s')}
        result['root_pos']=self.buffers['root_pos'][-1:];result['root_quat']=self.buffers['root_quat'][-1:]
        return result

    def reset(self):self.count=0
