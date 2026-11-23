import math
import torch
from exec_comp.common import ACTIVE,HEAD_OFFSET,CHUNK_FPS
from exec_comp.geometry import pose_normalize,six_to_q,qapply,qconj,qmul,from_anchor,integrate_root
from exec_comp.diff.obs import build_obs


def sample_pose(chunk,frames):
    frames=frames.clamp(0,chunk.shape[1]-1)
    lower=frames.floor().long();upper=(lower+1).clamp_max(chunk.shape[1]-1)
    weight=(frames-lower)[None,:,None,None]
    return pose_normalize(chunk[:,lower]+weight*(chunk[:,upper]-chunk[:,lower]))


class DifferentiableChain(torch.nn.Module):
    def __init__(self,policy,body_model):
        super().__init__();self.policy=policy;self.body=body_model
        if not body_model.privileged:
            raise ValueError("Full world/root optimization needs the explicitly privileged 68-D model; default64 is base-relative only")

    def forward(self,state_hist,command_chunk):
        device=command_chunk.device
        s=torch.as_tensor(state_hist["s"],device=device,dtype=torch.float32)[None,-3:]
        privileged=torch.as_tensor(state_hist["privileged"],device=device,dtype=torch.float32)
        if privileged.ndim==1:privileged=privileged[None].expand(3,-1)
        else:privileged=privileged[-3:]
        state=torch.cat((s[:,-1],privileged[-1:]),-1)
        action_hist=torch.as_tensor(state_hist["prev_action"],device=device,dtype=torch.float32)[None,-3:]
        root_p=torch.as_tensor(state_hist["root_pos"],device=device,dtype=torch.float32)
        root_q=torch.as_tensor(state_hist["root_quat"],device=device,dtype=torch.float32)
        root_p=root_p[-1:] if root_p.ndim==2 else root_p[None]
        root_q=root_q[-1:] if root_q.ndim==2 else root_q[None]
        quat_hist=root_q[:,None].expand(-1,3,-1)
        steps=math.ceil(command_chunk.shape[1]*50/CHUNK_FPS)
        predicted=[]
        for step in range(steps):
            frames=(torch.arange(6,device=device)+step)*(CHUNK_FPS/50)
            active=sample_pose(command_chunk,frames)
            active_q=six_to_q(active[...,3:]);active_p=active[...,:3].clone()
            offset=torch.tensor(HEAD_OFFSET,device=device).expand(1,6,3)
            torso=active_p[:,:,1]-qapply(active_q[:,:,1],offset)
            active_p=torch.stack((active_p[:,:,0],torso,active_p[:,:,2],active_p[:,:,3]),2)
            positions=torch.zeros(1,6,14,3,device=device);rotations=torch.zeros(1,6,14,4,device=device);rotations[...,0]=1
            positions[:,:,ACTIVE]=active_p;rotations[:,:,ACTIVE]=active_q
            inverse=qconj(root_q)[:,None,None].expand(1,6,14,4)
            local_p=qapply(inverse,positions-root_p[:,None,None]);local_q=qmul(inverse,rotations)
            _,task,_=build_obs(self.policy.fk,s[:,:,:29],s[:,:,29:58],quat_hist,s[:,:,58:61],action_hist,local_p,local_q)
            prop=torch.cat((s[:,:,61:64],s[:,:,58:61],s[:,:,:29]-self.policy.default,s[:,:,29:58]*.05),-1)
            action=self.policy.from_obs(prop,task,action_hist)
            next_state=self.body(state,action)
            root_p,root_q=integrate_root(root_p,root_q,state,next_state)
            predicted.append(from_anchor(self.policy.fk.keypoints(next_state[:,:29]),root_p,root_q))
            s=torch.cat((s[:,1:],next_state[:,None,:64]),1)
            action_hist=torch.cat((action_hist[:,1:],action[:,None]),1)
            quat_hist=torch.cat((quat_hist[:,1:],root_q[:,None]),1)
            state=next_state
        return torch.stack(predicted,1)
