"""Batched frozen-policy body chain over the actual17-step execution horizon."""
import torch
from exec_comp.common import ACTIVE,HEAD_OFFSET
from exec_comp.geometry import six_to_q,qconj,qmul,qapply,from_anchor,integrate_root
from exec_comp.diff.obs import build_obs
from exec_comp.diff.chain import sample_pose
from exec_comp.models.body_wm import project_gravity_state as project


class ExecutionChain(torch.nn.Module):
    def __init__(self,policy,body,steps=17):
        super().__init__();self.policy=policy;self.body=body;self.steps=steps
        if not body.privileged:raise ValueError('World chain needs an explicitly supplied root-height/velocity estimate; default64 is not world-observable')
        self.register_buffer('active_index',torch.tensor(ACTIVE,device=policy.default.device,dtype=torch.long))
        self.register_buffer('head_offset',torch.tensor(HEAD_OFFSET,device=policy.default.device))
        self.register_buffer('com_offset',torch.tensor([0.,0.,-.07605],device=policy.default.device))

    def forward(self,state_hist,privileged,root_pos,root_quat_hist,action_hist,command):
        n=len(command);device=command.device
        history=state_hist[:,-3:];quats=root_quat_hist[:,-3:];actions=action_hist[:,-3:]
        state=torch.cat((history[:,-1],privileged),-1)
        position=root_pos;quat=quats[:,-1];predictions=[]
        for step in range(self.steps):
            active=sample_pose(command,(torch.arange(6,device=device)+step)*.6)
            target_q=six_to_q(active[...,3:])
            torso=active[:,:,1,:3]-qapply(target_q[:,:,1],self.head_offset.expand(n,6,3))
            active_p=torch.stack((active[:,:,0,:3],torso,active[:,:,2,:3],active[:,:,3,:3]),2)
            targets=torch.zeros(n,6,14,3,device=device)
            rotations=torch.zeros(n,6,14,4,device=device);rotations[...,0]=1
            targets.index_copy_(2,self.active_index,active_p);rotations.index_copy_(2,self.active_index,target_q)
            inverse=qconj(quat)[:,None,None].expand(n,6,14,4)
            local_p=qapply(inverse,targets-position[:,None,None]);local_q=qmul(inverse,rotations)
            _,task,_=build_obs(self.policy.fk,history[:,:,:29],history[:,:,29:58],quats,history[:,:,58:61],actions,local_p,local_q,default_override=self.policy.default)
            prop=torch.cat((history[:,:,61:64],history[:,:,58:61],history[:,:,:29]-self.policy.default,history[:,:,29:58]*.05),-1)
            action=self.policy.from_obs(prop,task,actions)
            ns=project(self.body(state,action))
            position,quat=integrate_root(position,quat,state,ns,com_offset=self.com_offset)
            predictions.append(from_anchor(self.policy.fk.keypoints(ns[:,:29]),position,quat))
            history=torch.cat((history[:,1:],ns[:,None,:64]),1)
            quats=torch.cat((quats[:,1:],quat[:,None]),1)
            actions=torch.cat((actions[:,1:],action[:,None]),1);state=ns
        return torch.stack(predictions,1)
