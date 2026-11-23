"""Differentiable rollout that re-calls G the way deployment would.

``ExecutionChain`` from round 2 takes one fixed command chunk for the whole
horizon, which cannot express a correction that is recomputed per control step
or re-issued at a switch. This rollout keeps exactly the same body chain --
frozen cerebellum, frozen privileged world model, gravity projection,
COM-corrected root integration -- and only adds the callback that produces the
command at each step. Round 2's chain is left untouched so its results stay
reproducible.
"""
import torch
from exec_comp.common import ACTIVE,HEAD_OFFSET
from exec_comp.diff.graph_policy import prepare
from exec_comp.diff.chain import sample_pose
from exec_comp.diff.obs import build_obs
from exec_comp.geometry import from_anchor,integrate_root,qapply,qconj,qmul,six_to_q
from exec_comp.models.body_wm import project_gravity_state as project

CHUNK_RATE=.6  # chunk frames advanced per 50 Hz control step at 30 Hz spacing


class CorrectionRollout(torch.nn.Module):
    def __init__(self,policy,body,steps=17):
        super().__init__();self.policy=policy;self.body=body;self.steps=int(steps)
        prepare(self.policy)
        if not body.privileged:
            raise ValueError("The world rollout needs the privileged 68-D body model")
        device=policy.default.device
        self.register_buffer("active_index",torch.tensor(ACTIVE,device=device,dtype=torch.long))
        self.register_buffer("head_offset",torch.tensor(HEAD_OFFSET,device=device))
        self.register_buffer("com_offset",torch.tensor([0.,0.,-.07605],device=device))

    def forward(self,start,command_at):
        """``command_at(step, context) -> [B,chunk,4,9]`` world command window.

        ``context`` carries the measurable state the correction may read:
        predicted joint history, base angular velocity, gravity, and the
        keypoints forward kinematics gives from the predicted joints. The
        privileged root state is used to advance the world model, never handed
        to the correction.
        """
        history=start['s']
        if history.shape[1]!=10:raise ValueError('G must retain10 real state samples')
        quats=start['root_quat_history']
        actions=start["prev_action"][:,-3:]
        state=torch.cat((history[:,-1],start["privileged"]),-1)
        position=start["root_pos"];quat=start["root_quat"]
        count=len(position);device=position.device
        keypoint_history=start["kp_world"]
        predictions=[];commands=[]
        for step in range(self.steps):
            context={"s":history,"quats":quats,"actions":actions,"root_pos":position,
                     "root_quat":quat,"kp_world":keypoint_history}
            command=command_at(step,context)
            commands.append(command)
            # command_at already returns horizons aligned to NOW. Advancing
            # by step again skips time and trains against the wrong target.
            active=command if command.shape[1]==6 else sample_pose(command,torch.arange(6,device=device)*CHUNK_RATE)
            target_q=six_to_q(active[...,3:])
            torso=active[:,:,1,:3]-qapply(target_q[:,:,1],self.head_offset.expand(count,6,3))
            active_p=torch.stack((active[:,:,0,:3],torso,active[:,:,2,:3],active[:,:,3,:3]),2)
            targets=torch.zeros(count,6,14,3,device=device)
            rotations=torch.zeros(count,6,14,4,device=device);rotations[...,0]=1
            targets.index_copy_(2,self.active_index,active_p)
            rotations.index_copy_(2,self.active_index,target_q)
            inverse=qconj(quat)[:,None,None].expand(count,6,14,4)
            local_p=qapply(inverse,targets-position[:,None,None]);local_q=qmul(inverse,rotations)
            actor_history=history[:,-3:];actor_quats=quats[:,-3:]
            _,task,_=build_obs(self.policy.fk,actor_history[:,:,:29],actor_history[:,:,29:58],actor_quats,
                               actor_history[:,:,58:61],actions,local_p,local_q,
                               default_override=self.policy.default)
            proprio=torch.cat((actor_history[:,:,61:64],actor_history[:,:,58:61],
                               actor_history[:,:,:29]-self.policy.default,actor_history[:,:,29:58]*.05),-1)
            action=self.policy.from_obs(proprio,task,actions)
            # Each step projects gravity back onto the unit sphere: round 2
            # measured that leaving the physical manifold is what makes long
            # rollouts diverge.
            next_state=project(self.body(state,action))
            position,quat=integrate_root(position,quat,state,next_state,com_offset=self.com_offset)
            keypoints=from_anchor(self.policy.fk.keypoints(next_state[:,:29]),position,quat)
            predictions.append(keypoints)
            history=torch.cat((history[:,1:],next_state[:,None,:64]),1)
            quats=torch.cat((quats[:,1:],quat[:,None]),1)
            actions=torch.cat((actions[:,1:],action[:,None]),1)
            keypoint_history=torch.cat((keypoint_history[:,1:],keypoints[:,None]),1)
            state=next_state
        return torch.stack(predictions,1),torch.stack(commands,1)
