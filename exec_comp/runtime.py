"""Opt-in chunk interfaces. Fingertip and unrelated chunk fields pass through."""
import numpy as np
import torch
from exec_comp.common import ROOT,KP_NAMES,HISTORY
from exec_comp.geometry import pose9,six_to_q,to_anchor,from_anchor,heading_anchor
from exec_comp.train_common import load_model
from exec_comp.diff.chain import sample_pose


def correct_encoded(model,state,goal):
    # Training labels ach[t] are measured AFTER action[t], i.e. at t+20ms.
    # Request K*(t+20ms), while the returned command is still timestamped t.
    frames=torch.arange(goal.shape[1],device=goal.device,dtype=goal.dtype)+.6
    desired_response=sample_pose(goal,frames)
    corrected=model(state,desired_response)
    delta=corrected[...,:3]-goal[...,:3]
    delta=delta*torch.minimum(torch.ones_like(delta[...,:1]),.05/delta.norm(dim=-1,keepdim=True).clamp_min(1e-8))
    return torch.cat((goal[...,:3]+delta,corrected[...,3:]),-1)


def pack_chunk(chunk,device):
    if isinstance(chunk,dict):
        if "pose" in chunk:return torch.as_tensor(chunk["pose"],device=device,dtype=torch.float32)
        if float(chunk.get("fps",30))!=30:raise ValueError("Expected the audited 30-Hz HAT chunk")
        return torch.stack([pose9(torch.as_tensor(chunk[n+"_pos"],device=device,dtype=torch.float32),
                                  torch.as_tensor(chunk[n+"_quat_wxyz"],device=device,dtype=torch.float32)) for n in KP_NAMES],-2)
    return torch.as_tensor(chunk,device=device,dtype=torch.float32)


def unpack_chunk(original,poses):
    if not isinstance(original,dict):return poses if torch.is_tensor(original) else poses.detach().cpu().numpy()
    result=dict(original)
    if "pose" in original:
        result["pose"]=poses if torch.is_tensor(original["pose"]) else poses.detach().cpu().numpy()
    else:
        rotations=six_to_q(poses[...,3:])
        for index,name in enumerate(KP_NAMES):
            for suffix,value in (("_pos",poses[:,index,:3]),("_quat_wxyz",rotations[:,index])):
                result[name+suffix]=value if torch.is_tensor(original[name+suffix]) else value.detach().cpu().numpy()
    return result


def encode_history(history,device):
    values={k:torch.as_tensor(history[k],device=device,dtype=torch.float32) for k in ("q","dq","kp","root_pos","root_quat")}
    origin=values["root_pos"][-1] if values["root_pos"].ndim==2 else values["root_pos"]
    quat=values["root_quat"][-1] if values["root_quat"].ndim==2 else values["root_quat"]
    origin,quat=heading_anchor(origin,quat)
    kp=to_anchor(values["kp"][-HISTORY:],origin,quat)
    state=torch.cat((kp.flatten(1),values["q"][-HISTORY:],values["dq"][-HISTORY:]),-1).flatten()[None]
    return state,origin,quat


class KeypointCompensator:
    def __init__(self,method="A1",checkpoint=None,device="cuda:0"):
        self.device=device;name="inverse" if method=="A1" else "adapter"
        self.model=load_model(checkpoint or ROOT/"runs/models"/(name+"_best.pt"),device).requires_grad_(False)

    @torch.no_grad()
    def correct_kp(self,state_hist,K_star_chunk):
        state,origin,quat=encode_history(state_hist,self.device)
        goal=pack_chunk(K_star_chunk,self.device)
        if goal.shape!=(self.model.chunk,4,9):raise ValueError(f"Expected {self.model.chunk}x4x9 chunk, got {goal.shape}")
        corrected=correct_encoded(self.model,state,to_anchor(goal,origin,quat)[None])[0]
        return unpack_chunk(K_star_chunk,from_anchor(corrected,origin,quat))


_default_corrector=None
def correct_kp(state_hist,K_star_chunk):
    global _default_corrector
    if _default_corrector is None:_default_corrector=KeypointCompensator("A1")
    return _default_corrector.correct_kp(state_hist,K_star_chunk)
