"""World-pose sampling with quaternion SLERP, matching the online HAT interpolator."""
import torch
from exec_comp.geometry import six_to_q,qnormalize,pose9


def quaternion_slerp(a,b,t):
    dot=(a*b).sum(-1,keepdim=True)
    b=torch.where(dot<0,-b,b);dot=dot.abs().clamp(0,1)
    theta=torch.acos(dot.clamp_max(1-1e-7))
    sine=torch.sin(theta)
    curved=torch.sin((1-t)*theta)/sine*a+torch.sin(t*theta)/sine*b
    return qnormalize(torch.where(dot>.9995,a+(b-a)*t,curved))


def sample(chunk,frames):
    """chunk[B,T,4,9], frames[B,F] -> pose[B,F,4,9]."""
    frames=frames.clamp(0,chunk.shape[1]-1);lower=frames.floor().long();upper=(lower+1).clamp_max(chunk.shape[1]-1)
    batch=torch.arange(len(chunk),device=chunk.device)[:,None]
    lo=chunk[batch,lower];hi=chunk[batch,upper];weight=(frames-lower)[...,None,None]
    position=lo[...,:3]+(hi[...,:3]-lo[...,:3])*weight
    quat=quaternion_slerp(six_to_q(lo[...,3:]),six_to_q(hi[...,3:]),weight)
    return pose9(position,quat)
