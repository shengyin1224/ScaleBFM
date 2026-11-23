import torch
from exec_comp.common import metadata
from exec_comp.geometry import qconj,qapply,qmul,q_to_6d


def build_obs(fk,q_hist,dq_hist,root_quat_hist,ang_hist,action_hist,target_pos,target_quat,phase=None,default_override=None):
    """Deployment contract: caller supplies targets in the current base frame."""
    n=q_hist.shape[0]
    default=default_override if default_override is not None else torch.as_tensor(metadata()['default_dof_pos'],device=q_hist.device,dtype=q_hist.dtype)
    gravity=torch.zeros_like(ang_hist);gravity[...,2]=-1
    prop=torch.cat((qapply(qconj(root_quat_hist),gravity),ang_hist,q_hist-default,dq_hist*.05),-1)
    current_pos,current_quat=fk(q_hist[:,-1])
    current_pos=current_pos.index_select(1,fk.selected_index);current_quat=current_quat.index_select(1,fk.selected_index)
    error=target_pos-current_pos[:,None]
    relative=qmul(target_quat,qconj(current_quat[:,None]).expand_as(target_quat))
    offsets=torch.arange(6,device=q_hist.device,dtype=q_hist.dtype).view(1,6,1).expand(n,-1,-1)
    if phase is None:
        phase=torch.zeros(n,6,2,device=q_hist.device,dtype=q_hist.dtype);phase[...,1]=2
    task=torch.cat((target_pos.flatten(2),error.flatten(2),q_to_6d(target_quat).flatten(2),
                    q_to_6d(relative).flatten(2),offsets,phase),-1)
    mode=torch.zeros(n,14,device=q_hist.device,dtype=q_hist.dtype)
    for index in (0,7,10,13):mode[:,index]=1
    mapping=torch.cat([mode.repeat_interleave(d,dim=-1) for d in (3,3,6,6)]+[torch.ones(n,3,device=q_hist.device,dtype=q_hist.dtype)],-1)
    return prop,torch.cat((task*mapping[:,None],mode[:,None].expand(-1,6,-1)),-1),action_hist
