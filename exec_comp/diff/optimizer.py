import torch
from exec_comp.common import ROOT
from exec_comp.runtime import pack_chunk,unpack_chunk,encode_history
from exec_comp.train_common import load_model
from exec_comp.policy import FrozenPolicy
from exec_comp.geometry import to_anchor,from_anchor,pose_normalize,qmul,qconj,qapply
from exec_comp.diff.chain import DifferentiableChain,sample_pose


class CommandOptimizer:
    """Simulation privileged upper-bound optimizer, gated by B1 before use."""
    def __init__(self,checkpoint=None,iterations=5,device="cuda:0",check_gate=True):
        import json
        self.device=device;self.iterations=iterations
        if check_gate:
            result=json.loads((ROOT/"runs/body_privileged_validation.json").read_text())
            if not result["gate_passed"]:raise RuntimeError("B1 prediction gate failed; online optimization is disabled")
        self.policy=FrozenPolicy().to(device)
        self.body=load_model(checkpoint or ROOT/"runs/models/body_privileged_best.pt",device).requires_grad_(False)
        self.chain=DifferentiableChain(self.policy,self.body)
        self.last_delta=None;self.last_quat=None

    def optimize_cmd(self,state_hist,K_star_chunk,cmd_init=None):
        _,origin,quat=encode_history(state_hist,self.device)
        world_goal=pack_chunk(K_star_chunk,self.device)
        goal=to_anchor(world_goal,origin,quat)[None]
        scale=torch.tensor([.05,.05,.05,.25,.25,.25,.25,.25,.25],device=self.device)
        if cmd_init is not None:
            initial=(to_anchor(pack_chunk(cmd_init,self.device),origin,quat)[None]-goal)/scale
        elif self.last_delta is not None:
            shifted=torch.cat((self.last_delta[:,10:],self.last_delta[:,-1:].expand(-1,10,-1,-1)),1)
            rotation=qmul(qconj(quat),self.last_quat).view(1,1,1,4).expand(1,goal.shape[1],4,4)
            initial=torch.cat([qapply(rotation,shifted[...,i:i+3]) for i in (0,3,6)],-1)
        else:initial=torch.zeros_like(goal)
        delta=torch.nn.Parameter(initial.detach().clone())
        optimizer=torch.optim.Adam([delta],lr=.2)
        for _ in range(self.iterations):
            corrected=pose_normalize(goal+delta*scale)
            predicted=self.chain(state_hist,from_anchor(corrected,origin,quat))
            target=sample_pose(world_goal[None],(torch.arange(predicted.shape[1],device=self.device)+1)*.6)
            loss=((predicted[...,:3]-target[...,:3])/.05).square().mean()+.1*(predicted[...,3:]-target[...,3:]).square().mean()
            loss=loss+.01*delta.square().mean()+.05*delta.diff(dim=1).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("B optimization produced a non-finite loss")
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([delta],1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                delta.clamp_(-1,1)
                delta[...,:3]*=torch.minimum(torch.ones_like(delta[...,:1]),1/delta[...,:3].norm(dim=-1,keepdim=True).clamp_min(1e-8))
        self.last_delta=delta.detach();self.last_quat=quat.detach()
        return unpack_chunk(K_star_chunk,from_anchor(pose_normalize(goal+delta.detach()*scale)[0],origin,quat))


_default_optimizer=None
def optimize_cmd(state_hist,K_star_chunk,cmd_init=None):
    global _default_optimizer
    if _default_optimizer is None:_default_optimizer=CommandOptimizer()
    return _default_optimizer.optimize_cmd(state_hist,K_star_chunk,cmd_init)
