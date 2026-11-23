"""A1 supervised coefficient regression: same inputs/data/optimizer as R2 A1.

Positions keep24 coefficients for ramp (4nodes x3axes x2). The original A1
orientation correction also gets a low-dimensional6D-axis branch; it is not
silently removed to make a position-only model appear more stable.
"""
import torch
from exec_comp.models.base import mlp,STATE_HIST_DIM
from exec_comp.common import CHUNK
from exec_comp.geometry import pose_normalize
from exec_comp.round4.basis import TemporalBasis


class BasisInverse(torch.nn.Module):
    def __init__(self,basis):
        super().__init__();self.chunk=CHUNK;self.is_forward=False
        self.basis=TemporalBasis(basis);self.basis_name=basis
        rank=self.basis.matrix.shape[1]
        self.net=mlp([STATE_HIST_DIM+CHUNK*4*9,512,512,256,rank*4*9])
        self.register_buffer('input_mean',torch.zeros(STATE_HIST_DIM+CHUNK*4*9))
        self.register_buffer('input_std',torch.ones(STATE_HIST_DIM+CHUNK*4*9))
        self.register_buffer('residual_scale',torch.tensor([.05]*3+[.25]*6))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)

    def initialize_normalization(self,state,goal):
        values=torch.cat((state,goal.flatten(1)),1)
        self.input_mean.copy_(values.mean(0));self.input_std.copy_(values.std(0).clamp_min(.02))

    def coefficients(self,state,goal):
        values=torch.cat((state,goal.flatten(1)),1)
        return self.net((values-self.input_mean)/self.input_std).reshape(-1,self.basis.matrix.shape[1],4,9)*self.residual_scale

    def residual(self,state,goal):
        residual=self.basis.decode(self.coefficients(state,goal))
        position=residual[...,:3]
        position=position*torch.minimum(torch.ones_like(position[...,:1]),.05/position.norm(dim=-1,keepdim=True).clamp_min(1e-8))
        return torch.cat((position,residual[...,3:].clamp(-.25,.25)),-1)

    def forward(self,state,goal):return pose_normalize(goal+self.residual(state,goal))


class BasisPredictor:
    def __init__(self,path,device):
        from exec_comp.common import SOURCE_CHECKPOINT_HASH
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved['source_checkpoint_sha256']!=SOURCE_CHECKPOINT_HASH:raise ValueError('Frozen cerebellum mismatch')
        self.model=BasisInverse(saved['basis']).to(device)
        self.model.load_state_dict(saved['state_dict']);self.model.eval().requires_grad_(False)
        self.checkpoint=str(path)

    @torch.no_grad()
    def __call__(self,history,window):
        from exec_comp.geometry import heading_anchor,to_anchor,from_anchor,six_to_q,qmul,qconj,qnormalize
        from exec_comp.runtime import correct_encoded
        origin,quat=heading_anchor(history['root_pos'],history['root_quat'])
        kp=to_anchor(history['kp'],origin,quat)
        state=torch.cat((kp.flatten(2),history['q'],history['dq']),-1).flatten(1)
        corrected=from_anchor(correct_encoded(self.model,state,to_anchor(window,origin,quat)),origin,quat)
        return corrected[...,:3]-window[...,:3],qnormalize(qmul(six_to_q(corrected[...,3:]),qconj(six_to_q(window[...,3:]))))
