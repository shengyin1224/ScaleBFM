import torch
from exec_comp.common import HISTORY,CHUNK
from exec_comp.geometry import pose_normalize

STATE_HIST_DIM=HISTORY*(4*9+29+29)


def mlp(dims):
    layers=[]
    for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
        layers.append(torch.nn.Linear(a,b))
        if i<len(dims)-2:layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers)


class KeypointNet(torch.nn.Module):
    def __init__(self,chunk=CHUNK,forward=False):
        super().__init__();self.chunk=chunk;self.is_forward=forward
        self.net=mlp([STATE_HIST_DIM+chunk*4*9,512,512,256,chunk*4*9])
        self.register_buffer("input_mean",torch.zeros(STATE_HIST_DIM+chunk*4*9))
        self.register_buffer("input_std",torch.ones(STATE_HIST_DIM+chunk*4*9))
        self.register_buffer("residual_scale",torch.tensor([.05,.05,.05,.25,.25,.25,.25,.25,.25]))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)

    def initialize_normalization(self,state,goal):
        values=torch.cat((state,goal.flatten(1)),1)
        with torch.no_grad():
            self.input_mean.copy_(values.mean(0));self.input_std.copy_(values.std(0).clamp_min(.02))

    def residual(self,state,goal):
        value=torch.cat((state,goal.flatten(1)),1)
        raw=self.net((value-self.input_mean)/self.input_std).view(-1,self.chunk,4,9)
        if self.is_forward:
            return raw*self.residual_scale
        residual=torch.tanh(raw)*self.residual_scale
        position=residual[...,:3]
        position=position*torch.minimum(torch.ones_like(position[...,:1]),.05/position.norm(dim=-1,keepdim=True).clamp_min(1e-8))
        return torch.cat((position,residual[...,3:]),-1)

    def forward(self,state,goal):
        return pose_normalize(goal+self.residual(state,goal))


def pose_loss(predicted,target,rotation_weight=.1):
    position=((predicted[...,:3]-target[...,:3])/.05).square().mean()
    orientation=((predicted[...,3:]-target[...,3:])/.25).square().mean()
    return position+rotation_weight*orientation
