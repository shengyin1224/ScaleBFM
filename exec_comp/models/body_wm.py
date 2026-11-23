import torch
from exec_comp.models.base import mlp


def project_gravity_state(s):
    return torch.cat((s[...,:61],torch.nn.functional.normalize(s[...,61:64],dim=-1),s[...,64:]),-1)


class BodyWorldModel(torch.nn.Module):
    def __init__(self,privileged=False,width=1024):
        super().__init__();self.privileged=privileged;self.dim=68 if privileged else 64
        self.net=mlp([self.dim+29,width,width,width,self.dim])
        self.register_buffer("state_mean",torch.zeros(self.dim));self.register_buffer("state_std",torch.ones(self.dim))
        self.register_buffer("action_mean",torch.zeros(29));self.register_buffer("action_std",torch.ones(29))
        self.register_buffer("delta_std",torch.ones(self.dim)*.1)
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)

    def initialize_normalization(self,s,ns,action):
        with torch.no_grad():
            s=s.flatten(0,1);ns=ns.flatten(0,1);action=action.flatten(0,1)
            self.state_mean.copy_(s.mean(0));self.state_std.copy_(s.std(0).clamp_min(.01))
            self.action_mean.copy_(action.mean(0));self.action_std.copy_(action.std(0).clamp_min(.05))
            self.delta_std.copy_((ns-s).std(0).clamp_min(.0001))

    def forward(self,s,action):
        inputs=torch.cat(((s-self.state_mean)/self.state_std,(action-self.action_mean)/self.action_std),-1)
        return s+self.net(inputs)*self.delta_std

    def rollout(self,s,actions,project_gravity=False):
        result=[]
        for action in actions.unbind(1):
            s=self(s,action)
            if project_gravity:s=project_gravity_state(s)
            result.append(s)
        return torch.stack(result,1)
