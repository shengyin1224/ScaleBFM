import torch
from exec_comp.models.base import mlp
from exec_comp.geometry import pose_normalize


class WristA1(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=mlp([940+3600,512,512,256,1800]);self.register_buffer('mean',torch.zeros(4540));self.register_buffer('std',torch.ones(4540))
        self.register_buffer('scale',torch.tensor([.05,.05,.05,.25,.25,.25,.25,.25,.25]));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def features(self,s,kp):return torch.cat((kp.flatten(2),s[...,:58]),-1).flatten(1)
    def initialize(self,state,goal):
        x=torch.cat((state,goal.flatten(1)),1)
        with torch.no_grad():self.mean.copy_(x.mean(0));self.std.copy_(x.std(0).clamp_min(.02))
    def forward(self,state,goal):
        x=torch.cat((state,goal.flatten(1)),1);d=torch.tanh(self.net((x-self.mean)/self.std)).view(-1,100,2,9)*self.scale
        pos=d[...,:3];pos=pos*torch.clamp(.05/pos.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1)
        corrected=pose_normalize(goal[:,:,2:]+torch.cat((pos,d[...,3:]),-1))
        return torch.cat((goal[:,:,:2],corrected),2) # root and head EXACT passthrough


class WristB1(torch.nn.Module):
    def __init__(self,scale=.05):
        super().__init__();self.scale=scale;self.net=mlp([4600,512,512,256,12]);self.register_buffer('mean',torch.zeros(4600));self.register_buffer('std',torch.ones(4600));self.register_buffer('phase',torch.linspace(0,1,100)[None,:,None,None]);torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def features(self,s,kp):return torch.cat((kp.flatten(2),s),-1).flatten(1)
    def initialize(self,state,goal):
        x=torch.cat((state,goal.flatten(1)),1)
        with torch.no_grad():self.mean.copy_(x.mean(0));self.std.copy_(x.std(0).clamp_min(.02))
    def coefficients(self,state,goal):return torch.tanh(self.net((torch.cat((state,goal.flatten(1)),1)-self.mean)/self.std)).view(-1,2,2,3)*self.scale
    def forward(self,state,goal):
        c=self.coefficients(state,goal);d=c[:,0,None]+self.phase*c[:,1,None];d=d*torch.clamp(.05/d.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1)
        wrists=torch.cat((goal[:,:,2:,:3]+d,goal[:,:,2:,3:]),-1)
        return torch.cat((goal[:,:,:2],wrists),2)


class Body(torch.nn.Module):
    # Context:3 past requests(87), physical hand q/dq(48), domain(6).
    def __init__(self,width=512):
        super().__init__();self.width=width;self.net=mlp([238,width,width,width,68]);self.register_buffer('mean',torch.zeros(238));self.register_buffer('std',torch.ones(238));self.register_buffer('delta_std',torch.ones(68)*.1);self.register_buffer('state_std',torch.ones(68));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def initialize(self,s,action,context,ns):
        x=torch.cat((s,action,context),-1).flatten(0,-2);s=s.flatten(0,-2);ns=ns.flatten(0,-2)
        with torch.no_grad():self.mean.copy_(x.mean(0));self.std.copy_(x.std(0).clamp_min(.01));self.delta_std.copy_((ns-s).std(0).clamp_min(.0001));self.state_std.copy_(s.std(0).clamp_min(.01))
    def forward(self,s,action,context):
        out=s+self.net((torch.cat((s,action,context),-1)-self.mean)/self.std)*self.delta_std
        gravity=torch.nn.functional.normalize(out[...,61:64],dim=-1)
        return torch.cat((out[...,:61],gravity,out[...,64:]),-1)
