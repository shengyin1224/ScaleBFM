"""Same frozen actor computation with a preallocated three-state attention mask."""
import torch


class GraphActor(torch.nn.Module):
    def __init__(self,actor):
        super().__init__();self.actor=actor
        mask=torch.zeros(6,6,dtype=torch.bool,device=next(actor.parameters()).device)
        mask[:-1,-1]=True
        self.register_buffer('mask',mask);self.eval()

    def forward(self,prop_obs,action_obs,task_tokens):
        a=self.actor
        prop=a.prop_projection(prop_obs);actions=a.action_projection(action_obs)
        context=prop.new_empty(prop.shape[0],5,a.embed_dim)
        context[:,0::2]=prop;context[:,1::2]=actions[:,1:]
        x=torch.cat((context,a.empty_embedding.expand(prop.shape[0],-1,-1)),1)
        for block in a.transformer_blocks:x=block(x,task_tokens,self_attn_mask=self.mask)
        return a.projection_head(a.final_norm(x)[:,-1])


def prepare(policy):
    if not isinstance(policy.actor,GraphActor):policy.actor=GraphActor(policy.actor)
