import torch
from exec_comp.common import TRACK,checkpoint,import_path,metadata,digest,SOURCE_CHECKPOINT_HASH
from exec_comp.diff.fk import FK
from exec_comp.diff.obs import build_obs


class FrozenPolicy(torch.nn.Module):
    def __init__(self,source=None):
        super().__init__()
        source=source or checkpoint()
        if digest(source)!=SOURCE_CHECKPOINT_HASH:raise ValueError("Unexpected frozen checkpoint")
        network=import_path("exec_comp_frozen_network",TRACK/"source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py")
        self.actor=network.HumanoidTransformer(64,29,29,256,4,256,4)
        self.task_embedder=network.TaskEmbedder(269,256)
        state=torch.load(source,map_location="cpu",weights_only=False)["model_state_dict"]
        for name,module in (("actor",self.actor),("actor_task_embedder",self.task_embedder)):
            module.load_state_dict({k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+".")})
        self.fk=FK()
        meta=metadata()
        self.register_buffer("action_scale",torch.tensor(meta["action_scale"]))
        self.register_buffer("default",torch.tensor(meta["default_dof_pos"]))
        self.eval().requires_grad_(False)

    def from_obs(self,prop,task,action_hist):
        return self.actor(prop,action_hist,self.task_embedder(task))

    def forward(self,q_hist,dq_hist,root_quat_hist,ang_hist,action_hist,target_pos,target_quat,phase=None):
        return self.from_obs(*build_obs(self.fk,q_hist,dq_hist,root_quat_hist,ang_hist,action_hist,
                                       target_pos,target_quat,phase))

    def q_target(self,action):
        return self.default+self.action_scale*action
