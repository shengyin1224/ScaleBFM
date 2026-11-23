"""Short optimizer checks on real, completed train/validation shards; discard weights."""
import torch
from exec_comp.datasets import Rollouts
from exec_comp.models.inverse import InverseModel
from exec_comp.models.forward_kp import ForwardKeypointModel
from exec_comp.models.adapter import Adapter
from exec_comp.models.body_wm import BodyWorldModel
from exec_comp.models.base import pose_loss
from exec_comp.common import log


def check(name,model,loss_fn):
    opt=torch.optim.Adam(model.parameters(),lr=2e-4)
    for step in range(3):
        loss=loss_fn(model)
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();log("model_smoke",name=name,step=step+1,loss=float(loss.detach()),gradient_norm=float(norm))
    return model


def main():
    torch.set_num_threads(4);torch.manual_seed(42);device="cuda:0"
    train=Rollouts("train",device=device)
    batch=train.keypoint_batch(128)
    model=InverseModel().to(device);model.initialize_normalization(batch["state"],batch["goal"])
    check("inverse",model,lambda m:pose_loss(m(batch["state"],batch["goal"]),batch["cmd"]))
    model=ForwardKeypointModel().to(device);model.initialize_normalization(batch["state"],batch["cmd"])
    forward=check("forward",model,lambda m:pose_loss(m(batch["state"],batch["cmd"]),batch["ach"]))
    forward.eval().requires_grad_(False)
    adapter=Adapter().to(device);ref=train.keypoint_batch(128,"ref");adapter.initialize_normalization(ref["state"],ref["goal"])
    check("adapter",adapter,lambda m:pose_loss(forward(ref["state"],m(ref["state"],ref["goal"])),ref["goal"]))
    del train,batch,forward,adapter,model
    train=Rollouts("train",body=True,device=device)
    for privileged in (False,True):
        batch=train.body_batch(256,8,privileged)
        model=BodyWorldModel(privileged).to(device)
        model.initialize_normalization(batch["s"],batch["next_s"],batch["action"])
        check("body_privileged" if privileged else "body_default",model,
              lambda m:((m.rollout(batch["s"][:,0],batch["action"])-batch["next_s"])/m.state_std).square().mean((0,2)).sum())
    log("all_model_smokes_passed")


if __name__=="__main__":main()
