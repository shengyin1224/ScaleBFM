"""Check the complete input-gradient path with the trained body model; no optimization."""
import torch
from exec_comp.common import ROOT,log,save_json
from exec_comp.datasets import Rollouts
from exec_comp.train_common import load_model
from exec_comp.policy import FrozenPolicy
from exec_comp.diff.chain import DifferentiableChain,sample_pose


def main():
    torch.set_num_threads(4);torch.manual_seed(42);device="cuda:0"
    data=Rollouts("validation",body=True,device=device)
    batch=data.body_batch(1,167,privileged=True)
    s=batch["s"][:,0]
    history={"s":s[0,:64][None].repeat(10,1),"privileged":s[0,64:][None].repeat(10,1),
             "prev_action":torch.zeros(10,29,device=device),"root_pos":batch["root_pos"][0],"root_quat":batch["root_quat"][0]}
    goal=sample_pose(batch["ach"],torch.arange(100,device=device)*50/30).detach().requires_grad_()
    body=load_model(ROOT/"runs/models/body_privileged_best.pt",device).requires_grad_(False)
    policy=FrozenPolicy().to(device)
    predicted=DifferentiableChain(policy,body)(history,goal)
    gradient=torch.autograd.grad(predicted[...,:3].square().mean(),goal)[0]
    assert predicted.shape==(1,167,4,9)
    assert torch.isfinite(predicted).all() and torch.isfinite(gradient).all()
    assert gradient.abs().sum()>0
    result={"steps":167,"goal_frames":100,"gradient_max":float(gradient.abs().max()),"gradient_l1":float(gradient.abs().sum()),
            "finite":True,"source_policy_weights_frozen":all(not p.requires_grad for p in policy.parameters()),
            "body_weights_frozen":all(not p.requires_grad for p in body.parameters()),"does_not_establish_prediction_accuracy":True}
    save_json(ROOT/"configs/trained_chain_gradient.json",result);log("trained_chain_gradient_passed",**result)


if __name__=="__main__":main()
