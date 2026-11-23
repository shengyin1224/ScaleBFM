"""Frozen-weight counterfactual: does gravity normalization stop long-roll drift?"""
import torch
from exec_comp.common import ROOT,save_json,log
from exec_comp.datasets import Rollouts
from exec_comp.train_common import load_model
from exec_comp.diff.fk import FK
from exec_comp.geometry import integrate_root,from_anchor


def main():
    torch.set_num_threads(4)
    data=Rollouts("validation",body=True,device="cuda:0")
    batch=data.body_batch(1024,167,True)
    model=load_model(ROOT/"runs/models/body_privileged_best.pt","cuda:0").requires_grad_(False)
    fk=FK().cuda();results={}
    with torch.no_grad():
        for project in (False,True):
            state=batch["s"][:,0];p=batch["root_pos"];q=batch["root_quat"]
            for step in range(167):
                ns=model(state,batch["action"][:,step])
                if project:ns=torch.cat((ns[:,:61],torch.nn.functional.normalize(ns[:,61:64],dim=-1),ns[:,64:]),-1)
                p,q=integrate_root(p,q,state,ns);state=ns
            predicted=from_anchor(fk.keypoints(state[:,:29]),p,q)
            truth=batch["next_s"][:,-1]
            results["gravity_projection" if project else "original"]={
                "joint_mae_deg":float((state[:,:29]-truth[:,:29]).abs().mean()*180/torch.pi),
                "world_wrist_mean_cm":((predicted[:,2:,:3]-batch["ach"][:,-1,2:,:3]).norm(dim=-1).mean(0)*100).tolist(),
                "gravity_norm_mean":float(state[:,61:64].norm(dim=-1).mean()),
                "velocity_mae_m_s":float((state[:,65:68]-truth[:,65:68]).abs().mean())}
    save_json(ROOT/"configs/gravity_projection_diagnosis.json",results)
    log("gravity_projection_diagnosis",results=results,weights_unchanged=True,not_an_online_compensation_test=True)


if __name__=="__main__":main()
