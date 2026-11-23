import argparse
import numpy as np
import torch
from exec_comp.common import ROOT,log,save_json,KP_NAMES
from exec_comp.datasets import Rollouts
from exec_comp.train_common import load_model
from exec_comp.diff.fk import FK
from exec_comp.geometry import to_anchor,from_anchor,integrate_root


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--batches",type=int,default=16)
    p.add_argument("--optimizer-horizon",type=int,default=167);args=p.parse_args()
    torch.set_num_threads(4);model=load_model(args.checkpoint,args.device).requires_grad_(False)
    data=Rollouts("validation",body=True,device=args.device);data.rng=np.random.default_rng(77)
    fk=FK().to(args.device);scores={n:{"q":[],"pos":[]} for n in sorted(set((1,8,32,args.optimizer_horizon)))}
    with torch.no_grad():
        for _ in range(args.batches):
            batch=data.body_batch(256,args.optimizer_horizon,model.privileged)
            predicted=model.rollout(batch["s"][:,0],batch["action"])
            root_p=batch["root_pos"];root_q=batch["root_quat"];old=batch["s"][:,0]
            for step in range(args.optimizer_horizon):
                state=predicted[:,step]
                local=fk.keypoints(state[:,:29])
                if model.privileged:
                    root_p,root_q=integrate_root(root_p,root_q,old,state)
                    keypoints=from_anchor(local,root_p,root_q)
                    target=batch["ach"][:,step]
                else:
                    keypoints=local
                    target=to_anchor(batch["ach"][:,step],batch["next_root_pos"][:,step],batch["next_root_quat"][:,step])
                if step+1 in scores:
                    scores[step+1]["q"].append((state[:,:29]-batch["next_s"][:,step,:29]).abs().mean(-1).cpu().numpy()*180/np.pi)
                    scores[step+1]["pos"].append((keypoints[...,:3]-target[...,:3]).norm(dim=-1).cpu().numpy()*100)
                old=state
    summary={}
    for horizon,values in scores.items():
        pos=np.concatenate(values["pos"]);q=np.concatenate(values["q"])
        summary[horizon]={"joint_mae_deg":float(q.mean()),"joint_p95_deg":float(np.percentile(q,95)),
            "keypoint_mean_cm":dict(zip(KP_NAMES,pos.mean(0).tolist())),"keypoint_p95_cm":dict(zip(KP_NAMES,np.percentile(pos,95,axis=0).tolist()))}
    baseline=__import__("json").loads((ROOT/"runs/baseline/validated/metrics.json").read_text())
    frames=sum(c["frames"] for c in baseline["per_clip"])
    raw=[sum(c["frames"]*c["errors_cm"][name]["mean"] for c in baseline["per_clip"])/frames for name in ("left_wrist","right_wrist")]
    predicted=[summary[32]["keypoint_mean_cm"][name] for name in ("left_wrist","right_wrist")]
    long_predicted=[summary[args.optimizer_horizon]["keypoint_mean_cm"][name] for name in ("left_wrist","right_wrist")]
    gate=bool(model.privileged and all(a<.5*b for a,b in zip(predicted,raw)) and all(a<.5*b for a,b in zip(long_predicted,raw)))
    result={"checkpoint":args.checkpoint,"privileged":model.privileged,"coordinate_frame":"world" if model.privileged else "base_relative",
        "world_root_observable":model.privileged,"horizons":summary,"baseline_wrist_mean_cm":raw,
        "gate_passed":gate,"optimizer_control_steps":args.optimizer_horizon,
        "gate_definition":"world-frame 32-step AND actual optimizer-horizon mean wrist prediction error < 0.5 * raw tracking mean on both wrists",
        "default_state_limitation":None if model.privileged else "No world root translation/velocity in the default state; world-root compensation is unobservable"}
    name="body_privileged" if model.privileged else "body_default"
    save_json(ROOT/"runs"/(name+"_validation.json"),result);log("body_validation",**result)


if __name__=="__main__":main()
