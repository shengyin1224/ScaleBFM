"""Does repeated brain sampling carry usable uncertainty? (AAC precondition)

AAC decides an execution length from the spread of several samples drawn for
the SAME observation. That is only meaningful if the spread is non-zero. This
runs the released HAT checkpoint repeatedly on identical inputs and reports the
measured deviation, instead of asserting collapse from the architecture alone.
No sockets, no camera, no robot: the same offline loader A1-3 already used.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import h5py
import cv2
from exec_comp.common import ROOT,save_json,digest,log
from exec_comp.export_hat import inference_policy_class


def latent_sources(repo):
    """Locate the inference-time CVAE latent statements, quoted verbatim."""
    path=repo/"hdt/detr/models/detr_vae.py"
    lines=path.read_text().splitlines()
    hits=[]
    for index,line in enumerate(lines):
        if "latent_sample" in line and "zeros" in line:
            hits.append({"file":str(path),"line":index+1,"text":line.strip(),
                         "context":[l.strip() for l in lines[max(0,index-3):index+2]]})
    return hits


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",required=True);parser.add_argument("--bundle",required=True)
    parser.add_argument("--split",default=str(ROOT/"configs/hat_heldout_split.json"))
    parser.add_argument("--samples",type=int,default=8)
    parser.add_argument("--frames",type=int,default=4)
    parser.add_argument("--output",default=str(ROOT/"runs/round3/brain_sampling.json"))
    args=parser.parse_args();torch.manual_seed(42);torch.set_num_threads(4)
    repo=Path(args.repo);bundle=Path(args.bundle)
    sys.path[:0]=[str(repo),str(repo/"hdt")]
    import hdt.online_hat_service as service
    from hdt.data_utils_hdt import _detect_valid_start_from_actions
    config=next(bundle.glob("act_input*.yaml"))
    Policy=inference_policy_class(service,repo)
    engine=Policy(config_path=config,checkpoint=bundle/"policy_last.ckpt",stats_path=bundle/"dataset_stats.pkl",
                  embodiment="dex5",device="cuda:0",wrist_conversion="inspire_hand_base")
    entries=json.loads(Path(args.split).read_text())["val"]
    kind,path=next((e["type"],Path(e["dataset_path"])/name) for e in entries for name in e["file_list"] if e["type"]=="robot")
    engine.stats=service._load_stats(bundle/"dataset_stats.pkl","dex5")
    training_modules=[]
    for name,module in engine.policy.named_modules():
        if module.training:training_modules.append(name)
    stochastic=[name for name,module in engine.policy.named_modules()
                if isinstance(module,(torch.nn.Dropout,torch.nn.Dropout1d,torch.nn.Dropout2d,torch.nn.Dropout3d))]
    rows=[]
    with h5py.File(path) as f:
        start=int(_detect_valid_start_from_actions(np.asarray(f["action"][:20])))
        total=len(f["observation.state"])
        key=next(k for k in ("observation.image.top","observation.image.left") if k in f)
        for index in np.linspace(start,total-1,args.frames).astype(int):
            image=np.asarray(f[key][int(index)])
            if image.ndim==1:image=cv2.imdecode(image,cv2.IMREAD_COLOR)
            image=cv2.resize(image,(320,240)).transpose(2,0,1)[None]
            raw=np.asarray(f["observation.state"][int(index)],dtype=np.float32)
            normalized=((raw-engine.stats["qpos_mean"])/(engine.stats["qpos_std"]+1e-6)).astype(np.float32)
            draws=np.stack([engine.infer(image.copy(),None,normalized_state=normalized.copy())
                            for _ in range(args.samples)])
            spread=draws.std(0)
            rows.append({"source_frame":int(index),
                         "samples":int(args.samples),
                         "max_abs_deviation_from_first":float(np.abs(draws-draws[0]).max()),
                         "bitwise_identical":bool(np.array_equal(draws,np.broadcast_to(draws[0],draws.shape))),
                         "action_std_max":float(spread.max()),"action_std_mean":float(spread.mean())})
            log("brain_sampling_frame",**rows[-1])
    result={"checkpoint":str(bundle/"policy_last.ckpt"),"checkpoint_sha256":digest(bundle/"policy_last.ckpt"),
            "source_episode":str(path),"episode_type":kind,"frames":rows,
            "modules_left_in_training_mode":training_modules,"dropout_modules":stochastic,
            "inference_latent_code":latent_sources(repo),
            # float32 std() of identical values still returns ~1e-7, so identity
            # is decided on the deviation from the first draw, not on std().
            "aac_usable":bool(any(not r["bitwise_identical"] for r in rows)),
            "interpretation":("AAC estimates execution length from same-observation sample spread. "
                "A zero spread means the estimator is constant and cannot gate anything. "
                "action_std_* is reported only to show the float32 noise floor."),
            "robot_run":False}
    save_json(args.output,result)
    log("brain_sampling_complete",aac_usable=result["aac_usable"],
        max_std=max(r["action_std_max"] for r in rows))


if __name__=="__main__":main()
