import argparse
from pathlib import Path
import time
import numpy as np
import torch
from exec_comp.common import ROOT,log,save_json,SOURCE_CHECKPOINT_HASH
from exec_comp.datasets import Rollouts
from exec_comp.models.inverse import InverseModel
from exec_comp.models.forward_kp import ForwardKeypointModel
from exec_comp.models.adapter import Adapter
from exec_comp.models.body_wm import BodyWorldModel
from exec_comp.models.base import pose_loss


def load_model(path,device):
    saved=torch.load(path,map_location=device,weights_only=False)
    kind=saved["kind"]
    model={"inverse":InverseModel,"forward":ForwardKeypointModel,"adapter":Adapter,
           "body":lambda:BodyWorldModel(saved["privileged"],saved["width"])}[kind]()
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def run(kind):
    p=argparse.ArgumentParser();p.add_argument("--updates",type=int,default=10000)
    p.add_argument("--batch",type=int,default=1024);p.add_argument("--device",default="cuda:0")
    p.add_argument("--privileged",action="store_true");p.add_argument("--width",type=int,default=1024)
    p.add_argument("--eval-interval",type=int,default=500)
    p.add_argument("--residual-smooth-weight",type=float,default=0.)
    p.add_argument("--extra-train-data",action='append',default=[])
    p.add_argument("--extra-validation-data",action='append',default=[])
    p.add_argument("--balance-data-roots",action='store_true')
    p.add_argument("--initialize",default=None)
    p.add_argument('--body-horizon',type=int,default=8)
    p.add_argument('--body-curriculum-updates',type=int,default=0)
    p.add_argument('--velocity-input-noise',type=float,default=0.)
    p.add_argument('--project-gravity',action='store_true')
    p.add_argument("--name",default=kind);args=p.parse_args()
    torch.manual_seed(42);torch.set_num_threads(4)
    folder=ROOT/"runs/models";folder.mkdir(parents=True,exist_ok=True)
    if (folder/(args.name+"_last.pt")).exists():raise FileExistsError(args.name)
    log("loading_dataset",kind=kind,split="train")
    train=Rollouts("train",body=kind=="body",device=args.device,folders=[ROOT/'runs/data',*map(Path,args.extra_train_data)],balance_roots=args.balance_data_roots)
    val=Rollouts("validation",body=kind=="body",device=args.device,folders=[ROOT/'runs/data',*map(Path,args.extra_validation_data)],balance_roots=args.balance_data_roots)
    log("dataset_ready",kind=kind,train_transitions=train.total,validation_transitions=val.total)
    model={"inverse":InverseModel,"forward":ForwardKeypointModel,"adapter":Adapter,
           "body":lambda:BodyWorldModel(args.privileged,args.width)}[kind]().to(args.device)
    if kind=="body":
        initial=train.body_batch(4096,8,args.privileged)
        model.initialize_normalization(initial["s"],initial["next_s"],initial["action"])
    else:
        initial=train.keypoint_batch(4096,"ref" if kind=="adapter" else "ach")
        model.initialize_normalization(initial["state"],initial["cmd"] if kind=="forward" else initial["goal"])
    if args.initialize:
        saved=torch.load(args.initialize,map_location=args.device,weights_only=False)
        if saved['kind']!=kind:raise ValueError('Initialization model kind mismatch')
        model.load_state_dict(saved['state_dict'])
    forward=None
    if kind=="adapter":
        forward=load_model(folder/"forward_best.pt",args.device).requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
    start=time.monotonic();best=float("inf")
    for update in range(1,args.updates+1):
        model.train()
        if kind=="body":
            horizon=8 if update<=args.body_curriculum_updates else args.body_horizon
            batch=train.body_batch(args.batch,horizon,args.privileged)
            initial_state=batch['s'][:,0].clone()
            if args.velocity_input_noise:
                if not args.privileged:raise ValueError('Velocity-noise requires68D state')
                initial_state[:,65:68]+=(torch.rand_like(initial_state[:,65:68])*2-1)*args.velocity_input_noise
            predicted=model.rollout(initial_state,batch["action"],project_gravity=args.project_gravity)
            loss=((predicted-batch["next_s"])/model.state_std).square().mean((0,2)).sum()
        else:
            batch=train.keypoint_batch(args.batch,"ref" if kind=="adapter" else "ach")
            if kind=="forward":
                predicted=model(batch["state"],batch["cmd"]);loss=pose_loss(predicted,batch["ach"])
            elif kind=="inverse":
                predicted=model(batch["state"],batch["goal"]);loss=pose_loss(predicted,batch["cmd"])
                delta=(predicted-batch["goal"])/model.residual_scale
                loss=loss+args.residual_smooth_weight*delta.diff(dim=1).square().mean()
            else:
                corrected=model(batch["state"],batch["goal"])
                predicted=forward(batch["state"],corrected)
                delta=(corrected[...,:3]-batch["goal"][...,:3])/.05
                loss=pose_loss(predicted,batch["goal"])+.01*delta.square().mean()+.1*delta.diff(dim=1).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError(f"{kind}: non-finite loss")
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update==1 or update%100==0:log("model_update",kind=kind,name=args.name,update=update,loss=float(loss),grad_norm=float(norm),elapsed_s=time.monotonic()-start)
        if update%args.eval_interval==0 or update==args.updates:
            model.eval();val.rng=np.random.default_rng(20261025)
            values=[];position=[]
            with torch.no_grad():
                for _ in range(16):
                    if kind=="body":
                        b=val.body_batch(256,args.body_horizon,args.privileged)
                        initial_state=b['s'][:,0].clone()
                        if args.velocity_input_noise:
                            # Fixed validation perturbations, independent of training RNG.
                            noise=torch.tensor(val.rng.uniform(-args.velocity_input_noise,args.velocity_input_noise,(256,3)),device=args.device,dtype=initial_state.dtype)
                            initial_state[:,65:68]+=noise
                        predicted=model.rollout(initial_state,b["action"],project_gravity=args.project_gravity)
                        values.append(float(((predicted-b["next_s"])/model.state_std).square().mean()))
                    else:
                        b=val.keypoint_batch(256,"ref" if kind=="adapter" else "ach")
                        if kind=="inverse":predicted=model(b["state"],b["goal"]);target=b["cmd"]
                        elif kind=="forward":predicted=model(b["state"],b["cmd"]);target=b["ach"]
                        else:predicted=forward(b["state"],model(b["state"],b["goal"]));target=b["goal"]
                        validation_loss=pose_loss(predicted,target)
                        if kind=="inverse":
                            delta=(predicted-b["goal"])/model.residual_scale
                            validation_loss=validation_loss+args.residual_smooth_weight*delta.diff(dim=1).square().mean()
                        values.append(float(validation_loss))
                        position.append(float((predicted[...,:3]-target[...,:3]).norm(dim=-1).mean()*100))
            score=float(np.mean(values));result={"update":update,"validation_loss":score,"position_cm":float(np.mean(position)) if position else None}
            log("model_validation",kind=kind,name=args.name,**result)
            saved={"state_dict":model.state_dict(),"kind":kind,"privileged":args.privileged,"width":args.width,
                   "update":update,"validation":result,"source_checkpoint_sha256":SOURCE_CHECKPOINT_HASH,
                   "optimizer":optimizer.state_dict(),"config":vars(args)}
            for name in (args.name+"_last.pt",)+( (args.name+"_best.pt",) if score<best else () ):
                destination=folder/name;temporary=destination.with_suffix(".pt.tmp");torch.save(saved,temporary);temporary.replace(destination)
            if score<best:best=score
    save_json(folder/(args.name+"_training.json"),{"kind":kind,"updates":args.updates,"best_validation_loss":best,
        "elapsed_s":time.monotonic()-start,"train_transitions":train.total,"validation_transitions":val.total,"privileged":args.privileged,
        'initialize':args.initialize,'extra_train_data':args.extra_train_data,'balance_data_roots':args.balance_data_roots})
    log("model_training_complete",name=args.name,best=best)
