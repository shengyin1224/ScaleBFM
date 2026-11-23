"""Paired17-step prediction/actual diagnostics on unseen G TRAIN-motion rollouts.

The imagined trajectory uses the actions G actually sent (including collection
noise), not a different starting state or target. This isolates model optimism;
it is distinct from the differentiable, re-calling-G validation loss.
"""
import argparse
import numpy as np
import torch
from exec_comp.common import ROOT,log,save_json
from exec_comp.datasets import Rollouts
from exec_comp.train_common import load_model
from exec_comp.diff.fk import FK
from exec_comp.geometry import from_anchor,integrate_root

KEYS=('s','next_s','action','episode','done','root_pos','root_quat','privileged_root',
      'next_privileged_root','ach_kp','next_raw_kp')


@torch.no_grad()
def score(model,folder,device,batches):
    is_ordinary=folder==ROOT/'runs/data'
    keys=tuple('ref_kp' if k=='next_raw_kp' else k for k in KEYS) if is_ordinary else KEYS
    data=Rollouts('train' if is_ordinary else 'validation',device=device,folders=[folder],keys=keys)
    data.rng=np.random.default_rng(20261026);fk=FK().to(device)
    errors=[];imagined=[];actual=[];state_error=[];endpoint=[]
    for _ in range(batches):
        stream,e,t=data.indices(128,17,history=1)
        def get(key,offset=None):
            a=stream[key][e,t] if offset is None else stream[key][e[:,None],t[:,None]+offset]
            return torch.as_tensor(a,device=device,dtype=torch.float32)
        offsets=np.arange(17);s=torch.cat((get('s'),get('privileged_root')),-1)
        action=get('action',offsets);target=get('ref_kp',offsets+1) if is_ordinary else get('next_raw_kp',offsets);achieved=get('ach_kp',offsets)
        truth=torch.cat((get('next_s',offsets),get('next_privileged_root',offsets)),-1)
        predicted=model.rollout(s,action,project_gravity=True)
        position=get('root_pos');quat=get('root_quat');old=s
        for k in range(17):
            new=predicted[:,k];position,quat=integrate_root(position,quat,old,new,com_offset=new.new_tensor([0.,0.,-.07605]))
            kp=from_anchor(fk.keypoints(new[:,:29]),position,quat)
            errors.extend((kp[:,2:4,:3]-achieved[:,k,2:4,:3]).norm(dim=-1).flatten().cpu().tolist())
            imagined.extend((kp[:,2:4,:3]-target[:,k,2:4,:3]).norm(dim=-1).flatten().cpu().tolist())
            actual.extend((achieved[:,k,2:4,:3]-target[:,k,2:4,:3]).norm(dim=-1).flatten().cpu().tolist())
            if k==16:endpoint.append((kp[:,2:4,:3]-achieved[:,k,2:4,:3]).norm(dim=-1).cpu().numpy()*100)
            old=new
        state_error.append(float(((predicted-truth)/model.state_std).square().mean()))
    return {'world_wrist_error_cm':float(np.mean(errors)*100),'imagined_wrist_cm':float(np.mean(imagined)*100),
            'actual_wrist_cm':float(np.mean(actual)*100),'normalized_state_mse':float(np.mean(state_error)),
            'samples':len(errors),'horizon':17,
            'endpoint17_world_wrist_median_cm':np.median(np.concatenate(endpoint),axis=0).tolist(),
            'endpoint17_world_wrist_mean_cm':np.concatenate(endpoint).mean(0).tolist(),
            'source_split':'ordinary TRAIN' if is_ordinary else 'independent G replay on TRAIN motions',
            'tracking_target':'motion ground truth' if is_ordinary else 'uncompensated playing K* token at next control instant'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--body',required=True);p.add_argument('--data',required=True)
    p.add_argument('--name',required=True);p.add_argument('--iteration',type=int,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batches',type=int,default=8)
    p.add_argument('--controller',default=None);args=p.parse_args();torch.set_num_threads(4)
    model=load_model(args.body,args.device).requires_grad_(False)
    ordinary=score(model,ROOT/'runs/data',args.device,args.batches)
    online=score(model,ROOT/'runs/data'/args.data,args.device,args.batches)
    ratios=np.asarray(online['endpoint17_world_wrist_median_cm'])/np.maximum(np.asarray(ordinary['endpoint17_world_wrist_median_cm']),1e-8)
    ratio=float(ratios.max())
    # A2-sized optimism (actual >=2* imagined AND a material >=1cm gap) stops.
    optimism=online['actual_wrist_cm']>=2*online['imagined_wrist_cm'] and online['actual_wrist_cm']-online['imagined_wrist_cm']>=1.
    result=dict(online,iteration=args.iteration,body=args.body,ordinary=ordinary,
                model_error_ratio=ratio,model_error_ratio_per_wrist=ratios.tolist(),
                model_error_gate='each wrist WORLD endpoint median at17steps on G trajectories <=2 times ordinary TRAIN; same as B-4',gate_passed=ratio<=2 and not optimism,
                optimism_stop=optimism,
                imagination_definition='paired recorded G actions including sigma0.05; identical initial states/targets; not re-calling G on imagined states')
    if args.controller:
        saved=torch.load(args.controller,map_location='cpu',weights_only=False)
        result['training_closed_loop_imagined_wrist_cm']=saved['validation']['imagined_wrist_error_cm']
        # This value uses the augmented cache; do not compare its absolute
        # value as a paired optimism test against unrelated Isaac sequences.
    save_json(ROOT/'runs/round3'/f'{args.name}.json',result);log('b1_model_audit',**result)


if __name__=='__main__':main()
