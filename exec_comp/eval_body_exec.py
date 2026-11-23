"""B-1: 1/8/17/32-step gate, unit-gravity projection, no oracle world-root leakage."""
import argparse
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,log,KP_NAMES,digest
from exec_comp.datasets import Rollouts
from exec_comp.train_common import load_model
from exec_comp.diff.fk import FK
from exec_comp.geometry import integrate_root,from_anchor,to_anchor
from exec_comp.models.body_wm import project_gravity_state as project


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True)
    p.add_argument('--velocity-noise',type=float,default=0.);p.add_argument('--batches',type=int,default=16)
    p.add_argument('--data-root',action='append',default=[]);p.add_argument('--split',choices=('train','validation'),default='validation')
    args=p.parse_args();torch.manual_seed(77);torch.set_num_threads(4)
    model=load_model(args.checkpoint,'cuda:0').requires_grad_(False)
    data=Rollouts(args.split,body=True,device='cuda:0',folders=args.data_root or None);data.rng=np.random.default_rng(77)
    fk=FK().cuda();scores={h:{'q':[],'world':[],'local':[]} for h in (1,8,17,32)}
    with torch.no_grad():
        for _ in range(args.batches):
            batch=data.body_batch(256,32,model.privileged);state=batch['s'][:,0].clone()
            if args.velocity_noise:
                if not model.privileged:raise ValueError('Velocity-noise variant requires the explicitly augmented 68D state')
                state[:,65:68]+=(torch.rand_like(state[:,65:68])*2-1)*args.velocity_noise
            pos=batch['root_pos'];quat=batch['root_quat']
            for step in range(32):
                ns=project(model(state,batch['action'][:,step]))
                if model.privileged:pos,quat=integrate_root(pos,quat,state,ns)
                if step+1 in scores:
                    record=scores[step+1];local=fk.keypoints(ns[:,:29])
                    local_truth=to_anchor(batch['ach'][:,step],batch['next_root_pos'][:,step],batch['next_root_quat'][:,step])
                    record['q'].append(((ns[:,:29]-batch['next_s'][:,step,:29]).abs().mean(-1)*180/torch.pi).cpu().numpy())
                    record['local'].append(((local[...,:3]-local_truth[...,:3]).norm(dim=-1)*100).cpu().numpy())
                    if model.privileged:
                        world=from_anchor(local,pos,quat)
                        record['world'].append(((world[...,:3]-batch['ach'][:,step,:,:3]).norm(dim=-1)*100).cpu().numpy())
                state=ns
    horizons={}
    for h,record in scores.items():
        q=np.concatenate(record['q']);row={'joint_mae_deg':float(q.mean()),'joint_median_deg':float(np.median(q))}
        for field in ('world','local'):
            if not record[field]:row[field+'_keypoints_cm']=None;continue
            e=np.concatenate(record[field])
            row[field+'_keypoints_cm']={name:{'mean':float(e[:,i].mean()),'median':float(np.median(e[:,i])),'p95':float(np.percentile(e[:,i],95))} for i,name in enumerate(KP_NAMES)}
        horizons[h]=row
    world=horizons[17]['world_keypoints_cm']
    passed=bool(world and all(world[k]['median']<=1 for k in ('left_wrist','right_wrist')))
    result={'name':args.name,'checkpoint':args.checkpoint,'checkpoint_sha256':digest(args.checkpoint),'privileged':model.privileged,'initial_velocity_noise_uniform_m_s':args.velocity_noise,
            'H_exec':17,'gravity_projection_each_step':True,'world_gate_passed':passed,'horizons':horizons,'data_roots':args.data_root,'data_split':args.split,
            'gate':'both wrists WORLD endpoint median at17 control steps <=1cm',
            'world_limitation':None if model.privileged else 'N/A: default64 lacks root translation velocity. Only base-relative FK diagnostic is calculable; no true future root is fed to make up a world prediction.'}
    save_json(ROOT/'runs/round2/body_validation'/f'{args.name}.json',result);log('body_exec_validation',**result)


if __name__=='__main__':main()
