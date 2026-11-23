"""Heldout physical-FK check of frozen WM. Prints only; no real records read."""
import argparse
import numpy as np
import torch
from exec_comp.round6.data import Dataset
from exec_comp.round6.train import load
from exec_comp.policy import FrozenPolicy
from exec_comp.geometry import integrate_root,from_anchor
from exec_comp.common import log


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--samples',type=int,default=1024);a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_device(a.device)
    ds=Dataset(a.data,'validation',a.device);m,_=load(a.checkpoint,a.device);policy=FrozenPolicy().to(a.device);scores={i:[] for i in (1,4,8,16,17,32)}
    with torch.no_grad():
        for _ in range(int(np.ceil(a.samples/256))):
            s,e,t=ds.choose(256,32);state=ds.tensor(s['truth_s'][e,t]);pos=ds.tensor(s['true_root_pos'][e,t]);quat=ds.tensor(s['true_root_quat'][e,t]);past=ds.tensor(s['prev_action'][e[:,None],t[:,None]+np.arange(-2,1)])
            for j in range(32):
                action=ds.tensor(s['action'][e,t+j]);ctx=torch.cat((past.flatten(1),ds.tensor(s['hand'][e,t+j]),ds.tensor(s['domain'][e,t+j])),-1)
                ns=m(state,action,ctx);pos,quat=integrate_root(pos,quat,state,ns);pred=from_anchor(policy.fk.keypoints(ns[:,:29]),pos,quat);truth=ds.tensor(s['ach_kp'][e,t+j])
                if j+1 in scores:scores[j+1].append((pred[:,2:,:3]-truth[:,2:,:3]).square().sum(-1).cpu().numpy())
                state=ns;past=torch.cat((past[:,1:],action[:,None]),1)
    log('r6_world_physical_validation',wrist_rmse_cm={str(i):(np.sqrt(np.concatenate(v).mean(0))*100).tolist() for i,v in scores.items()},wrist_mean_cm={str(i):(np.sqrt(np.concatenate(v)).mean(0)*100).tolist() for i,v in scores.items()},wrist_p95_cm={str(i):(np.percentile(np.sqrt(np.concatenate(v)),95,axis=0)*100).tolist() for i,v in scores.items()},samples=int(np.ceil(a.samples/256))*256,actions='recorded requested actions,including action perturbation;future hand state known as WM exogenous input',no_real_records=True)


if __name__=='__main__':main()
