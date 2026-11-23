"""Extract a real episode-safe held-out history for export/latency contract checks."""
import numpy as np
import torch
from exec_comp.common import ROOT,log
from exec_comp.datasets import Rollouts
from exec_comp.geometry import pose_normalize


def main():
    data=Rollouts('validation',body=True,device='cpu')
    stream,e,t=data.indices(1,168,history=11);e=int(e[0]);t=int(t[0])
    history=slice(t-9,t+1)
    offsets=np.arange(100)*50/30;lower=offsets.astype(int);upper=np.ceil(offsets).astype(int)
    weight=torch.tensor(offsets-lower,dtype=torch.float32)[None,:,None,None]
    low=torch.tensor(stream['kp'][e,t+lower])[None];high=torch.tensor(stream['kp'][e,t+upper])[None]
    goal=pose_normalize(low+(high-low)*weight).numpy()
    s=stream['s'][e,history][None]
    destination=ROOT/'configs/heldout_contract_fixture.npz'
    np.savez_compressed(destination,s=s,q=s[...,:29],dq=s[...,29:58],kp=stream['kp'][e,history][None],
                        root_pos=stream['root_pos'][e,history][None],root_quat=stream['root_quat'][e,history][None],
                        privileged=stream['privileged_root'][e,t][None],previous_action=stream['action'][e,t-10:t][None],goal=goal)
    log('heldout_contract_fixture',path=str(destination),episode=int(stream['episode'][e,t]),time=t)


if __name__=='__main__':main()
