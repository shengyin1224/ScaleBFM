"""First gate: frozen cerebellum must work in unmodified-inertia Inspire physics."""
import argparse,time,json
import numpy as np
import torch
from exec_comp.round6.env import World
from exec_comp.round6.library import Clip,paths
from exec_comp.round6.policy_driver import Driver
from exec_comp.common import log


def main():
    p=argparse.ArgumentParser();p.add_argument('--clips',type=int,default=5);p.add_argument('--steps',type=int,default=500);p.add_argument('--device',default='cpu');p.add_argument('--fixture',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    source=paths('validation')
    if a.fixture:
        from exec_comp.common import ROOT
        source=sorted((ROOT/'round6/fixtures').glob('*.npz'))
    driver=Driver(a.device);allrows=[]
    for file in source[:a.clips]:
        clip=Clip(file);world=World();state=clip.reset(world);driver.reset([state]);errors=[];falls=False;start=time.monotonic()
        for k in range(min(a.steps,clip.frames-11)):
            times=(10+k)/clip.fps+np.arange(6)*.02;p,q=clip.sample(times)
            action=driver.action([state],p[None],q[None]);state=world.step(driver.targets(action)[0],clip.focus(times[0]));driver.append([state],action)
            target,_=clip.sample([times[0]+.02]);errors.append(np.linalg.norm(target[0,2:]-state['kp'][2:,:3],axis=-1)*100)
            if world.fallen():falls=True;break
        result=dict(clip=file.stem,steps=len(errors),fallen=falls,wrist_cm=np.mean(errors,0).tolist(),root_height=float(state['p'][2]),elapsed_s=time.monotonic()-start)
        allrows.append(result);log('r6_baseline_clip',**result)
    log('r6_baseline_done',all_survived=all(not r['fallen'] for r in allrows),clips=allrows,xml_inertias_preserved=True)


if __name__=='__main__':main()
