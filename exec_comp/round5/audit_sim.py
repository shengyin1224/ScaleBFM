"""Read composed USD mass/inertia and saved sim raw errors; no physics startup."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from exec_comp.common import ROOT,PROJECT,save_json,digest
from exec_comp.geometry import six_to_q
from exec_comp.round5.engine import rotate,qinv


def main():
    from pxr import Usd,UsdPhysics
    asset=PROJECT/'ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.usda'
    def masses(path):
        stage=Usd.Stage.Open(str(path));result=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MassAPI):
                api=UsdPhysics.MassAPI(prim);result.append(dict(path=str(prim.GetPath()),mass=api.GetMassAttr().Get(),inertia=list(api.GetDiagonalInertiaAttr().Get() or [])))
        return result
    robot=masses(asset);hands={}
    for side in ('left','right'):
        p=asset.parents[2]/'dexhand/inspire'/f'inspire_hand_{side}.usd'
        if p.exists():hands[side]=dict(path=str(p),bodies=masses(p))
    folder=ROOT/'runs/round3/brain_evaluation/raw_stall-on';meta=json.loads((folder/'metrics.json').read_text());errors=[]
    for i,clip in enumerate(meta['per_clip']):
        if clip['type']!='robot':continue
        with np.load(folder/f'episode_{i:02d}.npz') as f:
            actual=f['actual'][1:];target=f['target'][1:]
            quat=six_to_q(torch.as_tensor(actual[:,0,3:])).numpy()
            errors.append(rotate(qinv(quat)[:,None],target[:,2:4,:3]-actual[:,2:4,:3]))
    pooled=np.concatenate(errors)
    result=dict(usd=str(asset),usd_sha256=digest(asset),rigid_bodies=robot,total_mass_kg=sum(x['mass'] or 0 for x in robot),
        independent_hand_assets=hands,inspire_articulated_hand_bodies_attached=any(any(w in x['path'].lower() for w in ('finger','inspire','palm')) for x in robot),
        wrist_body_masses=[x for x in robot if 'wrist_yaw' in x['path']],
        mass_randomization='Original task defines hand_mass +U(0,1)kg on wrist_yaw, but exec_comp.env.make_env disables ALL event terms, then its DR adds friction/PD/torsoCOM only; no Inspire hand articulation is attached.',
        real_hand_mass='N/A: physical hand mass/inertia not measured in these logs; do not equate independent USD asset mass with measured hardware',
        sim_raw_fixed_bias_root_cm=(pooled.mean(0)*100).tolist(),sim_raw_error_cm=float(np.linalg.norm(pooled,axis=-1).mean()*100),frames=len(pooled),
        comparison_caveat='Sim10 brain clips and real46 raw runs are different trajectories; bias comparison is descriptive, not a paired model calibration.',
        evidence={'env_py_sha256':digest(ROOT/'env.py'),'randomization_sha256':digest(ROOT/'randomization.py')},real_robot_run=False)
    save_json(ROOT/'runs/round5/sim_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rigid_bodies','independent_hand_assets')}))


if __name__=='__main__':main()
