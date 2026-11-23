"""A deterministic clip-ID split of the single authorized unifolm corpus."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import yaml
from exec_comp.common import ROOT,TRACK,save_json


def prepare():
    paths=sorted((TRACK/"data/hat4_unifolm_focus_v3/processed").glob("*.npz"))
    if len(paths)!=3000:raise ValueError(f"Expected 3000 authorized clips, got {len(paths)}")
    paths.sort(key=lambda p:hashlib.sha256(("exec_comp_seed42:"+p.stem).encode()).hexdigest())
    val=paths[:300];train=paths[300:]
    assert not set(train)&set(val)
    output=ROOT/"configs";output.mkdir(exist_ok=True)
    for name,group in (("train",train),("validation",val)):
        (output/(name+".yaml")).write_text(yaml.safe_dump({p.stem:str(p) for p in sorted(group)},sort_keys=True))
    save_json(output/"split.json",{"seed":42,"method":"sha256_clip_id_90_10","train":[p.stem for p in train],
        "validation":[p.stem for p in val],"single_source":str(paths[0].parent)})
    print("PREPARED",len(train),len(val))


def audit_prefix():
    import concurrent.futures
    def inspect(path):
        with np.load(path,allow_pickle=False) as f:
            velocity=f["body_lin_vel_w"][:,0]
            angular=f["body_ang_vel_w"][:,0]
            position=f["body_pos_w"][:,0]
            n=min(10,len(velocity))
            invalid=(np.linalg.norm(velocity[:n],axis=-1)>3)|(np.linalg.norm(angular[:n],axis=-1)>5)
            start=int(np.flatnonzero(invalid)[-1]+1) if invalid.any() else 0
            displacement=np.linalg.norm(position[start:,:2]-position[start:start+1,:2],axis=-1)
            return path.stem,{"start":start,"startup_speed_m_s":float(np.linalg.norm(velocity[0])),
                "class":"walking" if displacement.max()>.2 else "standing","frames":len(position)}
    paths=sorted((TRACK/"data/hat4_unifolm_focus_v3/processed").glob("*.npz"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        result=dict(pool.map(inspect,paths))
    save_json(ROOT/"configs/prefix_audit.json",{"clips":result,"thresholds":{"root_speed_m_s":3,"root_angular_speed_rad_s":5,"prefix_frames":10},
        "trimmed_clips":sum(v["start"]>0 for v in result.values()),"max_startup_speed_m_s":max(v["startup_speed_m_s"] for v in result.values())})
    print("PREFIX_AUDITED",len(result),sum(v["start"]>0 for v in result.values()))


if __name__=="__main__":
    import sys
    audit_prefix() if "--audit-prefix" in sys.argv else prepare()
