"""Can the G1 reach what the brain asks for at all? (A1-5 item 8)

Separations measured from the root are invariant to world translation and
heading, so a target farther than the sum of the fixed link lengths on the way
to that link cannot be reached by any joint configuration. That is a property
of the request, not of the controller, and it is reported for the robot and
human source groups separately because only the robot group feeds the tracking
numbers.
"""
import argparse
import json
import numpy as np
from exec_comp.common import ROOT,save_json,log
from exec_comp.diff.fk import FK
import torch
from exec_comp.geometry import six_to_q,qapply

LINKS={"head":"torso_link","left_wrist":"left_wrist_yaw_link","right_wrist":"right_wrist_yaw_link"}


def chain_bound(fk,link,extra=0.):
    """Sum of fixed link translations from the root to ``link``."""
    index=fk.names.index(link);total=float(extra)
    while index>0:
        total+=float(fk.translations[index].norm());index=fk.parents[index]
    return total


def summarize(distance,bound):
    over=distance>bound
    gap=np.maximum(distance-bound,0.)*100
    return {"frames":int(distance.size),"bound_cm":bound*100,
            "fraction_beyond_reach":float(over.mean()),
            "distance_mean_cm":float(distance.mean()*100),
            "distance_p95_cm":float(np.percentile(distance,95)*100),
            "distance_max_cm":float(distance.max()*100),
            "excess_mean_cm":float(gap.mean()),"excess_p95_cm":float(np.percentile(gap,95))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predictions",default="runs/round3/hat_predictions")
    parser.add_argument("--output",default=str(ROOT/"configs/round3_reachability.json"))
    arguments=parser.parse_args()
    fk=FK()
    bounds={"head":chain_bound(fk,LINKS["head"],float(fk.head_offset.norm())),
            "left_wrist":chain_bound(fk,LINKS["left_wrist"]),
            "right_wrist":chain_bound(fk,LINKS["right_wrist"])}
    shoulders={side:fk.names.index(f'{side}_shoulder_pitch_link') for side in ('left','right')}
    torso=fk.names.index('torso_link')
    if any(fk.parents[i]!=torso for i in shoulders.values()):raise ValueError('Shoulder origin requires explicit torso chain audit')
    arm_bounds={side:chain_bound(fk,LINKS[f'{side}_wrist'])-chain_bound(fk,f'{side}_shoulder_pitch_link') for side in shoulders}
    folder=ROOT/arguments.predictions
    manifest=json.loads((folder/"manifest.json").read_text())
    episodes=[];pooled={kind:{name:[] for name in bounds} for kind in ("robot","human")}
    arm_pooled={kind:{side:[] for side in shoulders} for kind in ('robot','human')}
    for episode in manifest["episodes"]:
        with np.load(episode["file"]) as data:
            keypoints=data["keypoints"]
        # Every predicted frame of every chunk counts: a request the robot
        # cannot reach is a request whether or not it is later superseded.
        root=keypoints[...,0,:3]
        row={"episode":episode["episode"],"type":episode["type"],"source":episode["source"]}
        head_quat=six_to_q(torch.as_tensor(keypoints[...,1,3:]))
        torso_pos=torch.as_tensor(keypoints[...,1,:3])-qapply(head_quat,fk.head_offset.expand(*head_quat.shape[:-1],3))
        row['shoulder_arm']={}
        for index,side in enumerate(('left','right'),start=2):
            shoulder=torso_pos+qapply(head_quat,fk.translations[shoulders[side]].expand_as(torso_pos))
            distance=(torch.as_tensor(keypoints[...,index,:3])-shoulder).norm(dim=-1).numpy().reshape(-1)
            row['shoulder_arm'][side]=summarize(distance,arm_bounds[side]);arm_pooled[episode['type']][side].append(distance)
        for index,name in enumerate(("head","left_wrist","right_wrist"),start=1):
            distance=np.linalg.norm(keypoints[...,index,:3]-root,axis=-1).reshape(-1)
            row[name]=summarize(distance,bounds[name])
            pooled[episode["type"]][name].append(distance)
        episodes.append(row);log("round3_reachability_episode",episode=row["episode"],
                                 type=row["type"],
                                 head_beyond=row["head"]["fraction_beyond_reach"],
                                 right_wrist_beyond=row["right_wrist"]["fraction_beyond_reach"])
    groups={}
    for kind,links in pooled.items():
        if not links["head"]:continue
        groups[kind]={"episodes":sum(1 for e in episodes if e["type"]==kind),
                      **{name:summarize(np.concatenate(values),bounds[name])
                         for name,values in links.items()}}
        groups[kind]['head_nominal_44_4cm_fraction']=float((np.concatenate(links['head'])>.444).mean())
        groups[kind]['shoulder_arm']={side:summarize(np.concatenate(values),arm_bounds[side]) for side,values in arm_pooled[kind].items()}
    result={"chain_upper_bounds_cm":{k:v*100 for k,v in bounds.items()},
            'shoulder_arm_upper_bounds_cm':{k:v*100 for k,v in arm_bounds.items()},
            'nominal_head_threshold_cm':44.4,
            'nominal_threshold_note':'44.4cm is rounded; report it separately from the XML chain bound44.4178cm to avoid declaring rounding-scale deviations unreachable',
            "bound_meaning":("root-to-link distance cannot exceed the sum of the fixed "
                "translations along the chain; the head bound includes the 0.4 m virtual "
                "head offset above torso_link"),
            "per_episode":episodes,"groups":groups,
            "human_note":("human episodes are reported for diagnosis only and never enter a "
                "tracking number; no confirmed human-to-G1 retarget exists"),
            "predictions":str(folder)}
    save_json(arguments.output,result)
    log("round3_reachability",**{k:{"head":v["head"]["fraction_beyond_reach"],
        "left_wrist":v["left_wrist"]["fraction_beyond_reach"],
        "right_wrist":v["right_wrist"]["fraction_beyond_reach"]} for k,v in groups.items()})


if __name__=="__main__":main()
