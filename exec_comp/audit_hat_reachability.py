"""Rigid-chain geometric check independent of policy/controller performance."""
import json
import numpy as np
from exec_comp.common import ROOT,save_json,log
from exec_comp.diff.fk import FK
from exec_comp.run_round2_a1 import status


def main():
    fk=FK();index=fk.names.index('torso_link');bound=float(fk.head_offset.norm())
    while index>0:bound+=float(fk.translations[index].norm());index=fk.parents[index]
    folder=ROOT/'runs/round2/hat_predictions';manifest=json.loads((folder/'manifest.json').read_text());rows=[]
    for episode in manifest['episodes']:
        with np.load(episode['file']) as f:
            kp=f['keypoints'];distance=np.linalg.norm(kp[:,:,1,:3]-kp[:,:,0,:3],axis=-1)
            initial=f['initial_keypoints'];initial_distance=float(np.linalg.norm(initial[1,:3]-initial[0,:3]))
        gap=np.maximum(distance-bound,0)*100
        rows.append({'episode':episode['episode'],'type':episode['type'],'initial_root_head_distance_cm':initial_distance*100,
            'predicted_root_head_distance_mean_cm':float(distance.mean()*100),'necessary_position_error_gap_mean_cm':float(gap.mean()),
            'gap_p95_cm':float(np.percentile(gap,95)),'fraction_gap_above5cm':float((gap>5).mean())})
    result={'robot_root_to_virtual_head_chain_upper_bound_cm':bound*100,'episodes':rows,
            'meaning':'Euclidean head-root separation is invariant to world translation/heading. A target beyond the sum of fixed link lengths cannot be reached by joint-angle adjustment; it needs an explicit embodiment mapping.',
            'human_mapping_available':False,'human_raw_replay':'diagnostic only; no valid robot-domain tracking claim'}
    save_json(ROOT/'configs/hat_reachability.json',result);log('hat_reachability',**result)
    status('A1-4：机器人组已评估，人类组等待有效G1映射；继续离线打包',**{
        'A1-3':'未完成全部验收：机器人10条有效评估完成；人类15条原样人体目标超出G1几何范围，缺重定向N/A；抓取标记N/A'})


if __name__=='__main__':main()
