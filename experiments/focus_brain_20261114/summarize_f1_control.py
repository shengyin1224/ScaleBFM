"""Compare true focus/non-focus stages in native ScaleBFM VR-3."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path('/root/shengyin/ScaleBFM_focus_brain_20261114')
sys.path.insert(0,str(ROOT/'ScaleTrack/scripts/pretrain/rsl_rl'))
from focus_bucket_statistics import summarize

def main(directory):
    directory=Path(directory)
    keys=['official_v2','f1_v2','july_v2','august_v2']
    reports={k:json.loads((directory/(k+'.json')).read_text()) for k in keys}
    data={k:torch.load(directory/(k+'.per_motion.pt'),map_location='cpu',weights_only=False) for k in keys}
    names=data['official_v2']['motion_names']
    common=torch.ones(len(names),dtype=torch.bool)
    for key,d in data.items():
        lookup={name:i for i,name in enumerate(d['motion_names'])}
        index=torch.tensor([lookup[name] for name in names])
        d['success']=d['success'][index]
        d['partitioned']={k:v[index] for k,v in d['partitioned'].items()}
        common &= d['success']
    for key in ['f1_v2','july_v2','august_v2']:
        assert reports[key]['label_sha256']==reports['official_v2']['label_sha256']
        for bucket in ['focus_count','nonfocus_count']:
            assert torch.equal(data[key]['partitioned'][bucket],data['official_v2']['partitioned'][bucket])
    paired=common.clone()
    for d in data.values():
        paired &= (d['partitioned']['focus_count']>0) & (d['partitioned']['nonfocus_count']>0)
    result=dict(mode='VR-3',active_links=['pelvis','left_wrist_yaw_link','right_wrist_yaw_link'],
                seed=42,max_frames_per_motion=1000,motions=len(names),
                common_success_count=int(common.sum()),paired_stage_motion_count=int(paired.sum()),conditions={})
    for key in keys:
        result['conditions'][key]=dict(checkpoint=reports[key]['checkpoint'],
                    label_definition=reports[key]['label_definition'],
                    label_sha256=reports[key]['label_sha256'],
                    success_count=reports[key]['success_count'],success_rate=reports[key]['success_rate'],
                    own_success_buckets=reports[key]['success_buckets'],
                    common_success_buckets=summarize(data[key]['partitioned'],common),
                    matched_success_buckets=summarize(data[key]['partitioned'],paired))
    (directory/'f1_control_comparison.json').write_text(json.dumps(result,indent=2))
    print('VR3_BUCKET_COMPARISON',json.dumps(result),flush=True)

if __name__=='__main__':
    main(sys.argv[1])
