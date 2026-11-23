"""Pair later checkpoints with July/August on identical successful VR-3 motions."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path('/root/shengyin/ScaleBFM_focus_brain_20261114')
RUN = ROOT / 'experiments/focus_brain_20261114'
OUT = RUN / 'later_audit'
sys.path.insert(0, str(ROOT / 'ScaleTrack/scripts/pretrain/rsl_rl'))
from focus_bucket_statistics import summarize

paths = {name: RUN / 'vr3_buckets' / key for name, key in
         [('official', 'official_v2'), ('july', 'july_v2'), ('august', 'august_v2')]}
paths.update({name: OUT / (name + '_vr3') for name in
              ['w4', 'w4_only_hat4', 'oct_v3_focus', 'oct_v3_precision', 'oct_v2_precision']})
fixed5='--fixed5' in sys.argv
if fixed5:
    paths={name:OUT/(name+'_vr3_fixed5') for name in
           ['july','august','oct_student_a','oct_student_b','oct_student_c']}
reports = {k: json.loads(p.with_suffix('.json').read_text()) for k,p in paths.items()}
data = {k: torch.load(p.with_suffix('.per_motion.pt'), map_location='cpu', weights_only=False)
        for k,p in paths.items()}
baseline='july' if fixed5 else 'official'
names = data[baseline]['motion_names']
common = torch.ones(len(names), dtype=torch.bool)
for k,d in data.items():
    lookup = {name:i for i,name in enumerate(d['motion_names'])}
    index = torch.tensor([lookup[n] for n in names])
    d['success'] = d['success'][index]
    d['partitioned'] = {m:v[index] for m,v in d['partitioned'].items()}
    assert reports[k]['label_sha256'] == reports[baseline]['label_sha256']
    if fixed5:assert reports[k]['fixed_future_offset']==5
    for bucket in ('focus_count', 'nonfocus_count'):
        assert torch.equal(d['partitioned'][bucket], data[baseline]['partitioned'][bucket])
    common &= d['success']
paired = common & (data[baseline]['partitioned']['focus_count']>0) & (data[baseline]['partitioned']['nonfocus_count']>0)
result = dict(mode='VR-3', active_links=['pelvis','left_wrist_yaw_link','right_wrist_yaw_link'],
              seed=42, max_frames_per_motion=1000, motions=len(names),
              common_success_count=int(common.sum()), paired_stage_motion_count=int(paired.sum()),
              label_sha256=reports[baseline]['label_sha256'],
              fixed_future_offset=5 if fixed5 else None,conditions={})
for k,d in data.items():
    result['conditions'][k] = dict(checkpoint=reports[k]['checkpoint'],
        success_count=reports[k]['success_count'], success_rate=reports[k]['success_rate'],
        own_success_buckets=reports[k]['success_buckets'],
        matched_success_buckets=summarize(d['partitioned'], paired))
(OUT/('vr3_students_fixed5_comparison.json' if fixed5 else 'vr3_comparison.json')).write_text(json.dumps(result,indent=2))
print('matched motions',int(paired.sum()))
for k,d in result['conditions'].items():
    errors=d['matched_success_buckets']['focus']['errors_cm_deg']
    fields=[f'error_{side}_wrist_{frame}_{kind}' for frame,kind in
            [('global','pos'),('pelvis_local','pos'),('pelvis_local','rot')] for side in ['left','right']]
    print(k,d['success_count'],[round(errors[f]['mean_over_motions'],3) for f in fields])
