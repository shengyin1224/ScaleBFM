"""Apply the original October gate, using July as the same-mode baseline."""
import json
from pathlib import Path
import sys

RUN=Path('/root/shengyin/ScaleBFM_focus_brain_20261114/experiments/focus_brain_20261114')
OUT=RUN/'later_audit'
TRACK=Path('/root/shengyin/ScaleBFM/ScaleTrack')
sys.path.insert(0,str(TRACK/'scripts/precision_teacher_student'))
from supervise import stable

reports={p.stem:json.loads(p.read_text()) for p in OUT.glob('*_jitter1.json')}
historical=TRACK/'student_20261023_launch/validation.log'
for line in historical.read_text().splitlines():
    if not line.startswith('STUDENT_VALIDATION '):continue
    d=json.loads(line.split(' ',1)[1])
    if d['localization_jitter_rms_mm_per_axis']!=1:continue
    if 'student_regularized' not in d['checkpoint'] or not d['checkpoint'].endswith('model_1000.pt'):continue
    variant=Path(d['checkpoint']).parent.name[-1].lower()
    reports['oct_student_'+variant+'_mode4_jitter1']=d
output=dict(reference='G1_WBT_Inspire_Pick_Up_Drinks_ep_0006',hold_s=3,static_statistics_last_s=2,
            static_band_hz=[3,15],jitter_mm_per_axis=1,jitter_AR1_coefficient=.8,seed=42,
            scope='one-motion probe; not a population robustness estimate',conditions={})
for key,d in reports.items():
    baseline=reports['july_mode'+d['mode']+'_jitter1']
    local=d['tracking']['active_link_pose_pelvis_frame_local_twist']
    base_local=baseline['tracking']['active_link_pose_pelvis_frame_local_twist']
    a2=d['action_second_difference_rms'];hf=d['static_arm_hf_rms_rad']
    gates=dict(tracking=d['first_failure_frame'] is None and d['success_rate_frames']==1.,
        static_hf=hf<=1.2*baseline['static_arm_hf_rms_rad']+2e-5,
        action_second_difference=a2<=1.2*baseline['action_second_difference_rms'],
        local_position=all(local[n]['position_m']['mean']<=1.2*base_local[n]['position_m']['mean']+.002
                           for n in ['left_wrist_yaw_link','right_wrist_yaw_link']))
    assert all(gates.values())==stable(d,baseline)
    output['conditions'][key]=dict(checkpoint=d['checkpoint'],mode=d['mode'],
        success_rate_frames=d['success_rate_frames'],first_failure_frame=d['first_failure_frame'],
        static_arm_hf_rms_rad=hf,action_second_difference_rms=a2,
        static_hf_change_pct=100*(hf/baseline['static_arm_hf_rms_rad']-1),
        action_second_difference_change_pct=100*(a2/baseline['action_second_difference_rms']-1),
        gates=gates,passes_original_stability_gate=all(gates.values()),tracking=d['tracking'],
        dynamic_by_group=d.get('dynamic_by_group'))
(OUT/'stability_comparison.json').write_text(json.dumps(output,indent=2))
for key,d in output['conditions'].items():
    print(key,'action_change',round(d['action_second_difference_change_pct'],1),
          'hf_change',round(d['static_hf_change_pct'],1),d['gates'])
