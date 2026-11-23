"""Predeclared offline-only protocol. No parameters may be fitted to final real records."""
import json
from pathlib import Path
from exec_comp.common import ROOT

SETTINGS=dict(schema=2,state_fk_refreshed_after_integrate=True,seed=20261027,corpus='unifolm only; existing SHA256 clip-id90/10 split',physics_hz=500,control_hz=50,
    chunk_fps=15,chunk_frames=100,replan_s=.4,brain_age_s=[.05,.18],compute='accepted_switch_event_only',wrists_only=True,
    hand_load_scale=[.7,1.3],kp_scale=[.7,1.3],kd_scale=[.7,1.3],control_delay_ms=[0,60],sliding_friction=[.3,1.3],
    q_noise_rad=.01,dq_noise_rad_s=.5,gyro_noise_rad_s=.2,root_position_noise_m=.005,root_rotation_noise_deg=1.,
    command_ou_cap_m=.05,action_noise_sigma=.05,hand_servo_kp=5.,hand_servo_kd=.1,hand_servo_torque_limit_Nm=.8,
    pd_nominal='frozen cerebellum metadata; same software gains sent by original real backend',
    uncertainty='Hardware hand mass/inertia, actuator effective gains and total delay are not identifiable from existing control logs; ranges are declared engineering coverage, not fitted or certified physical bounds.',
    final_gate=dict(hand='right',phases=['approach','grasp'],cosine_mean_gt=.5,pooled_error_ratio_lt=.8,once_only=True,
        records_date='2026-10-26',no_posthoc_tuning=True,whole_phase_reconstruction_required=True),
    world_model='predict true68D body state conditioned on requested-action history, hand q/dq and simulation domain; no privileged inputs to A1/B1')


def guard_training():
    if (ROOT/'round6/FINAL_REAL_CHECK_STARTED').exists():raise RuntimeError('Final real-data check consumed; further R6 training/tuning is prohibited')


def settings():return json.loads(json.dumps(SETTINGS))
