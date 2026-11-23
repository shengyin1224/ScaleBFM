"""Freeze only reference sampling state; original switching decision stays live."""
import copy
import numpy as np
from exec_comp.round5.engine import rotate

FIELDS=('chunk','previous_chunk','chunk_started_ns','previous_chunk_started_ns','blend_started_ns',
        'held_sample','prestart_reference_sample','reference_play_gate','reference_playing','requires_restart',
        '_latched_sample','_play_ns','_play_real_ns','_play_scale','approach_slowdown','approach_slowdown_fingers_full_speed','localization_paused')


def snapshot(env):
    if bool(env.cfg.get('continuity_hand_event_guard',False)):
        raise ValueError('R5 snapshot for continuity_hand_event_guard is not validated; keep this experiment off')
    state={k:getattr(env,k,None) for k in FIELDS}
    for name in ('chunk','previous_chunk','held_sample','prestart_reference_sample','_latched_sample'):
        if isinstance(state[name],dict):state[name]=dict(state[name])
    handoff=getattr(env,'focus_handoff',None)
    if handoff is not None:
        handoff=copy.copy(handoff)
        if not hasattr(handoff,'controllers'):raise ValueError('R5 requires per-hand handoff when handoff is enabled')
        handoff.controllers={side:copy.copy(c) for side,c in handoff.controllers.items()}
        for c in handoff.controllers.values():
            for name in ('playback','hold','entry'):
                value=getattr(c,name,None)
                if isinstance(value,dict):setattr(c,name,dict(value))
    state['focus_handoff']=handoff
    state['cfg']={'chunk_blend_s':float(env.cfg.get('chunk_blend_s',.22)), 'continuity_hand_event_guard':False}
    return state


def sample_snapshot(state,now_ns,offsets):
    # No constructor runs: no interfaces, simulator, camera or robot are opened.
    from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
    proxy=HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
    proxy.__dict__.update(state);proxy.future_offsets_s=np.asarray(offsets)
    proxy._focus_handoff_phase=lambda *a:None  # phase is not an input to A1/B1
    if proxy.chunk is None:raise ValueError('No reference chunk')
    return HAT4OnlineMotionTrackingEnv._sample_reference(proxy,now_ns,apply_stall_assist=False)


def canonical7(sample):
    names=('root','head','left_wrist','right_wrist')
    return np.concatenate((np.stack([sample[k+'_pos'] for k in names],1),
        np.stack([sample[k+'_quat_wxyz'] for k in names],1)),-1).astype(np.float32)


def canonical9(sample):
    from exec_comp.round3.metrics import canonical
    return canonical(sample)
