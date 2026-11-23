"""Run the deployed chunk-switching logic itself inside the simulation.

Nothing here reimplements a switching rule. A bare instance of the deployed
``HAT4AdaptiveMotionTrackingEnv`` is created with ``__new__`` and given exactly
the attributes its switching methods read, so PACE horizon selection, the near
cadence, the anti-stall lookahead, the close boundary, continuity matching, RTC
and the overlap blend all execute as the original code, from the pinned
snapshot in ``exec_comp/vendor``. Only the transport around them (ZMQ receive,
protocol alignment, FOCUS worker IK) is replaced, because the simulation has no
sockets and no hand feedback; every such substitution is listed in
``LIMITATIONS`` and copied into the result files.

The same class drives both evaluation corpora, so the motion library and the
real brain outputs are switched by one implementation.
"""
import sys
from exec_comp.common import ROOT
sys.path.insert(0,str(ROOT/'round6/source'))

import copy
from types import SimpleNamespace
import numpy as np
import torch
from loguru import logger
from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from scalebridge.online.adaptive_chunk_execution import SIDES,StallLookahead,closing_boundary
from scalebridge.online.protocol import validate_hat_chunk
from scalebridge.online.reference import sample_chunk
# Exact-input memoization, restricted to this copied offline source tree.
from exec_comp.round3.boundary_cache import BoundaryCache
from scalebridge.online import adaptive_chunk_execution as _core
from scalebridge.env import motion_tracking_hat4_adaptive as _adaptive
from pathlib import Path
if not Path(_adaptive.__file__).is_relative_to(ROOT/'round6/source'):
    raise RuntimeError('R6 switcher must load its own source snapshot')
closing_boundary=BoundaryCache(_core.closing_boundary)
_core.closing_boundary=closing_boundary;_adaptive.closing_boundary=closing_boundary
from exec_comp.round3.signals import predicted_grasp_phase

LIMITATIONS=(
    "FOCUS handoff is inert: both corpora lack measured finger feedback, so no "
    "handoff window can open. Its geometric close boundary still runs and still "
    "caps the horizon.",
    "Chunks are supplied in simulator world coordinates; the ORIGINAL alignment "
    "function runs with an identity fixed anchor, not with an extra sensor calibration.",
    "Transport is an in-memory newest-packet subscriber; the ORIGINAL full acceptance "
    "method drains, retains, freshness-checks and consumes it at50Hz, including hold release.",
    "The Vive/DDS watchdogs, staleness latch and R1 gating are never triggered: "
    "the simulation has no outage to recover from.",
)
# Open, unchanging fingertips: a corpus with no finger channel must read as
# "no close in this chunk". All-zero fingertips would instead put every finger
# on the thumb and be read as an immediate grasp, which would silently force
# the whole corpus onto the near cadence. Spreads sit above the deployed
# focus_v3_closed_spread_m (0.11 m) and never contract.
OPEN_FINGERTIPS=np.array([[0.,0.,0.],[.16,0.,0.],[.17,0.,0.],[.17,0.,0.],[.16,0.,0.]],dtype=np.float32)
MEASURED_KEYS=("root","head","left_wrist","right_wrist")





def quiet_deployment_logging():
    """Silence the deployed per-switch logging for batched simulation.

    On the robot there is one environment and these lines are the operator's
    audit trail. With 300 simulated robots the same lines are several hundred
    formatted writes per replan and dominate the run time. Only this process's
    sink is switched off; the module is untouched and the switching decisions
    are recorded in the result files instead.
    """
    logger.disable("scalebridge")


def open_fingertips(frame_count):
    return np.repeat(OPEN_FINGERTIPS[None],frame_count,axis=0)


def make_chunk(sequence_id,source_state_sequence_id,generated_at_ns,fps,keypoints,
               fingertips=None):
    """Protocol-valid chunk from canonical [T,4,(pos,quat)] world keypoints.

    ``keypoints`` carries root/head/left_wrist/right_wrist, the same four links
    and the same head convention the deployed chunk uses.
    """
    positions,quaternions=keypoints
    frame_count=int(len(positions))
    fields={"type":"HatChunkV4","version":4,"sequence_id":int(sequence_id),
            "source_state_sequence_id":int(source_state_sequence_id),
            "generated_at_ns":int(generated_at_ns),"fps":float(fps),"frame_count":frame_count,
            "action_representation":"original"}
    for index,name in enumerate(MEASURED_KEYS):
        fields[f"{name}_pos"]=np.asarray(positions[:,index],dtype=np.float32)
        fields[f"{name}_quat_wxyz"]=np.asarray(quaternions[:,index],dtype=np.float32)
    for side_index,side in enumerate(SIDES):
        fields[f"{side}_fingertip_local"]=(open_fingertips(frame_count) if fingertips is None
                                           else np.asarray(fingertips[:,side_index],dtype=np.float32))
    return validate_hat_chunk(fields)


def measured_state(positions,quaternions):
    """The subset of the published robot state the pose matcher reads."""
    state={}
    for index,name in enumerate(MEASURED_KEYS):
        state[f"{name}_pos_world"]=np.asarray(positions[index],dtype=np.float64)
        state[f"{name}_quat_world_wxyz"]=np.asarray(quaternions[index],dtype=np.float64)
    return state


class _NoHandoff:
    """Stand-in for the FOCUS handoff when no finger feedback exists.

    Never bypasses and never activates, so every deployed branch guarded by the
    handoff takes its ordinary path instead of its grasp path.
    """
    bypass_continuity=False
    bypass_sides=()
    sides=()
    active=False
    active_sides=()
    controllers={}
    state='idle'
    entry=None

    def accepts(self,candidate):return True


class _MeasuredPoseFK:
    """Carries measured link poses in the attribute the deployed code reads."""

    def __init__(self):self.last_body_poses={}


class _MemorySubscriber:
    """Transport only: original acceptance consumes its newest retained packet."""
    def __init__(self):self.latest=None
    def send(self,chunk):
        if self.latest is None or chunk['sequence_id']>self.latest['sequence_id']:self.latest=chunk
    def receive_error(self):return None
    def receive_latest(self):
        value=self.latest;self.latest=None;return value


class DeployedSwitching:
    """One environment's chunk execution, decided by the deployment code.

    ``offer`` presents a freshly produced chunk; the deployed hold, matcher and
    RTC decide whether and where it enters. ``sample`` returns the reference the
    controller would be given at this control step.
    """

    def __init__(self,*,config,future_offsets_s,dt=.02,anti_stall=True):
        env=HAT4AdaptiveMotionTrackingEnv.__new__(HAT4AdaptiveMotionTrackingEnv)
        self.env=env
        self.config=dict(config)
        env.cfg=self.config
        env.dt=float(dt)
        env.future_offsets_s=np.asarray(future_offsets_s,dtype=np.float64)
        # Playback-rate scaling (approach_slowdown) is off in the deployed
        # configuration, so tau is the monotonic clock throughout.
        env._play_ns=None;env._play_real_ns=None;env._play_scale=1.
        env.approach_slowdown=bool(self.config.get("approach_slowdown",False))
        env.approach_slowdown_fingers_full_speed=bool(
            self.config.get("approach_slowdown_fingers_full_speed",True))
        env.focus_handoff_source=self.config.get("focus_handoff_source","keypoints")
        env.chunk_phase_alignment=bool(self.config.get("chunk_phase_alignment",True))
        env.chunk_start_reference_continuity=bool(self.config.get("chunk_start_reference_continuity",False))
        env.continuity_mode="legacy" if env.chunk_start_reference_continuity else "off"
        env.continuity_search_ahead_s=float(self.config.get("continuity_search_ahead_s",1.))
        env.phase_rotation_weight=float(self.config.get("phase_rotation_weight",.08))
        env.phase_velocity_weight=float(self.config.get("phase_velocity_weight",.03))
        env.phase_measured_weight=float(self.config.get("phase_measured_weight",.15))
        env.phase_max_root_jump_m=float(self.config.get("phase_max_root_jump_m",.2))
        env.chunk_alignment=self.config.get("chunk_alignment","fixed")
        env.finger_chunk_source=self.config.get("finger_chunk_source","playback")
        env.focus_handoff=_NoHandoff()
        env.focus_adaptive=None
        env.stall_wrist_assist=None
        env.hand_event_playback_boundary=None
        env.hand_event_chunk_preflight=None
        env.focus_worker=None
        env.simulator=SimpleNamespace()
        env.alignment_anchor=(np.zeros(3),np.array([1.,0.,0.,0.]))
        env.state_buffer={}
        env.current_pose_fk=_MeasuredPoseFK()
        env._focus_prediction_chunks={}
        env._adaptive_plan=None;env._adaptive_key=None
        # The anti-stall lookahead is part of the deployed adaptive execution;
        # A1-5 asks for a configuration with it switched off, which is done by
        # leaving the budget at zero rather than by editing the module.
        self.anti_stall=bool(anti_stall)
        env._adaptive_stall=StallLookahead(
            frames=int(self.config.get("adaptive_stall_frames",10)),
            motion_m=float(self.config.get("adaptive_stall_motion_m",.005)),
            increment_m=float(self.config.get("adaptive_stall_increment_m",.005)),
            maximum_m=float(self.config.get("adaptive_stall_max_lookahead_m",.05)))
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self):
        env=self.env
        env._play_ns=None;env._play_real_ns=None;env._play_scale=1.
        env.chunk=None;env.chunk_started_ns=None;env.chunk_start_frame=0
        env.previous_chunk=None;env.previous_chunk_started_ns=None;env.blend_started_ns=None
        env.last_chunk_sequence_id=-1;env.last_chunk_received_ns=None
        env._last_chunk_received_play_ns=None;env._chunk_hold_release_ns=None
        env.held_sample=None;env.awaiting_fresh_chunk=False;env.localization_paused=False
        env.requires_restart=False;env._latched_sample=None
        env.reference_playing=True;env.reference_play_gate=False
        env.prestart_reference_sample=None
        env.last_published_robot_state=None;env.last_reference_sample=None
        env.minimum_source_state_sequence_id=None
        env.state_sequence_id=0;env.state_timestamp_history=[];env.state_play_history=[]
        env._trajectory_source_chunk=None;env._trajectory_source_start_frame=None
        env._adaptive_plan=None;env._adaptive_key=None
        env._adaptive_stall.reset()
        env.chunk_subscriber=_MemorySubscriber();env._pending_raw_chunk=None
        self.events=[]
        self.stall_engaged_steps=0
        self.stall_events=dict.fromkeys(SIDES,0)
        self.stall_max_m=dict.fromkeys(SIDES,0.)
        self.last_accept=None
        self._boundaries={};self._boundary_key=None

    # -- inputs --------------------------------------------------------------
    def fork(self):
        """A sampling-only snapshot; clock cleanup and stall budget stay private."""
        other=copy.copy(self);other.env=copy.copy(self.env)
        other.env._adaptive_stall=copy.deepcopy(self.env._adaptive_stall)
        return other

    def publish_state(self,now_ns,positions,quaternions):
        """Register this control step's measured state, as the robot does."""
        env=self.env
        env.state_sequence_id+=1
        sequence=env.state_sequence_id-1
        env.state_timestamp_history.append((sequence,int(now_ns)))
        env.state_play_history.append((sequence,int(env._play_now_ns(now_ns))))
        del env.state_timestamp_history[:-256];del env.state_play_history[:-256]
        env.last_published_robot_state=measured_state(positions,quaternions)
        env.state_buffer['root_pos_buffer']=torch.as_tensor(positions[0],dtype=torch.float32)[None,None]
        env.state_buffer['root_quat_wxyz_buffer']=torch.as_tensor(quaternions[0],dtype=torch.float32)[None,None]
        env.current_pose_fk.last_body_poses={
            f"{side}_wrist_yaw_link":(np.asarray(positions[2+index],dtype=np.float64),
                                      np.asarray(quaternions[2+index],dtype=np.float64))
            for index,side in enumerate(SIDES)}
        if self.anti_stall and self.running():
            before=dict(env._adaptive_stall.budget)
            env._adaptive_stall.update(int(now_ns),
                {side:np.asarray(positions[2+index],dtype=np.float64) for index,side in enumerate(SIDES)},
                float(env.chunk["fps"]),blocked=env.focus_handoff.bypass_sides)
            for side,value in env._adaptive_stall.budget.items():
                # A stall event is the lookahead arming from rest; the budget
                # then grows while the wrist stays put.
                if before[side]<=0<value:self.stall_events[side]+=1
                self.stall_max_m[side]=max(self.stall_max_m[side],float(value))
        elif not self.running():
            env._adaptive_stall.reset();env._adaptive_key=None
        return sequence

    def running(self):
        return bool(self.env._adaptive_running())

    def hold_active(self,now_ns):
        return bool(self.env._min_execution_hold_active(now_ns))

    def horizon(self,now_ns):
        return self.env._adaptive_horizon(now_ns) if self.running() else None

    def horizon_start_frame(self):
        """The cursor PACE measured its 25-50 (or 8-12) frame window from."""
        env=self.env
        if env.chunk is None or env.chunk_started_ns is None:return None
        accepted=env._last_chunk_received_play_ns
        if accepted is None:accepted=env.last_chunk_received_ns
        if accepted is None:return None
        return max(0.,(accepted-env.chunk_started_ns)*1e-9)*float(env.chunk["fps"])

    def cursor_frame(self,now_ns):
        env=self.env
        if env.chunk is None or env.chunk_started_ns is None:return None
        return env._elapsed(now_ns,env.chunk_started_ns)*float(env.chunk["fps"])

    def offer(self,chunk,now_ns,*,training_oversample=False):
        """Publish then run the ORIGINAL complete acceptance method.

        The optional hold override is used only by the explicitly augmented
        short training cache. Evaluation and collection never request it.
        """
        self.env.chunk_subscriber.send(chunk)
        if not training_oversample:return self.poll(now_ns)
        own=self.env.__dict__.get('_min_execution_hold_active')
        self.env._min_execution_hold_active=lambda now:False
        try:return self.poll(now_ns)
        finally:
            if own is None:self.env.__dict__.pop('_min_execution_hold_active',None)
            else:self.env._min_execution_hold_active=own

    def poll(self,now_ns):
        """Called every50Hz tick, even when no new inference arrived."""
        env=self.env;held=self.hold_active(now_ns)
        accepted=bool(env._accept_latest_chunk(now_ns))
        if accepted:
            record={'accepted':True,'reason':'accepted','start_frame':int(env.chunk_start_frame),
                    'phase':dict(env.chunk_phase_state),'sequence_id':int(env.last_chunk_sequence_id)}
            self.events.append(dict(record,now_ns=int(now_ns)));self.last_accept=record
            return record
        return {'accepted':False,'reason':'min_execution_hold' if held else 'no_accepted_packet'}

    # -- output --------------------------------------------------------------
    def lookahead_active(self):
        """Is the anti-stall lookahead currently displacing anything?"""
        return self.anti_stall and max(self.env._adaptive_stall.budget.values())>0.

    def sample_raw(self,now_ns,deployed=None):
        """The same reference without the anti-stall lookahead: pure K*.

        With an unarmed lookahead the two are the same array by construction --
        the deployed override only edits wrist positions when a side has a
        positive budget -- so the second sampling pass is skipped rather than
        recomputed for every robot on every control step.
        """
        if deployed is not None and not self.lookahead_active():return deployed
        return self.env._sample_reference(now_ns,apply_stall_assist=False)

    def sample_window(self,now_ns,offsets_s):
        """The stitched command stream on an arbitrary horizon grid.

        A shallow copy carries the live clocks and chunks into the deployed
        sampler with different future offsets; it cannot write back, so asking
        for the A1 window never disturbs the control-step reference.
        """
        proxy=copy.copy(self.env)
        proxy.future_offsets_s=np.asarray(offsets_s,dtype=np.float64)
        return HAT4AdaptiveMotionTrackingEnv._sample_reference(proxy,now_ns,apply_stall_assist=False)

    def close_boundaries(self):
        """Deployed close detector, evaluated once per accepted chunk.

        The result depends only on the chunk and its hand events, so scanning
        all 100 frames again on every control step would change nothing but
        the runtime.
        """
        env=self.env
        if env.chunk is None:return {}
        key=(id(env.chunk),tuple(id(env._adaptive_hand(s)) for s in SIDES))
        if self._boundary_key!=key:
            self._boundaries={s:int(closing_boundary(env.chunk,s,0.,env._adaptive_hand(s)))
                              for s in SIDES}
            self._boundary_key=key
        return self._boundaries

    def grasp_phase(self,now_ns):
        """Is the playback cursor inside a predicted hand-close event?

        The deployed close detector is reused unchanged. With no finger
        feedback a FOCUS window cannot open, so this geometric boundary is the
        only grasp signal either corpus can supply.
        """
        env=self.env
        if not self.running():return False
        cursor=env._elapsed(now_ns,env.chunk_started_ns)*float(env.chunk["fps"])
        return predicted_grasp_phase(env.chunk,cursor,{s:env._adaptive_hand(s) for s in SIDES},
                                     closing_boundary,self.close_boundaries())

    def sample(self,now_ns):
        """The reference the controller receives at this control step."""
        env=self.env
        if self.anti_stall and self.running():
            if max(env._adaptive_stall.budget.values())>0:self.stall_engaged_steps+=1
        result=env._sample_reference(now_ns)
        env.last_reference_sample=result
        return result

    def state(self,now_ns):
        plan=self.horizon(now_ns)
        return {"cursor_frame":self.cursor_frame(now_ns),
                "hold_active":self.hold_active(now_ns),
                "horizon_start_frame":self.horizon_start_frame(),
                "horizon_frame":None if plan is None else float(plan.frame),
                "horizon_reason":None if plan is None else plan.reason,
                "horizon_travel_m":None if plan is None else float(plan.travel_m),
                "stall_budget_m":{s:float(v) for s,v in self.env._adaptive_stall.budget.items()},
                "sequence_id":None if self.env.chunk is None else int(self.env.chunk["sequence_id"])}


def provenance():
    return {"vendor":VENDOR,"limitations":list(LIMITATIONS),
            "implementation":"deployed HAT4AdaptiveMotionTrackingEnv methods, run directly",
            "pure_boundary_cache":closing_boundary.report()}


__all__=["DeployedSwitching","make_chunk","measured_state","open_fingertips",
         "provenance","sample_chunk","LIMITATIONS"]
