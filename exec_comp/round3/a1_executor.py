"""A1 as one independent stage: after chunk switching, before the cerebellum.

Everything here hangs off a single ``enable`` switch. The deployed switching
code decides what the reference is; this stage only adds a bounded residual to
it and never feeds anything back into PACE, continuity matching, the anti-stall
lookahead or the grasp protection -- those all run on the brain's raw K*.

Four behaviours distinguish it from the round-2 packaging:

* two residual call modes, stepwise at 50 Hz and one per real switch event;
* a grasp phase freezes the residual and ramps back over 150 ms, instead of
  bypassing A1 and dropping the residual to zero in one control step;
* the anti-stall lookahead and the residual share one 5 cm deviation budget
  measured against raw K*, so the two compensations cannot stack;
* NaN, a sustained cap, or a large single-step jump ramp the residual to zero
  over 150 ms and are logged, rather than being cut instantly.

``enable=False`` returns the deployed reference object untouched, so the whole
chain is bit-identical to running without this module.
"""
from dataclasses import dataclass,field
import copy
import torch
from exec_comp.common import CHUNK,CHUNK_FPS,DT,ROOT
from exec_comp.geometry import (from_anchor,heading_anchor,pose9,pose_normalize,
                                qconj,qmul,qnormalize,six_to_q,to_anchor)
from exec_comp.hat_playback import quaternion_slerp
from exec_comp.runtime import correct_encoded
from exec_comp.train_common import load_model

MODES=("stepwise","switch_event")
# A1 was trained on a 100-frame window spaced at 30 Hz. The deployed chunk is
# 100 frames at 15 Hz, so the window is resampled rather than the model being
# re-trained: this stage consumes the stitched command stream, not a chunk.
A1_WINDOW_DT=1./CHUNK_FPS


def identity_quat(*shape,device):
    quat=torch.zeros(*shape,4,device=device);quat[...,0]=1.;return quat


@dataclass
class ExecutorLimits:
    total_offset_cap_m:float=.05
    grasp_release_ramp_s:float=.15
    seam_crossfade_s:float=.12
    fallback_ramp_s:float=.15
    cap_hold_s:float=1.
    max_step_cm:float=1.
    lowpass:bool=True
    # Which reading of "the residual has been at the 5 cm limit for a second"
    # the fallback uses. "total" is the literal rule: lookahead plus residual
    # sitting on the shared budget. A1's own residual bound is also 5 cm, so a
    # raw error of 5 cm or more saturates it legitimately and the literal rule
    # fires during ordinary operation; "limited" instead requires the shared
    # budget to be actively truncating A1, and "off" disables this trigger.
    cap_guard:str="residual"


@dataclass
class ExecutorCounters:
    steps:int=0
    active_env_steps:int=0
    model_calls:int=0
    frozen_steps:int=0
    crossfade_steps:int=0
    ramp_steps:int=0
    cap_limited_steps:int=0
    window_padded_steps:int=0
    fallback_steps:int=0
    fallbacks:dict=field(default_factory=lambda:{"nan":0,"sustained_cap":0,"step_jump":0,'reference_reset':0})


class A1Executor:
    """Batched residual stage. ``batch`` is the number of simulated robots."""

    def __init__(self,*,enable,batch,device,mode="stepwise",checkpoint=None,
                 limits=None,wrist_compensation="none",tokens=6,predictor=None):
        """``predictor(state_hist, window) -> (delta_pos, delta_quat)``.

        Default is A1. B1's amortized network plugs in here, so both are
        measured by one stage with one set of limits and one fallback.
        """
        self.enabled=bool(enable);self.batch=int(batch);self.device=device
        self.tokens=int(tokens)
        self.limits=limits or ExecutorLimits()
        self.counters=ExecutorCounters()
        self.checkpoint=None;self.predictor=predictor
        if not self.enabled:
            self.mode=None;self.model=None;return
        if mode not in MODES:raise ValueError(f"exec_comp.mode must be one of {MODES}, got {mode!r}")
        # A1-5 item 6: the deployment wrist-compensation path and A1 correct the
        # same error; running both is refused rather than silently stacked.
        if str(wrist_compensation)!="none":
            raise ValueError("exec_comp.enable requires wrist_compensation=none; "
                             f"the deployed path is set to {wrist_compensation!r}")
        self.mode=mode
        if predictor is None:
            self.checkpoint=str(checkpoint or ROOT/"runs/models/inverse_smooth_r2_best.pt")
            self.model=load_model(self.checkpoint,device).requires_grad_(False)
            if int(self.model.chunk)!=CHUNK:raise ValueError(f"A1 expects {CHUNK} target frames")
        else:
            self.checkpoint=str(checkpoint) if checkpoint else getattr(predictor,"checkpoint",None)
            self.model=getattr(predictor,"model",None)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self,ids=None):
        if not self.enabled:return
        batch,tokens,device=self.batch,self.tokens,self.device
        if ids is None:
            self.delta_pos=torch.zeros(batch,CHUNK,4,3,device=device)
            self.delta_quat=identity_quat(batch,CHUNK,4,device=device)
            self.delta_age_s=torch.zeros(batch,device=device)
            self.previous_pos=torch.zeros(batch,CHUNK,4,3,device=device)
            self.previous_quat=identity_quat(batch,CHUNK,4,device=device)
            self.previous_age_s=torch.zeros(batch,device=device)
            self.crossfade_left_s=torch.zeros(batch,device=device)
            self.frozen=torch.zeros(batch,dtype=torch.bool,device=device)
            # One "hold this profile, then ramp away from it" mechanism serves
            # both grasp exit and fallback recovery, so neither can step.
            self.hold_pos=torch.zeros(batch,tokens,4,3,device=device)
            self.hold_quat=identity_quat(batch,tokens,4,device=device)
            self.ramp_left_s=torch.zeros(batch,device=device)
            self.ramp_total_s=torch.ones(batch,device=device)
            self.fallback_left_s=torch.zeros(batch,device=device)
            self.fallback_start_pos=torch.zeros(batch,tokens,4,3,device=device)
            self.fallback_start_quat=identity_quat(batch,tokens,4,device=device)
            self.cap_streak_s=torch.zeros(batch,device=device)
            self.have_delta=torch.zeros(batch,dtype=torch.bool,device=device)
            self.lowpass_state=torch.zeros(batch,4,3,device=device)
            self.applied_pos=torch.zeros(batch,tokens,4,3,device=device)
            self.applied_quat=identity_quat(batch,tokens,4,device=device)
            self.applied_valid=torch.zeros(batch,dtype=torch.bool,device=device)
            return
        index=torch.as_tensor(ids,device=device,dtype=torch.long)
        for name in ("delta_pos","previous_pos","hold_pos","lowpass_state","applied_pos","fallback_start_pos"):
            getattr(self,name)[index]=0.
        for name in ("delta_quat","previous_quat","hold_quat","applied_quat","fallback_start_quat"):
            tensor=getattr(self,name);tensor[index]=0.;tensor[index,...,0]=1.
        for name in ("delta_age_s","previous_age_s","crossfade_left_s","ramp_left_s",
                     "fallback_left_s","cap_streak_s"):
            getattr(self,name)[index]=0.
        self.ramp_total_s[index]=1.
        for name in ("frozen","have_delta","applied_valid"):
            getattr(self,name)[index]=False

    # -- model ---------------------------------------------------------------
    def fork(self):
        """Independent controller state, shared frozen predictor (for seam audit)."""
        other=copy.copy(self)
        for name,value in vars(self).items():
            if torch.is_tensor(value):setattr(other,name,value.clone())
        other.counters=copy.deepcopy(self.counters)
        return other

    @torch.no_grad()
    def _predict(self,state_hist,window):
        """Residual over the 100x30Hz stitched command window."""
        if self.predictor is not None:
            self.counters.model_calls+=1
            return self.predictor(state_hist,window)
        origin,quat=heading_anchor(state_hist["root_pos"],state_hist["root_quat"])
        keypoints=to_anchor(state_hist["kp"],origin,quat)
        encoded=torch.cat((keypoints.flatten(2),state_hist["q"],state_hist["dq"]),-1).flatten(1)
        corrected=from_anchor(correct_encoded(self.model,encoded,to_anchor(window,origin,quat)),origin,quat)
        self.counters.model_calls+=1
        rotation=qmul(six_to_q(corrected[...,3:]),qconj(six_to_q(window[...,3:])))
        return corrected[...,:3]-window[...,:3],qnormalize(rotation)

    def _sample_delta(self,positions,quaternions,age_s,offsets_s):
        """Sample a stored residual plan at its own age plus the policy horizons."""
        frames=((age_s[:,None]+offsets_s[None])/A1_WINDOW_DT).clamp(0.,CHUNK-1.)
        lower=frames.floor().long();upper=(lower+1).clamp_max(CHUNK-1)
        rows=torch.arange(len(positions),device=positions.device)[:,None]
        weight=(frames-lower)[...,None,None]
        low=positions[rows,lower];high=positions[rows,upper]
        position=low+(high-low)*weight
        rotation=quaternion_slerp(quaternions[rows,lower],quaternions[rows,upper],weight)
        return position,rotation

    def _arm_ramp(self,mask,seconds):
        """Ramp from the currently held residual to the live one."""
        self.ramp_left_s=torch.where(mask,torch.full_like(self.ramp_left_s,seconds),self.ramp_left_s)
        self.ramp_total_s=torch.where(mask,torch.full_like(self.ramp_total_s,max(seconds,1e-9)),self.ramp_total_s)

    # -- main ----------------------------------------------------------------
    @torch.no_grad()
    def __call__(self,*,deployed_pose,raw_pose,lookahead_offset,state_hist,window,
                 switched,grasp_phase,window_padded=None,offsets_s=None,dt=DT,active_mask=None,force_fallback=None):
        """Return the command the cerebellum receives at this control step.

        ``deployed_pose``   [B,F,4,9] what deployment would send (lookahead included)
        ``raw_pose``        [B,F,4,9] the same reference without the lookahead
        ``lookahead_offset``[B,F,4,3] deployed minus raw positions
        ``window``          [B,100,4,9] stitched K* at 30 Hz, the A1 input
        ``switched``        [B] a chunk switch was accepted on this step
        ``grasp_phase``     [B] the hand is inside a close/handoff phase
        """
        if not self.enabled:return deployed_pose,{}
        counters=self.counters;counters.steps+=1
        active_mask=(torch.ones(self.batch,device=self.device,dtype=torch.bool)
                     if active_mask is None else active_mask.to(self.device))
        counters.active_env_steps+=int(active_mask.sum())
        tokens=deployed_pose.shape[1]
        if tokens!=self.tokens:raise ValueError(f"Executor built for {self.tokens} future tokens, got {tokens}")
        if offsets_s is None:offsets_s=torch.arange(tokens,device=self.device,dtype=torch.float32)*dt
        switched=switched.to(self.device);grasp_phase=grasp_phase.to(self.device)
        if window_padded is not None:counters.window_padded_steps+=int((window_padded&active_mask).sum())
        self.delta_age_s+=dt;self.previous_age_s+=dt

        # A1-5 item 4. Entering a grasp phase freezes whatever is applied now;
        # leaving it ramps back to the live residual over 150 ms. The residual
        # is never set to zero in one step, which is what jolted the wrist in
        # the round-2 bypass.
        entering=grasp_phase&(~self.frozen)&self.applied_valid
        leaving=(~grasp_phase)&self.frozen
        if bool(entering.any()):
            self.hold_pos[entering]=self.applied_pos[entering]
            self.hold_quat[entering]=self.applied_quat[entering]
            self.ramp_left_s=torch.where(entering,torch.zeros_like(self.ramp_left_s),self.ramp_left_s)
        self._arm_ramp(leaving,self.limits.grasp_release_ramp_s)
        self.frozen=grasp_phase.clone()

        # What the plan in force before this step says about right now. The
        # seam measurement compares it with the refreshed plan at the same
        # instant, which is the only way to separate a switch discontinuity
        # from ordinary motion between two control steps.
        # The outgoing command already contains filtering/freeze/fallback.
        # Advance that issued profile by one control step for a same-time seam
        # comparison; never compare the new safe output with an unfiltered MLP.
        previous_position=torch.cat((self.applied_pos[:,1:],self.applied_pos[:,-1:]),1)
        previous_position=torch.where(self.applied_valid[:,None,None,None],previous_position,
                                      torch.zeros_like(previous_position))

        # A1-5 item 2: the model only ever sees the raw K* stream; switching has
        # already happened and is never re-decided from a corrected command.
        stepwise=torch.full_like(switched,self.mode=="stepwise")
        refresh=active_mask&(~grasp_phase)&(stepwise|switched|leaving|(~self.have_delta))
        if bool(refresh.any()):
            new_pos,new_quat=self._predict({k:v[refresh] for k,v in state_hist.items()},window[refresh])
            if self.mode=="switch_event":
                starting=refresh&self.have_delta&switched
                if bool(starting.any()):
                    self.previous_pos[starting]=self.delta_pos[starting]
                    self.previous_quat[starting]=self.delta_quat[starting]
                    self.previous_age_s[starting]=self.delta_age_s[starting]
                    self.crossfade_left_s[starting]=self.limits.seam_crossfade_s
            self.delta_pos[refresh]=new_pos;self.delta_quat[refresh]=new_quat
            self.delta_age_s[refresh]=0.;self.have_delta[refresh]=True

        position,rotation=self._sample_delta(self.delta_pos,self.delta_quat,self.delta_age_s,offsets_s)
        if self.mode=="switch_event":
            # A1-5 item 3 (ii): the crossfade is triggered by a real switch
            # event, not by a fixed cadence.
            active=self.crossfade_left_s>0
            if bool(active.any()):
                counters.crossfade_steps+=int((active&active_mask).sum())
                old_pos,old_quat=self._sample_delta(self.previous_pos,self.previous_quat,
                                                    self.previous_age_s,offsets_s)
                alpha=(1.-self.crossfade_left_s/max(self.limits.seam_crossfade_s,1e-9)).clamp(0.,1.)
                blend=torch.where(active,alpha,torch.ones_like(alpha))[:,None,None,None]
                position=old_pos+(position-old_pos)*blend
                rotation=quaternion_slerp(old_quat,rotation,blend)
            self.crossfade_left_s=(self.crossfade_left_s-dt).clamp_min(0.)
        position=torch.where(self.have_delta[:,None,None,None],position,torch.zeros_like(position))
        rotation=torch.where(self.have_delta[:,None,None,None],rotation,
                             identity_quat(*rotation.shape[:-1],device=self.device))

        if self.limits.lowpass:
            # 20 ms residual low pass at 50 Hz, as validated in round 2.
            filtered=[];state=self.lowpass_state
            for index in range(tokens):
                state=.5*position[:,index]+.5*state;filtered.append(state)
            position=torch.stack(filtered,1)

        frozen=self.frozen[:,None,None,None]
        if bool(self.frozen.any()):
            counters.frozen_steps+=int((self.frozen&active_mask).sum())
            position=torch.where(frozen,self.hold_pos,position)
            rotation=torch.where(frozen,self.hold_quat,rotation)
        ramping=(self.ramp_left_s>0)&(~self.frozen)
        if bool(ramping.any()):
            counters.ramp_steps+=int((ramping&active_mask).sum())
            alpha=(1.-self.ramp_left_s/self.ramp_total_s).clamp(0.,1.)
            blend=torch.where(ramping,alpha,torch.ones_like(alpha))[:,None,None,None]
            position=self.hold_pos+(position-self.hold_pos)*blend
            rotation=quaternion_slerp(self.hold_quat,rotation,blend)
        self.ramp_left_s=torch.where(self.frozen,self.ramp_left_s,(self.ramp_left_s-dt).clamp_min(0.))

        # A1-5 item 5: the lookahead offset and the residual share one budget.
        # Project the VECTOR SUM. Opposing corrections are allowed to cancel;
        # subtracting their lengths would discard useful compensation.
        candidate_total=lookahead_offset+position
        norm=candidate_total.norm(dim=-1,keepdim=True)
        scale=torch.minimum(torch.ones_like(norm),self.limits.total_offset_cap_m/norm.clamp_min(1e-9))
        limited=(scale<1.-1e-6).flatten(1).any(-1)
        counters.cap_limited_steps+=int((limited&active_mask).sum())
        position=candidate_total*scale-lookahead_offset
        total=(lookahead_offset+position).norm(dim=-1)
        if self.limits.cap_guard=='residual':
            at_cap=(position[:,0].norm(dim=-1)>=self.limits.total_offset_cap_m-1e-4).any(-1)
        elif self.limits.cap_guard=="total":
            at_cap=(total[:,0]>=self.limits.total_offset_cap_m-1e-4).any(-1)
        elif self.limits.cap_guard=="limited":at_cap=limited
        elif self.limits.cap_guard=="off":at_cap=torch.zeros_like(limited)
        else:raise ValueError(f"Unknown cap_guard {self.limits.cap_guard!r}")
        self.cap_streak_s=torch.where(at_cap&active_mask,self.cap_streak_s+dt,torch.zeros_like(self.cap_streak_s))

        # A1-6 item 2 safety fallback.
        finite=(torch.isfinite(position).flatten(1).all(-1)
                &torch.isfinite(rotation).flatten(1).all(-1))
        trigger_nan=~finite
        trigger_cap=self.cap_streak_s>=self.limits.cap_hold_s
        step=(position[:,0]-self.applied_pos[:,0]).norm(dim=-1).max(-1).values*100.
        trigger_step=self.applied_valid&finite&(step>self.limits.max_step_cm)
        trigger_reset=(torch.zeros_like(trigger_step) if force_fallback is None else force_fallback.to(self.device))
        trigger_reset=trigger_reset&self.applied_valid&(self.applied_pos.flatten(1).norm(dim=-1)>1e-7)
        idle=(self.fallback_left_s<=0)&active_mask
        for name,flag in (("nan",trigger_nan),("sustained_cap",trigger_cap),("step_jump",trigger_step),('reference_reset',trigger_reset)):
            counters.fallbacks[name]+=int((flag&idle).sum())
        trigger=(trigger_nan|trigger_cap|trigger_step|trigger_reset)&idle
        if bool(trigger.any()):
            valid=trigger&self.applied_valid
            self.fallback_start_pos[trigger]=0.
            self.fallback_start_quat[trigger]=identity_quat(int(trigger.sum()),tokens,4,device=self.device)
            self.fallback_start_pos[valid]=self.applied_pos[valid]
            self.fallback_start_quat[valid]=self.applied_quat[valid]
        self.fallback_left_s=torch.where(trigger,
            torch.full_like(self.fallback_left_s,self.limits.fallback_ramp_s),self.fallback_left_s)
        falling=self.fallback_left_s>0
        if bool(falling.any()):
            counters.fallback_steps+=int((falling&active_mask).sum())
            # Fade the LAST ISSUED finite command, independent of subsequent
            # invalid/oversized predictions. Include the trigger sample at
            # progress0 and reach exact zero within the configured duration.
            duration=max(self.limits.fallback_ramp_s-dt,dt)
            faded=torch.where(falling,((self.limits.fallback_ramp_s-self.fallback_left_s)/duration).clamp(0.,1.),
                              torch.zeros_like(self.fallback_left_s))[:,None,None,None]
            neutral=identity_quat(*rotation.shape[:-1],device=self.device)
            position=torch.where(falling[:,None,None,None],self.fallback_start_pos*(1.-faded),position)
            faded_rotation=quaternion_slerp(self.fallback_start_quat,neutral,faded)
            rotation=torch.where(falling[:,None,None,None],faded_rotation,rotation)
            self.cap_streak_s=torch.where(falling,torch.zeros_like(self.cap_streak_s),self.cap_streak_s)
            self.fallback_left_s=(self.fallback_left_s-dt).clamp_min(0.)
            recovered=falling&(self.fallback_left_s<=0)
            if bool(recovered.any()):
                # Rejoin the live residual through the same ramp, never a step.
                self.have_delta[recovered]=False
                self.lowpass_state[recovered]=0.
                self.hold_pos[recovered]=0.;self.hold_quat[recovered]=0.;self.hold_quat[recovered,...,0]=1.
                self.ramp_left_s=torch.where(recovered,
                    torch.full_like(self.ramp_left_s,self.limits.grasp_release_ramp_s),self.ramp_left_s)
                self.ramp_total_s=torch.where(recovered,
                    torch.full_like(self.ramp_total_s,max(self.limits.grasp_release_ramp_s,1e-9)),self.ramp_total_s)

        # A changing lookahead budget still cannot violate the shared cap.
        combined=lookahead_offset+position
        position=combined*torch.minimum(torch.ones_like(combined[...,:1]),
            self.limits.total_offset_cap_m/combined.norm(dim=-1,keepdim=True).clamp_min(1e-9))-lookahead_offset
        total=(lookahead_offset+position).norm(dim=-1)
        self.applied_pos=position.clone();self.applied_quat=rotation.clone();self.applied_valid[:]=True
        if self.limits.lowpass:self.lowpass_state=position[:,0].clone()
        command=pose9(raw_pose[...,:3]+lookahead_offset+position,
                      qnormalize(qmul(rotation,six_to_q(raw_pose[...,3:]))))
        telemetry={"residual_cm":position.norm(dim=-1)[:,0]*100.,
                   "lookahead_cm":lookahead_offset.norm(dim=-1)[:,0]*100.,
                   "total_offset_cm":total[:,0]*100.,
                   "frozen":self.frozen.clone(),"fallback":falling.clone(),
                   "cap_limited":limited,
                   "residual_position":position,"previous_residual_position":previous_position}
        return pose_normalize(command),telemetry

    def report(self):
        counters=self.counters
        return {"enabled":self.enabled,"mode":self.mode,"checkpoint":self.checkpoint,
                "steps":counters.steps,"model_calls":counters.model_calls,
                'active_env_steps':counters.active_env_steps,
                "model_calls_per_step":counters.model_calls/max(1,counters.steps),
                "frozen_env_steps":counters.frozen_steps,
                "crossfade_env_steps":counters.crossfade_steps,
                "ramp_env_steps":counters.ramp_steps,
                "cap_limited_env_steps":counters.cap_limited_steps,
                "window_padded_env_steps":counters.window_padded_steps,
                "fallback_env_steps":counters.fallback_steps,
                "fallback_events":dict(counters.fallbacks),
                "limits":vars(self.limits),
                "a1_window":"100 frames spaced at 30 Hz over the stitched command stream; "
                            "frames past the chunk end repeat its last frame"}
