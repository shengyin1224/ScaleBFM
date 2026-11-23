"""Body-target hooks over the deployed adaptive environment.

Nothing in the deployed switching path is altered: the hook reads the switch
event, the grasp phase and the stitched command stream, and writes only the
four body keypoint targets the cerebellum consumes, after every deployed
decision has already been made. Accepted chunks, finger commands and the
publishers are left exactly as they were.

The round-2 packaging bypassed A1 whenever a FOCUS handoff, a hand event or a
stall overlay was live, which dropped the residual to zero in one control step
and jolted the wrist. Here those states freeze the residual instead, and the
ramp back is owned by the executor.
"""
import copy
import numpy as np
import torch
from exec_comp.common import CHUNK,CHUNK_FPS
from exec_comp.geometry import pose9,qapply,qconj,qmul,six_to_q
from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from scalebridge.online.adaptive_chunk_execution import SIDES,closing_boundary
from exec_comp.round3.signals import predicted_grasp_phase

LINKS=("pelvis","torso_link","left_wrist_yaw_link","right_wrist_yaw_link")
HEAD_OFFSET=(0.,0.,.4)
A1_GRID=np.arange(CHUNK)/CHUNK_FPS


def assert_mutually_exclusive(config):
    """A1-5 item 6, checked before anything is built.

    The deployment wrist-compensation path and A1 close the same error. The
    pair is refused rather than silently stacked, and the refusal happens at
    configuration time so it cannot surprise anyone mid-run.
    """
    options=dict(config.get("exec_comp",{}) or {})
    compensation=str(config.get("wrist_compensation","none"))
    if bool(options.get("enable",False)) and compensation!="none":
        raise ValueError("exec_comp.enable and wrist_compensation are mutually exclusive; "
                         f"wrist_compensation is {compensation!r}")
    return options,compensation


class ExecCompRound3Hooks:
    # -- measurement ---------------------------------------------------------
    def _publish_robot_state(self):
        result=super()._publish_robot_state()
        adapter=getattr(self,"_exec_adapter",None)
        if adapter is not None and (adapter.enabled or getattr(adapter,'observing',False)):
            epoch=getattr(self,"reference_plan_epoch",0)
            if epoch!=getattr(self,"_exec_epoch",None):
                adapter.reset(preserve_residual=True);self._exec_epoch=epoch
            buffer=self.state_buffer
            adapter.observe(buffer["dof_pos_buffer"][0,-1],buffer["dof_vel_buffer"][0,-1],
                            buffer["root_pos_buffer"][0,-1],buffer["root_quat_wxyz_buffer"][0,-1])
        return result

    # -- read-only views of the deployed state -------------------------------
    def _exec_canonical_window(self):
        """The four body targets the cerebellum is about to receive."""
        positions=self.state_buffer["body_pos_w_future"][0]
        quaternions=self.state_buffer["body_quat_w_wxyz_future"][0]
        indices=[self.active_indices[name] for name in LINKS]
        positions=positions[:,indices].clone();quaternions=quaternions[:,indices]
        positions[:,1]+=qapply(quaternions[:,1],positions.new_tensor(HEAD_OFFSET).expand_as(positions[:,1]))
        return pose9(positions,quaternions)[None]

    def _exec_sample(self,now_ns,offsets_s,*,apply_stall_assist=True):
        """Run the deployed sampler on a shallow copy with other horizons.

        The copy shares the live chunks and clocks but cannot write back, so
        asking for the A1 window or for the lookahead-free reference never
        disturbs the control-step reference the robot is actually given.
        """
        proxy=copy.copy(self)
        proxy.future_offsets_s=np.asarray(offsets_s,dtype=np.float64)
        sample=HAT4AdaptiveMotionTrackingEnv._sample_reference(
            proxy,now_ns,apply_stall_assist=apply_stall_assist)
        positions=torch.tensor(np.stack([sample[k] for k in
            ("root_pos","torso_pos","left_wrist_pos","right_wrist_pos")],1),
            device=self.device,dtype=torch.float32)
        quaternions=torch.tensor(np.stack([sample[k] for k in
            ("root_quat_wxyz","torso_quat_wxyz","left_wrist_quat_wxyz","right_wrist_quat_wxyz")],1),
            device=self.device,dtype=torch.float32)
        positions[:,1]+=qapply(quaternions[:,1],positions.new_tensor(HEAD_OFFSET).expand_as(positions[:,1]))
        return pose9(positions,quaternions)[None]

    def _exec_grasp_phase(self,now_ns):
        """Any hand state in which the residual must hold still.

        A FOCUS handoff window, a live hand-event guard, and the brain's own
        predicted close boundary all qualify. The round-2 packaging treated
        exactly these as reasons to switch A1 off.
        """
        handoff=getattr(self,"focus_handoff",None)
        if handoff is not None and (handoff.active or handoff.bypass_sides):return True
        boundary=getattr(self,'hand_event_playback_boundary',None)
        if bool(self.cfg.get('continuity_hand_event_guard',False)) and boundary is not None and bool(boundary.holds):return True
        assist=getattr(self,"stall_wrist_assist",None)
        if assist is not None and any(state["mode"]!="watch" for state in assist.states.values()):return True
        chunk=getattr(self,"chunk",None)
        if chunk is None or getattr(self,"chunk_started_ns",None) is None:return False
        cursor=self._elapsed(now_ns,self.chunk_started_ns)*float(chunk["fps"])
        prediction=self._focus_prediction_chunks.get(chunk["sequence_id"],{})
        return predicted_grasp_phase(chunk,cursor,prediction.get('hand_focus',{}),closing_boundary)

    def _sample_reference(self,now_ns,*,apply_stall_assist=True):
        sample=super()._sample_reference(now_ns,apply_stall_assist=apply_stall_assist)
        if getattr(self,'_exec_capture_sampling',False) and apply_stall_assist:
            # Capture at the parent's sampling instant. Later in the original
            # gather method, a FOCUS exit can rebase its wrist chunk; sampling
            # again after that would mislabel the rebase as anti-stall offset.
            self._exec_raw_at_sample=self._exec_sample(now_ns,self.future_offsets_s,apply_stall_assist=False)
            adapter=getattr(self,'_exec_adapter',None)
            if adapter is not None and adapter.enabled:
                self._exec_window_at_sample=self._exec_sample(now_ns,A1_GRID,apply_stall_assist=False)
        return sample

    # -- correction ----------------------------------------------------------
    def _exec_write_body(self,corrected):
        positions=corrected[0,...,:3].clone();quaternions=six_to_q(corrected[0,...,3:])
        positions[:,1]-=qapply(quaternions[:,1],positions.new_tensor(HEAD_OFFSET).expand_as(positions[:,1]))
        self.state_buffer['body_pos_w_future']=self.state_buffer['body_pos_w_future'].clone()
        self.state_buffer['body_quat_w_wxyz_future']=self.state_buffer['body_quat_w_wxyz_future'].clone()
        for column,name in enumerate(LINKS):
            index=self.active_indices[name]
            self.state_buffer['body_pos_w_future'][0,:,index]=positions[:,column]
            self.state_buffer['body_quat_w_wxyz_future'][0,:,index]=quaternions[:,column]

    @torch.no_grad()
    def _gather_reference_state(self,now_ns):
        observer=getattr(self,'_exec_adapter',None)
        self._exec_raw_at_sample=None;self._exec_window_at_sample=None
        self._exec_capture_sampling=observer is not None and (observer.enabled or getattr(observer,'observing',False))
        try:result=super()._gather_reference_state(now_ns)
        finally:self._exec_capture_sampling=False
        adapter=getattr(self,"_exec_adapter",None)
        if adapter is None:return result
        if not adapter.enabled:
            if not getattr(adapter,'observing',False) or not adapter.history:return result
            # Logging-only off path. Never write body/finger buffers or run a
            # correction. The original parent completed its reference first.
            deployed=self._exec_canonical_window()
            active=bool(self._reference_plan_active()) and self.chunk is not None
            raw=self._exec_raw_at_sample if active else deployed
            if raw is None:raw=self._exec_sample(now_ns,self.future_offsets_s,apply_stall_assist=False)
            sequence=getattr(self,'last_chunk_sequence_id',None)
            switched=sequence is not None and sequence!=getattr(self,'_exec_sequence',None)
            self._exec_sequence=sequence
            grasp=self._exec_grasp_phase(now_ns) if active else False
            lookahead=(deployed[...,:3]-raw[...,:3]).norm(dim=-1)*100
            adapter.log_step(raw=raw[0,0],corrected=deployed[0,0],actual=adapter.history[-1]['kp'],
                telemetry={'lookahead_cm':lookahead[:,0]},chunk_sequence_id=sequence,
                switched=switched,grasp_phase=grasp,notes='A1 off; read-only baseline logger')
            return result
        if not adapter.history:return result
        deployed=self._exec_canonical_window()
        sequence=getattr(self,"last_chunk_sequence_id",None)
        switched=sequence is not None and sequence!=getattr(self,"_exec_sequence",None)
        self._exec_sequence=sequence
        active=bool(self._reference_plan_active()) and self.chunk is not None
        if not active:
            corrected,telemetry=adapter.fade_to_baseline(deployed)
            self._exec_write_body(corrected)
            adapter.log_step(raw=deployed[0,0],corrected=corrected[0,0],
                             actual=adapter.history[-1]["kp"],telemetry=telemetry,
                             chunk_sequence_id=sequence,notes="reference_not_playing")
            return result
        grasp=self._exec_grasp_phase(now_ns)
        raw=self._exec_raw_at_sample
        window=self._exec_window_at_sample
        if raw is None:raw=self._exec_sample(now_ns,self.future_offsets_s,apply_stall_assist=False)
        if window is None:window=self._exec_sample(now_ns,A1_GRID,apply_stall_assist=False)
        cursor=self._elapsed(now_ns,self.chunk_started_ns)*float(self.chunk["fps"])
        padded=cursor+A1_GRID[-1]*float(self.chunk["fps"])>int(self.chunk["frame_count"])-1
        corrected,telemetry=adapter.correct(deployed,raw,window,switched=switched,
                                            grasp_phase=grasp,window_padded=padded)
        # Write back only the four body targets, after every deployed decision.
        self._exec_write_body(corrected)
        adapter.log_step(raw=raw[0,0],corrected=corrected[0,0],
                         actual=adapter.history[-1]["kp"],telemetry=telemetry,
                         chunk_sequence_id=sequence,switched=switched,grasp_phase=grasp)
        return result

    def close_online(self):
        adapter=getattr(self,"_exec_adapter",None)
        if adapter is not None:adapter.close()
        return super().close_online()
