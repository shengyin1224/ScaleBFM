"""Batch-1 A1 stage for the robot. Importing this never opens an interface.

The simulation and the robot run the same ``A1Executor``; only the plumbing
around it differs. That is deliberate: the grasp freeze, the shared cap and the
safety fallback are the behaviours A1-5 measured, so they must not be a second
implementation here.
"""
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import torch
from exec_comp.common import ROOT,digest,metadata
from exec_comp.diff.fk import FK
from exec_comp.geometry import from_anchor,six_to_q,qmul,qconj,qnormalize
from exec_comp.round3.a1_executor import A1Executor,ExecutorLimits

HISTORY_FRAMES=10


class ScriptPredictor:
    """Use the released TorchScript itself, rather than a hidden raw checkpoint."""
    def __init__(self,artifact,device):
        self.checkpoint=str(artifact);self.model=torch.jit.load(str(artifact),map_location=device).eval()

    @torch.no_grad()
    def __call__(self,history,window):
        if len(window)!=1:raise ValueError('The released deployment interface is batch1')
        length=history['q'].shape[1]
        state={k:history[k][0] for k in ('q','dq','kp')}
        state['root_pos']=history['root_pos'][0,None].expand(length,-1)
        state['root_quat']=history['root_quat'][0,None].expand(length,-1)
        corrected=self.model.correct_kp(state,window[0])[None]
        return corrected[...,:3]-window[...,:3],qnormalize(qmul(six_to_q(corrected[...,3:]),qconj(six_to_q(window[...,3:]))))


class Round3Adapter:
    def __init__(self,options,controller_metadata=None,device="cuda:0",
                 wrist_compensation="none"):
        self.enabled=bool(options.get("enable",False));self.device=device
        self.options=dict(options)
        self.observing=self.enabled or bool(options.get('log_baseline',False))
        self.history=deque(maxlen=HISTORY_FRAMES)
        self.log_file=None;self.previous_fallback=False;self.pending_reset=False
        self.run_context=options.get('run_context','deployment')
        self.log_dir=Path(options.get('log_dir',ROOT/'runs/round3/deployment_logs'))
        if self.observing:self.fk=FK().to(device)
        if not self.enabled:
            self.executor=None;self.manifest={'artifact_sha256':None};return
        self.artifact=Path(options["model"])
        self.manifest=json.loads(self.artifact.with_name("manifest.json").read_text())
        if digest(self.artifact)!=self.manifest["artifact_sha256"]:
            raise ValueError("A1 artifact hash mismatch")
        cerebellum=options.get("cerebellum_model")
        if cerebellum and digest(cerebellum)!=self.manifest.get("cerebellum_deployment_sha256"):
            raise ValueError("A1 is bound to another cerebellum checkpoint")
        if controller_metadata:
            for key in ("joint_names","selected_body_names","action_scale","default_dof_pos",
                        "stiffness","damping","torque_limit"):
                if controller_metadata[key]!=metadata()[key]:
                    raise ValueError(f"Controller metadata differs: {key}")
        limits=ExecutorLimits(
            total_offset_cap_m=float(options.get("total_offset_cap_m",.05)),
            grasp_release_ramp_s=float(options.get("grasp_release_ramp_s",.15)),
            seam_crossfade_s=float(options.get("seam_crossfade_s",.12)),
            fallback_ramp_s=float(options.get("fallback_ramp_s",.15)),
            cap_hold_s=float(options.get("cap_hold_s",1.)),
            max_step_cm=float(options.get("max_step_cm",1.)),
            lowpass=bool(options.get("residual_lowpass",True)),
            cap_guard=str(options.get("cap_guard","residual")))
        self.executor=A1Executor(enable=True,batch=1,device=device,
                                 mode=str(options.get("mode","stepwise")),
                                 checkpoint=self.artifact,predictor=ScriptPredictor(self.artifact,device),
                                 limits=limits,wrist_compensation=wrist_compensation)
        self.fk=FK().to(device);self.history=deque(maxlen=HISTORY_FRAMES)
        self.log_dir=Path(options.get("log_dir",ROOT/"runs/round3/deployment_logs"))
        self.log_file=None;self.run_context=options.get("run_context","deployment")
        self.previous_fallback=False
        self.pending_reset=False

    # -- measurement ---------------------------------------------------------
    @torch.no_grad()
    def observe(self,q,dq,root_pos,root_quat):
        if not self.observing:return
        values=[torch.as_tensor(x,device=self.device,dtype=torch.float32).reshape(-1)
                for x in (q,dq,root_pos,root_quat)]
        q,dq,position,quat=values
        keypoints=from_anchor(self.fk.keypoints(q[None])[0],position,quat)
        self.history.append({"q":q.clone(),"dq":dq.clone(),"kp":keypoints.clone(),
                             "root_pos":position.clone(),"root_quat":quat.clone()})

    def measured_history(self):
        if not self.history:raise RuntimeError("No measured body history")
        padded=[self.history[0]]*(HISTORY_FRAMES-len(self.history))+list(self.history)
        stacked={k:torch.stack([frame[k] for frame in padded]) for k in padded[0]}
        return {"q":stacked["q"][None],"dq":stacked["dq"][None],"kp":stacked["kp"][None],
                "root_pos":stacked["root_pos"][-1][None],"root_quat":stacked["root_quat"][-1][None]}

    def reset(self,preserve_residual=False):
        if not self.observing:return
        self.history.clear()
        if not self.enabled:return
        if preserve_residual:self.pending_reset=True
        else:self.executor.reset();self.pending_reset=False

    # -- correction ----------------------------------------------------------
    @torch.no_grad()
    def correct(self,deployed_pose,raw_pose,window,*,switched,grasp_phase,window_padded=False,force_fallback=False):
        """deployed/raw [1,F,4,9]; window [1,100,4,9] on the 30 Hz A1 grid."""
        if not self.enabled or not self.history:return deployed_pose,{}
        if not bool(self.options.get('keep_lookahead',True)):deployed_pose=raw_pose
        flag=lambda value:torch.tensor([bool(value)],device=self.device)
        fade=self.pending_reset or force_fallback;self.pending_reset=False
        return self.executor(
            deployed_pose=deployed_pose,raw_pose=raw_pose,
            lookahead_offset=deployed_pose[...,:3]-raw_pose[...,:3],
            state_hist=self.measured_history(),window=window,
            switched=flag(switched),grasp_phase=flag(grasp_phase),
            window_padded=flag(window_padded),force_fallback=flag(fade))

    def fade_to_baseline(self,deployed_pose):
        window=deployed_pose[:,:1].expand(-1,100,-1,-1)
        return self.correct(deployed_pose,deployed_pose,window,switched=False,grasp_phase=True,force_fallback=True)

    # -- logging -------------------------------------------------------------
    def log_step(self,*,raw,corrected,actual,telemetry,chunk_sequence_id=None,
                 switched=False,grasp_phase=False,notes=None):
        """A1-6 item 3: one row per control step, every required channel."""
        if not self.observing:return
        if self.log_file is None:
            folder=self.log_dir/(time.strftime("%Y%m%d_%H%M%S")+"_"+str(time.time_ns()))
            folder.mkdir(parents=True,exist_ok=False)
            self.log_file=(folder/"control_steps.jsonl").open("w",buffering=8192)
            (folder/"schema.json").write_text(json.dumps({
                "control_hz":50,"tracking_alignment_control_steps":0,"a1_enabled":self.enabled,
                "keypoints":["root","head","left_wrist","right_wrist"],"coordinates":"world",
                "actual_timestamp":"measured state at t before action; instantaneous tracking compares raw_position[t] to actual_position[t]. Command response needs the appropriate future token, not a blind one-step shift.",
                "fields":{"raw_position":"brain K* after switching, before anti-stall lookahead and A1",
                          "corrected_position":"what the cerebellum received",
                          "actual_position":"measured keypoints",
                          "switch_event":"the deployed switcher accepted a chunk on this step",
                          "grasp_phase":"inside a close event or FOCUS window; the residual is frozen",
                          "fallback":"the safety fade is running",
                          "residual_cm":"per-keypoint applied residual",
                          "lookahead_cm":"per-keypoint anti-stall lookahead offset; a stall "
                                         "event is this going from zero to positive",
                          "total_offset_cm":"lookahead plus residual against raw K*"},
                "run_context":self.run_context,
                "artifact_sha256":self.manifest["artifact_sha256"]},indent=2))
        def positions(value):
            if torch.is_tensor(value):value=value.detach().cpu().numpy()
            return np.asarray(value)[...,:3].round(5).tolist()
        def scalar(key):
            value=telemetry.get(key)
            if value is None:return None
            if torch.is_tensor(value):
                value=value.detach().cpu().numpy()
                return value.reshape(-1).round(4).tolist() if value.size>1 else bool(value) if value.dtype==bool else float(value)
            return value
        fallback=bool(np.any(telemetry.get("fallback").detach().cpu().numpy())) if "fallback" in telemetry else False
        row={"a1_enabled":self.enabled,"monotonic_ns":time.monotonic_ns(),"wall_time_ns":time.time_ns(),
             "chunk_sequence_id":chunk_sequence_id,"switch_event":bool(switched),
             "grasp_phase":bool(grasp_phase),"fallback":fallback,
             "raw_position":positions(raw),"corrected_position":positions(corrected),
             "actual_position":positions(actual),
             "residual_cm":scalar("residual_cm"),"lookahead_cm":scalar("lookahead_cm"),
             "total_offset_cm":scalar("total_offset_cm"),"notes":notes}
        if fallback and not self.previous_fallback:
            row["fallback_counters"]=dict(self.executor.counters.fallbacks)
        self.previous_fallback=fallback
        self.log_file.write(json.dumps(row,separators=(",",":"))+"\n")

    def close(self):
        if self.observing and self.log_file:self.log_file.close();self.log_file=None
