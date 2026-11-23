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
from exec_comp.round4.executor import Round4Executor as A1Executor,ExecutorLimits
from exec_comp.round4.script_predictor import ScriptPredictor
from exec_comp.round4.device_history import DeviceHistory
from exec_comp.round4 import context

HISTORY_FRAMES=10


class Round4Adapter:
    def __init__(self,options,controller_metadata=None,device="cuda:0",
                 wrist_compensation="none"):
        self.enabled=bool(options.get("enable",False));self.device=device
        self.options=dict(options)
        self.method=str(options.get('method','b1'))
        if self.enabled and self.method not in ('a1','b1'):raise ValueError('exec_comp.method must be a1 or b1')
        if self.enabled and options.get('fallback_off'):raise ValueError('fallback_off is simulation-only')
        context.OPTIONS.clear();context.OPTIONS.update(population='deployment',method=self.method,
            coefficient_transition=bool(options.get('coefficient_transition',False)),
            activation_ramp=bool(options.get('activation_ramp',False)))
        self.observing=self.enabled or bool(options.get('log_baseline',False))
        self.history=deque(maxlen=HISTORY_FRAMES)
        self.log_file=None;self.previous_fallback=False;self.pending_reset=False
        self.run_context=options.get('run_context','deployment')
        self.log_dir=Path(options.get('log_dir',ROOT/'runs/round4/deployment_logs'))
        if self.observing:self.device_history=DeviceHistory(device)
        if not self.enabled:
            self.executor=None;self.manifest={'artifact_sha256':None};return
        self.artifact=Path(options.get("model") or options["models"][self.method])
        self.manifest=json.loads(self.artifact.with_name("manifest.json").read_text())
        if self.manifest.get('method',self.method)!=self.method:
            raise ValueError('exec_comp.method differs from artifact manifest')
        for option in ('coefficient_transition','activation_ramp'):
            if option in self.manifest and bool(options.get(option,False))!=self.manifest[option]:
                raise ValueError(f'{option} differs from artifact manifest')
        for filename,expected in self.manifest.get('runtime_source_sha256',{}).items():
            if digest(ROOT/'round4'/filename)!=expected:
                raise ValueError(f'Execution runtime source hash mismatch: {filename}')
        if digest(self.artifact)!=self.manifest["artifact_sha256"]:
            raise ValueError("Execution artifact hash mismatch")
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
                                 checkpoint=self.artifact,predictor=ScriptPredictor(self.artifact,device,self.method,graph=bool(options.get("cuda_graph",False))),
                                 limits=limits,wrist_compensation=wrist_compensation)
        self.executor.audit.enabled=False  # per-step deployment logger owns persistence
        self.history=deque(maxlen=HISTORY_FRAMES)
        self.log_dir=Path(options.get("log_dir",ROOT/"runs/round4/deployment_logs"))
        self.log_file=None;self.run_context=options.get("run_context","deployment")
        self.previous_fallback=False
        self.pending_reset=False
        # Compile/capture before control begins; the first live50Hz call must
        # never pay CUDA Graph construction or lazy TorchScript warmup.
        meta=metadata();q=torch.tensor(meta['default_dof_pos'],device=device)
        self.device_history.observe(q,torch.zeros_like(q),q.new_tensor([0.,0.,.8]),
            q.new_tensor([1.,0.,0.,0.]),q.new_zeros(3),q.new_tensor([0.,0.,-1.]))
        dummy=self.device_history.buffers['kp'][-1][None,None].expand(1,100,4,9)
        self.executor.predictor(self.device_history.batched(),dummy)
        if options.get('coefficient_transition'):
            self.executor.predictor.predict_coefficients(self.device_history.batched(),dummy)
        self.device_history.reset();self.history.clear()

    # -- measurement ---------------------------------------------------------
    @torch.no_grad()
    def observe(self,q,dq,root_pos,root_quat,w=None,g=None):
        if not self.observing:return
        if self.enabled and self.method=='b1' and w is None:
            raise ValueError('B1 requires the same measured base angular velocity used by the policy')
        root_quat=torch.as_tensor(root_quat,device=self.device,dtype=torch.float32).reshape(-1)
        if w is None:w=torch.zeros(3,device=self.device)
        if g is None:
            from exec_comp.geometry import qapply
            g=qapply(qconj(root_quat),root_quat.new_tensor([0.,0.,-1.]))
        self.device_history.observe(q,dq,root_pos,root_quat,w,g)
        # Only the latest view is needed by logging. The true ten-frame input
        # history lives in DeviceHistory, not in this observer-presence deque.
        self.history.append({'kp':self.device_history.buffers['kp'][-1]})

    def measured_history(self):
        if self.device_history.count==0:raise RuntimeError('No measured history')
        return self.device_history.batched()

    def reset(self,preserve_residual=False):
        if not self.observing:return
        self.history.clear();self.device_history.reset()
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
                "run_context":self.run_context,"method":self.method,
                "artifact_sha256":self.manifest["artifact_sha256"]},indent=2))
        def positions(value):
            if torch.is_tensor(value):value=value.detach().cpu().numpy()
            return np.asarray(value)[...,:3].tolist()
        def scalar(key):
            value=telemetry.get(key)
            if value is None:return None
            if torch.is_tensor(value):
                value=value.detach().cpu().numpy()
                return value.reshape(-1).round(4).tolist() if value.size>1 else bool(value) if value.dtype==bool else float(value)
            return value
        fallback=bool(np.any(telemetry.get("fallback").detach().cpu().numpy())) if "fallback" in telemetry else False
        row={"method":self.method,"a1_enabled":self.enabled,"monotonic_ns":time.monotonic_ns(),"wall_time_ns":time.time_ns(),
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
