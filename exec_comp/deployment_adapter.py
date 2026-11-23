"""Opt-in deployed adapter. Importing this module never opens robot interfaces."""
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import torch
from exec_comp.common import ROOT,digest,metadata
from exec_comp.diff.fk import FK
from exec_comp.geometry import from_anchor
from exec_comp.runtime import pack_chunk,unpack_chunk
from exec_comp.jitter import ResidualLowPass


class DeploymentAdapter:
    def __init__(self,options,controller_metadata=None,device='cuda:0'):
        self.enabled=bool(options.get('enable',False));self.device=device
        if not self.enabled:return
        self.artifact=Path(options['model']);manifest=json.loads(self.artifact.with_name('manifest.json').read_text())
        if digest(self.artifact)!=manifest['artifact_sha256']:raise ValueError('A1 artifact hash mismatch')
        cerebellum=options.get('cerebellum_model')
        if cerebellum and digest(cerebellum)!=manifest.get('cerebellum_deployment_sha256'):raise ValueError('A1 is bound to another cerebellum checkpoint')
        if controller_metadata:
            for key in ('joint_names','selected_body_names','action_scale','default_dof_pos','stiffness','damping','torque_limit'):
                if controller_metadata[key]!=metadata()[key]:raise ValueError(f'Controller metadata differs: {key}')
        self.model=torch.jit.load(str(self.artifact),map_location=device).eval()
        self.fk=FK().to(device);self.history=deque(maxlen=10)
        self.lowpass=ResidualLowPass() if options.get('residual_lowpass',manifest['residual_lowpass_50hz']) else None
        self.log_dir=Path(options.get('log_dir',ROOT/'runs/deployment_logs'));self.log_file=None
        self.run_context=options.get('run_context','deployment')
        self.manifest=manifest

    @torch.no_grad()
    def observe(self,q,dq,root_pos,root_quat):
        if not self.enabled:return
        values=[torch.as_tensor(x,device=self.device,dtype=torch.float32).reshape(-1) for x in (q,dq,root_pos,root_quat)]
        q,dq,pos,quat=values
        kp=from_anchor(self.fk.keypoints(q[None])[0],pos,quat)
        self.history.append(dict(q=q.clone(),dq=dq.clone(),kp=kp.clone(),root_pos=pos.clone(),root_quat=quat.clone()))

    def reset(self):
        if not self.enabled:return
        self.history.clear()
        if self.lowpass:self.lowpass=ResidualLowPass()

    @torch.no_grad()
    def correct_kp(self,state_hist,K_star_chunk):
        if not self.enabled:return K_star_chunk
        history={k:torch.as_tensor(state_hist[k],device=self.device,dtype=torch.float32) for k in ('q','dq','kp','root_pos','root_quat')}
        if history['q'].shape!=(10,29):raise ValueError('Expected10 measured frames of29 joints')
        goal=pack_chunk(K_star_chunk,self.device)
        if goal.shape!=(100,4,9):raise ValueError('Expected100@30Hz canonical four-keypoint chunk')
        return unpack_chunk(K_star_chunk,self.model.correct_kp(history,goal))

    def measured_history(self):
        if not self.history:raise RuntimeError('No measured body history')
        values=[self.history[0]]*(10-len(self.history))+list(self.history)
        return {k:torch.stack([v[k] for v in values]) for k in values[0]}

    def filter_control_window(self,command,raw):
        if not self.enabled or self.lowpass is None:return command
        return self.lowpass(command,raw)

    def log_step(self,raw,corrected,actual,chunk_sequence_id=None,bypass_reason=None):
        if not self.enabled:return
        if self.log_file is None:
            folder=self.log_dir/(time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()));folder.mkdir(parents=True,exist_ok=False)
            self.log_file=(folder/'control_steps.jsonl').open('w',buffering=8192)
            (folder/'schema.json').write_text(json.dumps({'control_hz':50,'response_alignment_control_steps':1,
                'actual_timestamp':'measured state BEFORE the current command; compare command[t] with actual[t+1]',
                'keypoints':['root','head','left_wrist','right_wrist'],'coordinates':'world','run_context':self.run_context,'artifact_sha256':self.manifest['artifact_sha256']},indent=2))
        def positions(x):
            if torch.is_tensor(x):x=x.detach().cpu().numpy()
            return np.asarray(x)[...,:3].tolist()
        row={'monotonic_ns':time.monotonic_ns(),'wall_time_ns':time.time_ns(),'chunk_sequence_id':chunk_sequence_id,
             'raw_position':positions(raw),'corrected_position':positions(corrected),'actual_position':positions(actual),'bypass_reason':bypass_reason}
        self.log_file.write(json.dumps(row,separators=(',',':'))+'\n')

    def close(self):
        if self.enabled and self.log_file:self.log_file.close();self.log_file=None
