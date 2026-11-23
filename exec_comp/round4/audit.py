"""Exact active-step fallback accounting and per-trigger causal context."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import save_json

TYPES=('nan','sustained_cap','step_jump','reference_reset')
KEYPOINTS=('root','head','left_wrist','right_wrist')


class ExecutionAudit:
    def __init__(self,options,batch):
        self.options=dict(options);self.batch=batch;self.enabled=True
        self.active_steps=0;self.fallback_steps=0;self.type_steps=dict.fromkeys(TYPES,0)
        self.trigger_events=[];self.traces=[]

    def counterfactual(self):
        other=copy.copy(self);other.enabled=False;return other

    def disabled(self,mask,batch):
        if self.enabled:self.active_steps+=batch if mask is None else int(mask.sum())

    def events(self,*,step,trigger,reason,candidate,previous,switched,last_switch,entering,leaving,grasp,active):
        if not self.enabled:return
        for i in (trigger&active).nonzero().flatten().tolist():
            difference=(candidate[i,0]-previous[i,0]).detach().cpu().numpy()
            norm=np.linalg.norm(difference,axis=-1);point=int(np.argmax(norm))
            axis=int(np.argmax(np.abs(difference[point])))
            self.trigger_events.append(dict(step=step,env=i,
                types=[name for j,name in enumerate(TYPES) if int(reason[i])&(1<<j)],
                frames_since_switch=None if int(last_switch[i])<0 else step-int(last_switch[i]),
                switch_event=bool(switched[i]),grasp_enter=bool(entering[i]),grasp_exit=bool(leaving[i]),
                grasp_phase=bool(grasp[i]),keypoint=KEYPOINTS[point],axis='xyz'[axis],
                jump_norm_cm=float(norm[point]*100) if np.isfinite(norm[point]) else None,
                jump_axis_cm=float(difference[point,axis]*100) if np.isfinite(difference[point,axis]) else None))

    def observe(self,*,step,position,candidate,rotation,actual,raw,falling,reason,active,switched,grasp):
        if not self.enabled:return
        self.active_steps+=int(active.sum());self.fallback_steps+=int((falling&active).sum())
        for j,name in enumerate(TYPES):self.type_steps[name]+=int((falling&active&((reason&(1<<j))!=0)).sum())
        # Fixed batch rows include an active mask, so finished environments can
        # never inflate durations or contaminate a spectral/correlation trace.
        if self.options.get('population')!='brain':return
        cpu=lambda x:x.detach().cpu().numpy().copy()
        self.traces.append(dict(step=step,residual=cpu(position[:,0]),candidate=cpu(candidate[:,0]),
            rotation=cpu(rotation[:,0]),actual=cpu(actual[...,:3]),raw=cpu(raw[...,:3]),
            active=cpu(active),fallback=cpu(falling),reason=cpu(reason),switch=cpu(switched),grasp=cpu(grasp)))

    def report(self):
        denominator=self.active_steps
        return dict(active_env_steps=denominator,fallback_env_steps=self.fallback_steps,
            fallback_fraction=self.fallback_steps/denominator if denominator else None,
            fallback_env_steps_by_type=self.type_steps,
            fallback_fraction_by_type={k:v/denominator if denominator else None for k,v in self.type_steps.items()},
            type_overlap='simultaneous triggers count toward each matching type; total counts each active step once',
            triggers=len(self.trigger_events),projection_scope='position xyz and SO3-log rotation xyz projected on the same basis; wrists-only zeros both pose residuals on root/head')

    def flush(self,folder):
        if not self.enabled:return
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'fallback_events.jsonl').open('w') as stream:
            for event in self.trigger_events:stream.write(json.dumps(event,allow_nan=False)+'\n')
        if self.traces:
            keys=[k for k in self.traces[0] if k!='step']
            arrays={k:np.stack([row[k] for row in self.traces]) for k in keys}
            arrays['step']=np.asarray([row['step'] for row in self.traces])
            np.savez_compressed(folder/'executor_trace.npz',**arrays)
        save_json(folder/'fallback_statistics.json',self.report())
