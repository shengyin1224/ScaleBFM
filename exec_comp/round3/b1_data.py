"""Rollout sampling for B1, including the switch events G must survive.

The K* sequences come from the reference motion library as it was recorded
alongside each transition, so the target and the start state belong to the
same moment. Reference caches call the pinned deployment matcher, RTC and sampler.
Short TRAIN rollouts explicitly oversample hold eligibility to fit two switches;
Isaac evaluation and collection preserve the natural PACE hold.
"""
import json
import numpy as np
import torch
from exec_comp.common import CHUNK,CHUNK_FPS,HISTORY,ROOT
from exec_comp.datasets import Rollouts
from exec_comp.geometry import heading_anchor,pose_normalize,to_anchor

KEYS=("s","action","prev_action","episode","done","root_pos","root_quat",
      "privileged_root","kp","ref_kp")
CONTROL_DT=.02


def deployed_contract(path=None):
    return json.loads((path or ROOT/"configs/online_execution_contract.json").read_text())


class B1Rollouts(Rollouts):
    """Start states plus the measurable history and the K* window G reads."""

    def __init__(self,split,device="cuda:0",folders=None,balance_roots=False):
        super().__init__(split,body=False,device=device,folders=folders,
                         balance_roots=balance_roots,keys=KEYS)

    def batch(self,batch,horizon,chunk=CHUNK):
        """``horizon`` is the number of 50 Hz steps the chain will unroll."""
        offsets=np.arange(chunk)*50/CHUNK_FPS
        lower=offsets.astype(int);upper=np.ceil(offsets).astype(int)
        stream,envs,steps=self.indices(batch,max(int(upper[-1]),horizon+1))
        history=steps[:,None]+np.arange(-HISTORY+1,1)[None]
        recent=steps[:,None]+np.arange(-2,1)[None]
        root=self.tensor(stream["root_pos"][envs,steps])
        quat=self.tensor(stream["root_quat"][envs,steps])
        origin,heading=heading_anchor(root,quat)
        weight=self.tensor(offsets-lower)[None,:,None,None]
        low=self.tensor(stream["ref_kp"][envs[:,None],steps[:,None]+lower])
        high=self.tensor(stream["ref_kp"][envs[:,None],steps[:,None]+upper])
        window=pose_normalize(low+(high-low)*weight)
        state=self.tensor(stream["s"][envs[:,None],history])
        return {
            # Measurable history, in G's own anchored frame.
            "kp":to_anchor(self.tensor(stream["kp"][envs[:,None],history]),origin,heading),
            "q":state[...,:29],"dq":state[...,29:58],
            "angular_velocity":state[...,58:61],"gravity":state[...,61:64],
            "window_world":window,
            "window":to_anchor(window,origin,heading),
            "anchor_origin":origin,"anchor_heading":heading,
            # Training-only privileged quantities for the world model rollout.
            "s":state,"privileged":self.tensor(stream["privileged_root"][envs,steps]),
            "root_pos":root,"root_quat":quat,
            'root_quat_history':self.tensor(stream['root_quat'][envs[:,None],history]),
            "prev_action":self.tensor(stream["prev_action"][envs[:,None],recent]),
        }

    def switching_batch(self,batch,horizon=32):
        """Recorded TRAIN/heldout starts with enough reference for100@15Hz."""
        native_fps=float(deployed_contract()['chunk_fps'])
        future=int(np.ceil((CHUNK-1)*50/native_fps))+horizon+2
        stream,e,t=self.indices(batch,future)
        hist=t[:,None]+np.arange(-HISTORY+1,1)[None]
        recent=t[:,None]+np.arange(-2,1)[None]
        s=self.tensor(stream['s'][e[:,None],hist])
        root=self.tensor(stream['root_pos'][e,t]);quat=self.tensor(stream['root_quat'][e,t])
        return {'s':s,'kp_world':self.tensor(stream['kp'][e[:,None],hist]),
            'privileged':self.tensor(stream['privileged_root'][e,t]),'root_pos':root,'root_quat':quat,
            'root_quat_history':self.tensor(stream['root_quat'][e[:,None],hist]),
            'prev_action':self.tensor(stream['prev_action'][e[:,None],recent]),
            'source_reference':self.tensor(stream['ref_kp'][e[:,None],t[:,None]+np.arange(future)[None]]),
            'source_keypoints':self.tensor(stream['kp'][e[:,None],t[:,None]+np.arange(horizon)[None]])}


def switch_schedule(generator,horizon,contract,minimum_events=2):
    """Control steps at which a chunk switch lands inside one rollout.

    PACE accepts a chunk every 25-50 chunk frames, which at the deployed rate
    is 1.67-3.33 s, longer than a 17-step rollout. Forcing at least two events
    per rollout is therefore a deliberate oversampling of the seam, stated
    here rather than hidden: G is being trained specifically on the case the
    round-2 analysis showed to be the problem.
    """
    fps=float(contract["chunk_fps"])
    low=int(contract["switching_parameters"]["adaptive_far_min_frames"]["value"])
    high=int(contract["switching_parameters"]["adaptive_far_max_frames"]["value"])
    natural=(np.array([low,high])/fps/CONTROL_DT)
    spacing=max(2,int(horizon/(minimum_events+1)))
    steps=[];cursor=int(generator.integers(1,max(2,spacing)))
    while cursor<horizon:
        steps.append(cursor)
        cursor+=int(generator.integers(spacing,spacing*2+1))
    if len(steps)<minimum_events:
        steps=[int(round((index+1)*horizon/(minimum_events+1))) for index in range(minimum_events)]
    return {"steps":steps,"natural_spacing_control_steps":[float(v) for v in natural],
            "oversampled":bool(natural[0]>horizon)}


