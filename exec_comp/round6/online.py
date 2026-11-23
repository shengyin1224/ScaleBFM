"""15Hz100-frame chunks and original accepted-switch sampler, simulation-only."""
import json,copy
import numpy as np
from exec_comp.common import ROOT
from exec_comp.round6.switching import DeployedSwitching,make_chunk,quiet_deployment_logging
from exec_comp.round6.library import canonical
from exec_comp.round5.reference import canonical7,canonical9

class Stream:
    def __init__(self,clip,rng,perturb=False):
        config={k:v['value'] for k,v in json.loads((ROOT/'configs/online_execution_contract.json').read_text())['switching_parameters'].items()}
        config.update(focus_handoff_enabled=False,continuity_hand_event_guard=False)
        self.switch=DeployedSwitching(config=config,future_offsets_s=np.arange(6)*.02)
        quiet_deployment_logging();self.clip=clip;self.rng=rng;self.perturb=perturb;self.sequence=0;self.next_packet=0.;self.ou=np.zeros((4,3));self.accepted=0
    def update(self,elapsed,source_time,state):
        now=int((elapsed+1000)*1e9)
        # Geometry only from measured FK, already in the same world frame.
        from exec_comp.geometry import six_to_q
        import torch
        quat=six_to_q(torch.from_numpy(state['kp'][:,3:]).float()).numpy()
        seq=self.switch.publish_state(now,state['kp'][:,:3],quat)
        if elapsed+1e-9>=self.next_packet:
            self.next_packet+=.4;age=float(self.rng.uniform(.05,.18));origin=source_time-age
            times=origin+np.arange(100)/15;p,q=self.clip.sample(times)
            if self.perturb:
                # Smooth command exploration, applied to actual target stream,
                # distinct from fictitious observation/target measurement noise.
                for k in range(100):
                    self.ou=.985*self.ou+np.sqrt(1-.985**2)*.015*self.rng.normal(size=(4,3));self.ou[:2]=0
                    self.ou*=np.minimum(1,.05/np.maximum(np.linalg.norm(self.ou,axis=-1,keepdims=True),1e-8));p[k]+=self.ou
            mask=np.array([self.clip.focus(t) for t in times]);tips=np.repeat(np.array([[0.,0.,0.],[.16,0,0],[.17,0,0],[.17,0,0],[.16,0,0]])[None,None],100,0);tips=np.repeat(tips,2,1)
            tips[mask]*=.35
            source_seq=max(0,seq-int(round(age/.02)))
            chunk=make_chunk(self.sequence,source_seq,now-int(age*1e9),15,(p,q),tips)
            event=self.switch.offer(chunk,now);self.sequence+=1
        else:event=self.switch.poll(now)
        self.accepted+=int(event['accepted'])
        sample=self.switch.sample(now);raw=self.switch.sample_raw(now,sample)
        output=canonical7(sample);pure=canonical7(raw)
        window=None
        if event['accepted']:
            # The NN sees the incoming committed plan, not the outgoing plan
            # selected by blend alpha=0 at the exact switching instant.
            proxy=copy.copy(self.switch.env);proxy.previous_chunk=None;proxy.blend_started_ns=None;proxy.future_offsets_s=np.arange(100)/15
            from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
            window=canonical9(HAT4OnlineMotionTrackingEnv._sample_reference(proxy,now,apply_stall_assist=False))
        return output,pure,window,event,now
