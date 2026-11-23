"""Body-target hooks; leaves accepted HAT chunks, fingers and publishers unchanged."""
import numpy as np
import copy
import torch
from exec_comp.geometry import pose9,six_to_q,qmul,qconj,qapply
from exec_comp.hat_playback import sample


class ExecCompHooks:
    def _publish_robot_state(self):
        result=super()._publish_robot_state()
        adapter=getattr(self,'_exec_adapter',None)
        if adapter is not None and adapter.enabled:
            epoch=getattr(self,'reference_plan_epoch',0)
            if epoch!=getattr(self,'_exec_epoch',None):
                adapter.reset();self._exec_delta=None;self._exec_epoch=epoch
            adapter.observe(self.state_buffer['dof_pos_buffer'][0,-1],self.state_buffer['dof_vel_buffer'][0,-1],
                            self.state_buffer['root_pos_buffer'][0,-1],self.state_buffer['root_quat_wxyz_buffer'][0,-1])
        return result

    def _canonical_body_window(self):
        p=self.state_buffer['body_pos_w_future'][0];q=self.state_buffer['body_quat_w_wxyz_future'][0]
        indices=[self.active_indices[n] for n in ('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link')]
        p=p[:,indices].clone();q=q[:,indices]
        p[:,1]+=qapply(q[:,1],p.new_tensor([0.,0.,.4]).expand_as(p[:,1]))
        return pose9(p,q)[None]

    def _exec_full_raw_plan(self,now_ns):
        # Reuse the complete existing clock/RTC/overlap sampler on a shallow
        # state proxy. Its previous_chunk cleanup and playback-clock updates
        # cannot mutate the live environment. Stateful overlays are excluded
        # by the caller until a separate validation covers them.
        proxy=copy.copy(self);proxy.future_offsets_s=np.arange(100,dtype=np.float64)/30
        sampler=super(ExecCompHooks,self)._sample_reference.__func__
        full=sampler(proxy,now_ns,apply_stall_assist=False)
        p=torch.tensor(np.stack([full[k] for k in ('root_pos','torso_pos','left_wrist_pos','right_wrist_pos')],1),device=self.device,dtype=torch.float32)
        q=torch.tensor(np.stack([full[k] for k in ('root_quat_wxyz','torso_quat_wxyz','left_wrist_quat_wxyz','right_wrist_quat_wxyz')],1),device=self.device,dtype=torch.float32)
        p[:,1]+=qapply(q[:,1],p.new_tensor([0.,0.,.4]).expand_as(p[:,1]))
        return pose9(p,q)

    @torch.no_grad()
    def _gather_reference_state(self,now_ns):
        result=super()._gather_reference_state(now_ns)
        adapter=getattr(self,'_exec_adapter',None)
        if adapter is None or not adapter.enabled:return result
        raw=self._canonical_body_window();corrected=raw
        overlays=(getattr(self,'focus_handoff',None) is not None or getattr(self,'focus_adaptive',None) is not None
                  or getattr(self,'stall_wrist_assist',None) is not None
                  or bool(self.cfg.get('continuity_hand_event_guard',False))
                  or self.cfg.get('wrist_compensation','none')!='none')
        active=bool(self._reference_plan_active()) and self.chunk is not None and not overlays
        if active:
            sequence=int(self.last_chunk_sequence_id)
            due=(getattr(self,'_exec_delta',None) is None or sequence!=getattr(self,'_exec_sequence',None)
                 or now_ns-getattr(self,'_exec_started_ns',0)>=333333333)
            if due:
                full=self._exec_full_raw_plan(now_ns)
                if not torch.allclose(full[0],raw[0,0],atol=1e-5,rtol=1e-5):
                    self._exec_delta=None
                    adapter.log_step(raw[0,0],raw[0,0],adapter.history[-1]['kp'],sequence,'reference_sampler_mismatch')
                    return result
                predicted=adapter.correct_kp(adapter.measured_history(),full)
                rotation=qmul(six_to_q(predicted[...,3:]),qconj(six_to_q(full[...,3:])))
                self._exec_delta=pose9(predicted[...,:3]-full[...,:3],rotation)[None]
                self._exec_started_ns=now_ns;self._exec_sequence=sequence
            frames=torch.tensor((now_ns-self._exec_started_ns)*1e-9+np.asarray(self.future_offsets_s),device=self.device,dtype=torch.float32)[None]*30
            delta=sample(self._exec_delta,frames)
            rotation=qmul(six_to_q(delta[...,3:]),six_to_q(raw[...,3:]))
            corrected=pose9(raw[...,:3]+delta[...,:3],rotation)
            corrected=adapter.filter_control_window(corrected,raw)
            # Apply only after the existing clock, RTC, handoff and finger logic.
            # Keep their original chunks/goals intact for subsequent replanning.
            p=corrected[0,...,:3].clone();q=six_to_q(corrected[0,...,3:])
            p[:,1]-=qapply(q[:,1],p.new_tensor([0.,0.,.4]).expand_as(p[:,1]))
            self.state_buffer['body_pos_w_future']=self.state_buffer['body_pos_w_future'].clone()
            self.state_buffer['body_quat_w_wxyz_future']=self.state_buffer['body_quat_w_wxyz_future'].clone()
            for j,name in enumerate(('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link')):
                index=self.active_indices[name]
                self.state_buffer['body_pos_w_future'][0,:,index]=p[:,j]
                self.state_buffer['body_quat_w_wxyz_future'][0,:,index]=q[:,j]
        else:
            self._exec_delta=None
            if adapter.lowpass is not None:
                from exec_comp.jitter import ResidualLowPass
                adapter.lowpass=ResidualLowPass()
        if adapter.history:
            reason='unvalidated_live_overlay' if overlays else None
            adapter.log_step(raw[0,0],corrected[0,0],adapter.history[-1]['kp'],getattr(self,'last_chunk_sequence_id',None),reason)
        return result

    def close_online(self):
        adapter=getattr(self,'_exec_adapter',None)
        if adapter is not None:adapter.close()
        return super().close_online()
