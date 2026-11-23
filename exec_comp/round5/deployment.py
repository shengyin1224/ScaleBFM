"""Opt-in R5 hooks; no original deployed module is edited."""
import time
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import ROOT,metadata,digest
from exec_comp.round5.adapter import EventAdapter
from exec_comp.round5.reference import snapshot,sample_snapshot,canonical7
from exec_comp.round5.engine import rotate,qinv
from exec_comp.round3.signals import predicted_grasp_phase
from scalebridge.online.adaptive_chunk_execution import closing_boundary
from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv

LINKS=('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link')


class Hooks:
    def _publish_robot_state(self):
        result=super()._publish_robot_state()
        adapter=getattr(self,'_r5',None)
        if adapter is None or not adapter.enabled:return result
        packet=getattr(self,'last_published_robot_state',None)
        if packet is None:return result
        # Reuse MuJoCo FK that the parent ALREADY computed. No new FK/GPU read.
        fk=self.current_pose_fk;adr=fk.root_qpos_adr
        p=np.asarray(fk.data.qpos[adr:adr+3],dtype=np.float32).copy()
        r=np.asarray(fk.data.qpos[adr+3:adr+7],dtype=np.float32).copy()
        q=np.asarray(fk.data.qpos[fk.joint_qpos_adr],dtype=np.float32).copy()
        sim=self.simulator
        dq=np.asarray(sim.dof_vel_tmp)[sim.sim_to_env_joint_idx].astype(np.float32,copy=True)
        w=np.asarray(sim.base_ang_vel_tmp,dtype=np.float32).copy()
        kp=np.stack([np.r_[packet[k+'_pos_world'],packet[k+'_quat_world_wxyz']]
            for k in ('root','head','left_wrist','right_wrist')]).astype(np.float32)
        # Measurable10-frame network input uses6D columns, logger retainsquat.
        from exec_comp.round3.metrics import quat_columns
        x,z=quat_columns(kp[:,3:]);pose9=np.concatenate((kp[:,:3],x,z),-1)
        adapter.observe(packet['timestamp_ns'],dict(q=q,dq=dq,kp=pose9,root_pos=p,root_quat=r,w=w,
            g=rotate(qinv(r),np.array([0.,0.,-1.],dtype=np.float32))))
        self._r5_actual=kp
        return result

    def _gather_reference_state(self,now_ns):
        begin=time.perf_counter_ns()
        result=super()._gather_reference_state(now_ns)
        parent_end=time.perf_counter_ns()
        adapter=getattr(self,'_r5',None)
        if adapter is None or not adapter.enabled:return result
        if not hasattr(self,'_r5_actual') or self.chunk is None:return result
        active=bool(self._reference_plan_active());seq=int(self.last_chunk_sequence_id);epoch=int(getattr(self,'reference_plan_epoch',0))
        # Only6 existing policy tokens on control side. The100-frame network
        # window is sampled in the independent worker from an event snapshot.
        deployed=canonical7(self.last_reference_sample)
        ref=snapshot(self)
        raw=canonical7(sample_snapshot(ref,now_ns,self.future_offsets_s))
        sample_end=time.perf_counter_ns()
        handoff=getattr(self,'focus_handoff',None)
        grasp=bool(handoff is not None and (handoff.active or handoff.bypass_sides))
        if not grasp and active:
            cursor=self._elapsed(now_ns,self.chunk_started_ns)*float(self.chunk['fps'])
            hands=self._focus_prediction_chunks.get(seq,{}).get('hand_focus',{})
            grasp=predicted_grasp_phase(self.chunk,cursor,hands,closing_boundary)
        event=(seq!=getattr(self,'_r5_sequence',None) or epoch!=getattr(self,'_r5_epoch',None)
               or active and not getattr(self,'_r5_active',False))
        if event and active:adapter.event(now_ns,seq,epoch,reference=ref)
        self._r5_sequence=seq;self._r5_epoch=epoch;self._r5_active=active
        corrected,telemetry=adapter.correct(now_ns,deployed,raw,sequence=seq,epoch=epoch,
            grasp=grasp,active=active,offsets=self.future_offsets_s)
        correct_end=time.perf_counter_ns()
        if adapter.method!='raw':
            positions=corrected[...,:3].copy();rotations=corrected[...,3:7]
            positions[:,1]-=rotate(rotations[:,1],np.array([0.,0.,.4]))
            joined=np.ascontiguousarray(np.concatenate((positions,rotations),-1)[None],dtype=np.float32)
            transfer=torch.as_tensor(joined,device=self.device)
            self.state_buffer['body_pos_w_future'].index_copy_(2,self._r5_ids,transfer[...,:3])
            self.state_buffer['body_quat_w_wxyz_future'].index_copy_(2,self._r5_ids,transfer[...,3:])
        transfer_end=time.perf_counter_ns()
        adapter.log(now_ns,raw,corrected,self._r5_actual,telemetry)
        end=time.perf_counter_ns()
        self._r5_last_stages=dict(parent_reference_ms=(parent_end-begin)/1e6,cpu_window6_ms=(sample_end-parent_end)/1e6,
            protection_expand_ipc_ms=(correct_end-sample_end)/1e6,host_device_write_ms=(transfer_end-correct_end)/1e6,
            log_enqueue_ms=(end-transfer_end)/1e6,total_gather_ms=(end-begin)/1e6,network_forward_on_control_ms=0.,extra_fk_ms=0.)
        return result

    def close_online(self):
        if getattr(self,'_r5',None):self._r5.close()
        return super().close_online()


class Init:
    def __init__(self,config,metadata_dict,device):
        self._r5=None
        options=dict(config.get('exec_comp',{}))
        if options.get('enable',False):
            if options.get('method','raw')!='raw' and not options.get('direction_approved',False):
                raise ValueError('R5 T2 direction screen has not approved this model; no robot interfaces were opened')
            if str(config.get('wrist_compensation','none'))!='none':raise ValueError('Old wrist compensation must be none')
            if bool(config.get('continuity_hand_event_guard',False)):raise ValueError('R5 event-guard snapshot not validated')
            folder=ROOT/'runs/round5/deployment_logs'/str(time.time_ns())
            self._r5=EventAdapter(options,folder/'control_steps.jsonl')
        try:super().__init__(config,metadata_dict,device)
        except BaseException:
            if self._r5:self._r5.close()
            raise
        if self._r5 is not None:self._r5_ids=torch.tensor([self.active_indices[k] for k in LINKS],device=device)


class AdaptiveEnv(Init,Hooks,HAT4AdaptiveMotionTrackingEnv):pass
class OnlineEnv(Init,Hooks,HAT4OnlineMotionTrackingEnv):pass
