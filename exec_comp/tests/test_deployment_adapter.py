import unittest
import numpy as np
import torch
from exec_comp.deployment_adapter import DeploymentAdapter
from exec_comp.deployment_hooks import ExecCompHooks


class DeploymentTests(unittest.TestCase):
    def test_disabled_adapter_opens_no_model_or_log_and_preserves_identity(self):
        adapter=DeploymentAdapter({'enable':False,'model':'/path/that/does/not/exist'})
        goal={'fingertips':object()}
        self.assertIs(adapter.correct_kp(None,goal),goal)
        self.assertIs(adapter.filter_control_window(goal,None),goal)
        adapter.observe(None,None,None,None);adapter.log_step(None,None,None);adapter.reset();adapter.close()
        self.assertFalse(hasattr(adapter,'model'));self.assertFalse(hasattr(adapter,'log_file'))

    def test_disabled_hooks_delegate_without_touching_state(self):
        class Original:
            def _publish_robot_state(self):return self.sentinel
            def _gather_reference_state(self,now):return self.sentinel
            def close_online(self):return self.sentinel
        class Combined(ExecCompHooks,Original):pass
        env=Combined();env.sentinel=object();env._exec_adapter=None
        self.assertIs(env._publish_robot_state(),env.sentinel)
        self.assertIs(env._gather_reference_state(0),env.sentinel)
        self.assertIs(env.close_online(),env.sentinel)

    def test_full_reference_proxy_does_not_mutate_live_clock_or_chunk(self):
        class Original:
            def _sample_reference(self,now,apply_stall_assist=True):
                self.previous_chunk=None;self._play_ns=999
                n=len(self.future_offsets_s);q=np.tile([1.,0,0,0],(n,1)).astype(np.float32)
                result={}
                for name in ('root','torso','left_wrist','right_wrist'):
                    result[name+'_pos']=np.zeros((n,3),np.float32)
                    result[name+'_quat_wxyz']=q.copy()
                return result
        class Combined(ExecCompHooks,Original):pass
        env=Combined();env.future_offsets_s=np.arange(6)/50;env.device='cpu';env._play_ns=123;env.previous_chunk=object()
        previous=env.previous_chunk;full=env._exec_full_raw_plan(0)
        self.assertEqual(full.shape,(100,4,9));self.assertEqual(env._play_ns,123)
        self.assertIs(env.previous_chunk,previous);self.assertEqual(len(env.future_offsets_s),6)

    def test_enabled_body_hook_preserves_fingers_and_inactive_body_targets(self):
        from exec_comp.geometry import pose9
        class Original:
            def _gather_reference_state(self,now):return 'base-result'
            def _sample_reference(self,now,apply_stall_assist=True):
                n=len(self.future_offsets_s);q=np.tile([1.,0,0,0],(n,1)).astype(np.float32)
                return {key+suffix:(np.zeros((n,3),np.float32) if suffix=='_pos' else q.copy()) for key in ('root','torso','left_wrist','right_wrist') for suffix in ('_pos','_quat_wxyz')}
        class Adapter:
            enabled=True;lowpass=None
            history=[{'kp':pose9(torch.zeros(4,3),torch.tensor([[1.,0,0,0]]).repeat(4,1))}]
            def measured_history(self):return {}
            def correct_kp(self,history,full):
                value=full.clone();value[...,:3]+=.01;return value
            def filter_control_window(self,command,raw):return command
            def log_step(self,*args):self.logged=args
        class Combined(ExecCompHooks,Original):
            def _reference_plan_active(self):return True
        env=Combined();env._exec_adapter=Adapter();env.cfg={};env.chunk=object();env.last_chunk_sequence_id=7
        env._play_scale=1.;env.device='cpu';env.future_offsets_s=np.arange(6)/50
        env.active_indices={'pelvis':0,'torso_link':7,'left_wrist_yaw_link':10,'right_wrist_yaw_link':13}
        positions=torch.zeros(1,6,14,3);rotations=torch.zeros(1,6,14,4);rotations[...,0]=1
        positions[:,:,1]=.3
        env.state_buffer={'body_pos_w_future':positions,'body_quat_w_wxyz_future':rotations}
        fingers=object();env.last_reference_sample={'left_fingertip_local':fingers};chunk=env.chunk
        result=env._gather_reference_state(0)
        self.assertEqual(result,'base-result');self.assertIs(env.chunk,chunk)
        self.assertIs(env.last_reference_sample['left_fingertip_local'],fingers)
        torch.testing.assert_close(env.state_buffer['body_pos_w_future'][:,:,1],positions[:,:,1])
        for index in (0,7,10,13):torch.testing.assert_close(env.state_buffer['body_pos_w_future'][:,:,index],positions[:,:,index]+.01)


if __name__=='__main__':unittest.main()
