import unittest
import numpy as np
import torch
from exec_comp.jitter import spectral_energy,clip_jitter,aggregate_jitter,ResidualLowPass


class JitterTests(unittest.TestCase):
    def test_spectrum_separates_fast_and_slow_motion(self):
        t=np.arange(1000)/50
        slow=spectral_energy(np.sin(2*np.pi*t)[:,None])
        fast=spectral_energy(np.sin(2*np.pi*10*t)[:,None])
        self.assertLess(slow['high']/slow['total'],.001)
        self.assertGreater(fast['high']/fast['total'],.999)

    def test_linear_motion_has_no_acceleration_or_false_clip_boundary(self):
        c=np.zeros((100,4,9));c[:,:,0]=np.arange(100)[:,None]*.001
        d=clip_jitter(c,c,np.ones((100,29)),np.ones((100,29))*2,[])
        s=aggregate_jitter([d,d])
        self.assertLess(s['command_second_difference_cm'],1e-12)
        self.assertEqual(s['action_difference'],0.)
        self.assertEqual(s['joint_velocity_rms_rad_s'],2.)

    def test_lowpass_filters_only_residual_and_has_one_step_mean_delay(self):
        raw=torch.zeros(1,6,4,9);raw[...,3]=1;raw[...,8]=1
        cmd=raw.clone();cmd[...,0]+=.04
        f=ResidualLowPass();first=f(cmd,raw);second=f(cmd,raw)
        torch.testing.assert_close(first[:,0,:,0],torch.full((1,4),.02))
        torch.testing.assert_close(second[:,0,:,0],torch.full((1,4),.03))
        self.assertLess(float((second[...,:3]-raw[...,:3]).norm(dim=-1).max()),.05)
        impulse=ResidualLowPass();response=[]
        for step in range(64):response.append(float(impulse(cmd if step==0 else raw,raw)[0,0,0,0]))
        response=np.asarray(response)
        self.assertAlmostEqual(float((np.arange(64)*.02*response).sum()/response.sum()),.02,places=7)

    def test_randomized_action_delay_is_paired_and_uses_first_action_at_start(self):
        from exec_comp.randomization import ActionDelay
        a=ActionDelay(100,'cpu');b=ActionDelay(100,'cpu')
        torch.testing.assert_close(a.steps,b.steps)
        for step in range(5):
            result=a(torch.full((100,29),float(step)))
            expected=(step-a.steps).clamp_min(0).float()
            torch.testing.assert_close(result[:,0],expected)

    def test_policy_noise_preserves_inactive_task_features_and_metadata(self):
        from exec_comp.randomization import PolicyNoise
        class Capture:
            def from_obs(self,prop,task,actions):return prop,task,actions
        noise=PolicyNoise(Capture(),'cpu')
        prop=torch.zeros(2,3,64);task=torch.zeros(2,6,269);task[...,-14]=1;task[...,252]=3
        actions=torch.zeros(2,3,29)
        p,t,a=noise(prop,task,actions)
        self.assertGreater(float(p.abs().sum()),0.)
        torch.testing.assert_close(t[...,3:42],task[...,3:42])
        torch.testing.assert_close(t[...,252:],task[...,252:])
        self.assertIs(a,actions)

    def test_task_errors_follow_the_noisy_joint_measurement(self):
        from exec_comp.randomization import PolicyNoise
        from exec_comp.diff.fk import FK
        from exec_comp.diff.obs import build_obs
        from exec_comp.common import metadata
        class Capture:
            fk=FK();default=torch.tensor(metadata()['default_dof_pos'])
            def from_obs(self,prop,task,actions):return prop,task,actions
        policy=Capture();q=policy.default[None,None].repeat(2,3,1)
        p,r=policy.fk(q[:,-1]);p=p[:,policy.fk.selected][:,None].repeat(1,6,1,1);r=r[:,policy.fk.selected][:,None].repeat(1,6,1,1)
        prop,task,actions=build_obs(policy.fk,q,torch.zeros_like(q),torch.tensor([1.,0,0,0])[None,None].repeat(2,3,1),torch.zeros(2,3,3),torch.zeros(2,3,29),p,r)
        _,noisy,_=PolicyNoise(policy,'cpu')(prop,task,actions)
        torch.testing.assert_close(noisy[...,:42],task[...,:42])
        torch.testing.assert_close(noisy[...,84:168],task[...,84:168])
        self.assertGreater(float((noisy[...,42:84]-task[...,42:84]).abs().sum()),0.)


if __name__=='__main__':unittest.main()
