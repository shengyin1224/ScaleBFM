"""The deployment packaging must be inert when the switch is off.

The constraint for this round is a bit-identical chain with
``exec_comp.enable=false``, so these tests check identity, not similarity.
"""
import json
import unittest
import numpy as np
import torch
from exec_comp.common import ROOT,digest
from exec_comp.round3.vendor import activate
activate()
from exec_comp.round3.deployment_hooks import ExecCompRound3Hooks,assert_mutually_exclusive

RELEASE=ROOT/"releases/a1_round3"
CONFIG=ROOT/"deployment/configs/hat4_exec_comp_round3.yaml"


class _Parent:
    """Minimal stand-in for the deployed environment's two touched methods."""

    def __init__(self):
        self.state_buffer={"body_pos_w_future":torch.zeros(1,6,14,3),
                           "body_quat_w_wxyz_future":torch.zeros(1,6,14,4)}
        self.state_buffer["body_quat_w_wxyz_future"][...,0]=1.
        self.published=0;self.gathered=0
        self.sentinel=object()

    def _publish_robot_state(self):self.published+=1;return self.sentinel

    def _gather_reference_state(self,now_ns):self.gathered+=1;return self.sentinel

    def close_online(self):return "closed"


class _Disabled(ExecCompRound3Hooks,_Parent):
    pass


class DisabledChainTests(unittest.TestCase):
    def test_raw_signal_is_captured_before_parent_handoff_rebases_chunk(self):
        from types import SimpleNamespace
        class SamplingParent(_Parent):
            def _sample_reference(self,now_ns,**kw):return self.sentinel
            def _gather_reference_state(self,now_ns):
                sample=self._sample_reference(now_ns)
                if sample is not self.sentinel:raise AssertionError('Original sample changed')
                self.plan_value=99.  # Original FOCUS-exit rebase after sampling.
                return self.sentinel
        class Env(ExecCompRound3Hooks,SamplingParent):pass
        env=Env();env.plan_value=7.;observed=[]
        env.active_indices=dict(zip(('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link'),(0,7,10,13)))
        env.future_offsets_s=np.arange(6)*.02;env._elapsed=lambda now,start:0
        env.chunk={'fps':15,'frame_count':100};env.chunk_started_ns=0;env.last_chunk_sequence_id=1
        env._reference_plan_active=lambda:True;env._exec_grasp_phase=lambda now:False
        env._exec_sample=lambda now,offsets,**kw:torch.full((1,len(offsets),4,9),env.plan_value)
        def correct(deployed,raw,window,**kw):
            observed.append((raw,window));return deployed,{}
        env._exec_adapter=SimpleNamespace(enabled=True,history=[{'kp':torch.zeros(4,9)}],correct=correct,log_step=lambda **kw:None)
        self.assertIs(env._gather_reference_state(0),env.sentinel)
        self.assertEqual(env.plan_value,99.)
        self.assertEqual(float(observed[0][0].mean()),7.)
        self.assertEqual(float(observed[0][1].mean()),7.)

    def test_disabled_readonly_logging_preserves_every_target_buffer(self):
        from types import SimpleNamespace
        env=_Disabled();entries=[]
        env.active_indices=dict(zip(('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link'),(0,7,10,13)))
        env.chunk=object();env.last_chunk_sequence_id=3;env.future_offsets_s=np.arange(6)*.02
        env._reference_plan_active=lambda:True;env._exec_grasp_phase=lambda now:False
        env._exec_sample=lambda *a,**k:torch.zeros(1,6,4,9)
        env.state_buffer['fingertips']=torch.randn(2,6,5,3)
        original={k:(v,v.clone()) for k,v in env.state_buffer.items()}
        env._exec_adapter=SimpleNamespace(enabled=False,observing=True,history=[{'kp':torch.zeros(4,9)}],
            log_step=lambda **kw:entries.append(kw),close=lambda:None)
        self.assertIs(env._gather_reference_state(0),env.sentinel)
        self.assertEqual(len(entries),1)
        for key,(identity,expected) in original.items():
            self.assertIs(env.state_buffer[key],identity)
            torch.testing.assert_close(env.state_buffer[key],expected,atol=0,rtol=0)

    def test_lookahead_off_is_only_a_post_switch_output_choice(self):
        from exec_comp.round3.deployment_adapter import Round3Adapter
        adapter=Round3Adapter.__new__(Round3Adapter);adapter.enabled=True;adapter.device='cpu'
        adapter.history=[object()];adapter.options={'keep_lookahead':False};adapter.pending_reset=False
        adapter.measured_history=lambda:{};captured=[]
        adapter.executor=lambda **kw:(captured.append(kw) or kw['deployed_pose'],{})
        raw=torch.zeros(1,6,4,9);deployed=raw.clone();deployed[...,:3]=.04
        out,_=adapter.correct(deployed,raw,torch.zeros(1,100,4,9),switched=False,grasp_phase=False)
        self.assertIs(out,raw);self.assertEqual(float(captured[0]['lookahead_offset'].abs().sum()),0)
        self.assertEqual(float(deployed[0,0,0,0]),torch.tensor(.04).item())
    def test_real_sampler_uses_same_class_namespace_as_wrapper(self):
        from exec_comp.round3.deployment_env import ExecCompRound3Env
        from exec_comp.round3.switching import DeployedSwitching,make_chunk,open_fingertips
        from exec_comp.round3.metrics import canonical
        from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
        self.assertTrue(issubclass(ExecCompRound3Env,HAT4AdaptiveMotionTrackingEnv))
        import copy
        cfg=json.loads((ROOT/'configs/online_execution_contract.json').read_text())
        cfg={k:v['value'] for k,v in cfg['switching_parameters'].items()}
        sw=DeployedSwitching(config=cfg,future_offsets_s=np.arange(6)*.02)
        p=np.zeros((100,4,3),dtype=np.float32);p[:,:,2]=.8;p[:,1,2]=1.3
        q=np.zeros((100,4,4),dtype=np.float32);q[...,0]=1
        sw.publish_state(0,p[0],q[0]);sw.offer(make_chunk(0,0,0,15,(p,q)),0)
        wrapper=ExecCompRound3Env.__new__(ExecCompRound3Env)
        wrapper.__dict__.update(copy.deepcopy(sw.env.__dict__));wrapper.device='cpu'
        actual=wrapper._exec_sample(0,np.arange(6)*.02,apply_stall_assist=False)[0]
        torch.testing.assert_close(actual,torch.tensor(canonical(sw.sample(0))),atol=1e-6,rtol=0)
    def test_paused_enabled_module_keeps_measurement_history_for_logging(self):
        from types import SimpleNamespace
        env=_Disabled();entries=[];resets=[]
        env.active_indices=dict(zip(('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link'),(0,7,10,13)))
        history=[{'kp':torch.zeros(4,9)}]
        env._exec_adapter=SimpleNamespace(enabled=True,history=history,
            fade_to_baseline=lambda pose:(resets.append(True) or pose,{}),log_step=lambda **kw:entries.append(kw))
        env.chunk=None;env._reference_plan_active=lambda:False
        self.assertIs(env._gather_reference_state(0),env.sentinel)
        self.assertEqual(len(history),1);self.assertEqual(len(entries),1);self.assertEqual(len(resets),1)

    def test_existing_but_inactive_hand_guard_does_not_freeze_everything(self):
        from types import SimpleNamespace
        env=_Disabled();env.cfg={'continuity_hand_event_guard':True}
        env.focus_handoff=SimpleNamespace(active=False,bypass_sides=())
        env.hand_event_playback_boundary=SimpleNamespace(holds={})
        env.chunk=None
        self.assertFalse(env._exec_grasp_phase(0))
        env.hand_event_playback_boundary.holds={'left':object()}
        self.assertTrue(env._exec_grasp_phase(0))

    def test_hook_touches_nothing_without_an_adapter(self):
        env=_Disabled();env._exec_adapter=None
        positions=env.state_buffer["body_pos_w_future"]
        quaternions=env.state_buffer["body_quat_w_wxyz_future"]
        self.assertIs(env._publish_robot_state(),env.sentinel)
        self.assertIs(env._gather_reference_state(0),env.sentinel)
        # Identity, not equality: the buffers were never even cloned.
        self.assertIs(env.state_buffer["body_pos_w_future"],positions)
        self.assertIs(env.state_buffer["body_quat_w_wxyz_future"],quaternions)
        self.assertEqual((env.published,env.gathered),(1,1))
        self.assertEqual(env.close_online(),"closed")

    def test_disabled_adapter_is_also_inert(self):
        from exec_comp.round3.deployment_adapter import Round3Adapter
        adapter=Round3Adapter({"enable":False})
        env=_Disabled();env._exec_adapter=adapter
        positions=env.state_buffer["body_pos_w_future"]
        self.assertIs(env._gather_reference_state(0),env.sentinel)
        self.assertIs(env.state_buffer["body_pos_w_future"],positions)
        self.assertIsNone(adapter.observe(torch.zeros(29),torch.zeros(29),
                                          torch.zeros(3),torch.tensor([1.,0.,0.,0.])))
        self.assertIsNone(adapter.log_step(raw=None,corrected=None,actual=None,telemetry={}))


class MutualExclusionTests(unittest.TestCase):
    def test_enabled_with_wrist_compensation_is_refused(self):
        with self.assertRaises(ValueError):
            assert_mutually_exclusive({"exec_comp":{"enable":True},"wrist_compensation":"offset"})

    def test_disabled_leaves_the_old_path_usable(self):
        options,compensation=assert_mutually_exclusive(
            {"exec_comp":{"enable":False},"wrist_compensation":"offset"})
        self.assertFalse(options["enable"]);self.assertEqual(compensation,"offset")

    def test_enabled_without_wrist_compensation_passes(self):
        options,compensation=assert_mutually_exclusive(
            {"exec_comp":{"enable":True},"wrist_compensation":"none"})
        self.assertTrue(options["enable"]);self.assertEqual(compensation,"none")


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        if not (RELEASE/"manifest.json").exists():self.skipTest("release not built")
        self.manifest=json.loads((RELEASE/"manifest.json").read_text())

    def test_artifact_matches_its_manifest(self):
        self.assertEqual(digest(RELEASE/"a1_correct_kp.pt"),self.manifest["artifact_sha256"])

    def test_manifest_states_the_round3_contract(self):
        self.assertFalse(self.manifest["enabled_by_default"])
        self.assertFalse(self.manifest["real_robot_run"])
        self.assertEqual(self.manifest["deployed_chunk"]["fps"],15.)
        self.assertEqual(self.manifest["a1_input_window"]["spacing_hz"],30)
        self.assertTrue(self.manifest["grasp_behaviour"]["freeze"])
        self.assertEqual(self.manifest["shared_cap"]["total_offset_cap_m"],.05)
        self.assertEqual(self.manifest["mutual_exclusion"]["wrist_compensation"],"none")

    def test_generated_config_is_off_by_default(self):
        if not CONFIG.exists():self.skipTest("configuration not packaged")
        import yaml
        config=yaml.safe_load(CONFIG.read_text())
        self.assertIs(config["exec_comp"]["enable"],False)
        self.assertEqual(config["env"]["_target_"],
                         "exec_comp.round3.deployment_env.ExecCompRound3Env")

    def test_torchscript_still_matches_the_eager_model(self):
        if not torch.cuda.is_available():self.skipTest("no CUDA")
        from exec_comp.export_a1 import fixture
        from exec_comp.geometry import from_anchor,to_anchor
        from exec_comp.runtime import correct_encoded,encode_history
        from exec_comp.train_common import load_model
        device="cuda:0"
        history,goal=fixture(device)
        script=torch.jit.load(str(RELEASE/"a1_correct_kp.pt"),map_location=device).eval()
        model=load_model(self.manifest["a1_training_checkpoint"],device).requires_grad_(False)
        with torch.no_grad():
            encoded,origin,quat=encode_history(history,device)
            expected=from_anchor(correct_encoded(model,encoded,to_anchor(goal,origin,quat)[None])[0],origin,quat)
            actual=script.correct_kp(history,goal)
        self.assertLess(float((expected-actual).abs().max()),1e-5)
        self.assertLessEqual(float((actual[...,:3]-goal[...,:3]).norm(dim=-1).max()),.050001)


if __name__=="__main__":unittest.main()
