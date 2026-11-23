"""A1-5 items 2-6 and the A1-6 fallback, checked as behaviour, not as wiring."""
import unittest
import torch
from exec_comp.common import CHUNK,HISTORY,ROOT,metadata
from exec_comp.geometry import pose9
from exec_comp.models.inverse import InverseModel
from exec_comp.round3.a1_executor import A1Executor,ExecutorLimits

DEVICE="cpu"
TOKENS=6


def identity_pose(batch,frames):
    quat=torch.zeros(batch,frames,4,4);quat[...,0]=1.
    positions=torch.zeros(batch,frames,4,3);positions[...,2]=.8
    return pose9(positions,quat)


class _ConstantResidual(InverseModel):
    """A stand-in A1 with a known residual, so the stage is tested, not A1."""

    def __init__(self,offset_m=.02):
        super().__init__();self.offset_m=float(offset_m)

    def forward(self,state,goal):
        shifted=goal.clone();shifted[...,0]=shifted[...,0]+self.offset_m
        return shifted


def make_executor(mode="stepwise",offset_m=.02,limits=None,enable=True,batch=2):
    executor=A1Executor(enable=enable,batch=batch,device=DEVICE,mode=mode,
                        limits=limits or ExecutorLimits(lowpass=False))
    if enable:executor.model=_ConstantResidual(offset_m).requires_grad_(False)
    return executor


def inputs(batch=2,lookahead_m=0.):
    deployed=identity_pose(batch,TOKENS);raw=identity_pose(batch,TOKENS)
    offset=torch.zeros(batch,TOKENS,4,3);offset[...,0]=lookahead_m
    deployed=deployed.clone();deployed[...,0]+=lookahead_m
    history={"q":torch.zeros(batch,HISTORY,29),"dq":torch.zeros(batch,HISTORY,29),
             "kp":identity_pose(batch,HISTORY),
             "root_pos":torch.zeros(batch,3),"root_quat":torch.tensor([[1.,0.,0.,0.]]).repeat(batch,1)}
    return dict(deployed_pose=deployed,raw_pose=raw,lookahead_offset=offset,
                state_hist=history,window=identity_pose(batch,CHUNK))


class ExecutorTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def run_steps(self,executor,count,*,grasp=None,switched=None,lookahead_m=0.,batch=2):
        trace=[]
        for index in range(count):
            data=inputs(batch,lookahead_m)
            flags=torch.zeros(batch,dtype=torch.bool) if switched is None else switched(index)
            phase=torch.zeros(batch,dtype=torch.bool) if grasp is None else grasp(index)
            command,telemetry=executor(switched=flags,grasp_phase=phase,**data)
            trace.append((command,telemetry))
        return trace

    def test_disabled_returns_the_deployed_reference_object(self):
        executor=make_executor(enable=False)
        data=inputs()
        command,telemetry=executor(switched=torch.zeros(2,dtype=torch.bool),
                                   grasp_phase=torch.zeros(2,dtype=torch.bool),**data)
        self.assertIs(command,data["deployed_pose"])
        self.assertEqual(telemetry,{})
        self.assertFalse(executor.report()["enabled"])

    def test_mutual_exclusion_with_the_deployment_wrist_path(self):
        with self.assertRaises(ValueError):
            A1Executor(enable=True,batch=1,device=DEVICE,wrist_compensation="offset")
        # Disabled, the assertion must not fire: the old path stays usable.
        A1Executor(enable=False,batch=1,device=DEVICE,wrist_compensation="offset")

    def test_stepwise_calls_every_control_step_and_events_do_not(self):
        stepwise=make_executor("stepwise");self.run_steps(stepwise,20)
        events=make_executor("switch_event")
        self.run_steps(events,20,switched=lambda i:torch.tensor([i in (5,12),False]))
        self.assertEqual(stepwise.report()["model_calls"],20)
        # One warm-up call plus the two switch events, batched into one call each.
        self.assertEqual(events.report()["model_calls"],3)

    def test_switch_event_crossfade_has_no_step(self):
        executor=make_executor("switch_event",offset_m=.02)
        trace=self.run_steps(executor,30,switched=lambda i:torch.tensor([i==10,False]))
        residual=torch.stack([t[1]["residual_cm"][0,2] for t in trace])
        jumps=(residual[1:]-residual[:-1]).abs()
        self.assertLess(float(jumps.max()),1.)
        self.assertGreater(executor.report()["crossfade_env_steps"],0)

    def test_grasp_freezes_the_residual_and_ramps_back(self):
        executor=make_executor("stepwise",offset_m=.02)
        grasp=lambda i:torch.tensor([10<=i<20,False])
        trace=self.run_steps(executor,40,grasp=grasp)
        residual=torch.stack([t[1]["residual_cm"][0,2] for t in trace])
        held=residual[11:20]
        self.assertLess(float((held-held[0]).abs().max()),1e-5,"grasp must freeze the residual")
        self.assertGreater(float(held[0]),0.5,"a frozen zero would defeat the point")
        # No instant return to zero at any time, entering or leaving.
        self.assertLess(float((residual[1:]-residual[:-1]).abs().max()),1.)
        self.assertGreater(executor.report()["frozen_env_steps"],0)
        self.assertGreater(executor.report()["ramp_env_steps"],0)

    def test_shared_cap_bounds_lookahead_plus_residual(self):
        executor=make_executor("stepwise",offset_m=.2,limits=ExecutorLimits(lowpass=False))
        trace=self.run_steps(executor,5,lookahead_m=.04)
        total=torch.stack([t[1]["total_offset_cm"] for t in trace])
        self.assertLessEqual(float(total.max()),5.0+1e-3)
        self.assertGreater(executor.report()["cap_limited_env_steps"],0)

    def test_shared_cap_preserves_valid_opposing_corrections(self):
        e=make_executor(offset_m=-.04,limits=ExecutorLimits(lowpass=False))
        out=self.run_steps(e,1,lookahead_m=.04)[0]
        torch.testing.assert_close(out[0][...,0],torch.zeros_like(out[0][...,0]),atol=1e-6,rtol=0)

    def test_default_cap_guard_refers_to_residual_not_lookahead(self):
        e=make_executor(offset_m=.01,limits=ExecutorLimits(lowpass=False))
        self.run_steps(e,60,lookahead_m=.04)
        self.assertEqual(e.counters.fallbacks['sustained_cap'],0)

    def test_reference_pause_fades_instead_of_resetting(self):
        e=make_executor(offset_m=.02,limits=ExecutorLimits(lowpass=False))
        self.run_steps(e,3);issued=[]
        for _ in range(12):
            _,t=e(**inputs(),switched=torch.zeros(2,dtype=torch.bool),grasp_phase=torch.ones(2,dtype=torch.bool),
                force_fallback=torch.ones(2,dtype=torch.bool))
            issued.append(t['residual_position'][0,0,2,0])
        issued=torch.stack(issued)
        self.assertAlmostEqual(float(issued[0]),.02,places=6)
        self.assertLess(float(issued.diff().abs().max()),.0031)
        self.assertEqual(float(issued[-1]),0.)
        self.assertEqual(e.counters.fallbacks['reference_reset'],2)

    def test_sustained_cap_fades_the_residual_out(self):
        # Deployed ramp: fading the whole 5 cm budget over 150 ms is 0.67 cm per
        # control step, which stays under the 1 cm single-step guard.
        limits=ExecutorLimits(lowpass=False,cap_hold_s=.2)
        executor=make_executor("stepwise",offset_m=.2,limits=limits)
        trace=self.run_steps(executor,40,lookahead_m=0.)
        residual=torch.stack([t[1]["residual_cm"][0,2] for t in trace])
        # The cause persists, so the guard keeps re-firing; what matters is
        # that each fade is gradual and actually reaches zero.
        self.assertGreaterEqual(executor.report()["fallback_events"]["sustained_cap"],2)
        self.assertLess(float(residual.min()),1e-6)
        self.assertLess(float((residual[1:]-residual[:-1]).abs().max()),1.)

    def test_nan_output_fades_out_and_is_counted(self):
        executor=make_executor("stepwise",offset_m=.02,
                               limits=ExecutorLimits(lowpass=False,fallback_ramp_s=.1))
        self.run_steps(executor,5)
        executor.model=_ConstantResidual(float("nan")).requires_grad_(False)
        trace=self.run_steps(executor,10)
        self.assertGreaterEqual(executor.report()["fallback_events"]["nan"],2)
        for command,_ in trace:self.assertTrue(bool(torch.isfinite(command).all()))
        residual=torch.stack([t[1]["residual_cm"][0,2] for t in trace])
        self.assertLess(float(residual[-1]),1e-6)

    def test_nan_fades_from_last_issued_residual_without_first_step_drop(self):
        executor=make_executor(offset_m=.02,limits=ExecutorLimits(lowpass=False))
        before=self.run_steps(executor,3)[-1][1]['residual_position'][0,0,2,0].clone()
        executor.model=_ConstantResidual(float('nan')).requires_grad_(False)
        trace=self.run_steps(executor,8)
        issued=torch.stack([t[1]['residual_position'][0,0,2,0] for t in trace])
        torch.testing.assert_close(issued[0],before)
        self.assertLessEqual(float(issued.diff().abs().max()),.0031)
        self.assertLess(float(issued[-1].abs()),1e-7)

    def test_large_candidate_is_never_issued_at_fallback_trigger(self):
        executor=make_executor(offset_m=.02,limits=ExecutorLimits(lowpass=False))
        before=self.run_steps(executor,3)[-1][1]['residual_position'][0,0,2,0].clone()
        executor.model=_ConstantResidual(-.04).requires_grad_(False)
        trace=self.run_steps(executor,8)
        torch.testing.assert_close(trace[0][1]['residual_position'][0,0,2,0],before)
        self.assertGreater(executor.report()['fallback_events']['step_jump'],0)

    def test_nonfinite_rotation_also_falls_back(self):
        executor=make_executor(offset_m=.02,limits=ExecutorLimits(lowpass=False))
        self.run_steps(executor,3)
        class BadRotation(_ConstantResidual):
            def forward(self,state,goal):
                result=super().forward(state,goal);result[...,3:]=float('nan');return result
        executor.model=BadRotation(.02).requires_grad_(False)
        trace=self.run_steps(executor,8)
        self.assertTrue(all(bool(torch.isfinite(c).all()) for c,_ in trace))
        self.assertGreater(executor.report()['fallback_events']['nan'],0)

    def test_seam_uses_previously_applied_filtered_profile(self):
        executor=make_executor(limits=ExecutorLimits(lowpass=True))
        first=self.run_steps(executor,1)[0][1]['residual_position'].clone()
        second=self.run_steps(executor,1)[0][1]['previous_residual_position']
        torch.testing.assert_close(second[:,0],first[:,1])

    def test_event_mode_refreshes_model_when_grasp_ends(self):
        e=make_executor('switch_event',offset_m=.02)
        self.run_steps(e,3)
        self.run_steps(e,5,grasp=lambda _:torch.ones(2,dtype=torch.bool))
        e.model=_ConstantResidual(.04).requires_grad_(False)
        out=self.run_steps(e,10)[-1][1]['residual_position']
        torch.testing.assert_close(out[...,0],torch.full_like(out[...,0],.04),atol=1e-6,rtol=0)
        self.assertEqual(e.counters.model_calls,2)

    def test_counterfactual_fork_cannot_advance_live_state(self):
        e=make_executor();self.run_steps(e,3)
        snapshot=e.applied_pos.clone();age=e.delta_age_s.clone();steps=e.counters.steps
        other=e.fork();self.run_steps(other,4)
        torch.testing.assert_close(e.applied_pos,snapshot,atol=0,rtol=0)
        torch.testing.assert_close(e.delta_age_s,age,atol=0,rtol=0)
        self.assertEqual(e.counters.steps,steps);self.assertIs(e.model,other.model)

    def test_finished_environments_do_not_contribute_safety_counters(self):
        e=make_executor(offset_m=.2,limits=ExecutorLimits(lowpass=False,cap_hold_s=.2))
        for _ in range(20):
            e(**inputs(),switched=torch.zeros(2,dtype=torch.bool),grasp_phase=torch.zeros(2,dtype=torch.bool),
              active_mask=torch.tensor([True,False]))
        self.assertEqual(e.counters.active_env_steps,20)
        self.assertFalse(bool(e.have_delta[1]));self.assertEqual(e.counters.fallbacks['sustained_cap'],1)

    def test_reset_clears_per_environment_state(self):
        executor=make_executor("stepwise",offset_m=.02)
        self.run_steps(executor,5,grasp=lambda i:torch.tensor([True,False]))
        executor.reset([0])
        self.assertFalse(bool(executor.have_delta[0]))
        self.assertFalse(bool(executor.applied_valid[0]))
        self.assertTrue(bool(executor.have_delta[1]))


class RealModelTests(unittest.TestCase):
    """The frozen round-2 A1 weights must drive this stage without changes."""

    def test_real_a1_runs_and_respects_the_cap(self):
        checkpoint=ROOT/"runs/models/inverse_smooth_r2_best.pt"
        if not checkpoint.exists():self.skipTest(f"missing {checkpoint}")
        executor=A1Executor(enable=True,batch=2,device=DEVICE,mode="stepwise",checkpoint=checkpoint)
        data=inputs(2)
        data["state_hist"]["q"]=torch.tensor(metadata()["default_dof_pos"])[None,None].repeat(2,HISTORY,1)
        for _ in range(5):
            command,telemetry=executor(switched=torch.zeros(2,dtype=torch.bool),
                                       grasp_phase=torch.zeros(2,dtype=torch.bool),**data)
        self.assertEqual(command.shape,(2,TOKENS,4,9))
        self.assertTrue(bool(torch.isfinite(command).all()))
        self.assertLessEqual(float(telemetry["total_offset_cm"].max()),5.0+1e-3)


if __name__=="__main__":unittest.main()
