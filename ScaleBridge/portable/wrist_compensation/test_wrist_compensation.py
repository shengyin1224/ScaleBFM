import numpy as np
import torch
import unittest

from scalebridge.env.motion_tracking_hat4 import HAT4MotionTrackingEnv

PELVIS = 0
LEFT_WRIST = 10
RIGHT_WRIST = 13
BODIES = 14
FUTURE = 6


def make_env(mode, focus=1.0, root_error=(0.03, -0.02, 0.01)):
    env = HAT4MotionTrackingEnv.__new__(HAT4MotionTrackingEnv)
    env.wrist_compensation = mode
    env.wrist_compensation_clamp_m = 0.08
    env.wrist_compensation_gain = 0.0
    env._wrist_offset = torch.zeros(2, 3)
    env.root_override_blend_s = 0.4
    env._override_alpha = 0.0
    env._pelvis_body_index = PELVIS
    env._wrist_body_indices = [LEFT_WRIST, RIGHT_WRIST]
    env._comp_last = None
    env.dt = 0.02
    env.reference_playing = True
    env.motion_complete = False
    env.localization_paused = False
    env.reference_forcing = True
    # _completion_return_enabled() dependencies
    env.deployment_hardening = False
    env.motion_complete_local_hold = True
    env.motion_complete_return_to_start_steps = 0

    body_pos = torch.zeros(1, FUTURE, BODIES, 3)
    ref_root = torch.tensor([1.0, 2.0, 0.8])
    body_pos[0, :, PELVIS] = ref_root
    body_pos[0, :, LEFT_WRIST] = torch.tensor([1.3, 2.2, 1.0])
    body_pos[0, :, RIGHT_WRIST] = torch.tensor([1.3, 1.8, 1.0])
    body_pos[0, :, 7] = torch.tensor([1.0, 2.0, 1.1])  # torso
    env.state_buffer = {
        "body_pos_w_future": body_pos,
        "root_pos_buffer": ref_root[None, None].clone(),
    }
    env.focus_phase = torch.tensor([[focus, 0.5]])
    env.measured_root = (ref_root - torch.tensor(root_error))[None]
    return env


class WristCompensationTest(unittest.TestCase):
    def test_none_mode_changes_nothing_but_records_baseline(self):
        env = make_env("none")
        before = env.state_buffer["body_pos_w_future"].clone()
        env._apply_wrist_compensation(env.measured_root, 0)
        torch.testing.assert_close(env.state_buffer["body_pos_w_future"], before)
        self.assertIsNotNone(env._comp_last)
        np.testing.assert_allclose(
            env._comp_last["root_error"], [0.03, -0.02, 0.01], atol=1e-6
        )
        np.testing.assert_allclose(env._comp_last["applied_offset"], [0.0, 0.0, 0.0])

    def test_offset_moves_only_the_wrists_by_the_root_error(self):
        env = make_env("offset")
        before = env.state_buffer["body_pos_w_future"].clone()
        env._apply_wrist_compensation(env.measured_root, 0)
        after = env.state_buffer["body_pos_w_future"]
        e = torch.tensor([0.03, -0.02, 0.01])
        torch.testing.assert_close(after[0, :, LEFT_WRIST], before[0, :, LEFT_WRIST] + e)
        torch.testing.assert_close(after[0, :, RIGHT_WRIST], before[0, :, RIGHT_WRIST] + e)
        # Every other body target, and the forced root observation, stay put:
        # the root loop must keep chasing its own error.
        untouched = [i for i in range(BODIES) if i not in (LEFT_WRIST, RIGHT_WRIST)]
        torch.testing.assert_close(after[0, :, untouched], before[0, :, untouched])
        torch.testing.assert_close(
            env.state_buffer["root_pos_buffer"][0, -1], torch.tensor([1.0, 2.0, 0.8])
        )

    def test_offset_is_clamped_per_component(self):
        env = make_env("offset", root_error=(0.30, -0.30, 0.0))
        env._apply_wrist_compensation(env.measured_root, 0)
        np.testing.assert_allclose(
            env._comp_last["applied_offset"], [0.08, -0.08, 0.0], atol=1e-7
        )

    def test_offset_integral_accumulates_the_wrist_residual_until_clamped(self):
        env = make_env("offset")
        env.wrist_compensation_gain = 0.1

        class FakeSim:
            pass

        env.simulator = FakeSim()  # no mujoco_data on purpose: fall back below

        # Without measured wrists the integral silently falls back to
        # feedforward so a missing backend can never zero the compensation.
        env._apply_wrist_compensation(env.measured_root, 0)
        np.testing.assert_allclose(
            env._comp_last["applied_offset"], [0.03, -0.02, 0.01], atol=1e-6
        )

        # With measured wrists 2cm short of the target, each call adds
        # gain * residual, independently per wrist, until the clamp.
        # Fresh env: the fallback call above already shifted the targets.
        env = make_env("offset")
        env.wrist_compensation_gain = 0.1
        measured = torch.zeros(2, 3)
        measured[0] = torch.tensor([1.3, 2.2, 1.0]) - torch.tensor([0.02, 0.0, 0.0])
        measured[1] = torch.tensor([1.3, 1.8, 1.0]) - torch.tensor([0.0, 0.02, 0.0])
        env._measured_wrist_pos_world = lambda: measured
        env._wrist_offset = torch.zeros(2, 3)
        env._apply_wrist_compensation(env.measured_root, 0)
        np.testing.assert_allclose(
            env._comp_last["applied_offset_lw"], [0.002, 0.0, 0.0], atol=1e-7
        )
        np.testing.assert_allclose(
            env._comp_last["applied_offset_rw"], [0.0, 0.002, 0.0], atol=1e-7
        )
        for _ in range(100):
            env._apply_wrist_compensation(env.measured_root, 0)
        np.testing.assert_allclose(
            env._comp_last["applied_offset_lw"], [0.08, 0.0, 0.0], atol=1e-6
        )

    def test_root_override_ramps_in_focus_and_reforces_the_root_obs(self):
        env = make_env("root_override", focus=1.0)
        measured = env.measured_root[0].clone()
        ref_root = torch.tensor([1.0, 2.0, 0.8])
        wrist_before = env.state_buffer["body_pos_w_future"][0, :, LEFT_WRIST].clone()

        rate = env.dt / env.root_override_blend_s  # 0.05 per step
        env._apply_wrist_compensation(env.measured_root, 0)
        self.assertAlmostEqual(env._override_alpha, rate)
        pelvis = env.state_buffer["body_pos_w_future"][0, 0, PELVIS]
        torch.testing.assert_close(pelvis, (1 - rate) * ref_root + rate * measured)
        # Under reference_forcing the root observation must follow the moved
        # pelvis target, and the wrist world targets must not move.
        torch.testing.assert_close(env.state_buffer["root_pos_buffer"][0, -1], pelvis)
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, :, LEFT_WRIST], wrist_before
        )

        # 20 steps at 50Hz cover the 0.4s blend: alpha saturates at 1 and the
        # pelvis reference equals the measured root exactly.
        for _ in range(25):
            env.state_buffer["body_pos_w_future"][0, :, PELVIS] = ref_root
            env._apply_wrist_compensation(env.measured_root, 0)
        self.assertAlmostEqual(env._override_alpha, 1.0)
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0, PELVIS], measured
        )

    def test_root_override_ramps_back_out_of_focus(self):
        env = make_env("root_override", focus=0.0)
        env._override_alpha = 1.0
        env._apply_wrist_compensation(env.measured_root, 0)
        self.assertAlmostEqual(env._override_alpha, 1.0 - env.dt / 0.4)
        for _ in range(30):
            env._apply_wrist_compensation(env.measured_root, 0)
        self.assertAlmostEqual(env._override_alpha, 0.0)

    def test_compensation_is_inert_when_not_playing_or_complete(self):
        for field in ("reference_playing", "motion_complete"):
            env = make_env("offset")
            env._override_alpha = 0.7
            setattr(env, field, field == "motion_complete")
            before = env.state_buffer["body_pos_w_future"].clone()
            env._apply_wrist_compensation(env.measured_root, 0)
            torch.testing.assert_close(env.state_buffer["body_pos_w_future"], before)
            self.assertIsNone(env._comp_last)
            self.assertEqual(env._override_alpha, 0.0)


if __name__ == "__main__":
    unittest.main()
