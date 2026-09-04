import numpy as np
import torch
import unittest

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv

PELVIS = 0
TORSO = 7
LEFT_WRIST = 10
RIGHT_WRIST = 13
BODIES = 14
FUTURE = 6

REF_ROOT = torch.tensor([1.0, 2.0, 0.8])
REF_LW = torch.tensor([1.3, 2.2, 1.0])
REF_RW = torch.tensor([1.3, 1.8, 1.0])


class FakeFK:
    def __init__(self, poses):
        self.last_body_poses = poses


def wrist_poses(short_lw=(0.02, 0.0, 0.0), short_rw=(0.0, 0.02, 0.0)):
    return {
        "left_wrist_yaw_link": (
            (REF_LW - torch.tensor(short_lw)).numpy(), np.array([1, 0, 0, 0.0])
        ),
        "right_wrist_yaw_link": (
            (REF_RW - torch.tensor(short_rw)).numpy(), np.array([1, 0, 0, 0.0])
        ),
    }


def make_env(mode, focus=1.0, root_error=(0.03, -0.02, 0.01), gain=0.1, poses=None):
    env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
    env.wrist_compensation = mode
    env.wrist_compensation_gain = gain
    env.wrist_compensation_clamp_m = 0.15
    env.wrist_compensation_fade_s = 0.4
    env._wrist_offset = torch.zeros(2, 3)
    env._comp_fade_alpha = 0.0
    env._comp_last = None
    env.dt = 0.02
    env.reference_playing = True
    env.localization_paused = False
    env.requires_restart = False
    env.awaiting_fresh_chunk = False
    env.held_sample = None
    env.active_indices = {
        "pelvis": PELVIS,
        "torso_link": TORSO,
        "left_wrist_yaw_link": LEFT_WRIST,
        "right_wrist_yaw_link": RIGHT_WRIST,
    }
    env.current_pose_fk = FakeFK(wrist_poses() if poses is None else poses)

    body_pos = torch.zeros(1, FUTURE, BODIES, 3)
    body_pos[0, :, PELVIS] = REF_ROOT
    body_pos[0, :, LEFT_WRIST] = REF_LW
    body_pos[0, :, RIGHT_WRIST] = REF_RW
    body_pos[0, :, TORSO] = torch.tensor([1.0, 2.0, 1.1])
    focus_phase = torch.tensor([[focus, 0.5]]).expand(FUTURE, 2)[None].clone()
    env.state_buffer = {
        "body_pos_w_future": body_pos,
        "focus_phase": focus_phase,
    }
    env.measured_root = (REF_ROOT - torch.tensor(root_error))[None]
    return env


def restore_targets(env):
    """Undo the in-place mutation; the real loop rebuilds body_pos each step."""
    body_pos = env.state_buffer["body_pos_w_future"]
    body_pos[0, :, PELVIS] = REF_ROOT
    body_pos[0, :, LEFT_WRIST] = REF_LW
    body_pos[0, :, RIGHT_WRIST] = REF_RW


class OnlineWristCompensationTest(unittest.TestCase):
    def test_none_mode_changes_nothing_but_records_baseline(self):
        env = make_env("none")
        before = env.state_buffer["body_pos_w_future"].clone()
        env._apply_wrist_compensation(env.measured_root)
        torch.testing.assert_close(env.state_buffer["body_pos_w_future"], before)
        self.assertIsNotNone(env._comp_last)
        np.testing.assert_allclose(
            env._comp_last["root_error"], [0.03, -0.02, 0.01], atol=1e-6
        )
        np.testing.assert_allclose(env._comp_last["applied_offset_lw"], [0, 0, 0])
        # The trace still carries the measured wrists for baseline analysis.
        np.testing.assert_allclose(
            env._comp_last["meas_lw"], (REF_LW - torch.tensor([0.02, 0, 0])).tolist()
        )

    def test_integral_accumulates_inside_focus_and_fades_in(self):
        env = make_env("offset")
        env._apply_wrist_compensation(env.measured_root)
        # First step: offset = gain * residual, applied scaled by alpha = 0.05.
        self.assertAlmostEqual(env._comp_fade_alpha, 0.05)
        np.testing.assert_allclose(
            env._wrist_offset[0], [0.002, 0.0, 0.0], atol=1e-7
        )
        np.testing.assert_allclose(
            env._wrist_offset[1], [0.0, 0.002, 0.0], atol=1e-7
        )
        np.testing.assert_allclose(
            env._comp_last["applied_offset_lw"], [0.05 * 0.002, 0.0, 0.0], atol=1e-8
        )
        after = env.state_buffer["body_pos_w_future"]
        # Only the two wrists move, every future token equally; root untouched.
        torch.testing.assert_close(
            after[0, :, LEFT_WRIST],
            REF_LW[None].expand(FUTURE, 3) + torch.tensor([1e-4, 0.0, 0.0]),
        )
        torch.testing.assert_close(after[0, :, PELVIS], REF_ROOT[None].expand(FUTURE, 3))
        torch.testing.assert_close(
            after[0, :, TORSO], torch.tensor([1.0, 2.0, 1.1])[None].expand(FUTURE, 3)
        )

    def test_integral_is_clamped_per_component(self):
        env = make_env("offset", poses=wrist_poses((0.30, 0.0, 0.0), (0.0, -0.30, 0.0)))
        for _ in range(200):
            restore_targets(env)
            env._apply_wrist_compensation(env.measured_root)
        np.testing.assert_allclose(env._wrist_offset[0], [0.15, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(env._wrist_offset[1], [0.0, -0.15, 0.0], atol=1e-6)

    def test_offset_fades_out_and_holds_outside_focus(self):
        env = make_env("offset", focus=0.0)
        env._comp_fade_alpha = 1.0
        env._wrist_offset = torch.tensor([[0.04, 0.0, 0.0], [0.0, 0.04, 0.0]])
        env._apply_wrist_compensation(env.measured_root)
        self.assertAlmostEqual(env._comp_fade_alpha, 0.95)
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0, LEFT_WRIST],
            REF_LW + torch.tensor([0.95 * 0.04, 0.0, 0.0]),
        )
        for _ in range(30):
            restore_targets(env)
            env._apply_wrist_compensation(env.measured_root)
        self.assertEqual(env._comp_fade_alpha, 0.0)
        # Fully faded out: targets untouched, but the integrator holds its
        # value so a re-entered window does not restart convergence from zero.
        restore_targets(env)
        before = env.state_buffer["body_pos_w_future"].clone()
        env._apply_wrist_compensation(env.measured_root)
        torch.testing.assert_close(env.state_buffer["body_pos_w_future"], before)
        np.testing.assert_allclose(env._wrist_offset[0], [0.04, 0.0, 0.0])

    def test_feedforward_gain_zero_uses_clamped_root_error(self):
        env = make_env("offset", gain=0.0, root_error=(0.30, -0.02, 0.01))
        env._comp_fade_alpha = 1.0
        env._apply_wrist_compensation(env.measured_root)
        np.testing.assert_allclose(
            env._comp_last["applied_offset_lw"], [0.15, -0.02, 0.01], atol=1e-6
        )

    def test_missing_fk_poses_holds_the_offset_without_crashing(self):
        env = make_env("offset", poses={})
        env._wrist_offset = torch.tensor([[0.03, 0.0, 0.0], [0.0, 0.03, 0.0]])
        env._comp_fade_alpha = 1.0
        env._apply_wrist_compensation(env.measured_root)
        np.testing.assert_allclose(env._wrist_offset[0], [0.03, 0.0, 0.0])
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0, LEFT_WRIST],
            REF_LW + torch.tensor([0.03, 0.0, 0.0]),
        )
        self.assertNotIn("meas_lw", env._comp_last)

    def test_compensation_is_inert_on_every_guard(self):
        guards = (
            ("reference_playing", False),
            ("localization_paused", True),
            ("requires_restart", True),
            ("awaiting_fresh_chunk", True),
            ("held_sample", {"root_pos": np.zeros((1, 3))}),
        )
        for field, value in guards:
            env = make_env("offset")
            env._comp_fade_alpha = 0.7
            env._wrist_offset = torch.full((2, 3), 0.05)
            setattr(env, field, value)
            before = env.state_buffer["body_pos_w_future"].clone()
            env._apply_wrist_compensation(env.measured_root)
            torch.testing.assert_close(env.state_buffer["body_pos_w_future"], before)
            self.assertIsNone(env._comp_last, msg=field)
            self.assertEqual(env._comp_fade_alpha, 0.0, msg=field)
            torch.testing.assert_close(env._wrist_offset, torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
