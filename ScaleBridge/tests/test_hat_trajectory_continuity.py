"""Trajectory progress, reversal, legacy isolation, and live clock wiring."""

import unittest
from unittest.mock import Mock

import numpy as np

from scalebridge.online.trajectory_continuity import continuity_mode, match_wrist_trajectory
from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.reference import sample_chunk
import test_hat_focus_handoff as base


def wrist_chunk(x, fps=20., sequence=1):
    data = base.chunk(sequence)
    count = len(x)
    data["frame_count"] = count
    data["fps"] = fps
    for key, values in tuple(data.items()):
        if isinstance(values, np.ndarray):
            data[key] = np.repeat(values[:1], count, axis=0)
    data["right_wrist_pos"][:] = 0.
    data["right_wrist_pos"][:, 0] = x
    return data


def curves():
    old = wrist_chunk(np.linspace(.016, .084, 41))
    new = wrist_chunk(np.r_[np.linspace(.033, .016, 11),
                            np.linspace(.016, .067, 31)[1:]], sequence=2)
    return old, new


class TrajectoryMatchingTests(unittest.TestCase):
    def test_same_point_on_wrong_branch_is_not_shared_progress(self):
        old, new = curves()
        result = match_wrist_trajectory(old, new, 0, 10, 2)
        self.assertEqual(result["selected_frame"], 20)
        self.assertAlmostEqual(result["matched_start_frame"], 10.)
        self.assertGreater(new["right_wrist_pos"][30, 0],
                           new["right_wrist_pos"][20, 0])

    def test_legitimate_backward_history_is_not_forced_forward(self):
        _, new = curves()
        old = wrist_chunk(np.linspace(.033, -.035, 41))
        result = match_wrist_trajectory(old, new, 0, 10, 2)
        self.assertEqual(result["selected_frame"], 10)
        self.assertLess(new["right_wrist_pos"][10, 0],
                        new["right_wrist_pos"][0, 0])

    def test_history_uses_duration_when_frame_rates_differ(self):
        _, new = curves()
        old = wrist_chunk(.016 + np.arange(40) / 15. * .034, fps=15.)
        result = match_wrist_trajectory(old, new, 0, 7.5, 2)
        self.assertEqual(result["selected_frame"], 20)
        self.assertAlmostEqual(result["history_duration_s"], .5)

    def test_other_body_poses_do_not_choose_wrist_progress(self):
        old, new = curves()
        for side in ("root", "head", "left_wrist"):
            new[side + "_pos"][:] = 100.
        result = match_wrist_trajectory(old, new, 0, 10, 2)
        self.assertEqual(result["selected_frame"], 20)

    def test_quaternion_sign_does_not_change_match(self):
        old, new = curves()
        new["right_wrist_quat_wxyz"][::2] *= -1
        result = match_wrist_trajectory(old, new, 0, 10, 2)
        self.assertEqual(result["selected_frame"], 20)

    def test_no_match_for_missing_motion_window_or_large_pose_error(self):
        old, new = curves()
        self.assertIsNone(match_wrist_trajectory(old, new, 0, 0, 2))
        self.assertIsNone(match_wrist_trajectory(old, new, 0, 10, 2, reserve_s=3.))
        new["right_wrist_pos"][:, 1] = .1
        self.assertIsNone(match_wrist_trajectory(old, new, 0, 10, 2))

    def test_old_boolean_option_remains_distinct(self):
        self.assertEqual(continuity_mode(True), "legacy")
        self.assertEqual(continuity_mode(False), "off")
        self.assertEqual(continuity_mode("trajectory"), "trajectory")
        with self.assertRaises(ValueError):
            continuity_mode("trajectroy")


class TrajectoryWiringTests(unittest.TestCase):
    def env(self):
        old, new = curves()
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.cfg = {"min_chunk_execution_fraction": .1}
        env.continuity_mode = "trajectory"
        env.chunk_start_reference_continuity = True
        env.continuity_search_ahead_s = 1.
        env.phase_max_root_jump_m = .2
        env.phase_rotation_weight = .08
        env.phase_velocity_weight = .03
        env.phase_measured_weight = .15
        env.chunk_phase_alignment = True
        env.reference_playing = True
        env.localization_paused = False
        env.awaiting_fresh_chunk = False
        env.held_sample = None
        env.chunk = old
        env.chunk_started_ns = 1_000_000_000
        env._trajectory_source_chunk = old
        env._trajectory_source_start_frame = 0.
        env._latency_start_frame = Mock(return_value=(2, .1, "timestamp"))
        env.last_published_robot_state = {}
        for prefix in ("root", "head", "left_wrist", "right_wrist"):
            posname = "root_pos_world" if prefix == "root" else prefix + "_pos_world"
            qname = "root_quat_world_wxyz" if prefix == "root" else prefix + "_quat_world_wxyz"
            env.last_published_robot_state[posname] = old[prefix + "_pos"][10]
            env.last_published_robot_state[qname] = old[prefix + "_quat_wxyz"][10]
        return env, new

    def test_new_selector_finds_segment_end_outside_legacy_window(self):
        env, new = self.env()
        selected, state = env._phase_match_start_frame(new, 1_500_000_000)
        self.assertEqual(selected, 20)
        self.assertEqual(state["mode"], "trajectory_continuity")
        self.assertEqual(state["trajectory_match"]["old_sequence_id"], 1)
        env.continuity_mode = "legacy"
        _, legacy = env._phase_match_start_frame(new, 1_500_000_000)
        self.assertEqual(legacy["mode"], "reference_continuity")
        self.assertEqual(legacy["search_range"], [2, 22])

    def test_missing_raw_history_falls_back_to_original_selector(self):
        env, new = self.env()
        env._trajectory_source_chunk = None
        _, state = env._phase_match_start_frame(new, 1_500_000_000)
        self.assertEqual(state["mode"], "reference_continuity")

    def test_root_guard_and_focus_overlay_keep_existing_path(self):
        env, new = self.env()
        new["root_pos"][:] = [1., 0., 0.]
        _, state = env._phase_match_start_frame(new, 1_500_000_000)
        self.assertNotEqual(state["mode"], "trajectory_continuity")
        env, new = self.env()
        env.focus_handoff = Mock(bypass_continuity=True)
        _, state = env._phase_match_start_frame(new, 1_500_000_000)
        self.assertNotEqual(state["mode"], "trajectory_continuity")

    def test_acceptance_shifts_clock_anchors_seam_and_keeps_raw_history(self):
        fixture = base.FocusHandoffHandTests()
        env = fixture.acceptance_env()
        env.continuity_mode = "trajectory"
        env.focus_handoff = None
        env.reference_play_gate = False
        env.previous_chunk = None
        env.previous_chunk_started_ns = None
        env.blend_started_ns = None
        env.future_offsets_s = np.arange(6) / 50.
        env.cfg["min_chunk_execution_fraction"] = 0.
        env._min_execution_hold_active = Mock(return_value=False)
        env._phase_match_start_frame.return_value = (20, dict(
            mode="trajectory_continuity", expected_frame=3, source_age_s=.113,
            pose_cost=0., transition_valid=True))
        raw = fixture.wire_chunk(43)
        env.chunk_subscriber.receive_latest.return_value = raw
        old_target = env._sample_reference(1_100_000_000)["right_wrist_pos"][0].copy()
        self.assertTrue(env._accept_latest_chunk(1_100_000_000))
        self.assertAlmostEqual(env._trajectory_source_start_frame, 20., places=6)
        self.assertAlmostEqual(env._trajectory_source_chunk["right_wrist_pos"][20, 0], .12)
        elapsed = (1_100_000_000 - env.chunk_started_ns) * 1e-9
        np.testing.assert_allclose(sample_chunk(env.chunk, elapsed, [0.])["right_wrist_pos"][0],
                                   old_target, atol=1e-7)
        self.assertAlmostEqual(raw["right_wrist_pos"][20][0], .12)


if __name__ == "__main__":
    unittest.main()
