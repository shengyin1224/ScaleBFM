"""Offline adaptive2 regressions; no model, DDS, sockets, or robot startup."""

import unittest
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from scalebridge.env.motion_tracking_hat4_adaptive2 import (
    HAT4Adaptive2MotionTrackingEnv,
)
from scalebridge.online.focus_handoff import (
    GraspFingerChunkControl,
    PerHandFocusHandoff,
)


def chunk(sequence_id=1, count=100, direction=1.0):
    result = dict(frame_count=count, fps=30.0, sequence_id=sequence_id)
    for side in ("left", "right"):
        result[f"{side}_wrist_pos"] = np.c_[
            direction * np.arange(count) * .003, np.zeros((count, 2))
        ]
        result[f"{side}_wrist_quat_wxyz"] = np.tile(
            [1.0, 0.0, 0.0, 0.0], (count, 1)
        )
        tips = np.zeros((count, 5, 3))
        tips[:, 1:, 0] = .15
        result[f"{side}_fingertip_local"] = tips
    return result


def filter_env():
    env = HAT4Adaptive2MotionTrackingEnv.__new__(HAT4Adaptive2MotionTrackingEnv)
    env.cfg = {}
    env._adaptive2_hold_enabled = True
    env._adaptive2_hold_timeout_s = 15.0
    env._adaptive2_grasp = {"left": None, "right": None}
    env._adaptive2_hold_suppressed = {"left": False, "right": False}
    env._adaptive2_last_hand_motor = {
        "left": np.ones(6), "right": np.ones(6)
    }
    env._adaptive2_grasp_serial = 0
    env._handoff_finger_control = GraspFingerChunkControl("hold_progress")
    env.hand_dof12_to_motor = lambda value: np.asarray(value, dtype=float)
    env.hand_motor_to_dof12 = lambda value: np.asarray(value, dtype=float)
    return env


def solved(sequence_id, close=(), opening=()):
    return {
        "_context": {
            "adaptive2": {
                "sequence_id": sequence_id,
                "close": tuple(close),
                "open": tuple(opening),
                "timestamp_ns": 0,
            }
        }
    }


class Adaptive2Tests(unittest.TestCase):
    def test_close_does_not_cap_scheduler_without_full_handoff(self):
        env = HAT4Adaptive2MotionTrackingEnv.__new__(
            HAT4Adaptive2MotionTrackingEnv
        )
        env.focus_handoff = None
        self.assertIsNone(env._adaptive_schedule_boundary(12, 0))
        env.focus_handoff = PerHandFocusHandoff()
        self.assertEqual(env._adaptive_schedule_boundary(12, 0), 12)

    def test_stall_budget_crosses_only_consistent_replans_and_is_capped(self):
        env = HAT4Adaptive2MotionTrackingEnv.__new__(
            HAT4Adaptive2MotionTrackingEnv
        )
        env.cfg = {
            "rtc_enabled": True,
            "rtc_prefix_frames": 5,
            "adaptive2_stall_carry_max_m": .02,
            "adaptive2_stall_carry_min_cosine": .5,
        }
        old = chunk(1)
        forward = chunk(2)
        reverse = chunk(3, direction=-1.0)
        budget = {"left": .05, "right": .015}
        self.assertEqual(
            env._adaptive2_carried_stall_budget(
                old, 20, forward, 10, budget
            ),
            {"left": .02, "right": .015},
        )
        self.assertEqual(
            env._adaptive2_carried_stall_budget(
                old, 20, reverse, 10, budget
            ),
            {"left": 0.0, "right": 0.0},
        )

    def test_hold_progress_survives_chunk_change_and_open_releases_it(self):
        env = filter_env()
        closing = np.array([.5, .5, .5, .5, .5, .4])
        idle = np.ones(6)
        left, _ = env._filter_raw_hand_targets(
            (closing, idle), solved(10, close=("left",))
        )
        np.testing.assert_allclose(left, closing)

        reopening = np.array([.8, .8, .8, .8, .8, .6])
        left, _ = env._filter_raw_hand_targets(
            (reopening, idle), solved(11)
        )
        np.testing.assert_allclose(left[:5], closing[:5])
        self.assertEqual(left[5], reopening[5])

        left, _ = env._filter_raw_hand_targets(
            (reopening, idle), solved(12, opening=("left",))
        )
        np.testing.assert_allclose(left, reopening)
        self.assertIsNone(env._adaptive2_grasp["left"])

    def test_per_hand_focus_result_drives_only_matching_event_window(self):
        env = filter_env()
        env._adaptive2_hand_focus = {}
        payload = {
            "hands": {
                "left": {"events": [
                    {"kind": "close", "start": 10, "end": 20}
                ]},
                "right": {"events": [
                    {"kind": "open", "start": 30, "end": 40}
                ]},
            }
        }
        env._on_chunk_focus_result(7, payload)
        self.assertEqual(env._adaptive2_event_sides(7, 15), (("left",), ()))
        self.assertEqual(env._adaptive2_event_sides(7, 35), ((), ("right",)))

    def test_config_is_opt_in_and_original_adaptive_is_unchanged(self):
        config = str(Path(__file__).resolve().parents[1] / "scalebridge/config")
        with initialize_config_dir(version_base=None, config_dir=config):
            old = compose(
                config_name="hat4_online",
                overrides=["env=motion_tracking_hat4_adaptive"],
            )
            new = compose(
                config_name="hat4_online",
                overrides=["env=motion_tracking_hat4_adaptive2"],
            )
        self.assertEqual(old.env.config.rtc_prefix_frames, 10)
        self.assertEqual(new.env.config.rtc_prefix_frames, 5)
        self.assertEqual(new.env.config.rtc_hard_prefix_frames, 1)
        self.assertEqual(new.env.config.rtc_blend_frames, 4)
        self.assertFalse(new.env.config.focus_handoff_enabled)
        self.assertTrue(new.env.config.adaptive2_independent_hold_progress)
        self.assertEqual(new.env.config.focus_handoff_finger_mode, "hold_progress")


if __name__ == "__main__":
    unittest.main()
