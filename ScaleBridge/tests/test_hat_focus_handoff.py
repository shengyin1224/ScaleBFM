"""Offline handoff regressions; no DDS, network, or robot execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.focus_handoff import FocusHandoff, FocusAdaptiveExecution, rebase_wrist_chunk
from scalebridge.online.reference import sample_chunk, quat_mul
from scalebridge.online.protocol import make_hat_chunk


def chunk(sequence=1, source=10):
    count = 60
    out = {"fps": 30.0, "frame_count": count,
           "sequence_id": sequence, "source_state_sequence_id": source}
    for name in ("root", "head", "left_wrist", "right_wrist"):
        out[f"{name}_pos"] = np.zeros((count, 3), dtype=np.float32)
        out[f"{name}_quat_wxyz"] = np.tile([1., 0., 0., 0.], (count, 1))
    out["right_wrist_pos"][:, 0] = .08 + np.arange(count) * .002
    for name in ("left", "right"):
        out[f"{name}_fingertip_local"] = np.zeros((count, 5, 3), dtype=np.float32)
    return out


def phase():
    out = np.tile([0., 2.], (60, 1))
    out[10:45] = [1., -1.]
    return out


class FocusHandoffTests(unittest.TestCase):
    def controller(self):
        return FocusHandoff(position_tolerance_m=.01, rotation_tolerance_deg=5.)

    def arm(self, control):
        control.observe_prediction(chunk(1), phase())
        control.observe_prediction(chunk(2), phase())

    def test_multiple_chunks_and_actual_orientation_required(self):
        control = self.controller()
        control.observe_prediction(chunk(), phase())
        self.assertIsNone(control.entry)
        self.arm(control)
        sample = sample_chunk(chunk(), 0, [0.])
        poses = {"right_wrist_yaw_link": (np.array([.04, 0, 0]), np.array([1., 0, 0, 0]))}
        self.assertFalse(control.arrive(poses, 20, sample))
        poses["right_wrist_yaw_link"] = (np.array([.10, 0, 0]), np.array([0., 0, 0, 1.]))
        self.assertFalse(control.arrive(poses, 20, sample))
        poses["right_wrist_yaw_link"] = (np.array([.10, 0, 0]), np.array([1., 0, 0, 0]))
        self.assertTrue(control.arrive(poses, 20, sample))
        self.assertEqual(control.state, "await_chunk")
        self.assertFalse(control.active)
        self.assertTrue(control.bypass_continuity)
        self.assertFalse(control.accepts(chunk(3, 19)))
        self.assertTrue(control.accepts(chunk(4, 20)))
        control.start()
        self.assertTrue(control.active)

    def test_inconsistent_predictions_do_not_unlock(self):
        control = self.controller()
        control.observe_prediction(chunk(1), phase())
        other = chunk(2)
        other["right_wrist_pos"] += .1
        control.observe_prediction(other, phase())
        self.assertIsNone(control.entry)

    def test_late_and_duplicate_predictions_cannot_supply_consensus(self):
        control = self.controller()
        control.observe_prediction(chunk(5), phase())
        control.observe_prediction(chunk(5), phase())
        control.observe_prediction(chunk(4), phase())
        self.assertIsNone(control.entry)

    def test_focus_ends_then_rearms_for_a_later_event(self):
        control = self.controller()
        control.state = "await_chunk"
        control.start()
        self.assertFalse(control.update_phase(False, 5))
        self.assertFalse(control.update_phase(True, 5))
        self.assertTrue(control.update_phase(False, 5))
        self.assertFalse(control.active)
        self.assertFalse(control.bypass_continuity)
        ongoing = phase(); ongoing[:10, 0] = 1
        control.observe_prediction(chunk(6), ongoing)
        self.assertEqual(control.state, "cooldown")
        control.observe_prediction(chunk(7), phase())
        control.observe_prediction(chunk(8), phase())
        self.assertEqual(control.state, "approach")
        self.assertIsNotNone(control.entry)

    def test_fixed_offset_preserves_wrist_increments_and_raw_prediction(self):
        raw = chunk()
        outgoing = sample_chunk(raw, 0, [0.])
        outgoing["right_wrist_pos"][0, 0] = .12
        result = rebase_wrist_chunk(raw, 0, outgoing, ["right"])
        np.testing.assert_allclose(result["right_wrist_pos"][:3, 0], [.12, .122, .124])
        np.testing.assert_allclose(raw["right_wrist_pos"][:3, 0], [.08, .082, .084])
        np.testing.assert_allclose(np.diff(result["right_wrist_pos"], axis=0),
                                   np.diff(raw["right_wrist_pos"], axis=0), atol=2e-8)
        self.assertIs(result["root_pos"], raw["root_pos"])
        self.assertIs(result["right_fingertip_local"], raw["right_fingertip_local"])

    def test_release_can_rearm_despite_history_focus_at_frame_zero(self):
        control = self.controller()
        control.state = "focus"
        control.update_phase(True, 5)
        control.update_phase(False, 5)
        release = phase();release[0, 0] = 1
        control.observe_prediction(chunk(6), release)
        control.observe_prediction(chunk(7), release)
        self.assertEqual(control.state, "approach")
        np.testing.assert_allclose(control.entry["right_wrist_pos"], [.10, 0., 0.])

    def test_replan_anchors_at_live_fractional_cursor_without_double_accumulation(self):
        raw = chunk()
        outgoing = sample_chunk(raw, 0, [0.]);outgoing["right_wrist_pos"][0, 0] = .12
        first = rebase_wrist_chunk(raw, .05, outgoing, ["right"])
        at_swap = sample_chunk(first, .15, [0.])
        second = rebase_wrist_chunk(chunk(2), .05, at_swap, ["right"])
        np.testing.assert_allclose(sample_chunk(second, .05, [0.])["right_wrist_pos"],
                                   at_swap["right_wrist_pos"], atol=1e-7)

    def test_orientation_anchor_keeps_new_rotation_deltas(self):
        raw = chunk()
        angles = np.arange(60) * .001
        raw["right_wrist_quat_wxyz"] = np.stack([np.cos(angles), angles*0, angles*0, np.sin(angles)], axis=1)
        outgoing = sample_chunk(raw, 0, [0.])
        rotation = np.array([np.cos(.2), 0, 0, np.sin(.2)])
        outgoing["right_wrist_quat_wxyz"][0] = rotation
        result = rebase_wrist_chunk(raw, 0, outgoing, ["right"])
        np.testing.assert_allclose(result["right_wrist_quat_wxyz"],
                                   quat_mul(rotation, raw["right_wrist_quat_wxyz"]), atol=1e-7)


class FocusHandoffHandTests(unittest.TestCase):
    def env(self):
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.focus_handoff = FocusHandoff()
        env._handoff_hand_hold = None
        env._handoff_ik_min_sequence = None
        env.open_hand_dof12 = np.zeros(12)
        env.startup_right_hand_dof12 = np.full(12, .1)
        env.hand_ik_worker = Mock()
        env.hand_ik_worker.submit.return_value = 100
        env.hand_ik_worker.latest.return_value = None
        env.reference_playing = True
        env.localization_paused = False
        env.requires_restart = False
        env.awaiting_fresh_chunk = False
        env.held_sample = None
        env.last_reference_sample = sample_chunk(chunk(), 0, [0.])
        env.finger_chunk_source = "playback"
        env._hand_prediction_target = lambda q, side: q
        env.simulator = SimpleNamespace(set_hand_ik_target=Mock())
        return env

    def test_fingers_locked_from_first_step_including_opening_predictions(self):
        env = self.env()
        env.hand_ik_worker.latest.return_value = {"lh": np.ones(12), "rh": np.ones(12)}
        env._send_hand_target()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_array_equal(left, np.zeros(12))
        np.testing.assert_array_equal(right, np.full(12, .1))
        env.hand_ik_worker.latest.assert_not_called()
        env.hand_ik_worker.submit.assert_not_called()

    def test_unlock_rejects_old_ik_then_relocks_last_applied_grip(self):
        env = self.env()
        env.focus_handoff.state = "focus"
        env._send_hand_target()
        env.hand_ik_worker.latest.assert_called_with(min_sequence=100)
        np.testing.assert_array_equal(env.simulator.set_hand_ik_target.call_args.args[0], np.zeros(12))
        env.hand_ik_worker.latest.return_value = {"lh": np.full(12, .7), "rh": np.full(12, .8)}
        env._send_hand_target()
        env.focus_handoff.state = "cooldown"
        env._send_hand_target()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_array_equal(left, np.full(12, .7))
        np.testing.assert_array_equal(right, np.full(12, .8))

    def test_localization_outage_holds_grip_even_inside_focus(self):
        env = self.env()
        env.focus_handoff.state = "focus"
        env.localization_paused = True
        env._send_hand_target()
        env.hand_ik_worker.submit.assert_not_called()

    def test_late_focus_result_still_supplies_entry_for_retired_chunk(self):
        env = self.env()
        env.focus_worker = Mock()
        env.focus_worker.latest.side_effect = [(1, phase()), (2, phase())]
        env._focus_prediction_chunks = {1: chunk(1), 2: chunk(2)}
        env._focus_phase_prediction = None
        env.chunk = chunk(4)
        env.previous_chunk = chunk(3)
        env._update_chunk_focus();env._update_chunk_focus()
        self.assertIsNotNone(env.focus_handoff.entry)
        self.assertEqual(env._focus_phase_prediction["sequence_id"], 2)

    def acceptance_env(self):
        env = self.env()
        env.cfg = {"rtc_enabled": True, "min_chunk_execution_fraction": .66}
        env.state_sequence_id = 43
        env.last_chunk_sequence_id = 10
        env.minimum_source_state_sequence_id = None
        env.chunk_alignment = "fixed"
        env.alignment_anchor = (np.zeros(3), np.array([1., 0, 0, 0]))
        env.chunk_subscriber = Mock()
        env.chunk_subscriber.receive_error.return_value = None
        env._pending_raw_chunk = None
        env.chunk = chunk(10)
        env.chunk_started_ns = 900_000_000
        env.last_chunk_received_ns = 900_000_000
        env.state_buffer = {
            "root_pos_buffer": torch.tensor([[[0., 0., 0.]]]),
            "root_quat_wxyz_buffer": torch.tensor([[[1., 0., 0., 0.]]]),
        }
        env._phase_match_start_frame = Mock(return_value=(3, {
            "mode": "time_aligned_reference", "expected_frame": 3,
            "source_age_s": .1, "pose_cost": 0., "transition_valid": True,
        }))
        env._source_state_timestamp_ns = Mock(return_value=1_000_000_000)
        env._submit_chunk_focus = Mock()
        env._focus_prediction_chunks = {}
        env._focus_phase_prediction = None
        env.simulator = Mock(spec=[])
        return env

    def wire_chunk(self, source):
        raw = chunk(11, source)
        return make_hat_chunk(generated_at_ns=1_100_000_000,
                              action_representation="original", **raw)

    def test_fresh_acceptance_bypasses_rtc_and_hold_and_preserves_raw_entry_pose(self):
        env = self.acceptance_env()
        env.focus_handoff.state = "await_chunk"
        env.focus_handoff.source_fence = 43
        env.focus_handoff.hold = sample_chunk(chunk(), 0, [0.])
        env.focus_handoff.hold["right_wrist_pos"][0, 0] = .12
        self.assertFalse(env._min_execution_hold_active(1_100_000_000))
        env.chunk_subscriber.receive_latest.return_value = self.wire_chunk(42)
        self.assertFalse(env._accept_latest_chunk(1_100_000_000))
        self.assertEqual(env.focus_handoff.state, "await_chunk")
        env.chunk_subscriber.receive_latest.return_value = self.wire_chunk(43)
        self.assertTrue(env._accept_latest_chunk(1_100_000_000))
        self.assertTrue(env.focus_handoff.active)
        self.assertEqual(env.chunk["execution_alignment"]["rtc"]["bypass_reason"], "focus_handoff")
        self.assertEqual(env.chunk_started_ns, 1_000_000_000)
        live = sample_chunk(env.chunk, .1, [0.])
        np.testing.assert_allclose(live["right_wrist_pos"][0, 0], .12, atol=1e-7)
        np.testing.assert_allclose(env._focus_prediction_chunks[11]["right_wrist_pos"][3, 0], .086)

    def test_late_phase_uses_source_clock_and_excludes_prearrival_predictions(self):
        env = self.env()
        env.future_offsets_s = np.array([0., .1])
        env.focus_handoff.source_fence = 20
        prediction = chunk(5, 19)
        prediction["focus_phase"] = phase()
        prediction["_focus_source_started_ns"] = 1_000_000_000
        env._focus_phase_prediction = prediction
        self.assertIsNone(env._focus_handoff_phase(1_400_000_000))
        prediction["source_state_sequence_id"] = 20
        np.testing.assert_allclose(env._focus_handoff_phase(1_400_000_000)[:, 0], [1., 1.])
        np.testing.assert_allclose(env._focus_handoff_phase(2_600_000_000)[:, 0], [0., 0.])

    def test_focus_history_ends_at_prediction_source_time_not_reception(self):
        env = self.env()
        env.focus_worker = Mock()
        env._focus_history_snapshot = Mock(return_value=np.zeros((30, 12)))
        prediction = chunk()
        prediction["_focus_source_started_ns"] = 1_000_000_000
        env._submit_chunk_focus(prediction, 1_200_000_000)
        env._focus_history_snapshot.assert_called_once_with(1_000_000_000, 30.)


class FocusAdaptiveTests(unittest.TestCase):
    def test_adaptive_entry_uses_selected_hand_events_not_history_union(self):
        for kind in ("history_tail", "left_only", "right_close"):
            with self.subTest(kind=kind):
                env = FocusHandoffHandTests().env()
                env.focus_handoff = None
                env.focus_adaptive = FocusAdaptiveExecution()
                env.focus_worker = Mock()
                env._focus_phase_prediction = None
                env._focus_result_logged = False
                env._focus_prediction_chunks = {}
                env.chunk = chunk(9)
                env.previous_chunk = None
                results = []
                for sequence in range(1, 4):
                    prediction = chunk(sequence)
                    prediction["_focus_source_started_ns"] = 1_000_000_000
                    env._focus_prediction_chunks[sequence] = prediction
                    outside = np.tile([0., 2.], (60, 1)).astype(np.float32)
                    union = outside.copy()
                    union[0] = [1., -.05]  # Observed Oct-08 history-boundary tail.
                    left = phase() if kind == "left_only" else outside
                    right = phase() if kind == "right_close" else outside
                    results.append((sequence, dict(phase=union, hands={
                        "left": {"phase": left}, "right": {"phase": right},
                    })))
                env.focus_worker.latest.side_effect = results
                for _ in results:
                    env._update_chunk_focus()
                env.focus_adaptive.update(1_400_000_000, self.measured(.10), running=True)
                self.assertEqual(env.focus_adaptive.execution_frames,
                                 10 if kind == "right_close" else 30)
                self.assertEqual(env.focus_adaptive.active, kind == "right_close")

    def arm(self, control, start_ns=1_000_000_000, first=1, x_offset=0):
        for sequence in range(first, first + 3):
            prediction = chunk(sequence)
            prediction["right_wrist_pos"][:, 0] += x_offset
            prediction["_focus_source_started_ns"] = start_ns
            control.observe_prediction(prediction, phase())

    def measured(self, x):
        return {"right_wrist_yaw_link": (np.array([x, 0., 0.]), np.array([1., 0., 0., 0.]))}

    def test_consensus_actual_distance_and_focus_all_required(self):
        control = FocusAdaptiveExecution()
        prediction = chunk(1)
        prediction["_focus_source_started_ns"] = 1_000_000_000
        control.observe_prediction(prediction, phase())
        control.update(1_100_000_000, self.measured(.10), running=True)
        self.assertEqual(control.execution_frames, 30)
        self.assertFalse(control.active)
        self.arm(control, first=2)
        control.update(1_200_000_000, self.measured(.054), running=True)
        self.assertFalse(control.active)  # 4.6 cm away: keep approach cadence.
        self.assertEqual(control.execution_frames, 30)
        control.update(1_250_000_000, self.measured(.056), running=True)
        self.assertEqual(control.execution_frames, 10)  # 4.4 cm; no FOCUS yet.
        self.assertFalse(control.active)
        control.update(1_400_000_000, self.measured(.056), running=True)
        self.assertTrue(control.active)

    def test_freeze_after_focus_and_rearm_for_next_release_window(self):
        control = FocusAdaptiveExecution()
        self.arm(control)
        control.update(1_400_000_000, self.measured(.10), running=True)
        self.assertTrue(control.active)
        control.update(2_600_000_000, self.measured(.10), running=True)
        self.assertFalse(control.active)
        self.assertEqual(control.execution_frames, 10)
        self.arm(control, start_ns=3_000_000_000, first=4, x_offset=.3)
        control.update(3_400_000_000, self.measured(.10), running=True)
        self.assertFalse(control.active)
        control.update(3_400_000_000, self.measured(.40), running=True)
        self.assertTrue(control.active)

    def test_outage_and_expired_prediction_cannot_unlock(self):
        control = FocusAdaptiveExecution()
        self.arm(control)
        control.update(1_400_000_000, self.measured(.10), running=False)
        self.assertFalse(control.fast)
        self.assertFalse(control.active)
        control.update(4_000_000_000, self.measured(.10), running=True)
        self.assertFalse(control.active)
        self.assertIsNone(control.finger_sample)

    def test_invalid_parameters(self):
        for kwargs in ({"approach_frames": 0}, {"near_frames": 31},
                       {"near_distance_m": float("nan")}, {"consensus_chunks": 0}):
            with self.assertRaises(ValueError):
                FocusAdaptiveExecution(**kwargs)

    def test_identical_frame_hold_for_all_anchor_modes_and_representations(self):
        from test_hat_online_chunk_hold import make_env
        for mode in ("real", "real_wrist", "actual"):
            for representation in ("original", "relative_chunk"):
                env = make_env()
                env.cfg["relative_anchor_mode"] = mode
                env.chunk["action_representation"] = representation
                env.focus_adaptive = FocusAdaptiveExecution()
                self.assertTrue(env._min_execution_hold_active(990_000_000))
                self.assertFalse(env._min_execution_hold_active(1_000_000_000))
                env.focus_adaptive.fast = True
                self.assertTrue(env._min_execution_hold_active(330_000_000))
                self.assertFalse(env._min_execution_hold_active(340_000_000))

    def test_motor_hold_and_late_ik_are_gated_at_each_focus_window(self):
        env = FocusHandoffHandTests().env()
        env.focus_handoff = None
        env.focus_adaptive = FocusAdaptiveExecution()
        env.hand_ik_worker.latest.return_value = {"lh": np.ones(12), "rh": np.ones(12)}
        env._send_hand_target()
        env.hand_ik_worker.latest.assert_not_called()
        env.focus_adaptive.active = True
        env._send_hand_target()
        env.focus_adaptive.active = False
        env.hand_ik_worker.latest.reset_mock()
        env.hand_ik_worker.latest.return_value = {"lh": np.zeros(12), "rh": np.zeros(12)}
        env._send_hand_target()
        env.hand_ik_worker.latest.assert_not_called()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_array_equal(left, np.ones(12))
        np.testing.assert_array_equal(right, np.ones(12))
        env.focus_adaptive.active = True
        env.hand_ik_worker.submit.return_value = 200
        env._send_hand_target()
        env.hand_ik_worker.latest.assert_called_once_with(min_sequence=200)

    def test_held_chunks_still_supply_focus_without_body_acceptance(self):
        helper = FocusHandoffHandTests()
        env = helper.acceptance_env()
        env.focus_handoff = None
        env.focus_adaptive = FocusAdaptiveExecution()
        env.last_reference_sample = sample_chunk(chunk(), .1, [0.])
        env._sample_reference = Mock(return_value=env.last_reference_sample)
        env.chunk_subscriber.receive_latest.return_value = helper.wire_chunk(43)
        self.assertFalse(env._accept_latest_chunk(1_100_000_000))
        self.assertEqual(env.last_chunk_sequence_id, 10)
        self.assertEqual(env._pending_raw_chunk["sequence_id"], 11)
        self.assertIn(11, env._focus_prediction_chunks)
        env._submit_chunk_focus.assert_called_once()

    def test_new_chunk_rebases_position_to_live_target_without_rotating_it(self):
        helper = FocusHandoffHandTests()
        env = helper.acceptance_env()
        env.focus_handoff = None
        env.focus_adaptive = FocusAdaptiveExecution()
        env.cfg["rtc_enabled"] = False
        env.last_chunk_received_ns = 0
        env.last_reference_sample = sample_chunk(chunk(), .1, [0.])
        env.last_reference_sample["right_wrist_pos"][0, 0] = .12
        env._sample_reference = Mock(return_value=env.last_reference_sample)
        env.chunk_subscriber.receive_latest.return_value = helper.wire_chunk(43)
        self.assertTrue(env._accept_latest_chunk(1_100_000_000))
        live = sample_chunk(env.chunk, .1, [0.])
        np.testing.assert_allclose(live["right_wrist_pos"][0, 0], .12, atol=1e-7)
        np.testing.assert_allclose(env.chunk["right_wrist_quat_wxyz"], chunk()["right_wrist_quat_wxyz"])


if __name__ == "__main__":
    unittest.main()
