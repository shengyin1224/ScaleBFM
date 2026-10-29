"""Guard regressions: preparation, close/open, provenance and clock wiring."""

import unittest
from unittest.mock import Mock

import numpy as np

from scalebridge.online.continuity_hand_event_guard import ContinuityHandEventGuard, GuardChunkPreflight, HandEventPlaybackBoundary
from scalebridge.online.focus_handoff import PerHandFocusHandoff
from scalebridge.online.focus import HAND_BEND_COLUMNS
import test_hat_trajectory_continuity as trajectory_fixture
from scalebridge.online.reference import sample_chunk
import test_hat_focus_handoff as base


def prediction(bend, sequence=2, source=10):
    values = np.asarray(bend, dtype=np.float32)
    if values.ndim == 1:
        values = np.repeat(values[:, None], 5, axis=1)
    closure = np.zeros((len(values), 6), dtype=np.float32)
    closure[:, HAND_BEND_COLUMNS] = values
    chunk = {"sequence_id": sequence, "source_state_sequence_id": source,
             "frame_count": len(values), "fps": 20.}
    result = dict(chunk, hand_focus={"right": {"closure": closure}})
    return chunk, result


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = ContinuityHandEventGuard()

    def limit(self, bend, actual=0., lower=1):
        chunk, pred = prediction(bend)
        return self.guard.search_limit(chunk, lower, ("right",), pred,
                                       {"right": np.full(5, actual)})

    def test_success_prebend_is_not_a_conflict(self):
        # Entry values supplied by the user, including the old .206 prediction.
        for values in ([.099, .099, .105, .102, .169], [.206] * 5):
            bend = np.repeat(np.array(values)[None], 40, axis=0)
            self.assertEqual(self.limit(bend)[0], 39)

    def test_close_bounds_first_sustained_change_even_if_it_reopens(self):
        bend = np.r_[np.full(8, .12), np.full(8, .4), np.full(24, .12)]
        limit, state = self.limit(bend)
        self.assertEqual(limit, 7)
        self.assertEqual(state["events"],
                         [{"side": "right", "kind": "close", "conflict_frame": 8}])

    def test_open_is_symmetric(self):
        limit, state = self.limit(np.r_[np.full(8, .8), np.full(32, .4)], actual=.8)
        self.assertEqual(limit, 7)
        self.assertEqual(state["events"][0]["kind"], "open")

    def test_executed_change_small_change_and_twitch_do_not_block(self):
        self.assertEqual(self.limit(np.r_[np.full(8, .12), np.full(32, .4)], .4)[0], 39)
        self.assertEqual(self.limit(np.linspace(.1, .2, 40))[0], 39)
        self.assertEqual(self.limit(np.r_[np.zeros(8), [.5, .5], np.zeros(30)])[0], 39)

    def test_rotation_and_single_finger_do_not_block(self):
        chunk, pred = prediction(np.zeros(40))
        closure = pred["hand_focus"]["right"]["closure"]
        for column in (4, 0):
            closure[:] = 0.
            closure[8:, column] = .9
            self.assertEqual(self.guard.search_limit(
                chunk, 1, ("right",), pred, {"right": np.zeros(5)})[0], 39)

    def test_missing_stale_invalid_data_only_cancel_forward_search(self):
        chunk, pred = prediction(np.zeros(40))
        stale = dict(pred, sequence_id=1)
        for result in (None, stale, dict(pred, source_state_sequence_id=9)):
            limit, state = self.guard.search_limit(chunk, 3, ("right",), result, {})
            self.assertEqual(limit, 3)
            self.assertEqual(state["reason"], "prediction_pending")
        self.assertEqual(self.guard.search_limit(chunk, 3, ("right",), pred, {})[0], 3)

    def test_both_hands_use_earliest_fence_and_unlocked_hands_are_exempt(self):
        chunk, pred = prediction(np.r_[np.zeros(12), np.full(28, .5)])
        _, left = prediction(np.r_[np.zeros(8), np.full(32, .5)])
        pred["hand_focus"]["left"] = left["hand_focus"]["right"]
        actual = {"left": np.zeros(5), "right": np.zeros(5)}
        self.assertEqual(self.guard.search_limit(chunk, 1, ("left", "right"), pred, actual)[0], 7)
        self.assertEqual(self.guard.search_limit(chunk, 1, ("right",), pred, actual)[0], 11)
        self.assertEqual(self.guard.search_limit(chunk, 1, (), None, {})[0], 39)

    def test_already_changed_latency_frame_is_not_rewound(self):
        self.assertEqual(self.limit(np.full(40, .5))[0], 39)

    def test_config_validation(self):
        for args in ({"min_feedback_delta": 0}, {"sustain_s": float("nan")},
                     {"min_interval_delta": 2}, {"min_fingers": 1}):
            with self.assertRaises(ValueError):
                ContinuityHandEventGuard(**args)


class GuardWiringTests(unittest.TestCase):
    def env(self):
        env, chunk = trajectory_fixture.TrajectoryWiringTests().env()
        env.focus_handoff = PerHandFocusHandoff(sides=("right",))
        env._focus_prediction_chunks = {}
        env._handoff_measured_bends = Mock(return_value={"right": np.zeros(5)})
        return env, chunk

    def arm(self, env, chunk):
        env.continuity_hand_event_guard = ContinuityHandEventGuard()
        env.cfg["continuity_hand_event_guard"] = True
        _, pred = prediction(np.r_[np.zeros(8), np.full(chunk["frame_count"] - 8, .5)])
        pred.update({key: chunk[key] for key in
                     ("sequence_id", "source_state_sequence_id", "frame_count", "fps")})
        env._focus_prediction_chunks[chunk["sequence_id"]] = pred

    def test_disabled_selector_retains_legacy_and_trajectory_results(self):
        for mode in ("legacy", "trajectory"):
            env, chunk = self.env()
            env.continuity_mode = mode
            expected = env._phase_match_start_frame(chunk, 1_500_000_000)
            env.cfg["continuity_hand_event_guard"] = False
            env.continuity_hand_event_guard = ContinuityHandEventGuard()
            env._handoff_measured_bends.side_effect = AssertionError("Disabled guard read feedback")
            self.assertEqual(env._phase_match_start_frame(chunk, 1_500_000_000), expected)
            self.assertNotIn("continuity_hand_event_guard", expected[1])

    def test_legacy_reselects_inside_bound(self):
        env, chunk = self.env()
        env.continuity_mode = "legacy"
        self.arm(env, chunk)
        selected, state = env._phase_match_start_frame(chunk, 1_500_000_000)
        self.assertLessEqual(selected, 7)
        self.assertEqual(state["search_range"], [2, 7])
        self.assertTrue(state["continuity_hand_event_guard"]["restricted"])

    def test_trajectory_cannot_bypass_guard(self):
        env, chunk = self.env()
        self.arm(env, chunk)
        selected, state = env._phase_match_start_frame(chunk, 1_500_000_000)
        self.assertLessEqual(selected, 7)
        self.assertEqual(state["mode"], "reference_continuity")

    def test_pending_ik_uses_latency_frame_and_keeps_transition_checks(self):
        env, chunk = self.env()
        env.continuity_hand_event_guard = ContinuityHandEventGuard()
        env.cfg["continuity_hand_event_guard"] = True
        selected, state = env._phase_match_start_frame(chunk, 1_500_000_000)
        self.assertEqual(selected, 2)
        self.assertEqual(state["continuity_hand_event_guard"]["reason"], "prediction_pending")
        chunk["root_pos"][:] = [1., 0., 0.]
        _, state = env._phase_match_start_frame(chunk, 1_500_000_000)
        self.assertFalse(state["transition_valid"])

    def test_await_chunk_and_active_focus_keep_fresh_handoff_path(self):
        for phase in ("await_chunk", "focus"):
            env, chunk = self.env()
            expected = env._phase_match_start_frame(chunk, 1_500_000_000)
            env.focus_handoff.controllers["right"].state = phase
            env.continuity_hand_event_guard = ContinuityHandEventGuard()
            env.cfg["continuity_hand_event_guard"] = True
            env._handoff_measured_bends.side_effect = AssertionError("Unlocked hand read feedback")
            selected, state = env._phase_match_start_frame(chunk, 1_500_000_000)
            self.assertEqual(selected, expected[0])
            self.assertEqual(state["continuity_hand_event_guard"]["reason"], "hands_unlocked")


class PreflightTests(unittest.TestCase):
    def env(self):
        env = base.FocusHandoffHandTests().acceptance_env()
        env.cfg.update(continuity_hand_event_guard=True, rtc_enabled=False)
        env.focus_handoff = PerHandFocusHandoff()
        env.chunk_start_reference_continuity = True
        env.continuity_mode = "legacy"
        env.continuity_search_ahead_s = 1.
        env.chunk_phase_alignment = True
        env.phase_max_root_jump_m = .2
        env.phase_rotation_weight = .08
        env.phase_velocity_weight = .03
        env.phase_measured_weight = .15
        template, _ = trajectory_fixture.TrajectoryWiringTests().env()
        env.last_published_robot_state = template.last_published_robot_state
        # Use the real selector; after the result arrives it must be able to
        # choose a forward continuation, not be stuck at the pending frame.
        env._phase_match_start_frame = type(env)._phase_match_start_frame.__get__(env)
        env._latency_start_frame = Mock(return_value=(1, .04, "timestamp"))
        env._min_execution_hold_active = Mock(return_value=False)
        env._handoff_measured_bends = Mock(return_value={s: np.zeros(5) for s in ("left", "right")})
        env.continuity_hand_event_guard = ContinuityHandEventGuard()
        env.hand_event_chunk_preflight = GuardChunkPreflight(env.continuity_hand_event_guard)
        env.reference_play_gate = False
        env.future_offsets_s = np.array([0., .1])
        env.previous_chunk = None
        env.previous_chunk_started_ns = None
        env.blend_started_ns = None
        env._play_scale = 1.
        return env

    def result(self, env, sequence=11, event=False):
        pred = env._focus_prediction_chunks[sequence]
        closure = np.zeros((pred["frame_count"], 6))
        if event:
            closure[5:, HAND_BEND_COLUMNS] = .5
        pred["hand_focus"] = {s: {"closure": closure.copy()} for s in ("left", "right")}

    def start(self, env):
        env.chunk_subscriber.receive_latest.return_value = base.FocusHandoffHandTests().wire_chunk(43)
        return env._accept_latest_chunk(1_100_000_000)

    def test_pending_does_not_accept_reset_clock_or_stop_outgoing_motion(self):
        env = self.env()
        old = env.chunk
        before = env._sample_reference(1_100_000_000)["right_wrist_pos"][0, 0]
        self.assertFalse(self.start(env))
        self.assertIs(env.chunk, old)
        self.assertEqual(env.chunk_started_ns, 900_000_000)
        self.assertEqual(env.last_chunk_received_ns, 900_000_000)
        self.assertEqual(env.last_chunk_sequence_id, 10)
        self.assertGreater(env._sample_reference(1_200_000_000)["right_wrist_pos"][0, 0], before)
        env._submit_chunk_focus.assert_called_once()
        env.chunk_subscriber.receive_latest.return_value = None
        self.assertFalse(env._accept_latest_chunk(1_120_000_000))
        env._submit_chunk_focus.assert_called_once()

    def test_same_chunk_result_restores_forward_matching_and_acceptance(self):
        env = self.env()
        self.assertFalse(self.start(env))
        self.result(env)
        env.chunk_subscriber.receive_latest.return_value = None
        self.assertTrue(env._accept_latest_chunk(1_160_000_000))
        self.assertGreater(env.chunk_start_frame, 1)
        self.assertEqual(env.last_chunk_sequence_id, 11)
        self.assertAlmostEqual(env.chunk_phase_state["continuity_hand_event_check_wait_s"], .06)
        self.assertNotEqual(env.chunk_phase_state["continuity_hand_event_guard"]["reason"], "prediction_pending")

    def test_ready_candidate_still_obeys_real_event_boundary(self):
        env = self.env()
        self.assertFalse(self.start(env))
        self.result(env, event=True)
        env.chunk_subscriber.receive_latest.return_value = None
        self.assertTrue(env._accept_latest_chunk(1_160_000_000))
        self.assertLessEqual(env.chunk_start_frame, 4)
        self.assertEqual(env.chunk_phase_state["continuity_hand_event_guard"]["reason"], "unexecuted_hand_change")

    def test_newer_arrivals_do_not_starve_checked_candidate(self):
        env = self.env()
        self.assertFalse(self.start(env))
        raw = base.FocusHandoffHandTests().wire_chunk(43)
        raw["sequence_id"] = 12
        env.chunk_subscriber.receive_latest.return_value = raw
        self.assertFalse(env._accept_latest_chunk(1_120_000_000))
        self.assertEqual(env.hand_event_chunk_preflight.waiting["sequence_id"], 11)
        self.assertEqual(env._pending_raw_chunk["sequence_id"], 12)
        self.result(env)
        env.chunk_subscriber.receive_latest.return_value = None
        self.assertTrue(env._accept_latest_chunk(1_160_000_000))
        self.assertEqual(env.last_chunk_sequence_id, 11)
        self.assertEqual(env._pending_raw_chunk["sequence_id"], 12)

    def test_timeout_drops_candidate_without_refreshing_watchdog(self):
        env = self.env()
        self.assertFalse(self.start(env))
        env.chunk_subscriber.receive_latest.return_value = None
        self.assertFalse(env._accept_latest_chunk(1_510_000_000))
        self.assertIsNone(env.hand_event_chunk_preflight.waiting)
        self.assertEqual(env.last_chunk_sequence_id, 10)
        self.assertEqual(env.last_chunk_received_ns, 900_000_000)

    def test_stale_waiting_candidate_is_rejected_before_matching(self):
        env = self.env()
        self.assertFalse(self.start(env))
        env.chunk_subscriber.receive_latest.return_value = None
        env.state_sequence_id = 100
        self.assertFalse(env._accept_latest_chunk(1_160_000_000))
        self.assertIsNone(env.hand_event_chunk_preflight.waiting)
        self.assertEqual(env.last_chunk_sequence_id, 10)

    def test_preview_dropped_by_latest_only_worker_is_resubmitted_once(self):
        env = self.env()
        raw = base.FocusHandoffHandTests().wire_chunk(43)
        env._preview_chunk_focus(raw, 1_000_000_000, 1_080_000_000)
        env._submit_chunk_focus.reset_mock()
        self.assertFalse(self.start(env))
        env._submit_chunk_focus.assert_called_once()
        env.chunk_subscriber.receive_latest.return_value = None
        self.assertFalse(env._accept_latest_chunk(1_120_000_000))
        env._submit_chunk_focus.assert_called_once()

    def test_disabled_guard_keeps_old_acceptance_path(self):
        env = self.env()
        env.cfg["continuity_hand_event_guard"] = False
        self.assertTrue(self.start(env))
        self.assertIsNone(env.hand_event_chunk_preflight.waiting)

    def test_startup_recovery_and_focus_do_not_wait(self):
        for field, value in (("reference_playing", False), ("awaiting_fresh_chunk", True),
                             ("held_sample", {})):
            env = self.env()
            setattr(env, field, value)
            self.assertEqual(env.hand_event_chunk_preflight.check(env, env.chunk, env.chunk, 1_100_000_000), "bypass")
        env = self.env()
        env.focus_handoff.controllers["right"].state = "await_chunk"
        self.assertEqual(env.hand_event_chunk_preflight.check(env, env.chunk, env.chunk, 1_100_000_000), "bypass")

    def test_check_error_rejects_candidate_without_replacing_old_clock(self):
        env = self.env()
        env._latency_start_frame.side_effect = ValueError("invalid source clock")
        self.assertFalse(self.start(env))
        self.assertEqual(env.chunk_started_ns, 900_000_000)
        self.assertEqual(env.last_chunk_sequence_id, 10)
        self.assertIsNone(env.hand_event_chunk_preflight.waiting)


class PlaybackBoundaryTests(unittest.TestCase):
    def env(self, kind="close"):
        env = base.FocusHandoffHandTests().acceptance_env()
        env.cfg.update(continuity_hand_event_guard=True, chunk_blend_s=.22)
        env.focus_handoff = PerHandFocusHandoff()
        env.reference_play_gate = False
        env.future_offsets_s = np.array([0., .1, .4, 1.])
        env.previous_chunk = None
        env.previous_chunk_started_ns = None
        env.blend_started_ns = None
        env.prestart_reference_sample = None
        env._play_scale = 1.
        env.chunk = base.chunk(2)
        env.chunk["fps"] = 20.
        env.chunk["left_wrist_pos"][:, 1] = np.arange(60) * .002
        env.chunk_started_ns = 1_000_000_000
        env._focus_prediction_chunks = {}
        env.hand_event_playback_boundary = HandEventPlaybackBoundary(ContinuityHandEventGuard())
        self.pred(env, kind)
        env.hand_event_playback_boundary.register(env.chunk, 1)
        env.last_reference_sample = None
        return env

    def pred(self, env, kind="close", conflict=10):
        bend = np.r_[np.full(conflict, .1 if kind == "close" else .8),
                     np.full(env.chunk["frame_count"] - conflict, .5 if kind == "close" else .4)]
        _, pred = prediction(bend)
        pred.update({key: env.chunk[key] for key in
                     ("sequence_id", "source_state_sequence_id", "frame_count", "fps")})
        env._focus_prediction_chunks[env.chunk["sequence_id"]] = pred
        env._handoff_measured_bends = Mock(return_value={
            "left": np.zeros(5), "right": np.full(5, 0. if kind == "close" else .8)})
        # Left has no coordinated event, but has its own valid worker result.
        pred["hand_focus"]["left"] = {"closure": np.zeros((env.chunk["frame_count"], 6))}

    def test_frame8_start_cannot_play_or_expose_frame10_event(self):
        env = self.env()
        # A future token crosses frame 10 before the current token does.
        sample = env._sample_reference(1_400_000_000)
        expected = env.chunk["right_wrist_pos"][9, 0]
        self.assertAlmostEqual(sample["right_wrist_pos"][0, 0], env.chunk["right_wrist_pos"][8, 0])
        np.testing.assert_allclose(sample["right_wrist_pos"][1:, 0], expected)
        env.last_reference_sample = sample
        sample = env._sample_reference(1_500_000_000)
        np.testing.assert_allclose(sample["right_wrist_pos"][:, 0], expected)
        self.assertIn("right", env.hand_event_playback_boundary.holds)
        # The other wrist and body/source clock have not stopped.
        self.assertAlmostEqual(sample["left_wrist_pos"][0, 1], .02)
        self.assertAlmostEqual(env._elapsed(1_500_000_000, env.chunk_started_ns), .5)

    def test_release_has_same_playback_and_future_boundary(self):
        env = self.env("open")
        env._sample_reference(1_400_000_000)
        sample = env._sample_reference(1_800_000_000)
        np.testing.assert_allclose(sample["right_wrist_pos"][:, 0], env.chunk["right_wrist_pos"][9, 0])

    def test_pending_result_does_not_freeze_approach(self):
        env = self.env()
        env._focus_prediction_chunks.clear()
        sample = env._sample_reference(1_800_000_000)
        np.testing.assert_allclose(sample["right_wrist_pos"], sample_chunk(
            env.chunk, .8, env.future_offsets_s)["right_wrist_pos"])
        self.assertFalse(env.hand_event_playback_boundary.holds)

    def test_late_result_holds_last_sent_pose_without_rewinding(self):
        env = self.env()
        pred = env._focus_prediction_chunks.pop(2)
        env.last_reference_sample = env._sample_reference(1_800_000_000)
        last = env.last_reference_sample["right_wrist_pos"][0].copy()
        env._focus_prediction_chunks[2] = pred
        sample = env._sample_reference(1_820_000_000)
        np.testing.assert_allclose(sample["right_wrist_pos"], np.repeat(last[None], 4, axis=0))

    def test_known_boundary_survives_missing_feedback_or_worker_cache(self):
        env = self.env()
        env._sample_reference(1_400_000_000)
        env._focus_prediction_chunks.clear()
        env._handoff_measured_bends.return_value = {}
        sample = env._sample_reference(1_600_000_000)
        np.testing.assert_allclose(sample["right_wrist_pos"][:, 0], env.chunk["right_wrist_pos"][9, 0])

    def test_latch_survives_new_chunk_and_releases_for_fresh_focus(self):
        env = self.env()
        env._sample_reference(1_400_000_000)
        sample = env._sample_reference(1_600_000_000)
        hold = sample["right_wrist_pos"][0].copy()
        env.chunk = base.chunk(3)
        env.chunk_started_ns = 1_600_000_000
        env.chunk["right_wrist_pos"] += 1.
        env._focus_prediction_chunks.clear()
        sample = env._sample_reference(1_800_000_000)
        np.testing.assert_allclose(sample["right_wrist_pos"], np.repeat(hold[None], 4, axis=0))
        right = env.focus_handoff.controllers["right"]
        right.state = "await_chunk"
        right.hold = sample
        env._sample_reference(1_820_000_000)
        self.assertNotIn("right", env.hand_event_playback_boundary.holds)
        right.state = "focus"
        right.started_ns = 1_820_000_000
        right.playback = env.chunk
        right.playback_started_ns = env.chunk_started_ns
        sample = env._sample_reference(1_900_000_000)
        self.assertGreater(sample["right_wrist_pos"][0, 0], 1.)

    def test_completed_close_is_not_rearmed_from_old_prediction(self):
        env = self.env()
        env.focus_handoff.controllers["right"].last_kind = "close"
        sample = env._sample_reference(1_800_000_000)
        self.assertFalse(env.hand_event_playback_boundary.holds)
        self.assertGreater(sample["right_wrist_pos"][0, 0], env.chunk["right_wrist_pos"][9, 0])

    def test_rtc_cannot_copy_old_future_beyond_event_into_new_prefix(self):
        env = self.env()
        env.cfg.update(rtc_enabled=True, rtc_prefix_frames=10, rtc_hard_prefix_frames=2, rtc_blend_frames=8)
        new = base.chunk(3)
        new["fps"] = 20.
        # Inspect the old prefix at frame8, which would copy frames10..17.
        result, state = env._apply_rtc_overlap(new, 1, 1_400_000_000, False)
        self.assertTrue(state["applied"])
        boundary = env.chunk["right_wrist_pos"][9, 0]
        self.assertLessEqual(result["right_wrist_pos"][2, 0], boundary + 1e-7)
        np.testing.assert_array_equal(env.chunk["right_wrist_pos"], base.chunk(2)["right_wrist_pos"])

    def test_blend_assist_and_disabled_path_cannot_override_or_leak_boundary(self):
        env = self.env()
        env.previous_chunk = base.chunk(1)
        env.previous_chunk_started_ns = env.chunk_started_ns
        env.blend_started_ns = 1_400_000_000
        env.previous_chunk["right_wrist_pos"][:] = 5.
        _, pred = prediction(np.r_[np.full(10, .1), np.full(50, .5)], sequence=1)
        env._focus_prediction_chunks[1] = pred
        env.hand_event_playback_boundary.register(env.previous_chunk, 1)
        sample = env._sample_reference(1_400_000_000)
        env.last_reference_sample = sample
        sample = env._sample_reference(1_500_000_000)
        hold = sample["right_wrist_pos"].copy()
        env.stall_wrist_assist = Mock()
        env.stall_wrist_assist.apply.side_effect = lambda sample, sides: dict(sample, right_wrist_pos=sample["right_wrist_pos"] + 10.)
        np.testing.assert_allclose(env._sample_reference(1_600_000_000)["right_wrist_pos"], hold)
        env.cfg["continuity_hand_event_guard"] = False
        self.assertGreater(env._sample_reference(1_600_000_000)["right_wrist_pos"][0, 0], 10.)
        env.hand_event_playback_boundary.reset()
        self.assertFalse(env.hand_event_playback_boundary.holds)


if __name__ == "__main__":
    unittest.main()
