"""Regression coverage for false FOCUS edges and independent physical gates."""

import unittest
import threading
from unittest.mock import Mock

import numpy as np

from scalebridge.online.focus import derive_hand_focus_events
from scalebridge.online.focus_handoff import PerHandFocusHandoff, GraspFingerChunkControl
from scalebridge.online.reference import sample_chunk
from scalebridge.env.motion_tracking_hat4_online import _AsyncBilateralHandIK
import test_hat_focus_handoff as base

chunk = base.chunk


def hand(actual=0., predicted=.8, onset=0):
    history = np.full((30, 6), actual, dtype=np.float32)
    future = np.full((60, 6), actual, dtype=np.float32)
    future[onset:] = predicted
    return derive_hand_focus_events(history, future, 30.)


def payload(left=None, right=None):
    return {"hands": {"left": hand(predicted=0.) if left is None else left,
                      "right": hand() if right is None else right}}


class FingerChunkControlTests(unittest.TestCase):
    def test_async_ik_keeps_result_provenance_when_new_chunk_is_submitted(self):
        entered = threading.Event()
        release = threading.Event()

        def solve(target, previous):
            if target[0, 0] == 2:
                entered.set()
                if not release.wait(2.):
                    raise RuntimeError("test worker timed out")
            return np.full(12, target[0, 0]), None

        worker = _AsyncBilateralHandIK({side: Mock(solve_frame=solve)
                                      for side in ("lh", "rh")})
        try:
            first = worker.submit(np.ones((5, 3)), np.ones((5, 3)),
                                  context={"right": (123, 10)})
            with worker._condition:
                self.assertTrue(worker._condition.wait_for(
                    lambda: worker._result_sequence >= first, timeout=.2))
            second = worker.submit(np.full((5, 3), 2), np.full((5, 3), 2),
                                   context={"right": (123, 11)})
            self.assertTrue(entered.wait(1.))
            self.assertEqual(worker.latest()["_context"], {"right": (123, 10)})
            self.assertIsNone(worker.latest(min_sequence=second))
            release.set()
            with worker._condition:
                self.assertTrue(worker._condition.wait_for(
                    lambda: worker._result_sequence >= second, timeout=.2))
            self.assertEqual(worker.latest()["_context"], {"right": (123, 11)})
        finally:
            release.set()
            worker.close()

    def test_continuity_is_bounded_and_preserves_within_chunk_opening(self):
        control = GraspFingerChunkControl("continuity")
        previous = np.array([.2] * 5 + [.4])
        first = np.array([.9] * 5 + [.8])
        result = control.apply("right", 1, 10, first, previous)
        np.testing.assert_allclose(result[:5], .2)
        self.assertEqual(result[5], .8)  # thumb rotation is not rebased
        result = control.apply("right", 1, 10, np.zeros(6), result)
        np.testing.assert_allclose(result, 0.)
        result = control.apply("right", 1, 10, np.ones(6), result)
        np.testing.assert_allclose(result[:5], .3)  # opening inside chunk passes
        control.reset("right")
        result = control.apply("right", 2, 11, np.zeros(6), np.ones(6))
        np.testing.assert_allclose(result[:5], 1.)

    def test_progress_hold_is_per_finger_and_does_not_clamp_thumb_rotation(self):
        control = GraspFingerChunkControl("hold_progress")
        result = control.apply("right", 1, 1,
                               [.8, .1, .9, .2, .7, .9],
                               [.2, .8, .3, .6, .4, .1])
        np.testing.assert_allclose(result, [.2, .1, .3, .2, .4, .9])

    def test_each_hand_and_each_grasp_has_its_own_anchor(self):
        control = GraspFingerChunkControl("continuity")
        control.apply("right", 1, 10, np.full(6, .8), np.full(6, .2))
        result = control.apply("left", 1, 10, np.full(6, .1), np.full(6, .7))
        np.testing.assert_allclose(result[:5], .7)
        result = control.apply("right", 2, 10, np.full(6, .8), np.full(6, .5))
        np.testing.assert_allclose(result[:5], .5)
        control.reset()
        self.assertFalse(control.states)

    def test_off_keeps_targets_and_invalid_mode_is_rejected(self):
        proposed = np.linspace(0., 1., 6)
        np.testing.assert_array_equal(
            GraspFingerChunkControl("off").apply("right", 1, 1, proposed, np.ones(6)),
            proposed)
        with self.assertRaisesRegex(ValueError, "focus_handoff_finger_mode"):
            GraspFingerChunkControl("bad")


class PerHandEventsTests(unittest.TestCase):
    def test_thumb_rotation_and_constant_history_cannot_create_event(self):
        history = np.zeros((30, 6)); history[:, 4] = 1
        predicted = np.zeros((100, 6)); predicted[:, 4] = np.linspace(0., 1., 100)
        result = derive_hand_focus_events(history, predicted, 15.)
        self.assertEqual(result["events"], [])
        np.testing.assert_array_equal(result["phase"][:, 0], 0)

    def test_small_twitch_is_not_normalized_to_a_grasp(self):
        history = np.zeros((30, 6)); predicted = np.full((60, 6), .08)
        self.assertEqual(derive_hand_focus_events(history, predicted, 30.)["events"], [])
        predicted[:] = .5; history[:] = .48
        self.assertEqual(derive_hand_focus_events(history, predicted, 30.)["events"], [])

    def test_single_finger_and_short_predictions_do_not_unlock(self):
        history = np.zeros((30, 6)); predicted = np.zeros((60, 6))
        predicted[:, 0] = 1
        self.assertEqual(derive_hand_focus_events(history, predicted, 30.)["events"], [])
        predicted[:] = 0; predicted[:2, :2] = 1
        self.assertEqual(derive_hand_focus_events(history, predicted, 30.)["events"], [])

    def test_real_frame_zero_and_release_pre_window_are_preserved(self):
        close = hand()
        self.assertEqual(close["events"][0]["kind"], "close")
        self.assertEqual(close["events"][0]["start"], 0)
        opening = hand(actual=.8, predicted=0., onset=40)
        event = opening["events"][0]
        self.assertEqual(event["kind"], "open")
        self.assertEqual(event["motion_start"], 40)
        self.assertEqual(event["start"], 10)


class IndependentLifecycleTests(unittest.TestCase):
    def arm(self, control, data=None, first=1):
        data = payload() if data is None else data
        for seq in (first, first + 1):
            control.observe_prediction(chunk(seq), data)

    def start_right(self, control):
        self.arm(control)
        right = control.controllers["right"]
        poses = {"right_wrist_yaw_link": (np.array([.08, 0., 0.]), np.array([1., 0., 0., 0.]))}
        self.assertTrue(right.arrive(poses, 20, sample_chunk(chunk(), 0., [0.]), np.zeros(5)))
        self.assertEqual(control.start_fresh(chunk(3, 20), 1_000_000_000), ["right"])
        return right

    def test_left_event_does_not_create_right_entry(self):
        control = PerHandFocusHandoff()
        self.arm(control, payload(left=hand(), right=hand(predicted=0.)))
        self.assertIsNotNone(control.controllers["left"].entry)
        self.assertIsNone(control.controllers["right"].entry)
        self.assertFalse(control.controllers["right"].active)

    def test_right_unlock_does_not_wait_for_left_pose_or_source_fence(self):
        control = PerHandFocusHandoff()
        left = control.controllers["left"]
        left.state = "await_chunk"; left.source_fence = 100
        right = self.start_right(control)
        self.assertTrue(right.active)
        self.assertEqual(left.state, "await_chunk")
        self.assertTrue(control.accepts(chunk(4, 30)))

    def test_phase_disappearance_and_replans_do_not_finish_unmoved_hand(self):
        control = PerHandFocusHandoff()
        right = self.start_right(control)
        control.observe_prediction(chunk(4, 30), payload(right=hand(predicted=0.)))
        self.assertFalse(right.update_actual(1_200_000_000, np.zeros(5)))
        self.assertFalse(right.update_actual(2_500_000_000, np.zeros(5)))
        self.assertTrue(right.active)
        self.assertEqual(control.active_sides, ("right",))

    def test_completion_requires_actual_progress_and_one_second_hold(self):
        control = PerHandFocusHandoff()
        right = self.start_right(control)
        self.assertFalse(right.update_actual(1_200_000_000, np.full(5, .8)))
        self.assertFalse(right.update_actual(2_199_000_000, np.full(5, .8)))
        self.assertTrue(right.update_actual(2_200_000_000, np.full(5, .8)))
        self.assertEqual(right.end_reason, "feedback_complete")
        self.assertFalse(right.active)
        self.assertEqual(control.controllers["left"].state, "approach")

    def test_timeout_is_distinct_from_completed_grasp_and_can_retry(self):
        control = PerHandFocusHandoff()
        right = self.start_right(control)
        self.assertTrue(right.update_actual(6_000_000_000, None))
        self.assertEqual(right.end_reason, "timeout")
        self.arm(control, first=4)
        self.assertIsNotNone(right.entry)

    def test_completed_close_arms_a_release_window_and_then_a_close_again(self):
        """Close and release alternate; the alternation also clears cooldown."""
        control = PerHandFocusHandoff()
        right = self.start_right(control)
        right.update_actual(1_200_000_000, np.full(5, .8))
        right.update_actual(2_200_000_000, np.full(5, .8))
        self.assertEqual(right.end_reason, "feedback_complete")
        self.assertEqual(right.last_kind, "close")
        self.assertEqual(right.state, "cooldown")
        # A completed close may only be followed by a release window, and
        # arming it is what walks the controller back out of cooldown.
        self.arm(control, payload(right=hand(actual=.8, predicted=0.)), first=4)
        self.assertEqual(right.event["kind"], "open")
        self.assertIsNotNone(right.entry)
        self.assertEqual(right.state, "approach")
        # After the release completes, a close may arm again.
        poses = {"right_wrist_yaw_link": (np.array([.08, 0., 0.]), np.array([1., 0., 0., 0.]))}
        self.assertTrue(right.arrive(poses, 40, sample_chunk(chunk(), 0., [0.]), np.full(5, .8)))
        self.assertEqual(control.start_fresh(chunk(6, 40), 3_000_000_000), ["right"])
        right.update_actual(4_200_000_000, np.zeros(5))
        right.update_actual(5_200_000_000, np.zeros(5))
        self.assertEqual(right.last_kind, "open")
        self.arm(control, payload(right=hand(actual=.05, predicted=.8)), first=8)
        self.assertEqual(right.event["kind"], "close")
        self.assertIsNotNone(right.entry)

    def test_rebase_does_not_change_other_wrist(self):
        control = PerHandFocusHandoff()
        self.start_right(control)
        outgoing = sample_chunk(chunk(), 0., [0.])
        outgoing["right_wrist_pos"][0, 0] = .12
        raw = chunk(4)
        result = control.rebase(raw, .1, outgoing)
        np.testing.assert_array_equal(result["left_wrist_pos"], raw["left_wrist_pos"])
        np.testing.assert_allclose(sample_chunk(result, .1, [0.])["right_wrist_pos"], outgoing["right_wrist_pos"], atol=1e-7)


class IndependentMotorTests(unittest.TestCase):
    def env(self):
        env = base.FocusHandoffHandTests().env()
        env.focus_handoff = PerHandFocusHandoff()
        # Test convention: larger synthetic joint angles mean more closing.
        env.hand_dof12_to_motor = lambda q: 1. - np.asarray(q)[:6]
        env.hand_motor_to_dof12 = lambda motor: np.tile(1. - np.asarray(motor), 2)
        return env

    def grasp_env(self, mode, kind="close"):
        env = self.env()
        env._handoff_finger_control = GraspFingerChunkControl(mode)
        right = env.focus_handoff.controllers["right"]
        right.state = "focus"
        right.started_ns = 123
        right.playback = chunk(10)
        # Stitching is grasp-only, so the window's event kind decides it.
        right.event = dict(kind=kind, start=0, motion_start=0, end=10,
                           goal=np.full(5, .8 if kind == "close" else 0.))
        env._handoff_hand_hold = (np.full(12, .5), np.full(12, .3))
        return env

    def solve(self, env, closing, event=123, sequence=10):
        env.hand_ik_worker.latest.return_value = {
            "lh": np.full(12, .2), "rh": np.full(12, closing),
            "_context": {"right": (event, sequence)}}
        env._send_hand_target()
        return env.simulator.set_hand_ik_target.call_args.args[1]

    def test_progress_hold_prevents_actual_motor_reopening(self):
        env = self.grasp_env("hold_progress")
        np.testing.assert_allclose(self.solve(env, .5)[:5], .5)
        np.testing.assert_allclose(self.solve(env, .1, sequence=11)[:5], .5)
        np.testing.assert_allclose(self.solve(env, .6, sequence=11)[:5], .6)
        # The idle left hand is latched at its last command, not stitched.
        np.testing.assert_allclose(env.simulator.set_hand_ik_target.call_args.args[0], .5)

    def test_continuity_tracks_increments_without_reanchoring_each_tick(self):
        env = self.grasp_env("continuity")
        np.testing.assert_allclose(self.solve(env, .1)[:5], .3)
        np.testing.assert_allclose(self.solve(env, .2)[:5], .4)
        env.focus_handoff.controllers["right"].playback = chunk(11)
        np.testing.assert_allclose(self.solve(env, .05, sequence=11)[:5], .4)
        np.testing.assert_allclose(self.solve(env, .15, sequence=11)[:5], .5)
        # A repeated IK result must not accumulate the same delta again.
        np.testing.assert_allclose(self.solve(env, .15, sequence=11)[:5], .5)

    def test_both_modes_latch_the_applied_command_and_clear_old_grasp_state(self):
        for mode in ("continuity", "hold_progress"):
            with self.subTest(mode=mode):
                env = self.grasp_env(mode)
                held = self.solve(env, .7).copy()
                env.focus_handoff.controllers["right"].state = "cooldown"
                # Leaving the window latches the applied command: even a fully
                # open prediction can no longer move the now-idle hand.
                np.testing.assert_allclose(self.solve(env, 0.), held)
                self.assertNotIn("right", env._handoff_finger_control.states)

    def test_release_window_is_never_stitched_so_it_can_open(self):
        env = self.grasp_env("hold_progress", kind="open")
        # hold_progress clamps the bending motors against reopening, so a
        # release window must bypass stitching entirely or it stays shut.
        np.testing.assert_allclose(self.solve(env, 0.), 0.)
        self.assertNotIn("right", env._handoff_finger_control.states)

    def test_stale_ik_cannot_start_new_grasp_or_rebase_new_chunk(self):
        env = self.grasp_env("continuity")
        np.testing.assert_allclose(self.solve(env, .9, event=122), .3)
        np.testing.assert_allclose(self.solve(env, .1)[:5], .3)
        np.testing.assert_allclose(self.solve(env, .2)[:5], .4)
        env.focus_handoff.controllers["right"].playback = chunk(11)
        # Old chunk result still has its old offset, even after acceptance.
        np.testing.assert_allclose(self.solve(env, .3)[:5], .5)
        np.testing.assert_allclose(self.solve(env, .1, sequence=11)[:5], .5)
        np.testing.assert_allclose(self.solve(env, .9, sequence=10)[:5], .5)

    def test_submission_captures_each_hand_chunk_and_grasp_provenance(self):
        env = self.grasp_env("continuity")
        env.hand_ik_worker.latest.return_value = None
        env._send_hand_target()
        self.assertEqual(env.hand_ik_worker.submit.call_args.kwargs,
                         {"context": {"right": (123, 10)}})

    def test_right_ik_cannot_unlock_left_and_relock_keeps_applied_command(self):
        env = self.env()
        env.focus_handoff.controllers["right"].state = "focus"
        env.hand_ik_worker.latest.return_value = {"lh": np.full(12, .7), "rh": np.full(12, .8)}
        env._send_hand_target()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_array_equal(left, np.zeros(12))
        np.testing.assert_array_equal(right, np.full(12, .8))
        env.focus_handoff.controllers["right"].state = "cooldown"
        env._send_hand_target()
        np.testing.assert_array_equal(env.simulator.set_hand_ik_target.call_args.args[1], np.full(12, .8))

    def test_left_ik_cannot_unlock_right(self):
        env = self.env()
        env.focus_handoff.controllers["left"].state = "focus"
        env.hand_ik_worker.latest.return_value = {"lh": np.full(12, .7), "rh": np.full(12, .8)}
        env._send_hand_target()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_array_equal(left, np.full(12, .7))
        np.testing.assert_array_equal(right, np.full(12, .1))

    def test_new_side_unlock_has_its_own_ik_fence(self):
        env = self.env()
        env.focus_handoff.controllers["right"].state = "focus"
        env._send_hand_target()
        env.focus_handoff.controllers["left"].state = "focus"
        env._handoff_ik_min_sequence["left"] = None  # fresh handoff acceptance
        env.hand_ik_worker.submit.return_value = 200
        env._send_hand_target()
        env.hand_ik_worker.latest.assert_called_with(min_sequence=200)
        self.assertEqual(env._handoff_ik_min_sequence, {"left": 200, "right": 100})

    def test_idle_hands_never_follow_the_prediction_in_either_direction(self):
        """2026-10-10 regression: an open-only clamp on the raw prediction
        ratchets an idle hand fully open in seconds and can never re-close,
        because the prediction's chunk prefix is always looser than the
        achieved grasp. Idle hands must simply hold their last command."""
        env = self.env()
        env._handoff_hand_hold = (np.full(12, .5), np.full(12, .8))
        for prediction in (.4, 0., 1.):
            env.hand_ik_worker.latest.return_value = {
                "lh": np.full(12, .7), "rh": np.full(12, prediction)}
            env._send_hand_target()
            left, right = env.simulator.set_hand_ik_target.call_args.args
            np.testing.assert_array_equal(left, np.full(12, .5))
            np.testing.assert_array_equal(right, np.full(12, .8))
        self.assertFalse(env.focus_handoff.active)
        env.hand_ik_worker.submit.assert_not_called()

    def test_outage_holds_even_opening_predictions(self):
        env = self.env(); env.localization_paused = True
        env._handoff_hand_hold = (np.full(12, .5), np.full(12, .8))
        env.hand_ik_worker.latest.return_value = {"lh": np.zeros(12), "rh": np.zeros(12)}
        env._send_hand_target()
        env.hand_ik_worker.submit.assert_not_called()
        np.testing.assert_array_equal(env.simulator.set_hand_ik_target.call_args.args[1], np.full(12, .8))

    def test_phase_stays_active_after_prediction_disappears(self):
        env = self.env(); env.future_offsets_s = np.array([0., .1])
        right = env.focus_handoff.controllers["right"]
        right.state = "focus"; right.started_ns = 1_000_000_000
        env._focus_phase_prediction = None
        np.testing.assert_array_equal(env._focus_handoff_phase(1_200_000_000)[:, 0], 1.)

    def test_handoff_uses_own_clock_without_changing_other_hand_clock(self):
        helper = base.FocusHandoffHandTests()
        env = helper.acceptance_env()
        env.focus_handoff = PerHandFocusHandoff()
        env.cfg["rtc_enabled"] = False
        env.reference_play_gate = False
        env.future_offsets_s = np.array([0.])
        env.previous_chunk = None
        env.previous_chunk_started_ns = None
        env.blend_started_ns = None
        env.prestart_reference_sample = None
        env.hand_ik_worker.latest.return_value = None
        env._phase_match_start_frame.return_value = (15, {
            "mode": "reference_continuity", "expected_frame": 3,
            "source_age_s": .1, "pose_cost": 0., "transition_valid": True,
        })
        right = env.focus_handoff.controllers["right"]
        right.state = "await_chunk"; right.source_fence = 43
        right.hold = sample_chunk(chunk(), 0., [0.])
        right.hold["right_wrist_pos"][0, 0] = .12
        raw = helper.wire_chunk(43)
        positions = np.asarray(raw["left_wrist_pos"])
        positions[:, 0] = np.arange(60) * .002
        raw["left_wrist_pos"] = positions.tolist()
        env.chunk_subscriber.receive_latest.return_value = raw
        self.assertTrue(env._accept_latest_chunk(1_100_000_000))
        self.assertTrue(right.active)
        self.assertFalse(env.focus_handoff.controllers["left"].active)
        self.assertEqual(env.chunk_started_ns, 600_000_000)
        self.assertEqual(right.playback_started_ns, 1_000_000_000)
        live = env._sample_reference(1_100_000_000)
        np.testing.assert_allclose(live["right_wrist_pos"][0, 0], .12, atol=1e-7)
        # The other side retains its ordinary old/new chunk blend at swap.
        np.testing.assert_allclose(live["left_wrist_pos"][0, 0], 0., atol=1e-7)
        later = env._sample_reference(1_400_000_000)
        np.testing.assert_allclose(later["left_wrist_pos"][0, 0], .048, atol=1e-7)
        np.testing.assert_allclose(later["right_wrist_pos"][0, 0], .138, atol=1e-7)
        right.state = "cooldown"
        env._rejoin_handoff_wrist("right", later, 1_400_000_000)
        rejoined = env._sample_reference(1_400_000_000)
        np.testing.assert_allclose(rejoined["right_wrist_pos"], later["right_wrist_pos"], atol=1e-7)
        np.testing.assert_allclose(rejoined["left_wrist_pos"], later["left_wrist_pos"], atol=1e-7)

    def test_body_hold_still_previews_per_hand_events_without_rebasing_entry(self):
        helper = base.FocusHandoffHandTests()
        env = helper.acceptance_env()
        env.focus_handoff = PerHandFocusHandoff()
        env.last_reference_sample = sample_chunk(chunk(), .1, [0.])
        raw = helper.wire_chunk(43)
        env.chunk_subscriber.receive_latest.return_value = raw
        self.assertFalse(env._accept_latest_chunk(1_100_000_000))
        self.assertEqual(env.last_chunk_sequence_id, 10)
        np.testing.assert_array_equal(env._focus_prediction_chunks[11]["right_wrist_pos"], raw["right_wrist_pos"])
        env._submit_chunk_focus.assert_called_once()

    def test_fresh_feedback_is_required_for_actual_motion(self):
        env = self.env()
        env.hand_focus_history = [(1_000_000_000, np.r_[np.zeros(6), np.ones(6)])]
        self.assertEqual(env._handoff_measured_bends(1_600_000_000), {})
        actual = env._handoff_measured_bends(1_100_000_000)
        np.testing.assert_array_equal(actual["left"], np.zeros(5))
        np.testing.assert_array_equal(actual["right"], np.ones(5))


if __name__ == "__main__":
    unittest.main()
