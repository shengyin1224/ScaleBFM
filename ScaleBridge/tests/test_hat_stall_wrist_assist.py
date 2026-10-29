"""Offline rescue/FOCUS seam tests; no robot, sockets or IK workers."""

import unittest
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv, ACTIVE_LINKS
from scalebridge.online.focus_handoff import rebase_wrist_chunk
from scalebridge.online.reference import sample_chunk
from scalebridge.online.stall_wrist_assist import StallWristAssist, finger_intent_entry
import test_hat_focus_handoff as base

SECOND = 1_000_000_000


def sample(p=(0., 0., 0.)):
    out = sample_chunk(base.chunk(), 0., [0., .1])
    out["right_wrist_pos"][:] = p
    return out


def intent(target=(.2, 0., 0.), kind="close"):
    return dict(target=np.array(target), kind=kind, frame=10, sequence_id=1)


def arm(control, target=(.2, 0., 0.)):
    for i in range(101):
        control.update(i * 20_000_000, {"right": np.zeros(3)},
                       {"right": np.zeros(3)}, {"right": intent(target)})
    return control.states["right"]


def metadata_env(enabled=False, continuity=False, **options):
    env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
    env.cfg = OmegaConf.create(dict(future_idx=[0, 5], simulator=dict(config={}),
        focus_handoff_enabled=True, chunk_start_reference_continuity=continuity,
        stall_wrist_assist=enabled, **options))
    env.device = "cpu"
    env.metadata_dict = {"selected_body_names": list(ACTIVE_LINKS)}
    env.stall_wrist_assist = None
    env.focus_adaptive = None
    env._handoff_finger_control = SimpleNamespace(mode="off")
    env._setup_metadata()
    return env


class FingerIntentTests(unittest.TestCase):
    def prediction(self, closed=False):
        p = base.chunk()
        closure = np.ones((p["frame_count"], 6)) if closed else np.zeros((p["frame_count"], 6))
        closure[10:, [0, 1]] = .1 if closed else .8
        p["hand_focus"] = {"right": {"closure": closure}}
        return p

    def test_open_hand_finds_close_onset_and_closed_hand_finds_open_onset(self):
        for closed, expected in ((False, "close"), (True, "open")):
            p = self.prediction(closed)
            e = finger_intent_entry(p, "right", np.full(5, float(closed)), 0.)
            self.assertEqual((e["frame"], e["kind"]), (10, expected))
            np.testing.assert_array_equal(e["target"], p["right_wrist_pos"][10])

    def test_rotation_single_finger_and_two_frame_twitch_do_not_trigger(self):
        for channels, rows in (([4], slice(0, None)), ([0], slice(0, None)),
                               ([0, 1], slice(10, 12))):
            p = self.prediction()
            p["hand_focus"]["right"]["closure"][:] = 0.
            p["hand_focus"]["right"]["closure"][rows, channels] = .8
            self.assertIsNone(finger_intent_entry(p, "right", np.zeros(5), 0.))

    def test_elapsed_prefix_is_not_used_and_live_feedback_changes_requested_kind(self):
        p = self.prediction()
        p["hand_focus"]["right"]["closure"][20:] = 0.
        self.assertIsNone(finger_intent_entry(p, "right", np.zeros(5), 1.))
        e = finger_intent_entry(p, "right", np.ones(5), 0.)
        self.assertEqual(e["kind"], "open")
        self.assertEqual(e["frame"], 0)


class IntegrationTests(unittest.TestCase):
    def test_focus_respects_configured_continuity_independently_of_assist(self):
        for version in ("legacy", "v2", "v3"):
            for assist in (False, True):
                with self.subTest(version=version, assist=assist):
                    env = metadata_env(assist, focus_handoff_version=version)
                    self.assertFalse(env.chunk_start_reference_continuity)
                    self.assertEqual(env.continuity_mode, "off")
            for value, mode in ((True, "legacy"), ("trajectory", "trajectory")):
                with self.subTest(version=version, continuity=value):
                    env = metadata_env(continuity=value, focus_handoff_version=version)
                    self.assertTrue(env.chunk_start_reference_continuity)
                    self.assertEqual(env.continuity_mode, mode)
        enabled = metadata_env(True)
        self.assertFalse(enabled.chunk_start_reference_continuity)
        self.assertIsInstance(enabled.stall_wrist_assist, StallWristAssist)
        for cont in (True, "trajectory"):
            with self.assertRaises(ValueError):
                metadata_env(True, cont)

    def test_focus_with_false_never_searches_ahead_of_latency_frame(self):
        for assist in (False, True):
            env = metadata_env(assist)
            env.chunk = base.chunk()
            env.chunk_started_ns = 0
            env.reference_playing = True
            env.localization_paused = env.awaiting_fresh_chunk = False
            env.last_published_robot_state = None
            env._latency_start_frame = lambda chunk, now: (2, .1, "timestamp")
            for focus_state in ("approach", "await_chunk", "focus"):
                with self.subTest(assist=assist, focus_state=focus_state):
                    env.focus_handoff.controllers["right"].state = focus_state
                    selected, state = env._phase_match_start_frame(base.chunk(), 500_000_000)
                    self.assertEqual(selected, 2)
                    self.assertEqual(state["search_range"], [2, 2])
                    self.assertEqual(state["mode"], "time_aligned_reference")

    def test_disabled_sampling_and_update_do_not_change_reference(self):
        env = playback_env(enabled=False)
        before = env._sample_reference(0)
        env._update_stall_wrist_assist(0)
        after = env._sample_reference(0)
        for k in before:
            np.testing.assert_array_equal(before[k], after[k])

    def test_newest_live_prediction_is_used_even_when_body_chunk_is_older(self):
        env = playback_env()
        p = base.chunk(sequence=50)
        p["right_wrist_pos"][:] = [0., .2, 0.]
        closure = np.zeros((60, 6)); closure[10:, :2] = .8
        p["hand_focus"] = {"right": {"closure": closure}}
        p["_focus_source_started_ns"] = p["_focus_source_started_real_ns"] = 0
        env._focus_prediction_chunks[50] = p
        arm(env.stall_wrist_assist)
        env._update_stall_wrist_assist(20_000_000)
        s = env.stall_wrist_assist.states["right"]
        self.assertEqual(s["sequence_id"], 50)
        np.testing.assert_allclose(s["direction"], [0., 1., 0.])
        prior = s["offset"].copy()
        env._update_stall_wrist_assist(SECOND)  # prediction expired
        np.testing.assert_array_equal(s["offset"], prior)

    def test_focus_await_hold_then_rebased_playback_never_double_adds(self):
        env = playback_env()
        state = arm(env.stall_wrist_assist)
        state["offset"][:] = [.05, 0., 0.]
        before = env._sample_reference(0)
        right = env.focus_handoff.controllers["right"]
        right.state, right.hold = "await_chunk", before
        np.testing.assert_allclose(env._sample_reference(0)["right_wrist_pos"],
                                   np.repeat(before["right_wrist_pos"][0:1], 2, axis=0))
        env._update_stall_wrist_assist(0)
        self.assertEqual(state["mode"], "guard")
        right.playback = rebase_wrist_chunk(env.chunk, 0., before, ("right",))
        right.state, right.playback_started_ns = "focus", 0
        right.started_ns = 0
        np.testing.assert_allclose(env._sample_reference(0)["right_wrist_pos"][0],
                                   before["right_wrist_pos"][0])
        right.state = "cooldown"; right.playback = None
        env.chunk["right_wrist_pos"][:] = 0.  # fresh prediction is still behind
        guarded = env._sample_reference(0)
        self.assertGreaterEqual(guarded["right_wrist_pos"][0, 0], .04999)
        self.assertEqual(state["mode"], "guard")


def playback_env(enabled=True):
    env = metadata_env(enabled)
    env.requires_restart = env.localization_paused = env.awaiting_fresh_chunk = False
    env.reference_playing = True
    env._latched_sample = env.prestart_reference_sample = env.held_sample = None
    env.previous_chunk = env.previous_chunk_started_ns = env.blend_started_ns = None
    env._play_ns = env._play_real_ns = None
    env._play_scale = 1.
    env._focus_prediction_chunks = {}
    env.chunk = base.chunk()
    env.chunk["right_wrist_pos"][:] = 0.
    env.chunk_started_ns = 0
    env.current_pose_fk = SimpleNamespace(last_body_poses={
        "right_wrist_yaw_link": (np.zeros(3), np.array([1., 0., 0., 0.]))})
    env._handoff_measured_bends = lambda now: {"right": np.zeros(5)}
    return env


class AdvancementTests(unittest.TestCase):
    def test_no_advance_before_stall_and_speed_is_per_second(self):
        c = StallWristAssist()
        arm(c)
        self.assertEqual(c.states["right"]["mode"], "advance")
        start = c.states["right"]["offset"].copy()
        # Measured wrist follows the assisted reference after the stall.
        for i in range(1, 501):
            cmd = c.apply(sample())["right_wrist_pos"][0]
            c.update(2 * SECOND + i * 20_000_000, {"right": cmd},
                     {"right": cmd}, {"right": intent()})
        np.testing.assert_allclose(c.states["right"]["offset"] - start, [.05, 0., 0.], atol=1e-7)

    def test_moving_wrist_never_arms(self):
        c = StallWristAssist()
        for i in range(201):
            p = np.array([i * .001, 0., 0.])
            c.update(i * 20_000_000, {"right": p}, {"right": p}, {"right": intent()})
        self.assertEqual(c.states["right"]["mode"], "watch")
        np.testing.assert_array_equal(c.states["right"]["offset"], 0.)

    def test_live_direction_changes_and_only_selected_wrist_positions_translate(self):
        c = StallWristAssist()
        arm(c)
        before = c.states["right"]["offset"].copy()
        cmd = c.apply(sample())["right_wrist_pos"][0]
        c.update(2 * SECOND + 20_000_000, {"right": np.zeros(3)},
                 {"right": cmd}, {"right": intent((0., .2, 0.))})
        np.testing.assert_allclose(c.states["right"]["offset"] - before, [0., .0001, 0.], atol=1e-8)
        raw = sample(); copied = {k: v.copy() for k, v in raw.items()}
        out = c.apply(raw)
        for key in raw:
            np.testing.assert_array_equal(raw[key], copied[key])
            if key != "right_wrist_pos":
                np.testing.assert_array_equal(out[key], raw[key])
        np.testing.assert_allclose(out["right_wrist_pos"][1] - raw["right_wrist_pos"][1],
                                   out["right_wrist_pos"][0] - raw["right_wrist_pos"][0])

    def test_missing_intent_feedback_tracking_lag_and_holds_stop_integration(self):
        for feedback, command, events in (({}, {"right": np.zeros(3)}, {"right": intent()}),
                ({"right": np.zeros(3)}, {"right": np.zeros(3)}, {}),
                ({"right": np.zeros(3)}, {"right": np.array([.06, 0., 0.])}, {"right": intent()})):
            c = StallWristAssist(); arm(c)
            before = c.states["right"]["offset"].copy()
            c.update(2 * SECOND + 20_000_000, feedback, command, events)
            np.testing.assert_array_equal(c.states["right"]["offset"], before)
        c.suspend()
        c.update(100 * SECOND, {"right": np.zeros(3)}, {"right": np.zeros(3)}, {"right": intent()})
        np.testing.assert_array_equal(c.states["right"]["offset"], before)

    def test_offset_bound(self):
        c = StallWristAssist(max_offset_m=.002); arm(c)
        for i in range(1, 101):
            c.update(2 * SECOND + i * 20_000_000, {"right": np.zeros(3)},
                     {"right": np.zeros(3)}, {"right": intent()})
        self.assertLessEqual(np.linalg.norm(c.states["right"]["offset"]), .002000001)

    def test_reference_past_target_still_advances_until_measured_wrist_arrives(self):
        for sign in (1., -1.):
            c = StallWristAssist(); arm(c, (sign * .2, 0., 0.))
            before = c.states["right"]["offset"].copy()
            target = np.array([sign * .02, 0., 0.])
            c.update(2 * SECOND + 20_000_000, {"right": np.zeros(3)},
                     {"right": np.array([sign * .03, 0., 0.])},
                     {"right": intent(target)})
            np.testing.assert_allclose(c.states["right"]["offset"] - before,
                                       [sign * .0001, 0., 0.], atol=1e-8)
            before = c.states["right"]["offset"].copy()
            c.update(2 * SECOND + 40_000_000, {"right": target},
                     {"right": np.array([sign * .03, 0., 0.])},
                     {"right": intent(target)})
            np.testing.assert_array_equal(c.states["right"]["offset"], before)
            self.assertEqual(c.states["right"]["reason"], "at_target")

    def test_tracking_lag_pauses_and_catchup_resumes_without_dropping_offset(self):
        c = StallWristAssist(); arm(c)
        before = c.states["right"]["offset"].copy()
        command = np.array([.06, 0., 0.])
        c.update(2 * SECOND + 20_000_000, {"right": np.zeros(3)},
                 {"right": command}, {"right": intent((.04, 0., 0.))})
        np.testing.assert_array_equal(c.states["right"]["offset"], before)
        self.assertEqual(c.states["right"]["reason"], "tracking_lag")
        c.update(2 * SECOND + 40_000_000, {"right": np.array([.02, 0., 0.])},
                 {"right": command}, {"right": intent((.04, 0., 0.))})
        np.testing.assert_allclose(c.states["right"]["offset"] - before,
                                   [.0001, 0., 0.], atol=1e-8)

    def test_step_is_capped_by_measured_distance_not_reference_distance(self):
        c = StallWristAssist(); arm(c)
        before = c.states["right"]["offset"].copy()
        c.update(2 * SECOND + 20_000_000, {"right": np.zeros(3)},
                 {"right": np.array([.03, 0., 0.])},
                 {"right": intent((.00005, 0., 0.))})
        np.testing.assert_allclose(c.states["right"]["offset"] - before,
                                   [.00005, 0., 0.], atol=1e-8)

    def test_signed_guard_blocks_retreat_and_allows_forward_and_lateral_motion(self):
        for sign in (1., -1.):
            c = StallWristAssist(); arm(c, (sign * .2, 0., 0.))
            c.update(3 * SECOND, {}, {"right": np.array([sign * .05, 0., 0.])}, {}, ["right"])
            output = c.apply(sample((sign * .01, .02, 0.)))["right_wrist_pos"]
            np.testing.assert_allclose(output[0], [sign * .05, .02, 0.], atol=1e-7)
            forward = c.apply(sample((sign * .07, 0., .03)))["right_wrist_pos"][0]
            np.testing.assert_allclose(forward, [sign * .07, 0., .03])
            c.release_caught_up(sample((sign * .06, 0., 0.)), ["right"])
            self.assertEqual(c.states["right"]["mode"], "guard")
            c.release_caught_up(sample((sign * .01, 0., 0.)))
            self.assertEqual(c.states["right"]["mode"], "guard")
            c.release_caught_up(sample((sign * .06, 0., 0.)),
                               playback_sample=sample((sign * .01, 0., 0.)))
            self.assertEqual(c.states["right"]["mode"], "guard")
            c.release_caught_up(sample((sign * .06, 0., 0.)))
            self.assertEqual(c.states["right"]["mode"], "watch")
            c.reset()
            np.testing.assert_array_equal(c.states["right"]["offset"], 0.)


if __name__ == "__main__":
    unittest.main()
