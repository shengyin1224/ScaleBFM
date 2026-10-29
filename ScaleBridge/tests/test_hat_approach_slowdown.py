"""Entry-distance playback rate; offline, no DDS, network or robot execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.focus_handoff import PerHandFocusHandoff
from scalebridge.online.reference import sample_chunk
import test_hat_focus_handoff as base

SECOND = 1_000_000_000
ENTRY = np.array([.3, 0., 1.], dtype=np.float32)


def env(enabled=True, fingers_full_speed=True):
    out = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
    out.approach_slowdown = enabled
    out.approach_slowdown_distance_bands_m = np.array([.20, .30, .40])
    out.approach_slowdown_band_scales = np.array([.4, .6, .8])
    out.approach_slowdown_min_scale = .4
    out.approach_slowdown_fingers_full_speed = fingers_full_speed
    out.focus_handoff = PerHandFocusHandoff()
    out.reference_playing = True
    out.localization_paused = False
    out._play_ns = None
    out._play_real_ns = None
    out._play_scale = 1.0
    out._play_scale_hold = {}
    out._play_scale_logged = 1.0
    out._play_entry_distances = {}
    out.current_pose_fk = SimpleNamespace(last_body_poses={})
    return out


def approach(out, distance_m):
    """Arm the right hand's entry consensus and place the measured wrist."""
    right = out.focus_handoff.controllers["right"]
    right.state = "approach"
    right.entry = {"right_wrist_pos": ENTRY,
                   "right_wrist_quat_wxyz": np.array([1., 0., 0., 0.])}
    out.current_pose_fk.last_body_poses = {
        "right_wrist_yaw_link": (ENTRY - np.array([distance_m, 0., 0.]),
                                 np.array([1., 0., 0., 0.]))}
    out._update_playback_scale(0)
    return out._play_scale


class EntryDistanceRateTests(unittest.TestCase):
    def test_rate_uses_distance_bands_including_boundaries(self):
        for distance, expected in (
            (.50, 1.0), (.40, 1.0), (.400001, 1.0),
            (.399999, .8), (.35, .8), (.300001, .8),
            (.30, .6), (.299999, .6), (.25, .6), (.200001, .6),
            (.20, .4), (.199999, .4), (.01, .4), (0., .4),
        ):
            with self.subTest(distance=distance):
                self.assertAlmostEqual(approach(env(), distance), expected)

    def test_absent_consensus_keeps_last_rate_then_recomputes_on_return(self):
        out = env()
        approach(out, .02)
        out.focus_handoff.controllers["right"].entry = None
        out._update_playback_scale(0)
        self.assertEqual(out._play_scale, .4)
        self.assertAlmostEqual(approach(out, .35), .8)
        self.assertAlmostEqual(approach(out, .25), .6)
        self.assertAlmostEqual(approach(out, .50), 1.0)

    def test_no_entry_before_first_slowdown_keeps_full_speed(self):
        out = env()
        out._update_playback_scale(0)
        self.assertEqual(out._play_scale, 1.0)

    def test_missing_or_invalid_measured_pose_keeps_last_rate(self):
        out = env()
        approach(out, .25)
        out.current_pose_fk.last_body_poses = {}
        out._update_playback_scale(0)
        self.assertEqual(out._play_scale, .6)
        out.current_pose_fk.last_body_poses = {
            "right_wrist_yaw_link": (np.full(3, np.nan), np.array([1., 0., 0., 0.]))}
        out._update_playback_scale(0)
        self.assertEqual(out._play_scale, .6)

    def test_unlocked_window_holds_the_rate_it_unlocked_at(self):
        out = env()
        self.assertAlmostEqual(approach(out, .25), .6)
        # Unlocking consumes the entry pose, so there is no distance left to
        # measure: the window must keep crawling, not jump back to full speed.
        right = out.focus_handoff.controllers["right"]
        right.state, right.entry = "focus", None
        out._update_playback_scale(0)
        self.assertAlmostEqual(out._play_scale, .6)

        right.state = "cooldown"
        out._update_playback_scale(0)
        self.assertAlmostEqual(out._play_scale, .6)

    def test_slowest_hand_governs_the_shared_clock(self):
        out = env()
        approach(out, .35)
        left = out.focus_handoff.controllers["left"]
        left.entry = {"left_wrist_pos": ENTRY,
                      "left_wrist_quat_wxyz": np.array([1., 0., 0., 0.])}
        out.current_pose_fk.last_body_poses["left_wrist_yaw_link"] = (
            ENTRY - np.array([.25, 0., 0.]), np.array([1., 0., 0., 0.]))
        out._update_playback_scale(0)
        self.assertAlmostEqual(out._play_scale, .6)
        # Losing the slower hand's entry must not let the other hand restore
        # the shared clock to its faster rate.
        left.entry = None
        out._update_playback_scale(0)
        self.assertAlmostEqual(out._play_scale, .6)

    def test_localization_hold_preserves_rate_but_stop_clears_it(self):
        out = env()
        approach(out, .02)
        out.localization_paused = True
        out._update_playback_scale(0)
        self.assertEqual(out._play_scale, .4)
        out.localization_paused = False
        out.reference_playing = False
        out._update_playback_scale(0)
        self.assertEqual(out._play_scale, 1.0)
        self.assertEqual(out._play_scale_hold, {})


class PlaybackClockTests(unittest.TestCase):
    def test_clock_integrates_the_scaled_rate(self):
        out = env()
        self.assertEqual(out._play_now_ns(10 * SECOND), 10 * SECOND)
        out._play_scale = .25
        self.assertEqual(out._play_now_ns(14 * SECOND), 11 * SECOND)
        # A rate change never rewrites the phase already played.
        out._play_scale = 1.0
        self.assertEqual(out._play_now_ns(15 * SECOND), 12 * SECOND)

    def test_disabled_clock_is_the_monotonic_clock(self):
        out = env(enabled=False)
        out._play_now_ns(3 * SECOND)
        out._update_playback_scale(0)
        self.assertEqual(out._play_scale, 1.0)
        self.assertEqual(out._play_now_ns(9 * SECOND), 9 * SECOND)

    def test_latency_cursor_shrinks_with_the_slowed_clock(self):
        """The saved frames must not be skipped back by the next chunk."""
        out = env()
        out.state_timestamp_history = [(10, 0)]
        out.state_play_history = [(10, 0)]
        out.dt = .02
        out._play_now_ns(0)
        out._play_scale = .3
        chunk = base.chunk(source=10)
        frame, age_s, source = out._latency_start_frame(chunk, SECOND)
        self.assertEqual(source, "timestamp")
        self.assertAlmostEqual(age_s, .3, places=6)
        self.assertEqual(frame, 9)  # 30 frames unscaled.

    def test_missing_play_stamp_keeps_the_monotonic_fallback_when_disabled(self):
        out = env(enabled=False)
        out.state_timestamp_history = [(10, 0)]
        out.state_play_history = []
        out.dt = .02
        _, age_s, source = out._latency_start_frame(base.chunk(source=10), SECOND)
        self.assertEqual(source, "timestamp")
        self.assertAlmostEqual(age_s, 1.0, places=6)


class ActiveWindowFingerRateTests(unittest.TestCase):
    def playback_env(self, fingers_full_speed=True):
        out = env(fingers_full_speed=fingers_full_speed)
        out.requires_restart = False
        out._latched_sample = None
        out.reference_play_gate = False
        out.prestart_reference_sample = None
        out.held_sample = None
        out.previous_chunk = None
        out.previous_chunk_started_ns = None
        out.blend_started_ns = None
        out.future_offsets_s = np.zeros(1, dtype=np.float64)
        playback = base.chunk()
        # Fingertips close over the chunk, so the sampled phase is readable.
        playback["right_fingertip_local"][:, 1, 0] = np.linspace(
            0., .59, playback["frame_count"])
        out.chunk = playback
        out.chunk_started_ns = 0
        right = out.focus_handoff.controllers["right"]
        right.state = "focus"
        right.started_ns = 0
        right.playback = playback
        # One second of wall clock has passed; the slowed reference has played
        # 0.3s of it.
        right.playback_started_ns = 0
        right.playback_started_real_ns = -SECOND + 300_000_000
        out._play_now_ns(300_000_000)
        return out, playback

    def test_active_window_fingers_keep_the_trained_rate(self):
        out, playback = self.playback_env()
        sample = out._sample_reference(300_000_000)
        np.testing.assert_allclose(sample["right_wrist_pos"][0],
                                   playback["right_wrist_pos"][9], atol=1e-6)
        np.testing.assert_allclose(sample["right_fingertip_local"][0],
                                   playback["right_fingertip_local"][30], atol=1e-6)

    def test_opt_out_keeps_fingers_on_the_slowed_clock(self):
        out, playback = self.playback_env(fingers_full_speed=False)
        sample = out._sample_reference(300_000_000)
        np.testing.assert_allclose(sample["right_fingertip_local"][0],
                                   playback["right_fingertip_local"][9], atol=1e-6)

    def test_future_wrist_is_slowed_while_finger_future_stays_full_speed(self):
        out, playback = self.playback_env()
        out.future_offsets_s = np.array([0., .1])
        out._play_scale = .2
        sample = out._sample_reference(300_000_000)
        expected_wrist = sample_chunk(playback, .3, [0., .02])["right_wrist_pos"]
        expected_fingers = sample_chunk(playback, 1., [0., .1])["right_fingertip_local"]
        np.testing.assert_allclose(sample["right_wrist_pos"], expected_wrist, atol=1e-6)
        np.testing.assert_allclose(sample["right_fingertip_local"], expected_fingers, atol=1e-6)

    def test_finger_opt_out_scales_future_horizons_too(self):
        out, playback = self.playback_env(fingers_full_speed=False)
        out.future_offsets_s = np.array([0., .1])
        out._play_scale = .2
        sample = out._sample_reference(300_000_000)
        expected = sample_chunk(playback, .3, [0., .02])
        np.testing.assert_allclose(sample["right_fingertip_local"],
                                   expected["right_fingertip_local"], atol=1e-6)


class FutureHorizonTests(unittest.TestCase):
    def test_normal_future_token_matches_actual_later_playback(self):
        for scale in (.2, .3, .4, .5, .6, .8, 1.):
            with self.subTest(scale=scale):
                out, playback = ActiveWindowFingerRateTests().playback_env()
                out.focus_handoff.controllers["right"].state = "approach"
                out.future_offsets_s = np.array([0., .1])
                out._play_scale = scale
                # Exercise position and orientation for all body links.
                for name in ("root", "head", "left_wrist", "right_wrist"):
                    playback[f"{name}_pos"][:, 2] = np.arange(60) * .003
                    angles = np.arange(60) * .01
                    playback[f"{name}_quat_wxyz"][:, 0] = np.cos(angles / 2)
                    playback[f"{name}_quat_wxyz"][:, 3] = np.sin(angles / 2)
                sample = out._sample_reference(300_000_000)
                later = out._sample_reference(400_000_000)
                for key in sample:
                    np.testing.assert_allclose(sample[key][1], later[key][0], atol=1e-6,
                                               err_msg=key)
                np.testing.assert_array_equal(out.future_offsets_s, [0., .1])

    def test_overlap_samples_both_chunks_at_scaled_future_offsets(self):
        out, playback = ActiveWindowFingerRateTests().playback_env()
        out.focus_handoff.controllers["right"].state = "approach"
        out.future_offsets_s = np.array([0., .1])
        out._play_scale = .2
        old = base.chunk(sequence=0)
        old["right_wrist_pos"] *= 2
        out.previous_chunk = old
        out.previous_chunk_started_ns = 0
        out.blend_started_ns = 200_000_000
        out.cfg = {"chunk_blend_s": .2}
        sample = out._sample_reference(300_000_000)
        expected = (sample_chunk(old, .3, [0., .02])["right_wrist_pos"]
                    + sample_chunk(playback, .3, [0., .02])["right_wrist_pos"]) / 2
        np.testing.assert_allclose(sample["right_wrist_pos"], expected, atol=1e-6)

    def test_rate_change_preserves_current_phase_and_updates_future_same_step(self):
        out = env()
        out.future_offsets_s = np.array([0., .1])
        out._play_now_ns(0)
        out._play_scale = .2
        # Stub transport/state work; run the actual observation ordering.
        for name in ("_update_state_manager", "_publish_robot_state", "_update_chunk_focus",
                     "_poll_gpt_supervisor", "_maybe_begin_focus_handoff", "_accept_latest_chunk",
                     "_update_staleness", "_submit_gpt_snapshot", "_log_gpt_control_step",
                     "_update_observation_manager"):
            setattr(out, name, Mock())
        out._update_playback_scale = Mock(side_effect=lambda now: setattr(out, "_play_scale", .5))
        observed = []
        out._gather_reference_state = lambda now: observed.append(
            (out._play_now_ns(now), out._playback_future_offsets_s().copy()))
        with patch("scalebridge.env.motion_tracking_hat4_online.monotonic_ns", return_value=SECOND):
            out._compute_observation()
        self.assertEqual(observed[0][0], 200_000_000)
        np.testing.assert_array_equal(observed[0][1], [0., .05])
        self.assertEqual(out._play_now_ns(2 * SECOND), 700_000_000)


if __name__ == '__main__':
    unittest.main()
