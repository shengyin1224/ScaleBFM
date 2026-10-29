"""Offline regressions. No model inference, sockets, DDS or robot startup."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from hydra import compose, initialize_config_dir
from pathlib import Path

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from scalebridge.online.adaptive_chunk_execution import (
    StallLookahead, advance_on_path, closing_boundary, select_horizon,
)
from scalebridge.online.focus_handoff import PerHandFocusHandoff


def chunk(count=100):
    result = dict(frame_count=count, fps=30., sequence_id=1)
    for side in ("left", "right"):
        result[f"{side}_wrist_pos"] = np.c_[np.arange(count) * .003, np.zeros((count, 2))]
        result[f"{side}_wrist_quat_wxyz"] = np.tile([1., 0., 0., 0.], (count, 1))
        tips = np.zeros((count, 5, 3))
        tips[:, 1:, 0] = .15
        result[f"{side}_fingertip_local"] = tips
    return result


def env():
    e = HAT4AdaptiveMotionTrackingEnv.__new__(HAT4AdaptiveMotionTrackingEnv)
    e.cfg = {}
    e.reference_playing = True
    e.localization_paused = e.awaiting_fresh_chunk = e.requires_restart = False
    e.held_sample = None
    e.chunk = chunk()
    e.chunk_started_ns = 0
    e.chunk_start_frame = 0
    e.last_chunk_received_ns = 0
    e._play_now_ns = lambda now: now
    e.focus_handoff = PerHandFocusHandoff()
    e.current_pose_fk = SimpleNamespace(last_body_poses={})
    e._focus_prediction_chunks = {}
    e._adaptive_key = e._adaptive_plan = None
    e._adaptive_stall = StallLookahead()
    return e


class HorizonTests(unittest.TestCase):
    def test_speed_valley_and_minimum_motion(self):
        c = chunk()
        # A broad, prominent low-speed valley centred near frame 35.
        speed = .003 * np.ones(99)
        speed[30:40] = .0001
        for side in ("left", "right"):
            c[f"{side}_wrist_pos"][:, 0] = np.r_[0., np.cumsum(speed)]
        plan = select_horizon(c, 0)
        self.assertEqual(plan.reason, "speed_valley")
        self.assertTrue(25 <= plan.frame <= 50)
        self.assertGreaterEqual(plan.travel_m, .05)

    def test_far_and_near_count_from_accepted_cursor(self):
        c = chunk()
        self.assertEqual(select_horizon(c, 20).frame, 70)
        self.assertEqual(select_horizon(c, 20, near=True).frame, 32)

    def test_unreachable_motion_floor_is_bounded_and_reported(self):
        c = chunk()
        for side in ("left", "right"):
            c[f"{side}_wrist_pos"] *= 0
        plan = select_horizon(c, 0)
        self.assertEqual(plan.frame, 50)
        self.assertEqual(plan.reason, "motion_floor_unmet")

    def test_event_fence_beats_minimum_motion_and_length(self):
        plan = select_horizon(chunk(), 0, boundary=12)
        self.assertEqual(plan.frame, 12)
        self.assertEqual(plan.reason, "event_boundary")

    def test_geometry_fences_contraction_before_closed_pose(self):
        c = chunk()
        c['right_fingertip_local'][20:, 1:, 0] = .14
        self.assertEqual(closing_boundary(c, 'right'), 20)
        hand = {'events': [dict(kind='close', start=15, end=30)]}
        self.assertEqual(closing_boundary(c, 'right', hand=hand), 15)

    def test_short_chunk_and_nonfinite_paths(self):
        self.assertEqual(select_horizon(chunk(1), 0).frame, 0)
        c = chunk()
        c['right_wrist_pos'][5, 0] = np.nan
        with self.assertRaises(ValueError):
            select_horizon(c, 0)


class LookaheadTests(unittest.TestCase):
    def test_arc_length_follows_curve_and_stops_at_event(self):
        path = np.array([[0., 0., 0.], [.03, 0., 0.], [.03, .04, 0.]])
        np.testing.assert_allclose(advance_on_path(path, 0, .05, 2), [.03, .02, 0.])
        np.testing.assert_allclose(advance_on_path(path, 0, .05, 1), [.03, 0., 0.])
        np.testing.assert_allclose(advance_on_path(path, .5, 0, 2), [.015, 0., 0.])

    def test_stall_uses_time_and_ignores_repeated_sampling(self):
        s = StallLookahead()
        measured = dict(left=np.zeros(3), right=np.zeros(3))
        s.update(0, measured, 30.)
        for _ in range(100):
            s.update(0, measured, 30.)
        self.assertEqual(s.budget['right'], 0.)
        s.update(334_000_000, measured, 30.)
        self.assertEqual(s.budget['right'], .005)
        for i in range(2, 20):
            s.update(i * 334_000_000, measured, 30.)
        self.assertEqual(s.budget['right'], .05)

    def test_out_and_back_is_motion_and_missing_feedback_is_not_stall(self):
        s = StallLookahead()
        for stamp, x in ((0, 0.), (170_000_000, .02), (334_000_000, 0.)):
            s.update(stamp, dict(right=np.array([x, 0., 0.])), 30.)
        self.assertEqual(s.budget['right'], 0.)
        self.assertEqual(s.budget['left'], 0.)

    def test_active_hand_cannot_be_advanced(self):
        s = StallLookahead()
        s.update(0, dict(right=np.zeros(3)), 30.)
        s.update(400_000_000, dict(right=np.zeros(3)), 30., blocked=('right',))
        self.assertEqual(s.budget['right'], 0.)


class IntegrationTests(unittest.TestCase):
    def test_setup_without_handoff_does_not_construct_or_require_controller(self):
        e = env()
        e.cfg = {'focus_handoff_enabled': False, 'focus_handoff_source': 'gpt'}
        def setup(instance):instance.focus_handoff = None
        with patch.object(HAT4OnlineMotionTrackingEnv, '_setup_metadata', setup):
            e._setup_metadata()
        self.assertIsNone(e.focus_handoff)
        self.assertIsInstance(e._adaptive_stall, StallLookahead)
        self.assertEqual(e._adaptive_controllers(), {})

    def test_enabled_handoff_still_checks_source_and_controller(self):
        e = env()
        e.cfg = {'focus_handoff_enabled': True, 'focus_handoff_source': 'gpt'}
        with self.assertRaisesRegex(ValueError, 'keypoint handoff'):e._setup_metadata()
        e.cfg['focus_handoff_source'] = 'keypoints'
        with patch.object(HAT4OnlineMotionTrackingEnv, '_setup_metadata', lambda instance: setattr(instance, 'focus_handoff', None)):
            with self.assertRaisesRegex(ValueError, 'independent hand handoff'):e._setup_metadata()

    def test_handoff_off_keeps_far_hold_and_geometry_near_phase(self):
        e = env();e.focus_handoff = None
        self.assertTrue(e._min_execution_hold_active(1_000_000_000))
        self.assertFalse(e._min_execution_hold_active(1_700_000_000))
        e.chunk['right_fingertip_local'][20:, 1:, 0] = .14
        e.current_pose_fk.last_body_poses['right_wrist_yaw_link'] = (np.array([.058, 0, 0]), None)
        self.assertTrue(e._adaptive_near(0))
        self.assertLessEqual(e._adaptive_horizon(0).frame, 12)

    def test_handoff_off_accepts_chunk_and_resets_only_path_budget(self):
        e = env();e.focus_handoff = None
        e._adaptive_stall.budget['right'] = .05
        e._adaptive_stall.history['right'].append((0, np.zeros(3)))
        with patch.object(HAT4OnlineMotionTrackingEnv, '_accept_latest_chunk', return_value=True):
            self.assertTrue(e._accept_latest_chunk(100))
        self.assertEqual(e._adaptive_stall.budget['right'], 0.)
        self.assertEqual(len(e._adaptive_stall.history['right']), 1)
        e._adaptive_cap_close()  # no controller / playback to pin

    def test_handoff_off_updates_stall_budget_without_blocked_hands(self):
        e = env();e.focus_handoff = None
        with patch.object(HAT4OnlineMotionTrackingEnv, '_publish_robot_state'):
            for stamp in (0, 400_000_000):
                e.last_published_robot_state = dict(timestamp_ns=stamp,
                    left_wrist_pos_world=np.zeros(3), right_wrist_pos_world=np.zeros(3))
                e._publish_robot_state()
        self.assertEqual(e._adaptive_stall.budget, {'left': .005, 'right': .005})

    def test_handoff_off_rescue_uses_geometry_and_stops_before_close(self):
        e = env();e.focus_handoff = None;e.blend_started_ns = None
        e._playback_future_offsets_s = lambda: np.array([0., .1])
        e._adaptive_stall.budget['right'] = .05
        e.chunk['right_fingertip_local'][35:, 1:, 0] = .14
        sample = {f'{side}_wrist_pos': e.chunk[f'{side}_wrist_pos'][[30, 33]].copy()
                  for side in ('left', 'right')}
        with patch.object(HAT4OnlineMotionTrackingEnv, '_sample_reference', return_value=sample):
            out = e._sample_reference(1_000_000_000)
        np.testing.assert_allclose(out['right_wrist_pos'][:, 0], [.105, .105])
        np.testing.assert_allclose(out['left_wrist_pos'][:, 0], [.09, .099])
        # At the contraction boundary itself, no extra forward motion is issued.
        current = dict(right_wrist_pos=np.array([[.105, 0, 0], [.114, 0, 0]]),
                       left_wrist_pos=np.zeros((2, 3)))
        with patch.object(HAT4OnlineMotionTrackingEnv, '_sample_reference', return_value=current):
            out = e._sample_reference(int(35/30*1e9)+1)
        np.testing.assert_array_equal(out['right_wrist_pos'], current['right_wrist_pos'])

    def test_single_env_override_and_original_config(self):
        config = str(Path(__file__).resolve().parents[1] / 'scalebridge/config')
        with initialize_config_dir(version_base=None, config_dir=config):
            old = compose(config_name='hat4_online')
            new = compose(config_name='hat4_online', overrides=['env=motion_tracking_hat4_adaptive'])
        self.assertEqual(old.env._target_, 'env.motion_tracking_hat4_online.HAT4OnlineMotionTrackingEnv')
        self.assertFalse(old.env.config.chunk_start_reference_continuity)
        self.assertEqual(old.env.config.min_chunk_execution_fraction, .66)
        self.assertEqual(new.env.config.min_chunk_execution_fraction, 0.)
        self.assertTrue(new.env.config.chunk_start_reference_continuity)

    def test_dynamic_hold_and_recovery_bypass(self):
        e = env()
        self.assertTrue(e._min_execution_hold_active(1_000_000_000))
        self.assertFalse(e._min_execution_hold_active(1_700_000_000))
        for name in ('localization_paused', 'awaiting_fresh_chunk', 'requires_restart'):
            setattr(e, name, True)
            self.assertFalse(e._min_execution_hold_active(1))
            setattr(e, name, False)

    def test_horizon_does_not_slide_when_monotonic_clock_is_queried_in_past(self):
        e = env()
        e._last_chunk_received_play_ns = 0
        e._play_now_ns = lambda now: 1_700_000_000
        self.assertFalse(e._min_execution_hold_active(1_700_000_000))
        self.assertEqual(e._adaptive_plan.frame, 50)

    def test_stall_sampling_follows_path_and_caps_at_known_close(self):
        e = env()
        e.blend_started_ns = None
        e._playback_future_offsets_s = lambda: np.array([0., .1])
        e._adaptive_stall.budget['right'] = .05
        hand = dict(events=[dict(kind='close', start=35, end=50)])
        e._focus_prediction_chunks[1] = dict(hand_focus=dict(right=hand))
        # At frame 30 the event is only 15mm away, despite a 50mm budget.
        sample = {f'{side}_wrist_pos': e.chunk[f'{side}_wrist_pos'][[30, 33]].copy()
                  for side in ('left', 'right')}
        with patch.object(HAT4OnlineMotionTrackingEnv, '_sample_reference', return_value=sample):
            out = e._sample_reference(1_000_000_000)
        np.testing.assert_allclose(out['right_wrist_pos'][:, 0], [.105, .105])
        np.testing.assert_allclose(out['left_wrist_pos'][:, 0], [.09, .099])
        self.assertEqual(e._adaptive_stall.budget['right'], .05)

    def test_stall_sampling_without_same_chunk_ik_defers_rescue(self):
        e = env()
        e.blend_started_ns = None
        e._playback_future_offsets_s = lambda: np.array([0.])
        e._adaptive_stall.budget['right'] = .05
        sample = dict(right_wrist_pos=np.array([[.09, 0., 0.]]))
        with patch.object(HAT4OnlineMotionTrackingEnv, '_sample_reference', return_value=sample):
            out = e._sample_reference(1_000_000_000)
        np.testing.assert_allclose(out['right_wrist_pos'], [[.09, 0., 0.]])

    def test_near_entry_shortens_execution_and_arrival_bypasses_hold(self):
        e = env()
        c = e.focus_handoff.controllers['right']
        c.entry = dict(right_wrist_pos=np.array([.02, 0., 0.]))
        e.current_pose_fk.last_body_poses['right_wrist_yaw_link'] = (np.zeros(3), None)
        self.assertTrue(e._min_execution_hold_active(200_000_000))
        self.assertFalse(e._min_execution_hold_active(410_000_000))
        c.state = 'await_chunk'
        self.assertFalse(e._min_execution_hold_active(1))

    def test_close_overlay_and_clocks_survive_new_chunk(self):
        e = env()
        right = e.focus_handoff.controllers['right']
        right.state = 'focus'
        right.event = dict(kind='close')
        right.playback = original = chunk()
        right.playback_started_ns = 123
        right.playback_started_real_ns = 456
        e._adaptive_stall.history['right'].append((0, np.zeros(3)))
        def replace(*args):
            right.playback = chunk()
            right.playback_started_ns = 999
            right.playback_started_real_ns = 999
            return True
        with patch.object(HAT4OnlineMotionTrackingEnv, '_accept_latest_chunk', replace):
            self.assertTrue(e._accept_latest_chunk(100))
        self.assertIs(right.playback, original)
        self.assertEqual(right.playback_started_ns, 123)
        self.assertEqual(right.playback_started_real_ns, 456)
        self.assertEqual(len(e._adaptive_stall.history['right']), 1)

    def test_release_overlay_can_update(self):
        e = env()
        right = e.focus_handoff.controllers['right']
        right.state = 'focus'
        right.event = dict(kind='open')
        right.playback = chunk()
        newer = chunk()
        def replace(*args):
            right.playback = newer
            return True
        with patch.object(HAT4OnlineMotionTrackingEnv, '_accept_latest_chunk', replace):
            e._accept_latest_chunk(100)
        self.assertIs(right.playback, newer)

    def test_close_suffix_clamped_without_mutating_original(self):
        e = env()
        right = e.focus_handoff.controllers['right']
        right.state = 'focus'
        right.event = dict(kind='close')
        right.playback = original = chunk()
        before = original['right_wrist_pos'].copy()
        e._focus_prediction_chunks[1] = dict(hand_focus=dict(right=dict(events=[dict(kind='close', start=10, end=25)])))
        e._adaptive_cap_close()
        np.testing.assert_array_equal(original['right_wrist_pos'], before)
        np.testing.assert_array_equal(right.playback['right_wrist_pos'][26:], np.repeat(before[25:26], 74, axis=0))
        np.testing.assert_array_equal(right.playback['left_wrist_pos'], original['left_wrist_pos'])


if __name__ == '__main__':
    unittest.main()
