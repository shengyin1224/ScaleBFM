import unittest
import unittest.mock

import numpy as np

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.reference import rtc_blend_chunk_prefix

SECOND = 1_000_000_000


def make_env(frame_count=100, fps=30.0, fraction=0.66, end_margin_s=0.5,
             received_ns=0, started_ns=0):
    env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
    env.cfg = {
        "min_chunk_execution_fraction": fraction,
        "chunk_end_margin_s": end_margin_s,
        "chunk_hold_timeout_s": 0.5,
        "chunk_latch_timeout_s": 2.0,
    }
    env.reference_playing = True
    env.localization_paused = False
    env.awaiting_fresh_chunk = False
    env.requires_restart = False
    env.held_sample = None
    env.chunk = {"fps": fps, "frame_count": frame_count}
    env.chunk_started_ns = started_ns
    env.last_chunk_received_ns = received_ns
    env._chunk_hold_release_ns = None
    env.reference_play_control = "R1"
    return env


def make_rtc_chunk(frame_count=30, offset=0.0):
    frame = np.arange(frame_count, dtype=np.float32) + float(offset)
    pos = np.stack((frame, frame + 1.0, frame + 2.0), axis=-1)
    quat = np.zeros((frame_count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    fingers = np.repeat(pos[:, None, :], 5, axis=1)
    return {
        "fps": 30.0,
        "frame_count": frame_count,
        "root_pos": pos.copy(),
        "root_quat_wxyz": quat.copy(),
        "head_pos": pos.copy(),
        "head_quat_wxyz": quat.copy(),
        "left_wrist_pos": pos.copy(),
        "left_wrist_quat_wxyz": quat.copy(),
        "right_wrist_pos": pos.copy(),
        "right_wrist_quat_wxyz": quat.copy(),
        "left_fingertip_local": fingers.copy(),
        "right_fingertip_local": fingers.copy(),
    }


class MinChunkExecutionHoldTest(unittest.TestCase):
    def test_hold_covers_two_thirds_of_the_chunk_then_releases(self):
        # 100 frames at 30 Hz = 3.33 s; 0.66 of that = 2.2 s.  A replan
        # period (0.33 s) into playback the swap must be deferred; past the
        # execution floor it must not.
        env = make_env()
        self.assertTrue(env._min_execution_hold_active(int(0.33 * SECOND)))
        self.assertTrue(env._min_execution_hold_active(int(2.19 * SECOND)))
        self.assertFalse(env._min_execution_hold_active(int(2.21 * SECOND)))

    def test_end_margin_releases_before_the_chunk_clamps(self):
        # A 30-frame chunk lasts 1.0 s with a 0.66 s execution floor, but the
        # 0.5 s end margin wins: the successor must be consumed before the
        # reference clamps on the final frame.
        env = make_env(frame_count=30)
        self.assertTrue(env._min_execution_hold_active(int(0.45 * SECOND)))
        self.assertFalse(env._min_execution_hold_active(int(0.55 * SECOND)))

    def test_recovery_paths_are_never_held(self):
        # R1/outage recovery, a staleness-frozen reference, a latched restart,
        # and the pre-first-chunk state all want a fresh chunk immediately.
        now = int(0.33 * SECOND)
        for attribute, value in (
            ("awaiting_fresh_chunk", True),
            ("held_sample", object()),
            ("requires_restart", True),
            ("reference_playing", False),
            ("chunk", None),
            ("last_chunk_received_ns", None),
        ):
            env = make_env()
            setattr(env, attribute, value)
            self.assertFalse(env._min_execution_hold_active(now), attribute)

    def test_zero_fraction_restores_swap_on_every_replan(self):
        env = make_env(fraction=0.0)
        self.assertFalse(env._min_execution_hold_active(int(0.1 * SECOND)))

    def test_humi_relative_chunks_swap_on_every_replan_by_default(self):
        env = make_env(fraction=0.66)
        env.chunk["action_representation"] = "relative_chunk"
        self.assertFalse(env._min_execution_hold_active(int(0.1 * SECOND)))

        env.cfg["relative_chunk_min_execution_fraction"] = 0.5
        self.assertTrue(env._min_execution_hold_active(int(0.1 * SECOND)))

    def test_staleness_clock_counts_from_hold_release_not_acceptance(self):
        # During a deliberate 2.2 s hold no chunk is accepted, so judging the
        # stream by last_chunk_received_ns would freeze (0.5 s) and latch
        # (2.0 s) mid-hold.  The watchdog must instead count from the hold's
        # release.
        env = make_env()
        frozen = object()
        env._sample_reference = lambda now_ns: frozen

        env._update_staleness(int(2.0 * SECOND))  # hold active: marks release
        self.assertIsNone(env.held_sample)
        self.assertFalse(env.requires_restart)
        self.assertEqual(env._chunk_hold_release_ns, int(2.0 * SECOND))

        # 0.4 s after release: within chunk_hold_timeout_s, still healthy.
        env._update_staleness(int(2.4 * SECOND))
        self.assertIsNone(env.held_sample)
        self.assertFalse(env.requires_restart)

        # 0.6 s after release with no new chunk: the stream really is stale.
        env._update_staleness(int(2.6 * SECOND))
        self.assertIs(env.held_sample, frozen)
        self.assertFalse(env.requires_restart)

        # 2.0 s after release: latched restart.
        env._update_staleness(int(4.1 * SECOND))
        self.assertTrue(env.requires_restart)
        self.assertFalse(env.reference_playing)


class LatchedHoldTest(unittest.TestCase):
    """The stale-stream latch must hold the running pose, not the stand pose."""

    def _latched_env(self):
        env = make_env()
        env.reference_play_gate = True
        env.future_offsets_s = [0.0, 0.1]
        env._latched_sample = None
        env._latched_link_poses = None
        env.prestart_reference_sample = "STAND"
        stand_pose = (np.zeros(3, dtype=np.float32), np.array([1, 0, 0, 0], np.float32))
        live_pose = (np.ones(3, dtype=np.float32), np.array([0, 1, 0, 0], np.float32))
        env._prestart_link_poses = {"torso_link": stand_pose}
        env.current_pose_fk = type(
            "FK", (), {"last_body_poses": {"torso_link": live_pose}}
        )()
        env._repeat_current = lambda sample, count: ("repeat", sample, count)
        frozen = {"running": True}
        env._sample_reference_impl = lambda now_ns: frozen
        real_sample = HAT4OnlineMotionTrackingEnv._sample_reference
        env._sample_reference = lambda now_ns: (
            frozen if env.held_sample is None and not env._latched_hold_active()
            else real_sample(env, now_ns)
        )
        env._update_staleness(int(4.1 * SECOND))
        return env, frozen, real_sample

    def test_latch_serves_the_running_pose_not_prestart(self):
        env, frozen, real_sample = self._latched_env()
        self.assertTrue(env.requires_restart)
        self.assertIs(env._latched_sample, frozen)
        # Snapshot of the live (running) pose, not the pre-start stand.
        self.assertEqual(list(env._latched_link_poses), ["torso_link"])
        np.testing.assert_allclose(env._latched_link_poses["torso_link"][0], 1.0)
        # Reference is the frozen running pose even though the pre-start
        # stand snapshot is still available.
        self.assertEqual(real_sample(env, int(5 * SECOND)), ("repeat", frozen, 2))

    def test_latch_survives_a_chunk_accepted_after_recovery(self):
        # _accept_latest_chunk clears held_sample but not requires_restart, so
        # the hold must not fall back to the stand while waiting for R1.
        env, frozen, real_sample = self._latched_env()
        env.held_sample = None
        self.assertTrue(env._latched_hold_active())
        self.assertEqual(real_sample(env, int(5 * SECOND)), ("repeat", frozen, 2))

    def test_prestart_hold_is_unchanged(self):
        env = make_env()
        env.reference_play_gate = True
        env.reference_playing = False
        env.future_offsets_s = [0.0, 0.1]
        env._latched_sample = None
        env._latched_link_poses = None
        env.prestart_reference_sample = "STAND"
        env._repeat_current = lambda sample, count: ("repeat", sample, count)
        self.assertFalse(env._latched_hold_active())
        self.assertEqual(
            HAT4OnlineMotionTrackingEnv._sample_reference(env, int(1 * SECOND)),
            ("repeat", "STAND", 2),
        )

    def test_latch_holds_the_hands_instead_of_opening_them(self):
        # Dropping whatever is in the hand defeats the point of holding the
        # body pose, so the open-hands pre-start branch must not run.
        env, _frozen, _real_sample = self._latched_env()
        env.hand_ik_worker = unittest.mock.Mock()
        env.hand_ik_worker.latest.return_value = None
        env.open_hand_dof12 = np.zeros(12)
        env.simulator = unittest.mock.Mock()
        env.finger_wrist_gate = False
        env.finger_chunk_source = "playback"
        env._finger_chunk = None
        held = {
            "left_fingertip_local": np.full((1, 5, 3), 0.25, dtype=np.float32),
            "right_fingertip_local": np.full((1, 5, 3), 0.75, dtype=np.float32),
        }
        env.last_reference_sample = held

        HAT4OnlineMotionTrackingEnv._send_hand_target(env)

        env.simulator.set_hand_ik_target.assert_not_called()
        left, right = env.hand_ik_worker.submit.call_args[0]
        np.testing.assert_allclose(left, held["left_fingertip_local"][0])
        np.testing.assert_allclose(right, held["right_fingertip_local"][0])

    def test_prestart_still_opens_the_hands(self):
        env = make_env()
        env.reference_play_gate = True
        env.reference_playing = False
        env.requires_restart = False
        env._latched_sample = None
        env.hand_ik_worker = unittest.mock.Mock()
        env.open_hand_dof12 = np.zeros(12)
        env.simulator = unittest.mock.Mock()

        HAT4OnlineMotionTrackingEnv._send_hand_target(env)

        env.simulator.set_hand_ik_target.assert_called_once()
        env.hand_ik_worker.submit.assert_not_called()


class SelfRejectionStalenessTest(unittest.TestCase):
    """Chunks we deliberately discard must not age the chunk stream.

    2026-09-11 run 11-04-00: a flapping Vive made ScaleBridge drop arriving
    chunks and then wait for a fresh-state chunk; the clock ran through both
    windows and latched a perfectly healthy HAT stream 0.87s *after* Vive had
    already recovered.
    """

    def _env(self, **flags):
        env = make_env(fraction=0.0)
        frozen = object()
        env._sample_reference = lambda now_ns: frozen
        for key, value in flags.items():
            setattr(env, key, value)
        return env, frozen

    def test_vive_outage_does_not_age_the_stream(self):
        env, _ = self._env(localization_paused=True)
        env._update_staleness(int(3.0 * SECOND))
        self.assertIsNone(env.held_sample)
        self.assertFalse(env.requires_restart)
        self.assertEqual(env._chunk_hold_release_ns, int(3.0 * SECOND))

    def test_recovery_handshake_does_not_age_the_stream(self):
        env, _ = self._env(awaiting_fresh_chunk=True)
        env._update_staleness(int(3.0 * SECOND))
        self.assertFalse(env.requires_restart)

    def test_clock_restarts_when_consumption_resumes(self):
        env, frozen = self._env(localization_paused=True)
        env._update_staleness(int(5.0 * SECOND))  # 5s of self-rejection
        env.localization_paused = False
        # 0.4s after the rejection stops: still healthy despite 5.4s since the
        # last acceptance.
        env._update_staleness(int(5.4 * SECOND))
        self.assertIsNone(env.held_sample)
        self.assertFalse(env.requires_restart)
        # 0.6s: genuinely stale now, soft freeze as before.
        env._update_staleness(int(5.6 * SECOND))
        self.assertIs(env.held_sample, frozen)
        self.assertFalse(env.requires_restart)

    def test_genuine_stream_loss_still_latches(self):
        env, frozen = self._env()
        env._update_staleness(int(2.5 * SECOND))
        self.assertTrue(env.requires_restart)
        self.assertFalse(env.reference_playing)


class RTCChunkPrefixTest(unittest.TestCase):
    def test_uses_live_old_cursor_and_latency_aligned_new_prefix(self):
        old = make_rtc_chunk(offset=0.0)
        new = make_rtc_chunk(offset=100.0)
        out = rtc_blend_chunk_prefix(
            old,
            new,
            old_start_frame=12,
            new_start_frame=4,
            prefix_frames=10,
            hard_prefix_frames=2,
            blend_frames=8,
        )

        # The first two playback samples are old[12:14], not old[0:2].
        np.testing.assert_allclose(out["root_pos"][4], old["root_pos"][12])
        np.testing.assert_allclose(out["root_pos"][5], old["root_pos"][13])
        # The 8-frame ramp lands exactly on the new trajectory at its end.
        np.testing.assert_allclose(out["root_pos"][13], new["root_pos"][13])
        # Frames outside the RTC prefix are untouched, and inputs are not
        # mutated (apply_alignment shares fingertip arrays with the raw chunk).
        np.testing.assert_allclose(out["root_pos"][3], new["root_pos"][3])
        np.testing.assert_allclose(out["root_pos"][14], new["root_pos"][14])
        np.testing.assert_allclose(new["root_pos"][4, 0], 104.0)
        np.testing.assert_allclose(
            out["left_fingertip_local"][4], old["left_fingertip_local"][12]
        )

    def test_rotation_uses_slerp(self):
        old = make_rtc_chunk(frame_count=12)
        new = make_rtc_chunk(frame_count=12, offset=100.0)
        # 180 degrees about z. With a two-frame ramp, the first frame has
        # alpha=0.5 and must be a normalized 90-degree quaternion.
        new_quat = np.zeros((12, 4), dtype=np.float32)
        new_quat[:, 3] = 1.0
        for key in (
            "root_quat_wxyz", "head_quat_wxyz",
            "left_wrist_quat_wxyz", "right_wrist_quat_wxyz",
        ):
            new[key] = new_quat.copy()
        out = rtc_blend_chunk_prefix(
            old,
            new,
            old_start_frame=2,
            new_start_frame=3,
            prefix_frames=2,
            hard_prefix_frames=0,
            blend_frames=2,
        )
        expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        np.testing.assert_allclose(out["root_quat_wxyz"][3], expected, atol=1e-6)
        np.testing.assert_allclose(out["root_quat_wxyz"][4], new_quat[4])
        np.testing.assert_allclose(
            np.linalg.norm(out["root_quat_wxyz"], axis=1), 1.0, atol=1e-6
        )

    def test_disabled_path_is_exact_no_op(self):
        env = make_env()
        env.cfg["rtc_enabled"] = False
        candidate = make_rtc_chunk()
        result, state = env._apply_rtc_overlap(
            candidate, new_start_frame=3, now_ns=int(2.2 * SECOND),
            reset_transition=False,
        )
        self.assertIs(result, candidate)
        self.assertIsNone(state)

    def test_enabled_path_computes_old_start_from_live_playback_cursor(self):
        candidate = make_rtc_chunk(offset=100.0)
        env = make_env(frame_count=30, received_ns=0, started_ns=0)
        env.cfg.update({
            "rtc_enabled": True,
            "rtc_prefix_frames": 10,
            "rtc_hard_prefix_frames": 2,
            "rtc_blend_frames": 8,
        })
        env.chunk = make_rtc_chunk()

        # 0.405 s * 30 Hz = 12.15. RTC starts at ceil(cursor)=13, not at
        # frame zero and not at a fixed latency-derived index.
        result, state = env._apply_rtc_overlap(
            candidate, new_start_frame=3, now_ns=int(0.405 * SECOND),
            reset_transition=False,
        )
        self.assertTrue(state["applied"])
        self.assertEqual(state["old_start_frame"], 13)
        self.assertEqual(state["new_start_frame"], 3)
        np.testing.assert_allclose(result["root_pos"][3], env.chunk["root_pos"][13])
        np.testing.assert_allclose(result["root_pos"][4], env.chunk["root_pos"][14])

    def test_all_safety_states_bypass_rtc(self):
        candidate = make_rtc_chunk(offset=100.0)
        cases = (
            ("awaiting_fresh_chunk", True, "localization_recovery"),
            ("localization_paused", True, "localization_recovery"),
            ("requires_restart", True, "latched_restart"),
            ("held_sample", object(), "frozen_reference"),
            ("chunk", None, "no_old_chunk"),
        )
        for attribute, value, reason in cases:
            env = make_env(frame_count=30, received_ns=0, started_ns=0)
            env.cfg.update({
                "rtc_enabled": True,
                "rtc_prefix_frames": 10,
                "rtc_hard_prefix_frames": 2,
                "rtc_blend_frames": 8,
            })
            env.chunk = make_rtc_chunk()
            setattr(env, attribute, value)
            result, state = env._apply_rtc_overlap(
                candidate, new_start_frame=1, now_ns=int(0.2 * SECOND),
                reset_transition=False,
            )
            self.assertIs(result, candidate, attribute)
            self.assertEqual(state["bypass_reason"], reason, attribute)

    def test_recovery_and_insufficient_old_future_bypass(self):
        candidate = make_rtc_chunk()
        env = make_env(frame_count=30, received_ns=0, started_ns=0)
        env.cfg.update({
            "rtc_enabled": True,
            "rtc_prefix_frames": 10,
            "rtc_hard_prefix_frames": 2,
            "rtc_blend_frames": 8,
        })
        env.chunk = make_rtc_chunk()

        result, state = env._apply_rtc_overlap(
            candidate, new_start_frame=1, now_ns=int(0.5 * SECOND),
            reset_transition=True,
        )
        self.assertIs(result, candidate)
        self.assertEqual(state["bypass_reason"], "reset_or_not_playing")

        # At 30 Hz, 0.75 s puts the live cursor at 22.5 -> ceil 23, leaving
        # only seven old frames. RTC must accept the new chunk unchanged.
        result, state = env._apply_rtc_overlap(
            candidate, new_start_frame=1, now_ns=int(0.75 * SECOND),
            reset_transition=False,
        )
        self.assertIs(result, candidate)
        self.assertEqual(state["old_start_frame"], 23)
        self.assertEqual(state["bypass_reason"], "insufficient_old_future")


if __name__ == "__main__":
    unittest.main()
