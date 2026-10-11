import unittest
from unittest.mock import Mock

import numpy as np
import torch

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.protocol import make_hat_chunk


def prediction(sequence, source):
    n = 100
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 2] = 0.8
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1
    return make_hat_chunk(
        sequence_id=sequence, source_state_sequence_id=source,
        generated_at_ns=1_100_000_000, fps=30, frame_count=n,
        action_representation="relative_chunk",
        root_pos=pos, root_quat_wxyz=quat,
        head_pos=pos, head_quat_wxyz=quat,
        left_wrist_pos=pos, left_wrist_quat_wxyz=quat,
        right_wrist_pos=pos, right_wrist_quat_wxyz=quat,
        left_fingertip_local=np.zeros((n, 5, 3)),
        right_fingertip_local=np.zeros((n, 5, 3)),
    )


def environment():
    env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
    env.cfg = {"rtc_enabled": False}
    env.chunk = prediction(10, 40)
    env.reference_playing = False
    env.reference_plan_epoch = 1
    env._reference_start_pending = False
    env.prestart_last_target = torch.zeros(29)
    env.reference_play_control = "R1"
    env.local_to_global_transition_steps = 4
    env.reference_forcing = False
    env.localization_paused = False
    env.requires_restart = False
    env.held_sample = None
    env._latched_sample = None
    env._latched_link_poses = None
    env.state_sequence_id = 43
    env.last_chunk_sequence_id = 10
    env.finger_chunk_source = "playback"
    env.chunk_alignment = "fixed"
    env.alignment_anchor = None
    env.chunk_subscriber = Mock()
    env.chunk_subscriber.receive_error.return_value = None
    env.simulator = Mock(spec=[])
    env.state_buffer = {
        "root_pos_buffer": torch.tensor([[[0., 0., .8]]]),
        "root_quat_wxyz_buffer": torch.tensor([[[1., 0., 0., 0.]]]),
    }
    env._phase_match_start_frame = Mock(return_value=(3, {
        "mode": "measured_pose", "source_age_s": .1, "pose_cost": 0,
        "transition_valid": True,
    }))
    env._source_state_timestamp_ns = Mock(return_value=1_000_000_000)
    env._submit_chunk_focus = Mock()
    return env


class HuMIStartupTest(unittest.TestCase):
    def test_explicit_heading_uses_origin_without_root_pivot_or_yaw_reanchoring(self):
        env = environment()
        candidate = prediction(11, 43)
        origin = np.array([.3, -.2, 1.2], dtype=np.float32)
        heading = np.array([np.cos(.4), 0, 0, np.sin(.4)], dtype=np.float32)
        candidate["position_origin_world"] = origin
        candidate["frame_heading_world_wxyz"] = heading
        candidate["root_pos"] = np.tile([.2, -.1, .8], (100, 1))
        env.reference_playing = True
        # Neither the moving measured heading nor a previous legacy anchor
        # may override the frame in which the producer encoded these poses.
        env.protocol_alignment_anchor = (np.zeros(3), np.array([1, 0, 0, 0]))
        pos, quat = env._protocol_alignment(candidate, [2, 3, .8], [1, 0, 0, 0])
        np.testing.assert_array_equal(pos, origin)
        np.testing.assert_array_equal(quat, heading)

    def test_r1_holds_until_a_prediction_from_post_r1_state_arrives(self):
        env = environment()
        self.assertFalse(env._reference_plan_active())
        env._request_reference_playback(torch.ones(29), 1_000_000_000)
        self.assertFalse(env.reference_playing)
        self.assertTrue(env._reference_start_pending)
        self.assertTrue(env._reference_plan_active())
        self.assertEqual(env.reference_plan_epoch, 2)
        self.assertEqual(env.minimum_source_state_sequence_id, 43)

        env.chunk_subscriber.receive_latest.return_value = prediction(11, 42)
        self.assertFalse(env._accept_latest_chunk(1_100_000_000))
        self.assertFalse(env.reference_playing)
        self.assertTrue(env._reference_start_pending)
        env._phase_match_start_frame.assert_not_called()

        env.chunk_subscriber.receive_latest.return_value = prediction(12, 43)
        self.assertTrue(env._accept_latest_chunk(1_200_000_000))
        self.assertTrue(env.reference_playing)
        self.assertFalse(env._reference_start_pending)
        self.assertFalse(env.awaiting_fresh_chunk)
        self.assertIsNone(env.previous_chunk)
        self.assertEqual(env.takeover_started_ns, 1_200_000_000)

    def test_original_checkpoint_starts_immediately_as_before(self):
        env = environment()
        env.chunk["action_representation"] = "original"
        env._request_reference_playback(torch.ones(29), 1_000_000_000)
        self.assertTrue(env.reference_playing)
        self.assertFalse(env._reference_start_pending)
        self.assertEqual(env.takeover_started_ns, 1_000_000_000)

    def test_paused_or_latched_reference_does_not_advance_producer_plan(self):
        env = environment()
        env.reference_playing = True
        self.assertTrue(env._reference_plan_active())
        env.localization_paused = True
        self.assertFalse(env._reference_plan_active())
        env.localization_paused = False
        env.held_sample = object()
        self.assertFalse(env._reference_plan_active())
        env.held_sample = None
        env.requires_restart = True
        self.assertFalse(env._reference_plan_active())


if __name__ == "__main__":
    unittest.main()
