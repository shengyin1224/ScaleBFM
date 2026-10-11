import unittest
from unittest.mock import patch
import numpy as np
from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv


class LatestFingerClockTests(unittest.TestCase):
    def test_body_phase_jump_does_not_skip_finger_approach(self):
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.finger_chunk_source = 'latest'
        env.reference_playing = True
        env.localization_paused = False
        env.held_sample = None
        env.chunk_started_ns = -1_000_000_000  # Body search jumped forward one second.
        tips = np.zeros((100, 5, 3))
        tips[:, 1, 0] = np.linspace(.15, 0, 100)
        raw = {'fps': 30., 'frame_count': 100,
               'left_fingertip_local': tips, 'right_fingertip_local': tips}
        env.last_reference_sample = {
            'left_fingertip_local': np.zeros((1, 5, 3)),
            'right_fingertip_local': np.zeros((1, 5, 3))}
        env._set_latest_finger_chunk(raw, 0, 100_000_000)
        with patch('scalebridge.env.motion_tracking_hat4_online.monotonic_ns', return_value=1_000_000_000):
            _, right = env._finger_reference_targets()
        np.testing.assert_allclose(right, tips[30])
        self.assertIs(env._finger_chunk, raw)
        self.assertEqual(env._finger_chunk_started_ns, 0)
        self.assertGreater(np.linalg.norm(right[0] - right[1]),
                           np.linalg.norm(tips[60, 0] - tips[60, 1]))

    def test_missing_source_timestamp_keeps_unshifted_latency_clock(self):
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env._set_latest_finger_chunk({}, None, 1_000_000_000, source_age_s=.1)
        self.assertEqual(env._finger_chunk_started_ns, 900_000_000)


if __name__ == '__main__':
    unittest.main()
