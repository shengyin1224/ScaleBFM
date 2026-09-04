import sys
import unittest

import numpy as np

sys.path.insert(0, "/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM")

from focus_definition import focus_from_hand_state, true_segments
from scalebridge.online.focus import (
    derive_chunk_focus_phase,
    focus_phase_from_mask,
)


class HATOnlineFocusTest(unittest.TestCase):
    def test_canonical_focus_marks_only_grasp_endpoints(self):
        hand = np.zeros((150, 2), dtype=np.float32)
        hand[40:80, 0] = 1.0
        focus, grasp = focus_from_hand_state(hand, 30.0)
        self.assertEqual(true_segments(grasp), [(40, 79)])
        # +/-30 frame endpoint windows overlap into one FOCUS interval.
        self.assertEqual(true_segments(focus), [(10, 109)])

    def test_phase_is_signed_time_to_next_focus_edge(self):
        phase = focus_phase_from_mask(
            np.array([False, False, True, True, False]), fps=1.0
        )
        np.testing.assert_allclose(
            phase,
            [[0, 2], [0, 1], [1, -2], [1, -1], [0, 1]],
        )

    def test_history_preserves_recent_event_without_fake_chunk_end_event(self):
        history = np.zeros((30, 1), dtype=np.float32)
        history[15:] = 1.0
        predicted = np.ones((64, 1), dtype=np.float32)
        phase = derive_chunk_focus_phase(
            history, predicted, 30.0, focus_from_hand_state
        )
        # The enter event 15 frames in the past remains focused for 15 future
        # frames. A continuing grasp at chunk end must not create a fake release.
        np.testing.assert_allclose(phase[:15, 0], 1.0)
        np.testing.assert_allclose(phase[16:, 0], 0.0)
        self.assertEqual(float(phase[-1, 1]), 2.0)

    def test_no_hand_motion_stays_neutral(self):
        phase = derive_chunk_focus_phase(
            np.zeros((30, 12), dtype=np.float32),
            np.zeros((64, 12), dtype=np.float32),
            30.0,
            focus_from_hand_state,
        )
        np.testing.assert_allclose(phase[:, 0], 0.0)
        np.testing.assert_allclose(phase[:, 1], 2.0)


if __name__ == "__main__":
    unittest.main()
