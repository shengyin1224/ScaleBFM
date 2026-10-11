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
        # Closing starts immediately; release retains its +/-30 frame window.
        self.assertEqual(true_segments(focus), [(40, 109)])

    def test_phase_is_signed_time_to_next_focus_edge(self):
        phase = focus_phase_from_mask(
            np.array([False, False, True, True, False]), fps=1.0
        )
        np.testing.assert_allclose(
            phase,
            [[0, 2], [0, 1], [1, -2], [1, -1], [0, 1]],
        )

    def test_closing_intent_advances_focus_without_changing_grasp_or_window_end(self):
        hand = np.zeros((300, 1), dtype=np.float32)
        hand[40:70, 0] = np.linspace(0.1, 0.49, 30)
        hand[70:200, 0] = 1.0
        focus, grasp = focus_from_hand_state(hand, 30.0)
        self.assertEqual(true_segments(grasp), [(70, 199)])
        self.assertEqual(true_segments(focus), [(40, 100), (169, 229)])
        legacy_focus, legacy_grasp = focus_from_hand_state(
            hand, 30.0, focus_enter=0.5, focus_pre_pad_s=1.0
        )
        np.testing.assert_array_equal(grasp, legacy_grasp)
        self.assertEqual(true_segments(legacy_focus), [(40, 100), (169, 229)])

    def test_aborted_closing_attempt_is_focused_without_becoming_a_grasp(self):
        hand = np.zeros((250, 1), dtype=np.float32)
        hand[30:35, 0] = 0.2
        hand[100:200, 0] = 1.0
        focus, grasp = focus_from_hand_state(hand, 10.0)
        self.assertEqual(true_segments(grasp), [(100, 199)])
        self.assertEqual(true_segments(focus), [(30, 44), (100, 110), (189, 209)])

    def test_intent_can_start_without_pre_padding(self):
        hand = np.zeros((300, 1), dtype=np.float32)
        hand[40:70, 0] = np.linspace(0.1, 0.49, 30)
        hand[70:200, 0] = 1.0
        focus, _ = focus_from_hand_state(hand, 30.0, focus_pre_pad_s=0.0)
        self.assertEqual(true_segments(focus), [(40, 100), (169, 229)])

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

    def test_online_focus_includes_predicted_closing_intent_before_grasp(self):
        history = np.zeros((30, 1), dtype=np.float32)
        predicted = np.zeros((150, 1), dtype=np.float32)
        predicted[40:70, 0] = np.linspace(0.1, 0.49, 30)
        predicted[70:130, 0] = 1.0
        phase = derive_chunk_focus_phase(
            history, predicted, 30.0, focus_from_hand_state
        )
        self.assertEqual(true_segments(phase[:, 0] > 0), [(40, 149)])
        self.assertEqual(float(phase[39, 0]), 0.0)
        self.assertEqual(float(phase[40, 0]), 1.0)

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
