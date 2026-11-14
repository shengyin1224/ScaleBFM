#!/usr/bin/env python3
"""Small regression suite for the embodiment-aware FOCUS definition."""

from __future__ import annotations

import unittest

import numpy as np

from focus_definition import focus_from_hand_state, focus_window, hand_signal


class FocusDefinitionTest(unittest.TestCase):
    def test_brainco_thumb_tilt_drift_is_not_closure(self) -> None:
        hand = np.zeros((120, 12), dtype=np.float32)
        hand[:, 1] = np.linspace(0.72, 0.80, len(hand))
        hand[:, 7] = np.linspace(0.80, 0.70, len(hand))
        signal, active = hand_signal(hand, "brainco")
        self.assertEqual(active.tolist(), [])
        self.assertFalse(signal.any())

    def test_brainco_physical_closure_event(self) -> None:
        hand = np.zeros((120, 12), dtype=np.float32)
        hand[:, [1, 7]] = 0.8
        hand[40:81, [6, 8, 9, 10, 11]] = 0.8
        focus, grasp, signal, active = focus_from_hand_state(hand, 30.0, "brainco")
        self.assertEqual((np.flatnonzero(grasp)[0], np.flatnonzero(grasp)[-1]), (40, 80))
        self.assertEqual(active.tolist(), [6, 8, 9, 10, 11])
        self.assertAlmostEqual(float(signal.max()), 1.0)
        self.assertTrue(focus[10:111].all())
        self.assertFalse(focus[:10].any())
        self.assertFalse(focus[111:].any())

    def test_inspire_reversed_active_hand_ignores_static_closed_hand(self) -> None:
        hand = np.ones((120, 12), dtype=np.float32)
        hand[:, [5, 11]] = 0.0
        hand[:, :5] = 0.2  # unused hand parked closed
        hand[40:81, [6, 7, 8, 9, 10]] = 0.2
        focus, grasp, _, active = focus_from_hand_state(hand, 30.0, "inspire")
        self.assertEqual((np.flatnonzero(grasp)[0], np.flatnonzero(grasp)[-1]), (40, 80))
        self.assertEqual(active.tolist(), [6, 7, 8, 9, 10])
        self.assertTrue(focus[10:111].all())

    def test_one_hand_cannot_hide_other_hand_endpoints(self) -> None:
        hand = np.zeros((300, 12), dtype=np.float32)
        hand[:, [1, 7]] = 0.8
        hand[40:201, [0, 2, 3, 4, 5]] = 0.8
        hand[100:261, [6, 8, 9, 10, 11]] = 0.8
        focus, grasp, _, _ = focus_from_hand_state(hand, 30.0, "brainco")
        self.assertEqual((np.flatnonzero(grasp)[0], np.flatnonzero(grasp)[-1]), (40, 260))
        # These are away from the union-grasp endpoints but covered by the
        # right-hand enter and left-hand release endpoint windows.
        self.assertTrue(focus[125])
        self.assertTrue(focus[175])

    def test_episode_boundaries_are_censored(self) -> None:
        grasp = np.zeros(200, dtype=bool)
        grasp[:51] = True
        grasp[100:] = True
        focus = focus_window(grasp, 30.0)
        # First segment contributes release only; second contributes enter only.
        self.assertFalse(focus[:20].any())
        self.assertTrue(focus[20:81].all())
        self.assertTrue(focus[70:131].all())
        self.assertFalse(focus[131:].any())


if __name__ == "__main__":
    unittest.main()
