#!/usr/bin/env python3
"""Regression tests for v3 onset and unchanged v2 endpoint semantics."""
import unittest
import numpy as np
from focus_definition import (
    onset_focus_window, hysteresis, focus_window,
    focus_from_hand_state, focus_from_hand_state_v3, time_to_next_focus_edge,
)


class FocusV3Tests(unittest.TestCase):
    def test_onset_uses_point_one_instead_of_one_second_before_point_five(self):
        signal = np.zeros(500, dtype=np.float32)
        signal[100:121] = np.linspace(0.1, 0.5, 21)
        signal[121:301] = 0.8
        grasp = hysteresis(signal)
        focus = onset_focus_window(signal, grasp, 50)
        # 0.07s becomes 3 source frames at 50 Hz, so it never exceeds the cap.
        self.assertFalse(focus[:97].any())
        self.assertTrue(focus[97:171].all())  # old entry 120 + 50 frames, inclusive
        self.assertFalse(focus[171:250].any())
        self.assertTrue(focus[250:351].all())  # unchanged release +/-1s
        self.assertFalse(focus[351:].any())

    def test_aborted_close_is_not_reused_by_later_grasp(self):
        signal = np.zeros(400, dtype=np.float32)
        signal[30:51] = 0.2
        signal[100:111] = 0.1
        signal[111:301] = 0.8
        focus = onset_focus_window(signal, hysteresis(signal), 50)
        self.assertFalse(focus[:97].any())
        self.assertTrue(focus[97])

    def test_no_advance_is_configurable_and_long_advance_is_rejected(self):
        signal = np.zeros(400, dtype=np.float32)
        signal[100:301] = 0.8
        grasp = hysteresis(signal)
        focus = onset_focus_window(signal, grasp, 50, lead_s=0)
        self.assertFalse(focus[:100].any())
        self.assertTrue(focus[100])
        with self.assertRaises(ValueError):
            onset_focus_window(signal, grasp, 50, lead_s=0.11)

    def test_episode_cuts_do_not_create_events(self):
        signal = np.ones(300, dtype=np.float32)
        grasp = hysteresis(signal)
        self.assertFalse(onset_focus_window(signal, grasp, 30).any())
        signal[:]=0
        signal[:51]=0.8
        focus = onset_focus_window(signal, hysteresis(signal), 30)
        np.testing.assert_array_equal(focus, focus_window(hysteresis(signal), 30))

    def test_hand_polarity_and_grasp_detector_stay_unchanged(self):
        for kind in ("brainco", "inspire"):
            hand = np.zeros((400,12), dtype=np.float32) if kind=="brainco" else np.full((400,12),0.9,dtype=np.float32)
            hand[100:111,2] = 0.08 if kind=="brainco" else 0.83
            hand[111:301,2] = 0.8 if kind=="brainco" else 0.2
            old=focus_from_hand_state(hand,50,kind)
            new=focus_from_hand_state_v3(hand,50,kind)
            for i in (1,2,3):
                np.testing.assert_array_equal(new[i],old[i])
            self.assertFalse(new[0][:97].any())
            self.assertTrue(new[0][97])
            self.assertEqual(np.flatnonzero(new[0])[-1],np.flatnonzero(old[0])[-1])

    def test_hands_independent_and_static_closed_hand_ignored(self):
        hand=np.zeros((400,12),dtype=np.float32)
        hand[:,6]=0.8  # inactive right hand
        hand[100:111,2]=0.08
        hand[111:301,2]=0.8
        focus,_,_,active=focus_from_hand_state_v3(hand,50,"brainco")
        self.assertEqual(active.tolist(),[2])
        self.assertFalse(focus[:97].any())

    def test_next_edge_matches_new_mask(self):
        mask=np.zeros(200,dtype=bool); mask[97:150]=True
        edge=time_to_next_focus_edge(mask,50)
        self.assertAlmostEqual(float(edge[96]),0.02)
        self.assertAlmostEqual(float(edge[97]),-1.06,places=6)
        self.assertAlmostEqual(float(edge[149]),-0.02)


if __name__=="__main__":
    unittest.main()
