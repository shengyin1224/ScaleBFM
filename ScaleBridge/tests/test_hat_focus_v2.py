"""Offline A/B regressions for frame-zero entry and absolute completion."""

import unittest

import numpy as np

from scalebridge.online.focus import derive_hand_focus_events
from scalebridge.online.focus_handoff import HandEventHandoff


def prediction(fps=20., actual=0., initial=.2, final=.3):
    history = np.full((10, 6), actual, dtype=np.float32)
    future = np.full((40, 6), actual, dtype=np.float32)
    horizon = int(round(.3 * fps))
    for col in (0, 5):  # index and thumb bend; rotation is column 4
        future[:horizon + 1, col] = np.linspace(initial, final, horizon + 1)
        future[horizon + 1:, col] = final
    return history, future


def chunk(sequence, distance=.005, fps=20.):
    count = 40
    wrist = np.zeros((count, 3), dtype=np.float32)
    wrist[:, 0] = np.linspace(0., distance, int(round(.3 * fps)) + 1)[-1]
    wrist[:int(round(.3 * fps)) + 1, 0] = np.linspace(
        0., distance, int(round(.3 * fps)) + 1)
    return dict(sequence_id=sequence, frame_count=count, fps=fps,
                right_wrist_pos=wrist,
                right_wrist_quat_wxyz=np.tile([1., 0., 0., 0.], (count, 1)))


class FocusV2Tests(unittest.TestCase):
    def events(self, fps=20., **kwargs):
        history, future = prediction(fps=fps, **kwargs)
        return derive_hand_focus_events(history, future, fps, entry_mode="v2")

    def test_frame_zero_intent_and_progress_at_both_frame_rates(self):
        for fps in (20., 30.):
            with self.subTest(fps=fps):
                result = self.events(fps=fps, actual=.09, initial=.13, final=.22)
                self.assertEqual(result["events"][0]["kind"], "close")
                self.assertEqual(result["events"][0]["start"], 0)

    def test_future_closing_edge_is_legacy_only(self):
        history = np.zeros((10, 6), dtype=np.float32)
        future = np.zeros((40, 6), dtype=np.float32)
        future[10:, [0, 5]] = .8
        self.assertTrue(derive_hand_focus_events(history, future, 20.)["events"])
        self.assertFalse(derive_hand_focus_events(
            history, future, 20., entry_mode="v2")["events"])

    def test_no_entry_without_further_closing_or_measured_intent(self):
        self.assertFalse(self.events(initial=.2, final=.2)["events"])
        self.assertFalse(self.events(actual=.19, initial=.2, final=.3)["events"])
        self.assertFalse(self.events(initial=.08, final=.3)["events"])

    def test_rotation_cannot_supply_second_closing_finger(self):
        history, future = prediction()
        future[:, 5] = 0.
        future[:, 4] = np.linspace(.2, .8, len(future))
        self.assertFalse(derive_hand_focus_events(
            history, future, 20., entry_mode="v2")["events"])

    def test_two_adjacent_qualifying_chunks_and_steady_wrist_required(self):
        c = HandEventHandoff(side="right", handoff_version="v2", consensus_chunks=1)
        data = self.events()
        c.observe_event(chunk(1), data)
        self.assertIsNone(c.entry)
        c.observe_event(chunk(3), data)  # missing chunk 2 breaks consensus
        self.assertIsNone(c.entry)
        c.observe_event(chunk(4, distance=.013), data)
        self.assertIsNone(c.entry)
        c.observe_event(chunk(5), data)
        self.assertIsNone(c.entry)
        c.observe_event(chunk(6), data)
        self.assertIsNotNone(c.entry)
        sample = {"right_wrist_pos": np.zeros((1, 3)),
                  "right_wrist_quat_wxyz": np.array([[1., 0., 0., 0.]])}
        measured = {"right_wrist_yaw_link": (np.array([.04, 0., 0.]),
                                             np.array([1., 0., 0., 0.]))}
        self.assertFalse(c.arrive(measured, 20, sample, np.zeros(5)))
        measured["right_wrist_yaw_link"] = (np.zeros(3), np.array([1., 0., 0., 0.]))
        self.assertTrue(c.arrive(measured, 20, sample, np.zeros(5)))
        self.assertFalse(c.active)
        self.assertFalse(c.accepts(dict(source_state_sequence_id=19)))
        self.assertTrue(c.accepts(dict(source_state_sequence_id=20)))
        c.start(1_000_000_000)
        self.assertTrue(c.active)

    def active_controller(self, version, kind="close"):
        c = HandEventHandoff(side="right", handoff_version=version)
        c.event = dict(kind=kind, goal=np.full(5, .3 if kind == "close" else .6))
        c.baseline = np.full(5, 0. if kind == "close" else 1.)
        c.state = "await_chunk"
        c.start(1_000_000_000)
        return c

    def test_goal_proximity_finishes_legacy_but_not_v2(self):
        for kind, actual in (("close", .2), ("open", .7)):
            for version in ("legacy", "v2"):
                with self.subTest(version=version, kind=kind):
                    c = self.active_controller(version, kind)
                    feedback = np.full(5, actual)
                    self.assertFalse(c.update_actual(1_100_000_000, feedback))
                    self.assertEqual(c.update_actual(2_100_000_000, feedback),
                                     version == "legacy")

    def test_v2_half_closed_requires_two_fingers_progress_and_full_hold(self):
        c = self.active_controller("v2")
        feedback = np.zeros(5); feedback[0] = .5
        self.assertFalse(c.update_actual(1_100_000_000, feedback))
        feedback[4] = .5
        self.assertFalse(c.update_actual(1_200_000_000, feedback))
        self.assertFalse(c.update_actual(2_199_000_000, feedback))
        self.assertTrue(c.update_actual(2_200_000_000, feedback))
        self.assertEqual(c.end_reason, "feedback_complete")
        c = self.active_controller("v2")
        c.baseline[:] = .5
        self.assertFalse(c.update_actual(1_200_000_000, np.full(5, .5)))
        self.assertFalse(c.update_actual(2_200_000_000, np.full(5, .5)))

    def test_v2_timeout_is_an_abort_and_release_uses_absolute_threshold(self):
        c = self.active_controller("v2")
        self.assertTrue(c.update_actual(6_000_000_000, np.full(5, .2)))
        self.assertEqual(c.end_reason, "timeout")
        c = self.active_controller("v2", "open")
        self.assertFalse(c.update_actual(1_100_000_000, np.full(5, .4)))
        self.assertTrue(c.update_actual(2_100_000_000, np.full(5, .4)))
        self.assertEqual(c.end_reason, "feedback_complete")

    def test_invalid_version_is_rejected(self):
        with self.assertRaises(ValueError):
            HandEventHandoff(side="right", handoff_version="typo")
        history, future = prediction()
        with self.assertRaises(ValueError):
            derive_hand_focus_events(history, future, 20., entry_mode="typo")


if __name__ == "__main__":
    unittest.main()
