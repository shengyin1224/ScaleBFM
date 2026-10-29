"""Offline regressions for the v3 close-entry gate on predicted grasp travel."""

import unittest

import numpy as np

from scalebridge.online.focus import derive_hand_focus_events
from scalebridge.online.focus_handoff import (
    HandEventHandoff, closure_approach_distance,
)

COUNT = 40
OPEN_SPREAD = .14
CLOSED_SPREAD = .05


def tips(count=COUNT, grasp_frame=10, open_spread=OPEN_SPREAD,
         closed_spread=CLOSED_SPREAD):
    """Five fingertips whose spread collapses to a grasp at ``grasp_frame``."""
    spread = np.full(count, open_spread, dtype=np.float32)
    if grasp_frame is not None:
        spread[grasp_frame:] = closed_spread
    out = np.zeros((count, 5, 3), dtype=np.float32)
    # Thumb at the origin, four fingers one spread away along +x.
    out[:, 1:, 0] = spread[:, None]
    return out


def chunk(sequence=1, grasp_frame=10, travel=.0, fps=20., count=COUNT,
          **tip_kwargs):
    """A chunk whose wrist advances ``travel`` metres by its grasp frame."""
    wrist = np.zeros((count, 3), dtype=np.float32)
    stop = count - 1 if grasp_frame is None else grasp_frame
    if stop:
        wrist[:stop + 1, 0] = np.linspace(0., travel, stop + 1)
    wrist[stop:, 0] = travel
    return dict(sequence_id=sequence, frame_count=count, fps=fps,
                right_wrist_pos=wrist,
                right_wrist_quat_wxyz=np.tile([1., 0., 0., 0.], (count, 1)),
                right_fingertip_local=tips(count=count, grasp_frame=grasp_frame,
                                           **tip_kwargs))


def closing_hand(fps=20., count=COUNT, start=0):
    """A legacy-detectable close event beginning at ``start``."""
    history = np.zeros((10, 6), dtype=np.float32)
    future = np.zeros((count, 6), dtype=np.float32)
    for col in (0, 5):  # index and thumb bend; column 4 is thumb rotation
        future[start:, col] = .8
    return derive_hand_focus_events(history, future, fps, entry_mode="v3")


class ClosureApproachDistanceTests(unittest.TestCase):
    def test_measures_wrist_travel_up_to_the_grasp_frame(self):
        value = closure_approach_distance(
            chunk(grasp_frame=10, travel=.06), "right", spread_m=.11)
        self.assertAlmostEqual(value, .06, places=6)

    def test_is_zero_when_the_chunk_already_starts_closed(self):
        value = closure_approach_distance(
            chunk(grasp_frame=0, travel=.06), "right", spread_m=.11)
        self.assertAlmostEqual(value, .0, places=6)

    def test_is_none_when_the_chunk_never_closes(self):
        self.assertIsNone(closure_approach_distance(
            chunk(grasp_frame=None), "right", spread_m=.11))

    def test_is_invariant_to_rigid_alignment_of_the_chunk(self):
        raw = chunk(grasp_frame=12, travel=.045)
        angle = .7
        rot = np.array([[np.cos(angle), -np.sin(angle), 0.],
                        [np.sin(angle), np.cos(angle), 0.],
                        [0., 0., 1.]])
        aligned = dict(raw)
        aligned["right_wrist_pos"] = (
            raw["right_wrist_pos"] @ rot.T + np.array([1.3, -.4, .9]))
        self.assertAlmostEqual(
            closure_approach_distance(raw, "right", spread_m=.11),
            closure_approach_distance(aligned, "right", spread_m=.11), places=6)

    def test_rejects_malformed_fingertip_or_wrist_arrays(self):
        bad = chunk()
        bad["right_fingertip_local"] = bad["right_fingertip_local"][:, :4]
        self.assertIsNone(closure_approach_distance(bad, "right", spread_m=.11))
        nan = chunk()
        nan["right_wrist_pos"][3, 0] = np.nan
        self.assertIsNone(closure_approach_distance(nan, "right", spread_m=.11))


class V3EntryGateTests(unittest.TestCase):
    def handoff(self, version="v3", **kwargs):
        kwargs.setdefault("consensus_chunks", 1)
        return HandEventHandoff(side="right", handoff_version=version, **kwargs)

    def test_close_arms_when_the_grasp_frame_is_already_here(self):
        controller = self.handoff()
        controller.observe_event(chunk(grasp_frame=8, travel=.01), closing_hand())
        self.assertIsNotNone(controller.event)
        self.assertEqual(controller.event["kind"], "close")

    def test_close_is_rejected_when_the_grasp_frame_is_still_ahead(self):
        controller = self.handoff()
        controller.observe_event(chunk(grasp_frame=30, travel=.06), closing_hand())
        self.assertIsNone(controller.entry)
        self.assertAlmostEqual(controller.v3_last_approach, .06, places=6)

    def test_close_is_rejected_when_no_grasp_frame_exists(self):
        controller = self.handoff()
        controller.observe_event(chunk(grasp_frame=None), closing_hand())
        self.assertIsNone(controller.entry)
        self.assertIsNone(controller.v3_last_approach)

    def test_legacy_still_arms_on_the_same_distant_grasp(self):
        controller = self.handoff(version="legacy")
        controller.observe_event(chunk(grasp_frame=30, travel=.06), closing_hand())
        self.assertIsNotNone(controller.event)
        self.assertEqual(controller.event["kind"], "close")

    def test_threshold_boundary_follows_configuration(self):
        for travel, expected in ((.019, True), (.021, False)):
            with self.subTest(travel=travel):
                controller = self.handoff(v3_max_approach_m=.02)
                controller.observe_event(
                    chunk(grasp_frame=10, travel=travel), closing_hand())
                self.assertEqual(controller.event is not None, expected)

    def test_release_events_are_never_gated(self):
        history = np.full((10, 6), .9, dtype=np.float32)
        future = np.zeros((COUNT, 6), dtype=np.float32)
        hand = derive_hand_focus_events(history, future, 20., entry_mode="v3")
        self.assertEqual(hand["events"][0]["kind"], "open")
        controller = self.handoff()
        # A release must pass even though this chunk never reaches a grasp
        # spread and travels far; clamping it shut would strand the bottle.
        controller.observe_event(chunk(grasp_frame=None, travel=.2), hand)
        self.assertIsNotNone(controller.event)
        self.assertEqual(controller.event["kind"], "open")

    def test_v3_keeps_the_legacy_goal_proximity_exit(self):
        legacy, v3 = self.handoff(version="legacy"), self.handoff()
        for controller in (legacy, v3):
            controller.event = dict(kind="close", start=0, end=1,
                                    goal=np.full(5, .6, dtype=np.float32))
            controller.state = "focus"
            controller.started_ns = 0
            controller.baseline = np.zeros(5, dtype=np.float32)
        # Below the 0.5 absolute bar, but within 0.12 of the predicted goal:
        # only the goal-proximity branch can complete this.
        actual = np.full(5, .49, dtype=np.float32)
        for controller in (legacy, v3):
            controller.update_actual(0, actual)
            controller.update_actual(int(2e9), actual)
        self.assertEqual(legacy.end_reason, "feedback_complete")
        self.assertEqual(v3.end_reason, legacy.end_reason)

    def test_invalid_gate_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.handoff(v3_closed_spread_m=0.)
        with self.assertRaises(ValueError):
            self.handoff(v3_max_approach_m=-.01)


if __name__ == "__main__":
    unittest.main()
