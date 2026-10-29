"""Offline regressions for the commanded-close FOCUS probe."""

import unittest

import numpy as np

from scalebridge.online.focus_handoff import (
    GraspFingerChunkControl, HandEventHandoff,
)

SEC = 1_000_000_000


def motor(value=1.0, thumb_rot=0.5):
    """Six DDS motors, 0 closed and 1 open.

    DDS order is pinky, ring, middle, index, thumb bend, thumb rotation, so
    the five bending motors are [:5] and rotation is the last one.
    """
    out = np.full(6, value, dtype=np.float64)
    out[5] = thumb_rot
    return out


class ProbeFingerCommandTests(unittest.TestCase):
    def control(self, **kwargs):
        kwargs.setdefault("probe_close_s", 0.5)
        return GraspFingerChunkControl("probe", **kwargs)

    def ramp(self, control, elapsed_s, previous=None, proposed=None, event_id=0):
        return control.apply(
            "right", event_id, 7,
            motor(1.0) if proposed is None else proposed,
            motor(1.0) if previous is None else previous,
            now_ns=int(elapsed_s * SEC))

    def test_ramps_the_bending_motors_shut_over_the_window(self):
        control = self.control()
        start = self.ramp(control, 0.0)
        half = self.ramp(control, 0.25)
        done = self.ramp(control, 0.5)
        np.testing.assert_allclose(start[:5], 1.0)
        np.testing.assert_allclose(half[:5], 0.5, atol=1e-9)
        np.testing.assert_allclose(done[:5], 0.0, atol=1e-9)

    def test_stays_shut_after_the_window(self):
        control = self.control()
        self.ramp(control, 0.0)
        np.testing.assert_allclose(self.ramp(control, 3.0)[:5], 0.0, atol=1e-9)

    def test_ignores_a_prediction_that_reopens(self):
        """The 6 labelled 'hand in place, fingers never moved' failures."""
        control = self.control()
        self.ramp(control, 0.0)
        # Every replan proposes a wide-open hand; the probe must not follow it.
        reopened = self.ramp(control, 0.4, proposed=motor(1.0),
                             previous=motor(0.3))
        self.assertLess(reopened[:5].max(), 0.3)

    def test_thumb_rotation_is_left_alone(self):
        control = self.control()
        out = self.ramp(control, 0.5, proposed=motor(1.0, thumb_rot=0.42))
        self.assertAlmostEqual(out[5], 0.42)

    def test_a_new_grasp_restarts_the_ramp_from_its_own_pose(self):
        control = self.control()
        self.ramp(control, 0.0)
        self.ramp(control, 0.5)
        second = self.ramp(control, 0.0, previous=motor(0.8), event_id=9 * SEC)
        np.testing.assert_allclose(second[:5], 0.8, atol=1e-9)

    def test_other_modes_and_validation_are_unchanged(self):
        proposed, previous = motor(0.9), motor(0.4)
        off = GraspFingerChunkControl("off")
        np.testing.assert_allclose(
            off.apply("right", 0, 1, proposed, previous), proposed)
        hold = GraspFingerChunkControl("hold_progress")
        np.testing.assert_allclose(
            hold.apply("right", 0, 1, proposed, previous)[:5], previous[:5])
        with self.assertRaises(ValueError):
            GraspFingerChunkControl("probe", probe_close_s=0.)
        with self.assertRaises(ValueError):
            GraspFingerChunkControl("nonsense")


class ProbeClassificationTests(unittest.TestCase):
    def controller(self, **kwargs):
        kwargs.setdefault("probe_s", 0.6)
        controller = HandEventHandoff(side="right", consensus_chunks=1, **kwargs)
        controller.event = dict(kind="close", start=0, end=1,
                                goal=np.ones(5, dtype=np.float32))
        controller.state = "focus"
        controller.started_ns = 0
        controller.baseline = np.zeros(5, dtype=np.float32)
        return controller

    def closure(self, value):
        return np.full(5, value, dtype=np.float32)

    def test_withholds_any_decision_until_the_window_elapses(self):
        controller = self.controller()
        # Two motors already past the ordinary 0.5 bar: without the probe this
        # would complete. The probe must keep driving instead.
        self.assertFalse(controller.update_actual(int(0.3 * SEC), self.closure(.7)))
        self.assertIsNone(controller.end_reason)
        self.assertTrue(controller.active)

    def test_stalled_fingers_are_a_confirmed_grasp(self):
        controller = self.controller()
        self.assertTrue(controller.update_actual(int(0.6 * SEC), self.closure(.52)))
        self.assertEqual(controller.end_reason, "grasp_confirmed")
        self.assertAlmostEqual(controller.probe_closure, .52, places=5)
        self.assertFalse(controller.reopen_request)
        # A confirmed grasp still blocks another close until a release.
        self.assertEqual(controller.last_kind, "close")

    def test_fully_closed_fingers_are_an_empty_grasp(self):
        controller = self.controller()
        self.assertTrue(controller.update_actual(int(0.6 * SEC), self.closure(.97)))
        self.assertEqual(controller.end_reason, "closed_empty")
        self.assertTrue(controller.reopen_request)
        # And it must not block the retry.
        self.assertIsNone(controller.last_kind)

    def test_empty_grasp_can_be_left_clamped_when_release_is_disabled(self):
        controller = self.controller(probe_release_on_empty=False)
        controller.update_actual(int(0.6 * SEC), self.closure(.97))
        self.assertEqual(controller.end_reason, "closed_empty")
        self.assertFalse(controller.reopen_request)

    def test_timeout_still_wins_over_the_probe(self):
        controller = self.controller(timeout_s=0.4, completion_hold_s=0.1,
                                     probe_s=0.3)
        self.assertTrue(controller.update_actual(int(0.5 * SEC), self.closure(.97)))
        self.assertEqual(controller.end_reason, "timeout")
        self.assertIsNone(controller.probe_closure)

    def test_release_windows_keep_their_own_feedback_exit(self):
        controller = self.controller()
        controller.event = dict(kind="open", start=0, end=1,
                                goal=np.zeros(5, dtype=np.float32))
        controller.baseline = np.ones(5, dtype=np.float32)
        opened = self.closure(.1)
        controller.update_actual(0, opened)
        self.assertTrue(controller.update_actual(int(2 * SEC), opened))
        self.assertEqual(controller.end_reason, "feedback_complete")
        self.assertIsNone(controller.probe_closure)

    def test_probe_disabled_leaves_completion_identical(self):
        probing, plain = self.controller(), self.controller(probe_s=0.)
        held = self.closure(.6)
        for controller in (probing, plain):
            controller.update_actual(0, held)
            controller.update_actual(int(1.2 * SEC), held)
        self.assertEqual(plain.end_reason, "feedback_complete")
        self.assertEqual(probing.end_reason, "grasp_confirmed")

    def test_invalid_probe_configuration_is_rejected(self):
        for kwargs in (dict(probe_s=-1.), dict(probe_empty_closure=0.),
                       dict(probe_empty_closure=1.5)):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    HandEventHandoff(side="right", **kwargs)


if __name__ == "__main__":
    unittest.main()
