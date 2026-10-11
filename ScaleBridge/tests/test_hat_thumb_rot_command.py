"""Offline checks of predicted thumb-rot commands using the real motor LUT."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv


class ThumbRotCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mapping_path = Path(__file__).resolve().parents[3] / "TWIST2" / "deploy_real"
        sys.path.insert(0, str(mapping_path))
        from cmd_to_dofpos import cmd6_to_dofpos12
        from dofpos2cmd import dofpos12_to_q6
        cls.to_joint = staticmethod(cmd6_to_dofpos12)
        cls.to_motor = staticmethod(dofpos12_to_q6)
        cls.order = np.array([8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5])

    def env(self, sides):
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.hand_thumb_rot_invert_sides = sides
        env.hand_dof12_to_motor = lambda q: self.to_motor(q[self.order])
        env.hand_motor_to_dof12 = lambda m: self.to_joint(m * 1000)[np.argsort(self.order)]
        return env

    def test_dds_round_trip_changes_only_selected_thumb_rot(self):
        for sides in ([], ["left"], ["right"], ["left", "right"]):
            env = self.env(sides)
            for side in ("left", "right"):
                for thumb in (0.0, 0.2, 0.5, 0.8, 1.0):
                    with self.subTest(sides=sides, side=side, thumb=thumb):
                        motor = np.array([0.1, 0.3, 0.6, 0.7, 0.9, thumb])
                        q12 = env.hand_motor_to_dof12(motor)
                        original = q12.copy()
                        corrected = env._hand_prediction_target(q12, side)
                        # The actual DDS worker recovers commands via this LUT.
                        sent = self.to_motor(corrected[self.order])
                        expected = motor.copy()
                        if side in sides:
                            expected[5] = 1.0 - thumb
                        np.testing.assert_allclose(sent, expected, atol=1e-7)
                        np.testing.assert_array_equal(q12, original)

    def test_startup_open_command_bypasses_prediction_correction(self):
        env = self.env(["left", "right"])
        env.reference_play_gate = True
        env.reference_playing = False
        env._latched_hold_active = lambda: False
        env.hand_ik_worker = Mock()
        env.open_hand_dof12 = env.hand_motor_to_dof12(np.ones(6))
        env.simulator = Mock()
        env._send_hand_target()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        for q12 in (left, right):
            np.testing.assert_allclose(self.to_motor(q12[self.order]), 1.0)
        env.hand_ik_worker.latest.assert_not_called()

    def test_fk_feedback_inversion_changes_only_selected_thumb(self):
        for sides in ([], ["left"], ["right"], ["left", "right"]):
            env = self.env([])
            env.hand_fk_thumb_rot_invert_sides = sides
            for side in ("left", "right"):
                for thumb in (0.0, 0.2, 0.5, 1.0):
                    motor = np.array([0.1, 0.3, 0.6, 0.7, 0.9, thumb])
                    original = motor.copy()
                    corrected = env._hand_fk_motor_state(motor, side)
                    expected = original.copy()
                    if side in sides:
                        expected[5] = 1.0 - thumb
                    np.testing.assert_allclose(corrected, expected)
                    np.testing.assert_array_equal(motor, original)

    def test_measured_fk_uses_corrected_feedback_and_keeps_dds_state(self):
        env = self.env([])
        env.hand_fk_thumb_rot_invert_sides = ["left", "right"]
        motor = np.array([1., 1., 1., 1., 1., 0.])
        env.simulator = Mock()
        env.simulator.get_hand_motor_state.return_value = (motor.copy(), motor.copy())
        env.hand_state_fk_solvers = {side: Mock() for side in ("lh", "rh")}
        for solver in env.hand_state_fk_solvers.values():
            solver._fk_tips.return_value = np.zeros((5, 3))
        left_q, _, right_q, _ = env._measured_hand_state()
        for solver in env.hand_state_fk_solvers.values():
            q12 = solver._fk_tips.call_args.args[0]
            self.assertAlmostEqual(q12[8], 0.0)
            np.testing.assert_allclose(self.to_motor(q12[self.order]), np.ones(6))
        self.assertEqual(left_q[4], 0.0)
        self.assertEqual(right_q[4], 0.0)
        np.testing.assert_array_equal(motor, [1., 1., 1., 1., 1., 0.])

    def test_live_predictions_are_corrected_before_simulator_output(self):
        env = self.env(["right"])
        env.reference_play_gate = False
        env.last_reference_sample = {}
        q12 = env.hand_motor_to_dof12(np.array([0.1, 0.3, 0.6, 0.7, 0.9, 0.2]))
        env.hand_ik_worker = Mock()
        env.hand_ik_worker.latest.return_value = {"lh": q12, "rh": q12}
        env._finger_reference_targets = lambda: (np.zeros((5, 3)), np.zeros((5, 3)))
        env._apply_finger_wrist_gate = lambda left, right: (left, right)
        env.simulator = Mock()
        env._send_hand_target()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_allclose(self.to_motor(left[self.order])[5], 0.2)
        np.testing.assert_allclose(self.to_motor(right[self.order])[5], 0.8)


if __name__ == "__main__":
    unittest.main()
