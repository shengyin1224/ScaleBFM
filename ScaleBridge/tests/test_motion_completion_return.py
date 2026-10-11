import unittest

import torch

from scalebridge.env.motion_tracking import MotionTrackingEnv


class MotionCompletionReturnTest(unittest.TestCase):
    @staticmethod
    def _make_env(hardening=True):
        env = MotionTrackingEnv.__new__(MotionTrackingEnv)
        env.deployment_hardening = hardening
        env.motion_complete = True
        env.motion_complete_local_hold = True
        env.motion_complete_return_to_start_steps = 3
        env._completion_return_active = True
        env._completion_return_index = 0
        env._completion_return_finished = False
        env._completion_stand_body_pos = None
        env._completion_stand_body_quat = None
        env.reference_play_gate = False
        env.reference_playing = True
        env._root_transition_active = False
        env.localization_paused = False
        env.reference_forcing = False
        env.episode_length_buf = torch.tensor([1], dtype=torch.long)
        env.motion_len = 2
        env.future_frame_offset = torch.tensor([0, 1], dtype=torch.long)
        env.body_pos_w = torch.tensor([
            [[0.0, 0.0, 1.0]],
            [[2.0, 0.0, 1.0]],
        ])
        env.body_quat_w = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        ])
        env.state_buffer = {
            "root_pos_buffer": torch.zeros(1, 3, 3),
            "root_quat_wxyz_buffer": torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3]),
        }
        return env

    def test_hardened_completion_blends_final_to_frame_zero_then_holds(self):
        env = self._make_env()

        env._gather_reference_state()
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0], env.body_pos_w[-1]
        )

        env._gather_reference_state()
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0],
            torch.tensor([[1.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(
            env.state_buffer["root_pos_buffer"][0, -1],
            torch.tensor([1.0, 0.0, 1.0]),
        )
        midpoint_quat = env.state_buffer["body_quat_w_wxyz_future"][0, 0, 0]
        torch.testing.assert_close(
            torch.linalg.vector_norm(midpoint_quat), torch.tensor(1.0)
        )

        env._gather_reference_state()
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0], env.body_pos_w[0]
        )
        self.assertFalse(env._completion_return_active)
        self.assertTrue(env._completion_return_finished)

        env._gather_reference_state()
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0], env.body_pos_w[0]
        )

    def test_disabled_hardening_preserves_final_frame_hold(self):
        env = self._make_env(hardening=False)
        env._gather_reference_state()
        torch.testing.assert_close(
            env.state_buffer["body_pos_w_future"][0, 0], env.body_pos_w[-1]
        )

    def test_frame_zero_stand_is_anchored_to_completion_heading(self):
        env = self._make_env()
        env.body_pos_w = torch.tensor([
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        ])
        env.body_quat_w = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ])
        yaw90 = 2.0 ** -0.5
        env.state_buffer["root_quat_wxyz_buffer"][0, -1] = torch.tensor(
            [yaw90, 0.0, 0.0, yaw90]
        )

        env._prepare_completion_stand_target()

        torch.testing.assert_close(
            env._completion_stand_body_pos,
            torch.tensor([[2.0, 0.0, 1.0], [2.0, 1.0, 1.0]]),
            atol=1e-5,
            rtol=1e-5,
        )
        root_quat = env._completion_stand_body_quat[0]
        expected = torch.tensor([yaw90, 0.0, 0.0, yaw90])
        self.assertAlmostEqual(float(torch.abs(torch.dot(root_quat, expected))), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
