"""Offline OMEGA startup FK and prestart/full-body mode regressions."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scalebridge.env.base_env import BaseEnv
from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from scalebridge.online.current_pose import G1CurrentPoseFK
from scalebridge.online.omega_ready_pose import (
    OMEGA_JOINT_NAMES, OMEGA_READY_Q, omega_ready_dof,
)


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "scalebridge/data/robot/g1_29dof/g1_29dof.xml"


class OmegaReadyPoseTest(unittest.TestCase):
    def test_joint_mapping_rejects_partial_or_duplicate_order(self):
        for names in (OMEGA_JOINT_NAMES[:-1], (OMEGA_JOINT_NAMES[0],) * 29):
            with self.assertRaises(ValueError):
                omega_ready_dof("omega_wipe_table", names)

    def test_profiles_make_consistent_full_body_references_in_both_envs(self):
        import mujoco

        # Exercise an interleaved controller order instead of dataset leg/arm order.
        joint_names = OMEGA_JOINT_NAMES[::2] + OMEGA_JOINT_NAMES[1::2]
        links = (
            "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
            "torso_link", "left_shoulder_roll_link", "left_elbow_link",
            "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
            "right_wrist_yaw_link",
        )
        root_pos = np.array([1.0, -2.0, 0.78], dtype=np.float32)
        root_quat = np.array([np.cos(.06), 0, np.sin(.06), 0], dtype=np.float32)
        fk = G1CurrentPoseFK(XML, joint_names, link_names=links)
        baseline = fk.compute(root_pos, root_quat, np.zeros(29))
        for cls in (HAT4OnlineMotionTrackingEnv, HAT4AdaptiveMotionTrackingEnv):
            for profile in OMEGA_READY_Q:
                with self.subTest(env=cls.__name__, profile=profile):
                    env = cls.__new__(cls)
                    env.prestart_arm_pose = profile
                    env.metadata_dict = {"joint_names": joint_names}
                    env.state_buffer = {"dof_pos_buffer": torch.zeros(1, 1, 29)}
                    env.current_pose_fk = fk
                    env.open_hand_fingertips = {
                        "lh": np.zeros((5, 3), dtype=np.float32),
                        "rh": np.ones((5, 3), dtype=np.float32),
                    }
                    ready = env._make_prestart_reference(root_pos, root_quat, {})
                    expected = dict(zip(OMEGA_JOINT_NAMES, OMEGA_READY_Q[profile]))
                    for name, adr in zip(joint_names, fk.joint_qpos_adr):
                        self.assertAlmostEqual(fk.data.qpos[adr], expected[name], places=6)
                        jid = mujoco.mj_name2id(fk.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        lo, hi = fk.model.jnt_range[jid]
                        self.assertTrue(lo <= expected[name] <= hi, name)
                    np.testing.assert_array_equal(ready["root_pos"][0], root_pos)
                    np.testing.assert_array_equal(ready["root_quat_wxyz"][0], root_quat)
                    self.assertEqual(set(env._prestart_link_poses), set(links))
                    for side in ("left", "right"):
                        pos, quat = env._prestart_link_poses[f"{side}_wrist_yaw_link"]
                        np.testing.assert_allclose(ready[f"{side}_wrist_pos"][0], pos)
                        np.testing.assert_allclose(ready[f"{side}_wrist_quat_wxyz"][0], quat)
                        self.assertGreater(np.linalg.norm(pos - baseline[f"{side}_wrist_pos_world"]), .05)
                    torso_pos, torso_quat = env._prestart_link_poses["torso_link"]
                    np.testing.assert_allclose(ready["torso_pos"][0], torso_pos)
                    np.testing.assert_allclose(ready["head_quat_wxyz"][0], torso_quat)
                    self.assertGreater(np.linalg.norm(ready["head_pos"][0] - baseline["head_pos_world"]), .001)
                    np.testing.assert_array_equal(env.state_buffer["dof_pos_buffer"], 0)

    def test_full_body_mode_only_during_omega_prestart(self):
        for profile in (*OMEGA_READY_Q, "stand", "sonic_0904", "upright_stand"):
            for playing in (False, True):
                env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
                env.prestart_arm_pose = profile
                env.reference_play_gate = True
                env.reference_playing = playing
                env.device = "cpu"
                with patch.object(BaseEnv, "_update_observation_manager", return_value={}):
                    obs = env._update_observation_manager()
                full_body = not playing and (profile in OMEGA_READY_Q or profile == "upright_stand")
                self.assertEqual("policy_control_mode" in obs, full_body)
                if full_body:
                    self.assertEqual(obs["policy_control_mode"].item(), 7)
                self.assertEqual(obs["prestart_hold"], not playing)


if __name__ == "__main__":
    unittest.main()
