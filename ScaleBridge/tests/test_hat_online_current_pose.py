import unittest

import numpy as np

from scalebridge.online.current_pose import G1CurrentPoseFK


XML = (
    "/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge/"
    "scalebridge/data/robot/g1_29dof/g1_29dof.xml"
)


class G1CurrentPoseFKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco

        model = mujoco.MjModel.from_xml_path(XML)
        cls.joint_names = []
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name != "pelvis":
                cls.joint_names.append(name)
        cls.fk = G1CurrentPoseFK(
            XML, cls.joint_names,
            link_names=("left_elbow_link", "right_elbow_link"),
        )

    def test_head_is_fixed_offset_from_torso_and_poses_are_finite(self):
        result = self.fk.compute(
            [1.0, -2.0, 0.8], [1.0, 0.0, 0.0, 0.0],
            np.zeros(len(self.joint_names)),
        )
        self.assertEqual(set(result), {
            "head_pos_world", "head_quat_world_wxyz",
            "left_wrist_pos_world", "left_wrist_quat_world_wxyz",
            "right_wrist_pos_world", "right_wrist_quat_world_wxyz",
        })
        self.assertIn("left_elbow_link", self.fk.last_body_poses)
        self.assertIn("right_elbow_link", self.fk.last_body_poses)
        for key, value in result.items():
            self.assertTrue(np.isfinite(value).all(), key)
            if "quat" in key:
                self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0, places=5)

    def test_global_yaw_rotates_every_link_pose(self):
        joints = np.zeros(len(self.joint_names))
        base = self.fk.compute([0, 0, 0.8], [1, 0, 0, 0], joints)
        yaw90 = np.sqrt(0.5)
        rotated = self.fk.compute([3, 4, 0.8], [yaw90, 0, 0, yaw90], joints)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        for prefix in ("head", "left_wrist", "right_wrist"):
            expected = np.array([3, 4, 0.8]) + rotation @ (
                base[f"{prefix}_pos_world"] - np.array([0, 0, 0.8])
            )
            np.testing.assert_allclose(
                rotated[f"{prefix}_pos_world"], expected, atol=2e-5
            )


if __name__ == "__main__":
    unittest.main()
