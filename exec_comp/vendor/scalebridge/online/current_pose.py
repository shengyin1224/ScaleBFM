"""Measured G1 link poses for the online HAT state."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def quat_apply_wxyz(quat, vector):
    quat = np.asarray(quat, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    qvec = quat[1:]
    uv = np.cross(qvec, vector)
    uuv = np.cross(qvec, uv)
    return vector + 2.0 * (quat[0] * uv + uuv)


class G1CurrentPoseFK:
    """MuJoCo FK driven only by measured root pose and 29 joint positions."""

    LINK_NAMES = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")

    def __init__(
        self, xml_path, joint_names, *, head_offset=(0.0, 0.0, 0.4),
        link_names=(),
    ):
        import mujoco

        path = Path(xml_path)
        if not path.is_absolute():
            scale_bridge_root = Path(__file__).resolve().parents[2]
            path = scale_bridge_root / path
        if not path.is_file():
            raise FileNotFoundError(f"G1 FK XML does not exist: {path}")

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.head_offset = np.asarray(head_offset, dtype=np.float64)
        if self.head_offset.shape != (3,):
            raise ValueError(f"head_offset must have shape (3,), got {self.head_offset.shape}")

        root_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis"
        )
        if root_joint < 0:
            raise KeyError("G1 FK model has no pelvis free joint")
        self.root_qpos_adr = int(self.model.jnt_qposadr[root_joint])

        self.joint_qpos_adr = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, str(name)
            )
            if joint_id < 0:
                raise KeyError(f"G1 FK model has no joint {name!r}")
            self.joint_qpos_adr.append(int(self.model.jnt_qposadr[joint_id]))
        self.joint_qpos_adr = np.asarray(self.joint_qpos_adr, dtype=np.int64)

        self.body_ids = {}
        for name in dict.fromkeys((*self.LINK_NAMES, *tuple(link_names))):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id < 0:
                raise KeyError(f"G1 FK model has no body {name!r}")
            self.body_ids[name] = body_id
        self.last_body_poses = {}

    def compute(self, root_pos, root_quat_wxyz, dof_pos):
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_quat = np.asarray(root_quat_wxyz, dtype=np.float64)
        dof_pos = np.asarray(dof_pos, dtype=np.float64)
        if root_pos.shape != (3,) or root_quat.shape != (4,):
            raise ValueError(
                f"root pose must be (3,)/(4,), got {root_pos.shape}/{root_quat.shape}"
            )
        if dof_pos.shape != self.joint_qpos_adr.shape:
            raise ValueError(
                f"dof_pos must have shape {self.joint_qpos_adr.shape}, got {dof_pos.shape}"
            )
        if not (
            np.isfinite(root_pos).all()
            and np.isfinite(root_quat).all()
            and np.isfinite(dof_pos).all()
        ):
            raise ValueError("G1 FK input contains NaN or infinity")
        quat_norm = np.linalg.norm(root_quat)
        if quat_norm < 1e-6:
            raise ValueError("G1 FK received a zero root quaternion")
        root_quat = root_quat / quat_norm

        adr = self.root_qpos_adr
        self.data.qpos[adr : adr + 3] = root_pos
        self.data.qpos[adr + 3 : adr + 7] = root_quat
        self.data.qpos[self.joint_qpos_adr] = dof_pos
        self.mujoco.mj_forward(self.model, self.data)

        poses = {}
        for name, body_id in self.body_ids.items():
            poses[name] = (
                self.data.xpos[body_id].astype(np.float32).copy(),
                self.data.xquat[body_id].astype(np.float32).copy(),
            )
        # The online reference builder consumes these measured poses for every
        # ScaleBFM-selected link that HAT itself does not predict.  Publishing
        # the cache here avoids a second MuJoCo FK pass in the same 50 Hz step.
        self.last_body_poses = poses
        torso_pos, torso_quat = poses["torso_link"]
        head_pos = torso_pos + quat_apply_wxyz(torso_quat, self.head_offset)
        return {
            "head_pos_world": head_pos.astype(np.float32),
            "head_quat_world_wxyz": torso_quat.copy(),
            "left_wrist_pos_world": poses["left_wrist_yaw_link"][0],
            "left_wrist_quat_world_wxyz": poses["left_wrist_yaw_link"][1],
            "right_wrist_pos_world": poses["right_wrist_yaw_link"][0],
            "right_wrist_quat_world_wxyz": poses["right_wrist_yaw_link"][1],
        }
