import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scalebridge.utils.torch_utils import calc_heading_quat, calc_heading_quat_inv, quat_mul

# ---- constant: P frame expressed in tracker-2 body frame ----
# Tracker-2 body axes (after Y-up→Z-up), measured on 2026-08-14 from the
# mounted attitude plus a manual move-the-tracker calibration session
# (outputs/2026-08-14/12-44-10): x-forward, y-right, z-down.
# The previous assumption (x-down, y-right, z-back) left the P frame lying
# on its side (P-z ~94 deg from vertical in every recorded run), which made
# the extracted heading ill-conditioned and destabilised the global loop.
# P axes: x-forward, y-left, z-up  (standard Z-up world)
# Columns = T2-body basis vectors expressed in P/world frame:
#   T2_x (forward) → [1, 0, 0]
#   T2_y (right)   → [0, -1, 0]
#   T2_z (down)    → [0, 0, -1]
_R_P_FROM_T2BODY = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1],
], dtype=np.float64)

# Position of P origin in tracker-2 body coordinates (metres).
# Physical intent unchanged: pelvis centre sits ~7 cm in front of and
# ~4 cm above the tracker; re-expressed in the corrected body axes
# (forward = +x, up = -z).
_P_POS_IN_T2BODY = np.array([0.07, 0.0, -0.04], dtype=np.float64)


def _yup_to_zup_pos(pos: np.ndarray) -> np.ndarray:
    """Convert position from Y-up (X-left, Y-up, Z-forward)
    to Z-up (X-forward, Y-left, Z-up): new = (z, x, y)."""
    return np.array([pos[2], pos[0], pos[1]], dtype=np.float64)


def _yup_to_zup_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) from Y-up to Z-up.
    Applies the same axis permutation as _yup_to_zup_pos."""
    w, x, y, z = q
    return np.array([w, z, x, y], dtype=np.float64)


class ViveTrackerProcessor:

    def __init__(self):
        # Pre-compute the fixed rotation from T2-body to P as a Rotation object
        self._R_P_from_T2body = R.from_matrix(_R_P_FROM_T2BODY)
        self._p_pos_in_T2body = _P_POS_IN_T2BODY.copy()

        # Optional calibration: an offset rotation applied to the final result
        self._calib_pos: np.ndarray | None = None
        self._calib_rot: R | None = None

    def calibrate(self, robot_root_quat_wxyz: np.ndarray, root_pose_init: np.ndarray) -> None:
        target_heading = calc_heading_quat(torch.from_numpy(robot_root_quat_wxyz))
        source_q0 = torch.from_numpy(root_pose_init[3:7].copy())
        source_heading_inv = calc_heading_quat_inv(source_q0)
        self._calib_rot = quat_mul(target_heading, source_heading_inv).numpy()
        self._calib_rot = R.from_quat(self._calib_rot, scalar_first=True)

        self._calib_pos = root_pose_init[:3].copy()
        self._calib_pos[..., -1] = 0.0

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process a (2, 14) raw tracker frame and return the pelvis pose.

        Parameters
        ----------
        data : np.ndarray, shape (2, 14)
            Row 0 = tracker 1 (base), Row 1 = tracker 2.
            Each row: pos(3) + quat_wxyz(4) + vel(3) + ang_vel(3) + timestamp(1).

        Returns
        -------
        result : np.ndarray, shape (7,)
            [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
            of frame P in the gravity-aligned SteamVR frame, translated so the
            floor tracker is the origin.
        """
        # ---- unpack raw data ----
        pos1_raw = data[0, :3]
        pos2_raw = data[1, :3]
        quat2_raw = data[1, 3:7]

        # ==================================================================
        # Step 1: Y-up → Z-up
        #   Original : X-left,    Y-up,   Z-forward
        #   Target   : X-forward, Y-left, Z-up
        #   Mapping  : new = (old_z, old_x, old_y)
        # ==================================================================
        pos1 = _yup_to_zup_pos(pos1_raw)
        pos2 = _yup_to_zup_pos(pos2_raw)
        quat2 = _yup_to_zup_quat_wxyz(quat2_raw)

        # scipy Rotation uses scalar-last (x, y, z, w)
        R2 = R.from_quat([quat2[1], quat2[2], quat2[3], quat2[0]])

        # ==================================================================
        # Step 2: Tracker-2 pose relative to the floor tracker origin
        #
        # SteamVR's standing universe already supplies a gravity-aligned
        # coordinate system.  The floor tracker therefore defines only the
        # translation origin; its arbitrary physical orientation must not
        # rotate the world frame.  Using R1^{-1} here made the computed height
        # depend on which way the floor tracker happened to face, and could
        # turn a valid positive pelvis height into a negative value.
        # ==================================================================
        t_rel = pos2 - pos1

        # ==================================================================
        # Step 3: Frame P in the gravity-aligned frame
        #   T_P_in_T2 is the fixed transform of P in tracker-2 body:
        #       rotation  = R_P_from_T2body
        #       position  = p_pos_in_T2body
        #   T_P_in_W = T_T2_in_W * T_P_in_T2
        #       R_P_in_W = R2 * R_P_from_T2body
        #       t_P_in_W = t_rel + R2.apply(p_pos_in_T2body)
        # ==================================================================
        R_P_in_world = R2 * self._R_P_from_T2body
        t_P_in_world = t_rel + R2.apply(self._p_pos_in_T2body)

        # ---- optional calibration offset ----
        if self._calib_rot is not None:
            R_P_in_world = self._calib_rot * R_P_in_world
            t_P_in_world = self._calib_rot.apply(t_P_in_world - self._calib_pos)

        # ---- pack result as (7,): pos(3) + quat_wxyz(4) ----
        q_xyzw = R_P_in_world.as_quat()  # scipy → (x, y, z, w)
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        result = np.concatenate([t_P_in_world, q_wxyz])
        return result
