"""OMEGA-HOME opening 29-DOF references, in radians.

Source: omega_HOME_4tasks_20261025/hat, body_q_measured (not action/joint_pos).
For each episode in each task's training + validation manifest, take the
per-joint median of frames 0:10, then the median across episodes. Computed
2026-10-28 from omega_HOME_4tasks_geometry_noleftwrist_w0_20261025/configs.
Root world pose and Inspire finger motors are not part of these profiles.
"""

import numpy as np

from scalebridge.online.protocol import HAT_BODY_JOINT_NAMES


OMEGA_JOINT_NAMES = HAT_BODY_JOINT_NAMES[:15] + tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )
)

OMEGA_READY_Q = {
    "omega_wipe_table": (
        0.153347, 0.010951, 0.068694, 0.168955, -0.239830, 0.013321,
        0.156513, -0.033113, -0.139825, 0.168695, -0.205330, 0.018141,
        0.008269, -0.009177, 0.003844,
        0.240871, 0.102045, 0.073871, 1.081636, 0.264330, 0.125025, 0.135428,
        0.141186, -0.127979, 0.059184, 1.262065, -0.266673, 0.131695, 0.003523,
    ),
    "omega_classify_gadgets": (
        0.132530, 0.020915, 0.107788, 0.167928, -0.237640, 0.005680,
        0.134769, -0.016141, -0.184514, 0.149951, -0.184670, -0.001288,
        0.005524, -0.016850, 0.003106,
        0.237231, 0.101299, 0.076262, 1.119554, 0.010933, 0.032354, 0.146780,
        0.136246, -0.123554, 0.057129, 1.281108, 0.487147, 0.017116, 0.138613,
    ),
    "omega_grab_fruit_bucket": (
        0.114081, 0.007898, 0.081319, 0.187465, -0.232947, 0.009830,
        0.124495, -0.061355, -0.188873, 0.174710, -0.208385, 0.038173,
        0.001537, -0.022503, 0.003804,
        0.221103, 0.120986, 0.072211, 1.117898, 0.423651, 0.192784, -0.009755,
        0.138046, -0.143744, 0.020190, 1.265594, -0.281644, 0.188194, 0.066704,
    ),
    "omega_move_table_with_human": (
        0.109609, 0.031960, 0.138402, 0.151464, -0.211639, -0.004718,
        0.089734, -0.015990, -0.195555, 0.149521, -0.170548, -0.008834,
        0.001302, -0.016217, 0.005540,
        0.187682, 0.096242, 0.035024, 1.125984, -0.029370, -0.066336, 0.201916,
        0.109641, -0.120590, 0.057650, 1.259272, 0.463115, -0.033121, 0.151547,
    ),
}

OMEGA_READY_EPISODES = {
    "omega_wipe_table": 187,
    "omega_classify_gadgets": 200,
    "omega_grab_fruit_bucket": 204,
    "omega_move_table_with_human": 200,
}


def omega_ready_dof(profile, joint_names):
    """Reorder the complete reference into the controller's named joint order."""
    names = tuple(joint_names)
    if len(names) != 29 or set(names) != set(OMEGA_JOINT_NAMES):
        raise ValueError("OMEGA startup requires the complete 29 named G1 body joints")
    by_name = dict(zip(OMEGA_JOINT_NAMES, OMEGA_READY_Q[profile]))
    return np.asarray([by_name[name] for name in names], dtype=np.float32)
