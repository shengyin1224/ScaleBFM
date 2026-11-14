"""Inspire hand articulation configs for the dexhand GMT task."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from scaletrack.assets import ASSET_DIR


INSPIRE_JOINT_COUNT = 12
INSPIRE_ACTION_SCALE = 0.5
INSPIRE_STIFFNESS = 10.0
INSPIRE_DAMPING = 1.0
INSPIRE_EFFORT_LIMIT = 1.0
# The URDF declares 0.5 rad/s, but the source Isaac Gym environment uses
# effort-mode actuation and recorded successful rollouts routinely exceed it
# (heldout q finite-difference p95 is 1.87 rad/s). Applying 0.5 as PhysX's hard
# joint limit in IsaacLab makes the reference physically untrackable.
INSPIRE_VELOCITY_LIMIT = 100.0


def _usd_path(side: str) -> str:
    override = os.getenv(f"SCALETRACK_INSPIRE_{side.upper()}_USD")
    if override:
        return str(Path(override).expanduser().resolve())
    return str(Path(ASSET_DIR) / "dexhand" / "inspire" / f"inspire_hand_{side}.usd")


def _cfg(side: str) -> ArticulationCfg:
    prefix = "R" if side == "right" else "L"
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd_path(side),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=2.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={f"{prefix}_.*_joint": 0.0},
            joint_vel={f"{prefix}_.*_joint": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            # TWIST runs effort mode and explicitly computes/clips the PD
            # torque every simulation substep. IdealPDActuator reproduces that
            # path; a PhysX implicit position drive is not dynamically
            # equivalent for these moving wrist roots.
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=[f"{prefix}_.*_joint"],
                effort_limit=INSPIRE_EFFORT_LIMIT,
                effort_limit_sim=1.0e9,
                velocity_limit_sim=INSPIRE_VELOCITY_LIMIT,
                stiffness=INSPIRE_STIFFNESS,
                damping=INSPIRE_DAMPING,
            )
        },
    )


INSPIRE_HAND_RIGHT_CFG = _cfg("right")
INSPIRE_HAND_LEFT_CFG = _cfg("left")

RIGHT_TIP_BODY_NAMES = [
    "R_thumb_tip",
    "R_index_tip",
    "R_middle_tip",
    "R_ring_tip",
    "R_pinky_tip",
]
LEFT_TIP_BODY_NAMES = [name.replace("R_", "L_") for name in RIGHT_TIP_BODY_NAMES]

# The imported ``*_tip`` links are fixed, massless marker links without collision
# shapes.  Track their positions, but collect contact impulses from the nearest
# collision-bearing distal/intermediate links.
RIGHT_CONTACT_BODY_NAMES = [
    "R_thumb_distal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_ring_intermediate",
    "R_pinky_intermediate",
]
LEFT_CONTACT_BODY_NAMES = [name.replace("R_", "L_") for name in RIGHT_CONTACT_BODY_NAMES]

_JOINT_STEMS = [
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
]
RIGHT_JOINT_NAMES = [f"R_{name}" for name in _JOINT_STEMS]
LEFT_JOINT_NAMES = [f"L_{name}" for name in _JOINT_STEMS]
