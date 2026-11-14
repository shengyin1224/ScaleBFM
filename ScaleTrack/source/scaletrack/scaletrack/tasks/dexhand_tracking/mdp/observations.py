"""Object-blind actor and asymmetric critic observations."""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

from .commands import DexHandMotionCommand


def _command(env, command_name: str) -> DexHandMotionCommand:
    return env.command_manager.get_term(command_name)


def _rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    matrix = matrix_from_quat(quaternion)
    return matrix[..., :, :2].reshape(*quaternion.shape[:-1], 6)


def _future_for_hand(command: DexHandMotionCommand, hand: str) -> torch.Tensor:
    wrist = command.future_wrist_states(hand)
    tips = command.future_fingertips(hand)
    wrist_pos = wrist[..., :3]
    wrist_quat = wrist[..., 3:7]
    # TWIST semantics: global wrist position/RPY and five wrist-local tips.
    local_tip = quat_apply(
        quat_inv(wrist_quat)[:, :, None, :].expand(-1, -1, tips.shape[2], -1),
        tips - wrist_pos[:, :, None, :],
    )
    # Isaac Lab 2.3's helper accepts only a flat (N, 4) batch.
    flat_quat = wrist_quat.reshape(-1, 4)
    roll, pitch, yaw = euler_xyz_from_quat(flat_quat)
    wrist_rpy = torch.stack((roll, pitch, yaw), dim=-1).reshape(*wrist_quat.shape[:-1], 3)
    wrist_pos = wrist_pos - command._env.scene.env_origins[:, None, :]
    return torch.cat((wrist_pos, wrist_rpy, local_tip.flatten(start_dim=-2)), dim=-1)


def future_keypoint_targets(env, command_name: str = "motion") -> torch.Tensor:
    """20 x 42 target tokens; no object, force, or visual state."""

    command = _command(env, command_name)
    return torch.cat((_future_for_hand(command, "rh"), _future_for_hand(command, "lh")), dim=-1)


def actor_proprio(env, command_name: str = "motion") -> torch.Tensor:
    """Current proprio for both hands.

    Per hand: body angular velocity (3), full wrist rotation 6D (6), q (12),
    dq (12), and previous action (12) = 45 dimensions.  Isaac Lab's observation
    manager supplies the 11-frame history.
    """

    command = _command(env, command_name)
    last_action = env.action_manager.action
    if last_action is None:
        last_action = torch.zeros(env.num_envs, 24, device=env.device)
    values = []
    for hand_name, hand, action in (
        ("rh", command.right_hand, last_action[:, :12]),
        ("lh", command.left_hand, last_action[:, 12:]),
    ):
        values.append(
            torch.cat(
                (
                    hand.data.root_ang_vel_b,
                    _rotation_6d(hand.data.root_quat_w),
                    command.joint_pos(hand_name) - command.default_joint_pos(hand_name),
                    command.joint_vel(hand_name),
                    action,
                ),
                dim=-1,
            )
        )
    return torch.cat(values, dim=-1)


def bc_target_action(env, command_name: str = "motion") -> torch.Tensor:
    """Reference action exposed to the loss only; ActorCriticDexHand ignores this group."""

    return _command(env, command_name).reference_action


def _orientation_error_vector(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    delta = quat_mul(reference, quat_inv(actual))
    sign = torch.where(delta[..., :1] < 0, -1.0, 1.0)
    return 2.0 * sign * delta[..., 1:4]


def _tip_force(env, sensor_name: str) -> torch.Tensor:
    force = env.scene[sensor_name].data.net_forces_w
    if force.shape[1] != 5:
        raise RuntimeError(f"{sensor_name}: expected exactly five tip bodies, got {force.shape}")
    return force.flatten(start_dim=1)


def critic_privileged(env, command_name: str = "motion") -> torch.Tensor:
    """Real privileged object/contact features used only by the critic."""

    command = _command(env, command_name)
    rows = []
    for hand_name, hand, obj, ref_state, active, mass, friction, force_name in (
        (
            "rh",
            command.right_hand,
            command.object_rh,
            command.object_state_rh,
            command.object_active_rh,
            command.object_mass_rh,
            command.object_friction_rh,
            "right_tip_contact",
        ),
        (
            "lh",
            command.left_hand,
            command.object_lh,
            command.object_state_lh,
            command.object_active_lh,
            command.object_mass_lh,
            command.object_friction_lh,
            "left_tip_contact",
        ),
    ):
        wrist_pos = hand.data.root_pos_w
        wrist_quat = hand.data.root_quat_w
        object_state = obj.data.root_state_w
        relative_pos = quat_apply_inverse(wrist_quat, object_state[:, :3] - wrist_pos)
        relative_quat = quat_mul(quat_inv(wrist_quat), object_state[:, 3:7])
        relative_lin_vel = quat_apply_inverse(wrist_quat, object_state[:, 7:10] - hand.data.root_lin_vel_w)
        relative_ang_vel = quat_apply_inverse(wrist_quat, object_state[:, 10:13] - hand.data.root_ang_vel_w)
        pose_error = torch.cat(
            (
                quat_apply_inverse(wrist_quat, ref_state[:, :3] - object_state[:, :3]),
                _orientation_error_vector(ref_state[:, 3:7], object_state[:, 3:7]),
            ),
            dim=-1,
        )
        support_height = torch.where(command.has_table, command.table_heights, torch.zeros_like(command.table_heights))
        height_margin = object_state[:, 2] - support_height
        wrist_distance = torch.linalg.vector_norm(object_state[:, :3] - wrist_pos, dim=-1)
        rows.append(
            torch.cat(
                (
                    relative_pos,
                    relative_quat,
                    relative_lin_vel,
                    relative_ang_vel,
                    pose_error,
                    _tip_force(env, force_name),
                    mass[:, None],
                    friction[:, None],
                    active.float()[:, None],
                    height_margin[:, None],
                    wrist_distance[:, None],
                    command.contact_established.float()[:, None],
                ),
                dim=-1,
            )
        )
    return torch.cat(rows, dim=-1)
