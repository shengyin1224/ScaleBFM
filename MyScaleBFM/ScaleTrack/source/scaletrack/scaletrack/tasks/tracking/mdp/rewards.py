from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from scaletrack.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _phase_weighted_mean(
    command: MotionCommand,
    error: torch.Tensor,
    body_indexes: list[int],
    focus_weight: float,
    phase_weighted_body_names: list[str] | None,
) -> torch.Tensor:
    if focus_weight == 1.0 or not phase_weighted_body_names:
        return error.mean(-1)
    if focus_weight <= 0:
        raise ValueError(f"focus_weight must be positive, got {focus_weight}")
    selected_names = [command.cfg.body_names[index] for index in body_indexes]
    phase_columns = [
        index
        for index, name in enumerate(selected_names)
        if name in phase_weighted_body_names
    ]
    if not phase_columns:
        return error.mean(-1)
    weights = torch.ones_like(error)
    focus = (command.focus_flag & command.focus_valid).to(error.dtype)
    multiplier = 1.0 + focus * (float(focus_weight) - 1.0)
    weights[:, phase_columns] = multiplier.unsqueeze(-1)
    # Keep the factory denominator.  This literally multiplies the selected
    # wrist terms by w_focus while leaving every other link coefficient at
    # one; normalizing by sum(weights) would silently down-weight all other
    # links during FOCUS.
    return torch.mean(error * weights, dim=-1)


def global_body_height(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[..., 2] - command.robot_anchor_pos_w[..., 2])
    return torch.exp(-error / std**2)

def global_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    focus_weight: float = 1.0,
    phase_weighted_body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    weighted_error = _phase_weighted_mean(
        command,
        error,
        body_indexes,
        focus_weight,
        phase_weighted_body_names,
    )
    return torch.exp(-weighted_error / std**2)

def global_body_rot(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    focus_weight: float = 1.0,
    phase_weighted_body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    weighted_error = _phase_weighted_mean(
        command,
        error,
        body_indexes,
        focus_weight,
        phase_weighted_body_names,
    )
    return torch.exp(-weighted_error / std**2)

def global_body_lin_vel(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def global_body_ang_vel(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
