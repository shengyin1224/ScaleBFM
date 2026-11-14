"""TWIST tracking rewards ported without any object reward term."""

from __future__ import annotations

import torch

from .commands import DexHandMotionCommand


def _command(env, command_name: str) -> DexHandMotionCommand:
    return env.command_manager.get_term(command_name)


def joint_position_tracking(env, command_name: str = "motion", coefficient: float = -0.15) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.square(command.joint_pos("rh") - command.q_rh).sum(-1)
    error += torch.square(command.joint_pos("lh") - command.q_lh).sum(-1)
    return torch.exp(coefficient * error)


def joint_velocity_tracking(env, command_name: str = "motion", coefficient: float = -0.01) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.square(command.joint_vel("rh") - command.dq_rh).sum(-1)
    error += torch.square(command.joint_vel("lh") - command.dq_lh).sum(-1)
    return torch.exp(coefficient * error)


def fingertip_tracking(
    env,
    command_name: str = "motion",
    coefficient_start: float = -20.0,
    coefficient_end: float = -150.0,
    curriculum_steps: int = 250_000,
) -> torch.Tensor:
    """Tighten from broad exploration shaping to millimetric tracking."""

    command = _command(env, command_name)
    fraction = min(float(env.common_step_counter) / max(curriculum_steps, 1), 1.0)
    coefficient = coefficient_start + fraction * (coefficient_end - coefficient_start)
    actual_rh = command.right_hand.data.body_pos_w[:, command.right_tip_ids]
    actual_lh = command.left_hand.data.body_pos_w[:, command.left_tip_ids]
    error_rh = torch.square(actual_rh - command.fingertip_pos_rh).sum((-1, -2))
    error_lh = torch.square(actual_lh - command.fingertip_pos_lh).sum((-1, -2))
    error = torch.where(command.hand_active_rh, error_rh, torch.zeros_like(error_rh))
    error += torch.where(command.hand_active_lh, error_lh, torch.zeros_like(error_lh))
    return torch.exp(coefficient * error)


def action_rate_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def joint_velocity_l2(env, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    return torch.square(command.joint_vel("rh")).sum(-1) + torch.square(command.joint_vel("lh")).sum(-1)
