"""Dexhand tracking, motion-end, and contact-gated object-drop termination."""

from __future__ import annotations

import torch

from .commands import DexHandMotionCommand


def _command(env, command_name: str) -> DexHandMotionCommand:
    return env.command_manager.get_term(command_name)


def motion_finished(
    env, command_name: str = "motion", disable_flag: bool = False
) -> torch.Tensor:
    if disable_flag:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    command = _command(env, command_name)
    return command.time_steps.to(env.device) >= command.time_totals[command.motion_ids].to(env.device) - 1


def bad_fingertip_tracking(
    env,
    command_name: str = "motion",
    thresholds: tuple[float, float, float, float, float] = (0.06, 0.0675, 0.075, 0.09, 0.09),
    contact_relaxation: float = 1.5,
    disable_flag: bool = False,
) -> torch.Tensor:
    if disable_flag:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    command = _command(env, command_name)
    threshold = torch.tensor(thresholds, device=env.device)[None, :]
    relaxed = torch.where(
        command.contact_established[:, None],
        threshold * contact_relaxation,
        threshold,
    )
    error_rh = torch.linalg.vector_norm(
        command.right_hand.data.body_pos_w[:, command.right_tip_ids] - command.fingertip_pos_rh, dim=-1
    )
    error_lh = torch.linalg.vector_norm(
        command.left_hand.data.body_pos_w[:, command.left_tip_ids] - command.fingertip_pos_lh, dim=-1
    )
    bad_rh = command.hand_active_rh & torch.any(error_rh > relaxed, dim=-1)
    bad_lh = command.hand_active_lh & torch.any(error_lh > relaxed, dim=-1)
    return bad_rh | bad_lh


def bad_wrist_velocity(
    env,
    command_name: str = "motion",
    linear_threshold: float = 5.0,
    angular_threshold: float = 20.0,
    disable_flag: bool = False,
) -> torch.Tensor:
    if disable_flag:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    command = _command(env, command_name)
    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for hand, reference in (
        (command.right_hand, command.wrist_state_rh),
        (command.left_hand, command.wrist_state_lh),
    ):
        bad |= torch.linalg.vector_norm(hand.data.root_lin_vel_w - reference[:, 7:10], dim=-1) > linear_threshold
        bad |= (
            torch.linalg.vector_norm(hand.data.root_ang_vel_w - reference[:, 10:13], dim=-1) > angular_threshold
        )
    return bad


def object_dropped(
    env,
    command_name: str = "motion",
    contact_force_threshold: float = 2.0,
    support_tolerance: float = 0.05,
    flyaway_distance: float = 0.5,
    disable_flag: bool = False,
) -> torch.Tensor:
    """Terminate only after contact was established, never merely for pose error."""

    if disable_flag:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    command = _command(env, command_name)
    established = command.update_contact_established(contact_force_threshold)
    support_height = torch.where(command.has_table, command.table_heights, torch.zeros_like(command.table_heights))
    wrist_rh = command.right_hand.data.root_pos_w
    wrist_lh = command.left_hand.data.root_pos_w
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for obj, active in (
        (command.object_rh, command.object_active_rh),
        (command.object_lh, command.object_active_lh),
    ):
        pos = obj.data.root_pos_w
        distance = torch.minimum(
            torch.linalg.vector_norm(pos - wrist_rh, dim=-1),
            torch.linalg.vector_norm(pos - wrist_lh, dim=-1),
        )
        fell = pos[:, 2] < support_height - support_tolerance
        flew = distance > flyaway_distance
        dropped |= active & established & (command.attach_steps_remaining == 0) & (fell | flew)
    command.metrics["object_drop"][:] = dropped.float()
    return dropped
