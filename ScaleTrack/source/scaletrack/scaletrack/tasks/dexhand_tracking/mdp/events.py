"""Reset-time physics randomization and per-step kinematic reference driving."""

from __future__ import annotations

import torch

from .commands import DexHandMotionCommand


def drive_wrist_roots_and_soft_attach(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    disable_flag: bool = False,
) -> None:
    if disable_flag:
        return
    command: DexHandMotionCommand = env.command_manager.get_term(command_name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    command._write_wrist_roots(env_ids)
    attached = env_ids[command.attach_steps_remaining[env_ids] > 0]
    if len(attached):
        command._write_objects(attached)
        command.attach_steps_remaining[attached] -= 1


def randomize_object_properties(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Set recorded mass/friction with the plan's sole ±20% domain randomization."""

    command: DexHandMotionCommand = env.command_manager.get_term(command_name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids_cpu = env_ids.cpu()
    low, high = scale_range
    # This IsaacLab version cannot attach a material directly to UsdFileCfg.
    # Match DexHandManipBiH's hand friction by writing every imported hand
    # collider through the PhysX tensor view instead.
    for hand in (command.right_hand, command.left_hand):
        materials = hand.root_physx_view.get_material_properties()
        materials[env_ids_cpu, :, 0] = 4.0
        materials[env_ids_cpu, :, 1] = 4.0
        materials[env_ids_cpu, :, 2] = 0.0
        hand.root_physx_view.set_material_properties(materials, env_ids_cpu)

    for obj, nominal_mass, nominal_friction in (
        (command.object_rh, command.object_mass_rh, command.object_friction_rh),
        (command.object_lh, command.object_mass_lh, command.object_friction_lh),
    ):
        scale = torch.empty(len(env_ids_cpu), device="cpu").uniform_(low, high)
        masses = obj.root_physx_view.get_masses()
        old_mass = masses[env_ids_cpu, 0].clone().clamp_min(1.0e-6)
        new_mass = nominal_mass[env_ids].cpu() * scale
        masses[env_ids_cpu, 0] = new_mass
        obj.root_physx_view.set_masses(masses, env_ids_cpu)
        inertias = obj.root_physx_view.get_inertias()
        inertias[env_ids_cpu] *= (new_mass / old_mass)[:, None]
        obj.root_physx_view.set_inertias(inertias, env_ids_cpu)

        materials = obj.root_physx_view.get_material_properties()
        friction = nominal_friction[env_ids].cpu() * torch.empty(len(env_ids_cpu), device="cpu").uniform_(low, high)
        materials[env_ids_cpu, :, 0] = friction[:, None]
        materials[env_ids_cpu, :, 1] = friction[:, None]
        materials[env_ids_cpu, :, 2] = 0.0
        obj.root_physx_view.set_material_properties(materials, env_ids_cpu)
