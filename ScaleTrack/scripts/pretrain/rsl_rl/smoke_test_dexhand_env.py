#!/usr/bin/env python3
"""Create, reset, and step the dexhand environment with a tiny manifest."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--task", default="ScaleTrack-DexHand-RLScratch-v0")
parser.add_argument("--split", default="train")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import scaletrack.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.configure_manifest(args.manifest, split=args.split, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    try:
        observation, _ = env.reset()
        command = env.unwrapped.command_manager.get_term("motion")
        reset_joint_error = max(
            float(torch.abs(command.joint_pos("rh") - command.q_rh).max()),
            float(torch.abs(command.joint_pos("lh") - command.q_lh).max()),
        )
        if reset_joint_error > 1.0e-5:
            raise RuntimeError(
                f"canonical joint reset is misaligned: max error {reset_joint_error:.6g} rad"
            )
        reset_tip_errors = []
        for hand, tip_ids, reference, active in (
            (
                command.right_hand,
                command.right_tip_ids,
                command.fingertip_pos_rh,
                command.hand_active_rh,
            ),
            (
                command.left_hand,
                command.left_tip_ids,
                command.fingertip_pos_lh,
                command.hand_active_lh,
            ),
        ):
            per_tip = torch.linalg.vector_norm(
                hand.data.body_pos_w[:, tip_ids] - reference,
                dim=-1,
            )
            if active.any():
                reset_tip_errors.append(per_tip[active])
        reset_tip_errors = torch.cat(reset_tip_errors)
        reset_tip_mean = float(reset_tip_errors.mean())
        reset_tip_max = float(reset_tip_errors.max())
        expected = {"policy": (args.num_envs, 11, 90), "policy_task": (args.num_envs, 20, 42)}
        for name, shape in expected.items():
            actual = tuple(observation[name].shape)
            if actual != shape:
                raise RuntimeError(f"{name}: expected shape {shape}, got {actual}")
        if observation["critic"].shape[-1] != 80:
            raise RuntimeError(f"critic: expected 80 features, got {tuple(observation['critic'].shape)}")
        if observation["bc_target"].shape[-1] != 24:
            raise RuntimeError(
                f"bc_target: expected 24 features, got {tuple(observation['bc_target'].shape)}"
            )
        table_error = 0.0
        if command.has_table.any():
            table_ids = torch.nonzero(command.has_table, as_tuple=False).squeeze(-1)
            expected_table = env.unwrapped.scene.env_origins[table_ids].clone()
            expected_table[:, 0] += command.table_center_x[table_ids]
            expected_table[:, 2] = (
                command.table_heights[table_ids] - command.cfg.table_thickness * 0.5
            )
            table_error = float(
                torch.linalg.vector_norm(
                    command.table.data.root_pos_w[table_ids] - expected_table, dim=-1
                ).max()
            )
            if table_error > 1.0e-5:
                raise RuntimeError(f"DexManipNet table placement error: {table_error:.6g} m")
        scene_object_error = 0.0
        for obj, active, state in zip(
            command.scene_objects, command.scene_object_active, command.scene_object_states
        ):
            if active.any():
                active_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
                expected = state[active_ids, :3] + env.unwrapped.scene.env_origins[active_ids]
                scene_object_error = max(
                    scene_object_error,
                    float(
                        torch.linalg.vector_norm(
                            obj.data.root_pos_w[active_ids] - expected, dim=-1
                        ).max()
                    ),
                )
        if scene_object_error > 1.0e-5:
            raise RuntimeError(f"FAVOR scene object placement error: {scene_object_error:.6g} m")
        for _ in range(args.steps):
            observation, _, _, _, _ = env.step(torch.zeros(args.num_envs, 24, device=env.unwrapped.device))
            for name, value in observation.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"{name}: contains non-finite values")
        wrist_error = max(
            float(
                torch.linalg.vector_norm(
                    command.right_hand.data.root_pos_w - command.wrist_state_rh[:, :3], dim=-1
                ).max()
            ),
            float(
                torch.linalg.vector_norm(
                    command.left_hand.data.root_pos_w - command.wrist_state_lh[:, :3], dim=-1
                ).max()
            ),
        )
        if wrist_error > 0.05:
            raise RuntimeError(f"driven wrist root error is too large: {wrist_error:.6f} m")
        print(
            "DEXHAND_ENV_SMOKE_OK "
            f"steps={args.steps} actor={tuple(observation['policy'].shape)} "
            f"targets={tuple(observation['policy_task'].shape)} critic={tuple(observation['critic'].shape)} "
            f"reset_joint_error_rad={reset_joint_error:.3g} wrist_error_m={wrist_error:.6f} "
            f"reset_tip_mean_m={reset_tip_mean:.6f} reset_tip_max_m={reset_tip_max:.6f} "
            f"table_error_m={table_error:.3g} scene_object_error_m={scene_object_error:.3g} "
            f"right_contact_bodies={tuple(env.unwrapped.scene['right_tip_contact'].body_names)} "
            f"motion_ids={command.motion_ids.tolist()}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
