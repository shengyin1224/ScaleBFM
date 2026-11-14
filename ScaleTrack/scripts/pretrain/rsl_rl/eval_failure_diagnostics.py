"""Evaluate a manifest and record first-failure timing and per-link errors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-BFM-Transformer-Tracking")
parser.add_argument("--motion_file", required=True)
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode_index", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=600)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--failure_threshold_m", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

import scaletrack.tasks  # noqa: F401


def link_groups(body_names: list[str]) -> dict[str, list[int]]:
    groups = {
        "root_legs": [],
        "torso": [],
        "arms_without_wrists": [],
        "wrists": [],
    }
    for index, name in enumerate(body_names):
        if name == "pelvis" or any(token in name for token in ("hip", "knee", "ankle")):
            groups["root_legs"].append(index)
        elif name == "torso_link":
            groups["torso"].append(index)
        elif "wrist" in name:
            groups["wrists"].append(index)
        else:
            groups["arms_without_wrists"].append(index)
    if sorted(index for values in groups.values() for index in values) != list(range(len(body_names))):
        raise RuntimeError(f"Link grouping is not exhaustive: {body_names}")
    return groups


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    checkpoint = Path(args_cli.checkpoint_path).resolve()
    motion_file = Path(args_cli.motion_file).resolve()
    output = Path(args_cli.output).resolve()
    for path in (checkpoint, motion_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    env_cfg.seed = agent_cfg.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.enable_reset_disturbance = False
    env_cfg.commands.motion.debug_vis = False
    mode_candidates = list(env_cfg.commands.motion.mode_candidates.items())
    if not 0 <= args_cli.mode_index < len(mode_candidates):
        raise ValueError(f"Invalid mode_index={args_cli.mode_index}")
    mode_name, active_links = mode_candidates[args_cli.mode_index]
    env_cfg.commands.motion.mode_candidates = {mode_name: active_links}
    env_cfg.commands.motion.mode_sampling_weights = {mode_name: 1.0}

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    runner._set_env_is_evaluating()

    actor_critic = runner.alg.policy
    get_actor_obs = actor_critic.get_actor_obs

    def get_fixed_mode_actor_obs(obs, inference=False):
        return get_actor_obs(obs, inference=False)

    actor_critic.get_actor_obs = get_fixed_mode_actor_obs
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command = env.unwrapped.command_manager.get_term("motion")
    num_motions = int(command.num_motion)
    if args_cli.num_envs < num_motions:
        raise ValueError(
            f"This diagnostic requires num_envs >= motions ({args_cli.num_envs} < {num_motions})"
        )
    motion_ids = torch.arange(num_motions, device=command.motion_ids.device)
    command.motion_ids[:num_motions] = motion_ids
    if args_cli.num_envs > num_motions:
        command.motion_ids[num_motions:] = 0
    lengths = command.time_totals[motion_ids].long().clamp(max=args_cli.max_steps)
    max_length = int(lengths.max().item())
    body_names = list(command.cfg.body_names)
    groups = link_groups(body_names)
    count = len(body_names)

    first_failure = np.full(num_motions, -1, dtype=np.int64)
    first_failure_link_error = np.full((num_motions, count), np.nan, dtype=np.float32)
    peak_mean = np.zeros(num_motions, dtype=np.float32)
    peak_mean_step = np.full(num_motions, -1, dtype=np.int64)
    peak_link_error = np.zeros((num_motions, count), dtype=np.float32)
    peak_link_step = np.full((num_motions, count), -1, dtype=np.int64)
    elapsed = torch.zeros(num_motions, dtype=torch.long, device=lengths.device)

    obs, _ = env.reset()
    for _ in range(max_length):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        elapsed += 1
        active = elapsed < lengths
        if not bool(active.any()):
            break
        errors = torch.linalg.vector_norm(
            command.body_pos_w[:num_motions] - command.robot_body_pos_w[:num_motions],
            dim=-1,
        )
        means = errors.mean(dim=-1)
        active_cpu = active.detach().cpu().numpy()
        means_cpu = means.detach().cpu().numpy()
        errors_cpu = errors.detach().cpu().numpy()
        steps_cpu = elapsed.detach().cpu().numpy()

        improved = active_cpu & (means_cpu > peak_mean)
        peak_mean[improved] = means_cpu[improved]
        peak_mean_step[improved] = steps_cpu[improved]
        for link_index in range(count):
            link_improved = active_cpu & (errors_cpu[:, link_index] > peak_link_error[:, link_index])
            peak_link_error[link_improved, link_index] = errors_cpu[link_improved, link_index]
            peak_link_step[link_improved, link_index] = steps_cpu[link_improved]

        newly_failed = active_cpu & (first_failure < 0) & (means_cpu > args_cli.failure_threshold_m)
        first_failure[newly_failed] = steps_cpu[newly_failed]
        first_failure_link_error[newly_failed] = errors_cpu[newly_failed]

    motion_names = list(command.motion_names)
    fps = 50.0
    entries = []
    timing = Counter()
    dominant_groups = Counter()
    dominant_links = Counter()
    for motion_id, name in enumerate(motion_names):
        failed = bool(first_failure[motion_id] >= 0)
        if failed:
            step = int(first_failure[motion_id])
            if step <= int(fps):
                timing_bucket = "early_le_1s"
            else:
                timing_bucket = "mid_trajectory_gt_1s"
            timing[timing_bucket] += 1
            group_values = {
                group: float(np.mean(first_failure_link_error[motion_id, indices]))
                for group, indices in groups.items()
            }
            dominant_group = max(group_values, key=group_values.get)
            dominant_groups[dominant_group] += 1
            dominant_link_index = int(np.argmax(first_failure_link_error[motion_id]))
            dominant_link = body_names[dominant_link_index]
            dominant_links[dominant_link] += 1
        else:
            step = None
            timing_bucket = "passed"
            group_values = None
            dominant_group = None
            dominant_link = None
        entries.append(
            {
                "motion_id": motion_id,
                "motion": name,
                "evaluated_transitions": int(max(int(lengths[motion_id].item()) - 1, 0)),
                "failed": failed,
                "first_failure_transition": step,
                "first_failure_time_s": None if step is None else step / fps,
                "failure_timing_bucket": timing_bucket,
                "dominant_group_at_first_failure": dominant_group,
                "group_mean_error_at_first_failure_m": group_values,
                "dominant_link_at_first_failure": dominant_link,
                "link_error_at_first_failure_m": (
                    None
                    if not failed
                    else {
                        body_names[index]: float(first_failure_link_error[motion_id, index])
                        for index in range(count)
                    }
                ),
                "peak_14link_mean_error_m": float(peak_mean[motion_id]),
                "peak_14link_mean_transition": int(peak_mean_step[motion_id]),
                "per_link_peak_error_m": {
                    body_names[index]: float(peak_link_error[motion_id, index])
                    for index in range(count)
                },
            }
        )

    report = {
        "simulator": "IsaacLab",
        "checkpoint": str(checkpoint),
        "motion_file": str(motion_file),
        "mode": {"index": args_cli.mode_index, "name": mode_name, "active_links": active_links},
        "motions": num_motions,
        "max_steps_per_motion": args_cli.max_steps,
        "failure_definition": (
            f"first evaluated transition where mean {count}-link global position error "
            f"> {args_cli.failure_threshold_m} m"
        ),
        "body_names": body_names,
        "link_groups": {name: [body_names[index] for index in indices] for name, indices in groups.items()},
        "failed_motion_count": int(np.sum(first_failure >= 0)),
        "failure_timing": dict(timing),
        "dominant_group_at_first_failure": dict(dominant_groups),
        "dominant_link_at_first_failure": dict(dominant_links),
        "motions_detail": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"),
        motion_names=np.asarray(motion_names),
        body_names=np.asarray(body_names),
        first_failure_transition=first_failure,
        first_failure_link_error_m=first_failure_link_error,
        peak_14link_mean_error_m=peak_mean,
        peak_14link_mean_transition=peak_mean_step,
        per_link_peak_error_m=peak_link_error,
        per_link_peak_transition=peak_link_step,
    )
    print(output)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
