"""Evaluate one checkpoint on a complete motion manifest with a fixed link mode."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-BFM-Transformer-Tracking")
parser.add_argument("--motion_file", required=True)
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode_index", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

import scaletrack.tasks  # noqa: F401


BASE_METRICS = [
    "error_anchor_height",
    "error_anchor_rot",
    "error_anchor_pos",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos_g",
    "error_body_pos",
    "error_body_pos_relative",
    "error_body_rot",
    "error_body_rot_relative",
    "error_joint_pos",
    "error_joint_vel",
]
FOCUS_METRICS = [
    "error_anchor_pos",
    "cosine_pelvis_left_wrist_global_pos_error",
    "cosine_pelvis_right_wrist_global_pos_error",
    "error_left_wrist_global_pos",
    "error_left_wrist_global_rot",
    "error_right_wrist_global_pos",
    "error_right_wrist_global_rot",
    "error_head_proxy_global_pos",
    "error_head_proxy_global_rot",
    "error_left_wrist_pelvis_local_pos",
    "error_left_wrist_pelvis_local_rot",
    "error_right_wrist_pelvis_local_pos",
    "error_right_wrist_pelvis_local_rot",
    "error_head_proxy_pelvis_local_pos",
    "error_head_proxy_pelvis_local_rot",
]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    checkpoint = Path(args_cli.checkpoint_path).resolve()
    motion_file = Path(args_cli.motion_file).resolve()
    output = Path(args_cli.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not motion_file.is_file():
        raise FileNotFoundError(motion_file)

    env_cfg.seed = agent_cfg.seed
    # Keep IsaacLab tensors, the simulator, and the policy on the device chosen
    # by AppLauncher.  Without this override, `--device cuda:N` launches Kit on
    # N but leaves the environment and runner on their config default (cuda:0).
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.enable_reset_disturbance = False
    env_cfg.commands.motion.debug_vis = False

    mode_candidates = list(env_cfg.commands.motion.mode_candidates.items())
    if not 0 <= args_cli.mode_index < len(mode_candidates):
        raise ValueError(
            f"mode_index must be in [0, {len(mode_candidates) - 1}], "
            f"got {args_cli.mode_index}"
        )
    mode_name, active_link_names = mode_candidates[args_cli.mode_index]
    env_cfg.commands.motion.mode_candidates = {mode_name: active_link_names}
    env_cfg.commands.motion.mode_sampling_weights = {mode_name: 1.0}

    agent_cfg.eval_metric_keys = BASE_METRICS + FOCUS_METRICS
    agent_cfg.eval_focus_metric_keys = FOCUS_METRICS
    agent_cfg.eval_max_steps = 1000
    agent_cfg.success_metric_dict = {"error_body_pos_g": 0.5}

    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir = output.parent / f"{output.stem}_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(runtime_dir),
        device=agent_cfg.device,
    )
    runner.load(str(checkpoint))

    # ScaleBFM's inference observation path intentionally replaces the sampled
    # sparse-link mode with the complete 14-link mask.  For a fixed-mode
    # evaluation we must use the training observation path; otherwise different
    # mode_index values silently evaluate the same 14-link policy input.
    actor_critic = runner.alg.policy
    get_actor_obs = actor_critic.get_actor_obs

    def get_fixed_mode_actor_obs(obs, inference=False):
        return get_actor_obs(obs, inference=False)

    actor_critic.get_actor_obs = get_fixed_mode_actor_obs

    with torch.inference_mode():
        metrics = runner.evaluate_policy(test_set=False)

    command = env.unwrapped.command_manager.get_term("motion")
    failed_path = runtime_dir / "failed_train_motions.txt"
    failed_motions = (
        [line for line in failed_path.read_text().splitlines() if line]
        if failed_path.is_file()
        else []
    )
    report = {
        "simulator": "IsaacLab",
        "checkpoint": str(checkpoint),
        "motion_file": str(motion_file),
        "motions": int(command.num_motion),
        "eval_max_steps_per_motion": agent_cfg.eval_max_steps,
        "mode": {
            "index": args_cli.mode_index,
            "name": mode_name,
            "active_links": active_link_names,
        },
        "success_definition": (
            "max over evaluated frames of mean 14-link global position "
            "error <= 0.5 m"
        ),
        "metrics": metrics,
        "failed_motion_count": len(failed_motions),
        "failed_motions_file": str(failed_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
