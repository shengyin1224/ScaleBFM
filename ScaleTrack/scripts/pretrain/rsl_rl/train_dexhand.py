#!/usr/bin/env python3
"""Train dexhand GMT with static per-environment assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--split", default="train")
parser.add_argument("--task", default="ScaleTrack-DexHand-GMT-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--bc_checkpoint", default="")
parser.add_argument("--resume_checkpoint", default="")
parser.add_argument("--manifest-shard-count", type=int, default=1)
parser.add_argument("--manifest-shard-index", type=int, default=0)
parser.add_argument("--manifest-shard-seed", type=int, default=20260728)
parser.add_argument(
    "--bc_gate_report",
    default="",
    help="Online free-air gate JSON; defaults to <bc_checkpoint>.online_gate.json",
)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import scaletrack.tasks  # noqa: F401,E402
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.configure_manifest(
        args_cli.manifest,
        split=args_cli.split,
        num_envs=num_envs,
        manifest_shard_count=args_cli.manifest_shard_count,
        manifest_shard_index=args_cli.manifest_shard_index,
        manifest_shard_seed=args_cli.manifest_shard_seed,
    )
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{launcher.local_rank}"
        agent_cfg.device = f"cuda:{launcher.local_rank}"
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(launcher.local_rank)

    if "RLScratch" not in args_cli.task and not args_cli.bc_checkpoint:
        raise ValueError("BC-initialized dexhand runs require --bc_checkpoint; use RLScratch for the ablation")
    checkpoint = ""
    if args_cli.bc_checkpoint:
        checkpoint = os.path.abspath(args_cli.bc_checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        gate_path = Path(
            args_cli.bc_gate_report
            or (checkpoint + ".online_gate.json")
        ).expanduser().resolve()
        if not gate_path.is_file():
            raise FileNotFoundError(
                f"BC online gate report is required before PPO: {gate_path}"
            )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        expected_manifest = str(Path(args_cli.manifest).expanduser().resolve())
        expected_checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        expected_manifest_sha256 = hashlib.sha256(
            Path(expected_manifest).read_bytes()
        ).hexdigest()
        if gate.get("schema_version") != 2:
            raise ValueError(
                f"{gate_path}: expected gate schema_version=2 with content hashes"
            )
        if gate.get("checkpoint") != checkpoint:
            raise ValueError(
                f"{gate_path}: checkpoint mismatch: {gate.get('checkpoint')!r} != {checkpoint!r}"
            )
        if gate.get("manifest") != expected_manifest:
            raise ValueError(
                f"{gate_path}: manifest mismatch: {gate.get('manifest')!r} != {expected_manifest!r}"
            )
        if gate.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError(f"{gate_path}: checkpoint content hash mismatch")
        if gate.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError(f"{gate_path}: manifest content hash mismatch")
        if gate.get("threshold_mm") != 15.0 or not gate.get("passed"):
            raise RuntimeError(
                f"BC online gate did not pass: value={gate.get('value_mm')} mm, "
                f"threshold={gate.get('threshold_mm')} mm"
            )
        print(
            f"Verified BC online gate: {gate.get('value_mm'):.3f} mm "
            f"< {gate.get('threshold_mm'):.1f} mm ({gate_path})"
        )
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = agent_cfg.run_name or stamp
    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, run_name))
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if args_cli.resume_checkpoint:
        resume_checkpoint = os.path.abspath(args_cli.resume_checkpoint)
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(resume_checkpoint)
        runner.load(
            resume_checkpoint,
            load_optimizer=True,
            map_location=agent_cfg.device,
        )
    elif args_cli.bc_checkpoint:
        runner.load(checkpoint, load_optimizer=False, map_location=agent_cfg.device)
        runner.current_learning_iteration = 0
    runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=False)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
