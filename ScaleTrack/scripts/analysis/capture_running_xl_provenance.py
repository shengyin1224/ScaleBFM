#!/usr/bin/env python3
"""Capture immutable evidence for the already-running distributed XL job.

This intentionally reads the live process tree instead of reconstructing the
launch command from a report.  Source files are accepted as launch-time
evidence only when their mtime predates the runner-generated launch-time git
diff.  The corrected runner no longer emits the formerly off-by-one
``model_0.pt`` after its first PPO update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


TRACK = Path(__file__).resolve().parents[2]
DEFAULT_RUN = TRACK / "logs/rsl_rl/g1_hat4_focus_finetune/formal_xl_f3h_20260804"
DEFAULT_BASE = (
    TRACK / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_xl/model_22200.pt"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "run_provenance.json"
SOURCE_PATHS = (
    "scripts/analysis/run_xl_f3h_detached.sh",
    "scripts/pretrain/rsl_rl/train.py",
    "source/my_rsl_rl/my_rsl_rl/algorithms/ppo.py",
    "source/my_rsl_rl/my_rsl_rl/modules/actor_critic_humanoid_transformer.py",
    "source/my_rsl_rl/my_rsl_rl/runners/on_policy_runner.py",
    "source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/__init__.py",
    "source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/hat4_finetune_env_cfg.py",
    "source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/agents/rsl_rl_ppo_cfg.py",
    "source/scaletrack/scaletrack/tasks/tracking/mdp/commands.py",
    "source/scaletrack/scaletrack/tasks/tracking/mdp/observations.py",
    "source/scaletrack/scaletrack/tasks/tracking/mdp/rewards.py",
)
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proc_cmdline(pid: int) -> list[str]:
    return [
        value.decode(errors="replace")
        for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if value
    ]


def proc_environment(pid: int) -> dict[str, str]:
    values = {}
    for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        key_string = key.decode(errors="replace")
        if key_string in ENV_KEYS:
            values[key_string] = value.decode(errors="replace")
    return values


def manifest_evidence(path: Path) -> dict:
    manifest = yaml.safe_load(path.read_text())
    referenced = [Path(value) for value in manifest.values()]
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "motions": len(manifest),
        "all_references_exist": all(item.is_file() for item in referenced),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    start_diff = run_dir / "git/ScaleBFM.diff"
    for required in (start_diff, args.base_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)

    children_file = Path(f"/proc/{args.leader_pid}/task/{args.leader_pid}/children")
    if not children_file.is_file():
        raise ProcessLookupError(args.leader_pid)
    children = sorted(int(value) for value in children_file.read_text().split())
    ranks = []
    for pid in children:
        environment = proc_environment(pid)
        ranks.append(
            {
                "pid": pid,
                "cmdline": proc_cmdline(pid),
                "environment": environment,
            }
        )

    launch_evidence_mtime_ns = start_diff.stat().st_mtime_ns
    source_snapshots = []
    for relative in SOURCE_PATHS:
        path = TRACK / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_snapshots.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "mtime_ns": path.stat().st_mtime_ns,
                "unchanged_before_launch_evidence": path.stat().st_mtime_ns
                <= launch_evidence_mtime_ns,
            }
        )

    train_manifest = TRACK / "data/hat4_unifolm/manifests/unifolm_hat4_train.yaml"
    heldout_manifest = TRACK / "data/hat4_unifolm/manifests/unifolm_hat4_heldout.yaml"
    launch_log = TRACK / "logs/launch_logs/formal_xl_f3h_20260804.log"
    if not launch_log.is_file():
        raise FileNotFoundError(launch_log)
    launch_log_text = launch_log.read_text(errors="replace")
    # The same append-only log may contain a failed initialization attempt.
    # Validate only the most recent systemd launch segment while preserving the
    # previous segment as evidence.
    launch_attempt_count = launch_log_text.count("launch_mode=")
    launch_segment = launch_log_text.rsplit("launch_mode=", maxsplit=1)[-1]
    warm_start_path_line = (
        "[INFO]: Warm-starting model only (optimizer and iteration are NOT restored) "
        f"from: {args.base_checkpoint.resolve()}"
    )
    zero_expanded_line = (
        "[INFO]: Warm start complete; iteration=0; "
        "zero-expanded=['actor_task_embedder.task_projection.weight', "
        "'critic_task_embedder.task_projection.weight']"
    )
    warm_start_log_valid = (
        launch_segment.count(warm_start_path_line) == 5
        and launch_segment.count(zero_expanded_line) == 5
    )
    rank_environments = [row["environment"] for row in ranks]
    child_commands = [row["cmdline"] for row in ranks]
    required_args = {
        "--distributed",
        "--task",
        "G1-HAT4-F3H-Finetune",
        "--num_envs",
        "819",
        "--max_iterations",
        "20000",
        "--motion_file",
        str(train_manifest.resolve()),
        "--test_motion_file",
        str(heldout_manifest.resolve()),
        "--pretrained_checkpoint",
        str(args.base_checkpoint.resolve()),
        "agent.policy.embedding_dim=384",
        "agent.policy.num_heads=6",
        "agent.policy.ff_dim=384",
        "agent.policy.num_layers=6",
    }
    process_tree_valid = (
        len(ranks) == 5
        and {int(env["RANK"]) for env in rank_environments} == set(range(5))
        and {int(env["LOCAL_RANK"]) for env in rank_environments} == set(range(5))
        and {env["WORLD_SIZE"] for env in rank_environments} == {"5"}
        and {env["CUDA_VISIBLE_DEVICES"] for env in rank_environments}
        == {"0,1,2,3,5"}
        and all(required_args.issubset(command) for command in child_commands)
    )
    source_snapshots_valid = all(
        row["unchanged_before_launch_evidence"] for row in source_snapshots
    )
    leader_cgroup = Path(f"/proc/{args.leader_pid}/cgroup").read_text()
    detached_service_valid = "scalebfm-xl-f3h-20260804.service" in leader_cgroup
    report = {
        "capture_time_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "leader": {
            "pid": args.leader_pid,
            "cmdline": proc_cmdline(args.leader_pid),
            "cgroup": leader_cgroup.strip(),
        },
        "detached_systemd_service_valid": detached_service_valid,
        "ranks": ranks,
        "process_tree_valid": process_tree_valid,
        "source_snapshots": source_snapshots,
        "source_snapshots_valid": source_snapshots_valid,
        "launch_time_git_diff": {
            "path": str(start_diff),
            "sha256": sha256(start_diff),
            "size_bytes": start_diff.stat().st_size,
        },
        "warm_start_log": {
            "path": str(launch_log),
            "attempts_in_append_only_log": launch_attempt_count,
            "official_checkpoint_line_count": launch_segment.count(warm_start_path_line),
            "iteration_zero_and_zero_expand_line_count": launch_segment.count(
                zero_expanded_line
            ),
            "zero_expanded_keys": [
                "actor_task_embedder.task_projection.weight",
                "critic_task_embedder.task_projection.weight",
            ],
            "valid": warm_start_log_valid,
            "note": (
                "Counts refer to the latest launch segment. The corrected runner "
                "does not emit an off-by-one model_0 checkpoint."
            ),
        },
        "official_initialization": {
            "path": str(args.base_checkpoint.resolve()),
            "sha256": sha256(args.base_checkpoint),
            "size_bytes": args.base_checkpoint.stat().st_size,
        },
        "launch_evidence_cutoff": {
            "path": str(start_diff),
            "mtime_ns": launch_evidence_mtime_ns,
            "model_0_emitted": False,
            "reason": "periodic checkpoint names now equal completed PPO updates",
        },
        "manifests": {
            "train": manifest_evidence(train_manifest),
            "heldout": manifest_evidence(heldout_manifest),
        },
        "recipe_interpretation": {
            "task": "G1-HAT4-F3H-Finetune",
            "base_seed": 42,
            "rank_seeds": [42, 43, 44, 45, 46],
            "distributed_ranks": 5,
            "environments_per_rank": 819,
            "global_environments": 4095,
            "iterations": 20000,
            "architecture": {
                "embedding_dim": 384,
                "num_heads": 6,
                "ff_dim": 384,
                "num_layers": 6,
            },
        },
        "all_valid": process_tree_valid
        and detached_service_valid
        and warm_start_log_valid
        and source_snapshots_valid
        and sha256(args.base_checkpoint)
        == "8a7be5191de8bbcd3f7359615b935ea605ee47d8703135dc4202fa9eeefc1b2f"
        and all(
            item["all_references_exist"]
            for item in (
                manifest_evidence(train_manifest),
                manifest_evidence(heldout_manifest),
            )
        ),
        "caveat": (
            "The launch-time git diff is the exact tracked-worktree snapshot. "
            "Source hashes are additionally accepted only because their mtimes predate the "
            "runner-generated launch diff; "
            "this avoids presenting post-launch edits as the code loaded by the running ranks."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    if not report["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
