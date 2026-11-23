#!/usr/bin/env python3
"""Cycle memory-safe static-asset shards while preserving PPO optimizer state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--iterations-per-shard", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    bc_checkpoint = args.bc_checkpoint.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not bc_checkpoint.is_file():
        raise FileNotFoundError(bc_checkpoint)
    if args.shard_count <= 0 or args.cycles <= 0:
        raise ValueError("--shard-count and --cycles must be positive")
    if args.iterations_per_shard <= 0 or args.num_envs <= 0:
        raise ValueError("--iterations-per-shard and --num-envs must be positive")

    manifest_sha256 = _sha256(manifest)
    bc_sha256 = _sha256(bc_checkpoint)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("manifest") != str(manifest):
            raise ValueError("state manifest path does not match")
        if state.get("manifest_sha256") != manifest_sha256:
            raise ValueError("state manifest content hash does not match")
        if state.get("bc_checkpoint") != str(bc_checkpoint):
            raise ValueError("state BC checkpoint path does not match")
        if state.get("bc_checkpoint_sha256") != bc_sha256:
            raise ValueError("state BC checkpoint content hash does not match")
        expected_settings = {
            "shard_count": args.shard_count,
            "cycles": args.cycles,
            "iterations_per_shard": args.iterations_per_shard,
            "num_envs": args.num_envs,
            "seed": args.seed,
        }
        for key, value in expected_settings.items():
            if state.get(key) != value:
                raise ValueError(
                    f"state {key}={state.get(key)!r} does not match "
                    f"requested {value!r}"
                )
        segment = int(state["next_segment"])
        resume_checkpoint = Path(state["resume_checkpoint"]).resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
    else:
        segment = 0
        resume_checkpoint = None
        state = {
            "schema_version": 1,
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha256,
            "bc_checkpoint": str(bc_checkpoint),
            "bc_checkpoint_sha256": bc_sha256,
            "shard_count": args.shard_count,
            "cycles": args.cycles,
            "iterations_per_shard": args.iterations_per_shard,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "next_segment": 0,
            "resume_checkpoint": None,
            "completed": False,
        }
        _write_state(state_path, state)

    total_segments = args.shard_count * args.cycles
    train_script = Path(__file__).with_name("train_dexhand.py")
    repo_root = Path(__file__).resolve().parents[3]
    while segment < total_segments:
        cycle = segment // args.shard_count
        shard = segment % args.shard_count
        run_name = f"{args.run_prefix}_c{cycle:03d}_s{shard:02d}"
        command = [
            sys.executable,
            "-u",
            str(train_script),
            "--manifest",
            str(manifest),
            "--split",
            "train",
            "--task",
            "ScaleTrack-DexHand-GMT-v0",
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.iterations_per_shard),
            "--manifest-shard-count",
            str(args.shard_count),
            "--manifest-shard-index",
            str(shard),
            "--manifest-shard-seed",
            str(args.seed),
            "--bc_checkpoint",
            str(bc_checkpoint),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--run_name",
            run_name,
            "--headless",
        ]
        if resume_checkpoint is not None:
            command.extend(("--resume_checkpoint", str(resume_checkpoint)))
        print(
            f"[sharded-train] segment={segment}/{total_segments} "
            f"cycle={cycle} shard={shard} run={run_name}",
            flush=True,
        )
        subprocess.run(command, cwd=repo_root, check=True)

        expected_iteration = (segment + 1) * args.iterations_per_shard
        resume_checkpoint = (
            repo_root
            / "logs"
            / "rsl_rl"
            / "dexhand_gmt"
            / run_name
            / f"model_{expected_iteration}.pt"
        ).resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"training completed but final checkpoint is missing: "
                f"{resume_checkpoint}"
            )
        segment += 1
        state.update(
            {
                "next_segment": segment,
                "resume_checkpoint": str(resume_checkpoint),
                "resume_checkpoint_sha256": _sha256(resume_checkpoint),
                "completed": segment == total_segments,
            }
        )
        _write_state(state_path, state)

    print(
        f"[sharded-train] complete segments={total_segments} "
        f"checkpoint={resume_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
