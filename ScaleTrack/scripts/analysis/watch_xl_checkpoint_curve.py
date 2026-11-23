#!/usr/bin/env python3
"""Evaluate stable XL-F3h intermediate checkpoints in fixed-HAT4 mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path


TRACK = Path(__file__).resolve().parents[2]
RUN = TRACK / "logs/rsl_rl/g1_hat4_focus_finetune/formal_xl_f3h_20260804"
OUTPUT = TRACK / "action_error_results/official_xl_hat4/checkpoints"
ARTIFACTS = OUTPUT / "artifacts"
MANIFEST = TRACK / "data/hat4_unifolm/manifests/unifolm_hat4_heldout.yaml"
EVAL = TRACK / "scripts/pretrain/rsl_rl/eval_manifest_tracking.py"
PYTHON = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
STEPS = (2000, 8000, 12000, 16000)
METRICS = (
    "error_left_wrist_pelvis_local_pos_focus_success",
    "error_left_wrist_pelvis_local_rot_focus_success",
    "error_right_wrist_pelvis_local_pos_focus_success",
    "error_right_wrist_pelvis_local_rot_focus_success",
    "error_head_proxy_pelvis_local_pos_focus_success",
    "error_head_proxy_pelvis_local_rot_focus_success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def wait_until_stable(path: Path, poll_seconds: float) -> None:
    previous = -1
    while True:
        size = path.stat().st_size if path.is_file() else 0
        if size > 0 and size == previous:
            return
        previous = size
        print(f"waiting for {path.name}: size={size}", flush=True)
        time.sleep(poll_seconds)


def convert(key: str, value: float) -> float:
    return value * 100.0 if "_pos_" in key else math.degrees(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_checkpoint(source: Path, step: int, poll_seconds: float) -> Path:
    """Freeze a milestone before the legacy runner can overwrite its filename.

    The live runner writes ``model_N`` after update N for its internal evaluation,
    but its legacy periodic-save branch writes the same name once more after
    update N+1.  A hash-verified immutable copy makes the curve reproducible and
    removes the race between that overwrite and evaluator startup.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACTS / f"model_{step}.pt"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    partial = ARTIFACTS / f"model_{step}.pt.partial"
    while True:
        wait_until_stable(source, poll_seconds)
        source_size = source.stat().st_size
        source_hash = sha256(source)
        shutil.copy2(source, partial)
        copied_hash = sha256(partial)
        # Re-hash the live path so an overwrite during copy cannot pass.
        unchanged = (
            source.is_file()
            and source.stat().st_size == source_size
            and sha256(source) == source_hash
        )
        if unchanged and copied_hash == source_hash:
            os.replace(partial, destination)
            print(
                f"snapshotted {source.name}: sha256={source_hash}",
                flush=True,
            )
            return destination
        print(f"checkpoint changed while snapshotting {source.name}; retrying", flush=True)
        time.sleep(poll_seconds)


def write_summary() -> None:
    rows = []
    for step in STEPS:
        path = OUTPUT / f"model_{step}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        snapshot = ARTIFACTS / f"model_{step}.pt"
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        rows.append(
            {
                "step": step,
                "passed": report["motions"] - report["failed_motion_count"],
                "total": report["motions"],
                "checkpoint_snapshot": str(snapshot.resolve()),
                "checkpoint_size_bytes": snapshot.stat().st_size,
                "checkpoint_sha256": sha256(snapshot),
                **{
                    key: convert(key, float(report["metrics"][key]))
                    for key in METRICS
                },
            }
        )
    (OUTPUT / "curve_summary.json").write_text(
        json.dumps(
            {
                "scope": "IsaacLab fixed-HAT4, successful-motion FOCUS means",
                "position_unit": "cm",
                "rotation_unit": "deg",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    for required in (MANIFEST, EVAL, PYTHON):
        if not required.is_file():
            raise FileNotFoundError(required)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    environment["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

    for step in STEPS:
        checkpoint = RUN / f"model_{step}.pt"
        result = OUTPUT / f"model_{step}.json"
        if result.is_file():
            write_summary()
            continue
        checkpoint_snapshot = snapshot_checkpoint(
            checkpoint, step, args.poll_seconds
        )
        command = [
            str(PYTHON),
            str(EVAL),
            "--task",
            "G1-HAT4-F3H-Finetune",
            "--motion_file",
            str(MANIFEST),
            "--checkpoint_path",
            str(checkpoint_snapshot),
            "--output",
            str(result),
            "--mode_index",
            "8",
            "--num_envs",
            "600",
            "--device",
            f"cuda:{args.physical_gpu}",
            "--headless",
            "agent.policy.embedding_dim=384",
            "agent.policy.num_heads=6",
            "agent.policy.ff_dim=384",
            "agent.policy.num_layers=6",
        ]
        log_path = OUTPUT / f"model_{step}.log"
        with log_path.open("w") as log:
            subprocess.run(
                command,
                cwd=TRACK,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        write_summary()
        print(f"evaluated {checkpoint_snapshot} -> {result}", flush=True)


if __name__ == "__main__":
    main()
