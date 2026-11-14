#!/usr/bin/env python3
"""Wait for the formal XL-F3h checkpoint, then run the fixed-HAT4 final eval.

The watcher requires a checkpoint to have an unchanged, non-zero size across
two polls before handing it to PyTorch.  This avoids reading a partially
written multi-hundred-megabyte checkpoint.
"""

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
DEFAULT_RUN = (
    TRACK
    / "logs/rsl_rl/g1_hat4_focus_finetune/formal_xl_f3h_20260804"
)
DEFAULT_BASELINE = (
    TRACK
    / "action_error_results/official_xl_hat4/released_xl_model22200.json"
)
DEFAULT_OUTPUT = (
    TRACK
    / "action_error_results/official_xl_hat4/f3h_xl_model20000.json"
)
DEFAULT_RELEASE = TRACK / "releases/hat4_focus_release_xl_v1"
MANIFEST = TRACK / "data/hat4_unifolm/manifests/unifolm_hat4_heldout.yaml"
EVAL = TRACK / "scripts/pretrain/rsl_rl/eval_manifest_tracking.py"
PYTHON = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
FOCUS_METRICS = (
    "error_left_wrist_pelvis_local_pos_focus_success",
    "error_left_wrist_pelvis_local_rot_focus_success",
    "error_right_wrist_pelvis_local_pos_focus_success",
    "error_right_wrist_pelvis_local_rot_focus_success",
    "error_head_proxy_pelvis_local_pos_focus_success",
    "error_head_proxy_pelvis_local_rot_focus_success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "model_20000.pt")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--physical-gpu", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def wait_until_stable(path: Path, poll_seconds: float) -> int:
    previous_size = -1
    while True:
        size = path.stat().st_size if path.is_file() else 0
        if size > 0 and size == previous_size:
            return size
        previous_size = size
        print(f"waiting for stable checkpoint: {path} (size={size})", flush=True)
        time.sleep(poll_seconds)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_is_running(run_name: str) -> bool:
    """Return whether any live ScaleTrack training command owns this run."""
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            run_name in command
            and "scripts/pretrain/rsl_rl/train.py" in command
        ):
            return True
    return False


def wait_for_training_exit(run_name: str, poll_seconds: float) -> None:
    while training_is_running(run_name):
        print(f"waiting for training run to exit cleanly: {run_name}", flush=True)
        time.sleep(poll_seconds)


def snapshot_checkpoint(source: Path, destination: Path, poll_seconds: float) -> int:
    """Copy a stable post-training checkpoint and prove the copy is identical."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    while True:
        size = wait_until_stable(source, poll_seconds)
        source_hash = sha256(source)
        shutil.copy2(source, partial)
        copied_hash = sha256(partial)
        unchanged = (
            source.is_file()
            and source.stat().st_size == size
            and sha256(source) == source_hash
        )
        if unchanged and copied_hash == source_hash:
            os.replace(partial, destination)
            return size
        print("final checkpoint changed during snapshot; retrying", flush=True)
        time.sleep(poll_seconds)


def converted(metric: str, value: float) -> float:
    return value * 100.0 if "_pos_" in metric else math.degrees(value)


def write_release_docs(release_dir: Path, summary: dict) -> None:
    """Write a self-contained, machine-readable recipe and human release card."""
    fixed = summary["fixed_hat4"]
    values = fixed["focus_success_metrics"]
    recipe = {
        "release": "hat4_focus_release_xl_v1",
        "variant": "F3h",
        "task": summary["training"]["task"],
        "checkpoint": Path(summary["released_checkpoint"]).name,
        "sha256": summary["checkpoint_sha256"],
        "initialization": "official ScaleBFM humanoid_transformer_xl/model_22200.pt",
        "architecture": summary["architecture"],
        "distributed_training": summary["training"],
        "mode_sampling": {
            "HAT-4": 0.5,
            "each_of_eight_factory_modes": 0.0625,
        },
        "focus_reward_multiplier": 2.0,
        "focus_weighted_links": [
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "torso_link",
        ],
        "phase_observation_dims_per_token": 2,
        "heldout_fixed_hat4": {
            "baseline_passed": fixed["baseline_passed"],
            "candidate_passed": fixed["candidate_passed"],
            "total": fixed["total"],
            "focus_success_metrics": values,
        },
        "claim_scope": (
            "IsaacLab fixed-HAT4 precision and survival as measured; "
            "not a real-robot or general external-data claim"
        ),
    }
    (release_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

    metric_rows = []
    labels = {
        "error_left_wrist_pelvis_local_pos_focus_success": "left wrist position (cm)",
        "error_left_wrist_pelvis_local_rot_focus_success": "left wrist rotation (deg)",
        "error_right_wrist_pelvis_local_pos_focus_success": "right wrist position (cm)",
        "error_right_wrist_pelvis_local_rot_focus_success": "right wrist rotation (deg)",
        "error_head_proxy_pelvis_local_pos_focus_success": "head proxy position (cm)",
        "error_head_proxy_pelvis_local_rot_focus_success": "head proxy rotation (deg)",
    }
    for key in FOCUS_METRICS:
        row = values[key]
        metric_rows.append(
            f"| {labels[key]} | {row['baseline']:.4f} | "
            f"{row['candidate']:.4f} | {row['relative_change_percent']:+.2f}% |"
        )
    readme = f"""# hat4_focus_release_xl_v1

Official ScaleBFM XL initialization fine-tuned with the selected F3h recipe.

## Provenance and recipe

- Source checkpoint: `{summary['checkpoint']}`
- Released checkpoint: `model_20000.pt`
- SHA-256: `{summary['checkpoint_sha256']}`
- Task: `G1-HAT4-F3H-Finetune`
- Training: 20,000 iterations, 5 distributed ranks x 819 environments = 4,095 environments.
- Rank seeds: 42--46.
- HAT-4 sampling: 50%; each of eight factory modes: 6.25%.
- FOCUS multiplier: 2.0 on both wrists and the `torso_link` head proxy.
- Phase observation: `focus_flag` plus clipped time-to-edge per future token.
- XL architecture: embedding 384, 6 heads, FF 384, 6 layers.
- Input/output: actor task width 269; normalized 29-DoF action.

## Fixed-HAT4 evidence

Scope: 600 held-out motions, at most 1,000 transitions per motion, successful-motion
FOCUS pelvis-local means.

- Official XL baseline survival: {fixed['baseline_passed']}/{fixed['total']}.
- XL-F3h survival: {fixed['candidate_passed']}/{fixed['total']}.

| metric | official XL | XL-F3h | relative change |
| --- | ---: | ---: | ---: |
{chr(10).join(metric_rows)}

This release card reports measured IsaacLab results only. It does not claim real-robot
validation, and the two-motion SONIC audit is documented separately as insufficient
for a broad no-forgetting claim.
"""
    (release_dir / "README.md").write_text(readme)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    baseline_path = args.baseline.resolve()
    output = args.output.resolve()
    release_dir = args.release_dir.resolve()
    for required in (baseline_path, MANIFEST, EVAL, PYTHON):
        if not required.is_file():
            raise FileNotFoundError(required)

    wait_until_stable(checkpoint, args.poll_seconds)
    # model_20000 is first written before the runner's final internal
    # evaluation and written once more during shutdown.  Do not evaluate or
    # archive the live path until every rank has exited and the final write is
    # complete.
    wait_for_training_exit(checkpoint.parent.name, args.poll_seconds)
    release_dir.mkdir(parents=True, exist_ok=True)
    released_checkpoint = release_dir / "model_20000.pt"
    stable_size = snapshot_checkpoint(
        checkpoint, released_checkpoint, args.poll_seconds
    )
    checkpoint_sha256 = sha256(released_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    environment["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    command = [
        str(PYTHON),
        str(EVAL),
        "--task",
        "G1-HAT4-F3H-Finetune",
        "--motion_file",
        str(MANIFEST),
        "--checkpoint_path",
        str(released_checkpoint),
        "--output",
        str(output),
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
    subprocess.run(command, cwd=TRACK, env=environment, check=True)

    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(output.read_text())
    metrics = {}
    for key in FOCUS_METRICS:
        base_value = converted(key, float(baseline["metrics"][key]))
        candidate_value = converted(key, float(candidate["metrics"][key]))
        metrics[key] = {
            "unit": "cm" if "_pos_" in key else "deg",
            "baseline": base_value,
            "candidate": candidate_value,
            "relative_change_percent": (candidate_value / base_value - 1.0) * 100.0,
        }

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": stable_size,
        "checkpoint_sha256": checkpoint_sha256,
        "released_checkpoint": str(released_checkpoint),
        "baseline": str(baseline_path),
        "evaluation": str(output),
        "fixed_hat4": {
            "baseline_passed": baseline["motions"] - baseline["failed_motion_count"],
            "candidate_passed": candidate["motions"] - candidate["failed_motion_count"],
            "total": candidate["motions"],
            "focus_success_metrics": metrics,
        },
        "architecture": {
            "embedding_dim": 384,
            "num_heads": 6,
            "ff_dim": 384,
            "num_layers": 6,
            "actor_task_input": 269,
            "output_action_dof": 29,
        },
        "training": {
            "task": "G1-HAT4-F3H-Finetune",
            "iterations": 20000,
            "distributed_ranks": 5,
            "environments_per_rank": 819,
            "global_environments": 4095,
            "rank_seeds": [42, 43, 44, 45, 46],
        },
    }
    (release_dir / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_release_docs(release_dir, summary)
    print(release_dir / "final_summary.json", flush=True)


if __name__ == "__main__":
    main()
