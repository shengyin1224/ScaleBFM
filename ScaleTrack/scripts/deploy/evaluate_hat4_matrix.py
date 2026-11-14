#!/usr/bin/env python3
"""Run and aggregate the A/B/C/D HAT-4 MuJoCo evaluation matrix.

A and B default to the family-stratified UnifoLM manifests.  C must be supplied
as a YAML manifest containing the prepared SONIC HAT motions plus ten AMASS
motions.  D automatically reuses C with VR-3 and WholeBody-14.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml


SCRIPT = Path(__file__).with_name("mujoco_sim2sim_3point.py")
TRACK = Path(__file__).resolve().parents[2]
DEFAULT_MANIFESTS = TRACK / "data" / "hat4_unifolm" / "manifests"
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument(
        "--action-metadata-npz",
        type=Path,
        required=True,
        help="Any ScaleTrack Isaac rollout NPZ providing joint_names/action_scale/action_offset.",
    )
    parser.add_argument(
        "--bucket-a",
        type=Path,
        default=DEFAULT_MANIFESTS / "unifolm_hat4_heldout.yaml",
    )
    parser.add_argument(
        "--bucket-b",
        type=Path,
        default=DEFAULT_MANIFESTS / "unifolm_hat4_train.yaml",
    )
    parser.add_argument(
        "--bucket-c",
        type=Path,
        help="External SONIC HAT + ten-AMASS prepared-motion YAML.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-motions",
        type=int,
        help="Deterministic prefix cap for a quick validation; omit for the full table.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path, limit: int | None) -> list[tuple[str, Path]]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Empty/invalid manifest: {path}")
    motions = [(str(name), Path(value)) for name, value in sorted(payload.items())]
    if limit is not None:
        motions = motions[:limit]
    for _, motion in motions:
        if not motion.is_file():
            raise FileNotFoundError(motion)
    return motions


def run_motion(
    args: argparse.Namespace,
    bucket: str,
    mode: str,
    name: str,
    reference: Path,
) -> tuple[Path, Path]:
    run_dir = args.output_dir / bucket / f"mode_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_dir / f"{name}.json"
    trace = run_dir / f"{name}_trace.npz"
    if args.skip_existing and metrics.is_file() and trace.is_file():
        return metrics, trace
    command = [
        str(SCRIPT),
        "--checkpoint",
        str(args.checkpoint),
        "--reference",
        str(reference),
        "--isaac-result-npz",
        str(args.action_metadata_npz),
        "--xml",
        str(args.xml),
        "--metrics-out",
        str(metrics),
        "--video-out",
        str(run_dir / f"{name}.mp4"),
        "--trace-out",
        str(trace),
        "--tracking-mode",
        mode,
        "--device",
        args.device,
        "--no-video",
    ]
    subprocess.run(command, check=True)
    return metrics, trace


def stat(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"mean": None, "p95": None}
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
    }


def aggregate(entries: list[tuple[Path, Path]]) -> dict:
    arrays: dict[str, list[np.ndarray]] = {}
    success_weighted = 0.0
    transition_count = 0
    for metrics_path, trace_path in entries:
        report = json.loads(metrics_path.read_text())
        frames = int(report["evaluated_transitions"])
        success_weighted += float(report["success_rate_frames"]) * frames
        transition_count += frames
        with np.load(trace_path) as trace:
            names = [str(value) for value in trace["active_link_names"]]
            focus = np.asarray(trace["focus_mask"], dtype=bool)
            for side_name in WRISTS:
                if side_name not in names:
                    continue
                index = names.index(side_name)
                for scope, pos_key, rot_key in (
                    (
                        "global",
                        "active_global_position_m",
                        "active_global_rotation_deg",
                    ),
                    (
                        "local",
                        "active_pelvis_local_position_m",
                        "active_pelvis_local_rotation_deg",
                    ),
                ):
                    for phase, mask in (
                        ("full", np.ones(len(focus), dtype=bool)),
                        ("focus", focus),
                    ):
                        arrays.setdefault(
                            f"{side_name}/{scope}/{phase}/position_m", []
                        ).append(np.asarray(trace[pos_key])[:, index][mask])
                        arrays.setdefault(
                            f"{side_name}/{scope}/{phase}/rotation_deg", []
                        ).append(np.asarray(trace[rot_key])[:, index][mask])
            for scope, prefix in (
                ("global", "head_global"),
                ("local", "head_pelvis_local"),
            ):
                for phase, mask in (
                    ("full", np.ones(len(focus), dtype=bool)),
                    ("focus", focus),
                ):
                    for metric in ("position_m", "rotation_deg"):
                        arrays.setdefault(
                            f"head/{scope}/{phase}/{metric}", []
                        ).append(np.asarray(trace[f"{prefix}_{metric}"])[mask])
    return {
        "motions": len(entries),
        "frames": transition_count,
        "success_rate_frames_0.5m": (
            success_weighted / transition_count if transition_count else None
        ),
        "metrics": {
            key: stat(np.concatenate(values))
            for key, values in sorted(arrays.items())
        },
    }


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.xml, args.action_metadata_npz):
        if not path.is_file():
            raise FileNotFoundError(path)
    matrix = [
        ("A_unifolm_heldout", args.bucket_a, ("4",)),
        ("B_unifolm_train", args.bucket_b, ("4",)),
    ]
    if args.bucket_c is not None:
        matrix.extend(
            [
                ("C_external_heldout", args.bucket_c, ("4",)),
                ("D_old_mode_regression", args.bucket_c, ("3", "14")),
            ]
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "bucket_c_status": (
            "provided" if args.bucket_c is not None else "NOT RUN: no manifest provided"
        ),
        "buckets": {},
    }
    for bucket, manifest, modes in matrix:
        motions = load_manifest(manifest, args.max_motions)
        summary["buckets"][bucket] = {}
        for mode in modes:
            entries = [
                run_motion(args, bucket, mode, name, reference)
                for name, reference in motions
            ]
            summary["buckets"][bucket][f"mode_{mode}"] = aggregate(entries)
            (args.output_dir / "summary.partial.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
