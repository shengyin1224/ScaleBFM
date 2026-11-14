#!/usr/bin/env python3
"""Relate wrist reference angular speed to pelvis-local rotation error."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-glob", required=True)
    parser.add_argument("--action-metadata-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-only", action="store_true")
    return parser.parse_args()


def quat_speed_deg_s(quat: np.ndarray, fps: float) -> np.ndarray:
    quat = quat.astype(np.float64)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    dot = np.abs(np.sum(quat[1:] * quat[:-1], axis=-1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))) * fps


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.action_metadata_npz) as metadata:
        joint_names = [str(value) for value in metadata["joint_names"]]
    body_names = ["pelvis"] + [name.replace("_joint", "_link") for name in joint_names]
    body_names[body_names.index("waist_pitch_link")] = "torso_link"

    samples = []
    per_side = {side: {"speed": [], "error": []} for side in ("left", "right")}
    json_paths = sorted(Path(path) for path in glob.glob(args.results_glob))
    if not json_paths:
        raise FileNotFoundError(args.results_glob)
    for report_path in json_paths:
        report = json.loads(report_path.read_text())
        trace_path = Path(report["trace_out"])
        reference_path = Path(report["reference"])
        with np.load(trace_path) as trace, np.load(reference_path) as reference:
            active_names = [str(value) for value in trace["active_link_names"]]
            errors = np.asarray(trace["active_pelvis_local_rotation_deg"], dtype=np.float64)
            focus = np.asarray(trace["focus_mask"], dtype=bool)
            fps = float(reference["fps"])
            for side, wrist in zip(("left", "right"), WRISTS):
                active_index = active_names.index(wrist)
                reference_index = body_names.index(wrist)
                speed = quat_speed_deg_s(reference["body_quat_w"][:, reference_index], fps)
                length = min(len(speed), len(errors), len(focus))
                mask = focus[:length] if args.focus_only else np.ones(length, dtype=bool)
                selected_speed = speed[:length][mask]
                selected_error = errors[:length, active_index][mask]
                per_side[side]["speed"].append(selected_speed)
                per_side[side]["error"].append(selected_error)
                samples.extend(
                    zip(
                        [report_path.stem] * len(selected_speed),
                        [side] * len(selected_speed),
                        selected_speed.tolist(),
                        selected_error.tolist(),
                    )
                )

    for values in per_side.values():
        values["speed"] = np.concatenate(values["speed"])
        values["error"] = np.concatenate(values["error"])
    pooled_speed = np.concatenate([per_side[side]["speed"] for side in per_side])
    pooled_error = np.concatenate([per_side[side]["error"] for side in per_side])
    edges = np.asarray([0.0, 15.0, 30.0, 60.0, 120.0, np.inf])
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (pooled_speed >= low) & (pooled_speed < high)
        bins.append(
            {
                "low_deg_s": low,
                "high_deg_s": None if np.isinf(high) else high,
                "frames": int(mask.sum()),
                "speed_mean_deg_s": float(pooled_speed[mask].mean()) if mask.any() else None,
                "rotation_error_mean_deg": float(pooled_error[mask].mean()) if mask.any() else None,
                "rotation_error_p95_deg": float(np.percentile(pooled_error[mask], 95)) if mask.any() else None,
            }
        )
    nonempty_means = [entry["rotation_error_mean_deg"] for entry in bins if entry["frames"]]
    report = {
        "scope": "FOCUS frames" if args.focus_only else "all frames",
        "motions": len(json_paths),
        "frames_per_side": {side: len(values["speed"]) for side, values in per_side.items()},
        "per_side": {
            side: {
                "reference_angular_speed_mean_deg_s": float(values["speed"].mean()),
                "reference_angular_speed_p95_deg_s": float(np.percentile(values["speed"], 95)),
                "pelvis_local_rotation_error_mean_deg": float(values["error"].mean()),
                "pelvis_local_rotation_error_p95_deg": float(np.percentile(values["error"], 95)),
                "speed_error_spearman": spearman(values["speed"], values["error"]),
            }
            for side, values in per_side.items()
        },
        "right_to_left_mean_speed_ratio": float(
            per_side["right"]["speed"].mean() / per_side["left"]["speed"].mean()
        ),
        "pooled_speed_error_spearman": spearman(pooled_speed, pooled_error),
        "bins": bins,
        "bin_means_monotonic_non_decreasing": all(
            right >= left for left, right in zip(nonempty_means[:-1], nonempty_means[1:])
        ),
    }
    (args.output_dir / "wrist_speed_error.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "wrist_speed_error_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["motion", "side", "reference_speed_deg_s", "local_rotation_error_deg"])
        writer.writerows(samples)

    plot_bins = [entry for entry in bins if entry["frames"]]
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(
        [entry["speed_mean_deg_s"] for entry in plot_bins],
        [entry["rotation_error_mean_deg"] for entry in plot_bins],
        marker="o",
        label="mean",
    )
    axis.plot(
        [entry["speed_mean_deg_s"] for entry in plot_bins],
        [entry["rotation_error_p95_deg"] for entry in plot_bins],
        marker="s",
        label="P95",
    )
    axis.set_xlabel("Reference wrist angular speed (deg/s)")
    axis.set_ylabel("Pelvis-local wrist rotation error (deg)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "wrist_speed_error.png", dpi=180)
    plt.close(figure)
    print(args.output_dir / "wrist_speed_error.json")


if __name__ == "__main__":
    main()
