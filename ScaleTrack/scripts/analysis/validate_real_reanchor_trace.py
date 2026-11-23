#!/usr/bin/env python3
"""Validate a measured real-robot static re-anchor trace.

The endpoint measurement must be independent of the policy/FK chain by
default (for example wrist-mounted mocap markers).  A trace that merely runs
the commanded joints through the same FK model cannot pass the physical-chain
acceptance gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = (
    "time_s",
    "tracker_valid",
    "pelvis_measured_pos_w",
    "left_target_pos_w",
    "right_target_pos_w",
    "left_measured_pos_w",
    "right_measured_pos_w",
    "measurement_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--mean-threshold-cm", type=float, default=2.5)
    parser.add_argument("--max-tracker-dropout", type=float, default=0.01)
    parser.add_argument("--min-valid-frames", type=int, default=100)
    parser.add_argument(
        "--allow-fk-measurement",
        action="store_true",
        help="Diagnostic only: allow a non-independent FK endpoint source.",
    )
    return parser.parse_args()


def scalar_string(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("measurement_source must be a scalar string")
    return str(array.reshape(-1)[0])


def stats_cm(values_m: np.ndarray) -> dict[str, float]:
    values_cm = np.asarray(values_m, dtype=np.float64) * 100.0
    return {
        "mean": float(values_cm.mean()),
        "p95": float(np.percentile(values_cm, 95)),
        "max": float(values_cm.max()),
    }


def main() -> None:
    args = parse_args()
    if not args.trace.is_file():
        raise FileNotFoundError(args.trace)
    with np.load(args.trace, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED if key not in archive]
        if missing:
            raise KeyError(f"Missing required arrays: {missing}")
        payload = {key: np.asarray(archive[key]) for key in REQUIRED}

    time_s = payload["time_s"].astype(np.float64).reshape(-1)
    count = len(time_s)
    if count < 2 or not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing with at least two frames")
    tracker_valid = payload["tracker_valid"].astype(bool).reshape(-1)
    if len(tracker_valid) != count:
        raise ValueError("tracker_valid length does not match time_s")
    for key in REQUIRED[2:-1]:
        if payload[key].shape != (count, 3):
            raise ValueError(f"{key} must have shape ({count}, 3), got {payload[key].shape}")
        if not np.all(np.isfinite(payload[key])):
            raise ValueError(f"{key} contains non-finite values")

    source = scalar_string(payload["measurement_source"])
    source_lower = source.lower()
    independent = not any(token in source_lower for token in ("fk", "kinematic", "command"))
    after_warmup = time_s >= time_s[0] + args.warmup_seconds
    valid = tracker_valid & after_warmup
    valid_frames = int(valid.sum())
    if not valid_frames:
        raise ValueError("No tracker-valid frames remain after warmup")

    left_error = np.linalg.norm(
        payload["left_measured_pos_w"] - payload["left_target_pos_w"], axis=-1
    )[valid]
    right_error = np.linalg.norm(
        payload["right_measured_pos_w"] - payload["right_target_pos_w"], axis=-1
    )[valid]
    left = stats_cm(left_error)
    right = stats_cm(right_error)
    worst_mean = max(left["mean"], right["mean"])
    dropout = 1.0 - float(tracker_valid.mean())
    dt = np.diff(time_s)
    sample_rate = 1.0 / float(np.median(dt))
    jitter_ms_p95 = float(np.percentile(np.abs(dt - np.median(dt)), 95) * 1000.0)
    pelvis_span_cm = (
        np.ptp(payload["pelvis_measured_pos_w"][valid], axis=0) * 100.0
    ).tolist()

    gates = {
        "independent_endpoint_measurement": independent or args.allow_fk_measurement,
        "minimum_valid_frames": valid_frames >= args.min_valid_frames,
        "tracker_dropout": dropout <= args.max_tracker_dropout,
        "wrist_world_mean_error": worst_mean <= args.mean_threshold_cm,
    }
    report = {
        "trace": str(args.trace.resolve()),
        "measurement_source": source,
        "independent_endpoint_measurement": independent,
        "allow_fk_measurement": args.allow_fk_measurement,
        "frames": count,
        "valid_frames_after_warmup": valid_frames,
        "warmup_seconds": args.warmup_seconds,
        "sample_rate_hz_median": sample_rate,
        "sample_interval_jitter_p95_ms": jitter_ms_p95,
        "tracker_dropout_fraction": dropout,
        "pelvis_axis_span_cm_after_warmup": pelvis_span_cm,
        "left_wrist_world_position_error_cm": left,
        "right_wrist_world_position_error_cm": right,
        "acceptance": {
            "mean_threshold_cm": args.mean_threshold_cm,
            "max_tracker_dropout": args.max_tracker_dropout,
            "min_valid_frames": args.min_valid_frames,
            "worst_wrist_world_mean_cm": worst_mean,
            "gates": gates,
            "passed": all(gates.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
