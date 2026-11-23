#!/usr/bin/env python3
"""Compare external fixed-HAT4 metrics against the released checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METRICS = (
    "error_left_wrist_pelvis_local_pos",
    "error_left_wrist_pelvis_local_rot",
    "error_right_wrist_pelvis_local_pos",
    "error_right_wrist_pelvis_local_rot",
    "error_head_proxy_pelvis_local_pos",
    "error_head_proxy_pelvis_local_rot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def convert(key: str, value: float) -> float:
    return value * 100.0 if "_pos" in key else math.degrees(value)


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    baseline_metrics = baseline["metrics"]
    comparisons = []
    for path in args.candidate:
        candidate = json.loads(path.read_text())
        entries = {}
        degradations = []
        for key in METRICS:
            base = convert(key, baseline_metrics[key])
            value = convert(key, candidate["metrics"][key])
            change = (value / base - 1.0) * 100.0
            entries[key] = {
                "baseline": base,
                "candidate": value,
                "relative_change_percent": change,
            }
            degradations.append(change)
        comparisons.append(
            {
                "candidate": str(path.resolve()),
                "passed": candidate["motions"] - candidate["failed_motion_count"],
                "total": candidate["motions"],
                "metrics": entries,
                "worst_relative_degradation_percent": max(degradations),
                "all_degradations_le_10_percent": max(degradations) <= 10.0,
            }
        )
    report = {
        "baseline": str(args.baseline.resolve()),
        "motions": baseline["motions"],
        "comparison_scope": "all evaluated frames; pelvis-local wrist/head metrics",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
