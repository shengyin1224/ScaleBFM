#!/usr/bin/env python3
"""Summarize fixed-HAT4 held-out evaluations across checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path


METRICS = (
    "error_left_wrist_pelvis_local_pos_focus",
    "error_left_wrist_pelvis_local_rot_focus",
    "error_right_wrist_pelvis_local_pos_focus",
    "error_right_wrist_pelvis_local_rot_focus",
    "error_head_proxy_pelvis_local_pos_focus",
    "error_head_proxy_pelvis_local_rot_focus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model8k-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def convert(key: str, value: float) -> float:
    return value * 100.0 if "_pos_" in key else math.degrees(value)


def variant_from_name(name: str) -> str:
    match = re.match(r"(F1|F2o|F2|F3h|F3)_model8000", name)
    if not match:
        raise ValueError(name)
    return match.group(1)


def row(path: Path, variant: str, step: int) -> dict:
    report = json.loads(path.read_text())
    metrics = report["metrics"]
    result = {
        "variant": variant,
        "step": step,
        "checkpoint": report["checkpoint"],
        "passed": int(report["motions"] - report["failed_motion_count"]),
        "total": int(report["motions"]),
        "success_rate": float(metrics["success_rate"]),
    }
    for key in METRICS:
        result[f"{key}_success"] = convert(key, metrics[f"{key}_success"])
        result[f"{key}_all"] = convert(key, metrics[key])
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row(args.baseline, "released_hat4", 22200)]
    for path_value in sorted(glob.glob(str(args.model8k_dir / "F*_model8000.json"))):
        path = Path(path_value)
        rows.append(row(path, variant_from_name(path.stem), 8000))
    for path in sorted(args.checkpoint_root.glob("F*/model_*.json")):
        rows.append(row(path, path.parent.name, int(path.stem.removeprefix("model_"))))
    rows.sort(key=lambda value: (value["step"], value["variant"]))

    step20 = {value["variant"]: value for value in rows if value["step"] == 20000}
    decision = {"status": "pending_missing_model_20000"}
    if {"F3", "F3h"}.issubset(step20):
        f3 = step20["F3"]
        f3h = step20["F3h"]
        left_disadvantage = (
            f3h["error_left_wrist_pelvis_local_rot_focus_success"]
            - f3["error_left_wrist_pelvis_local_rot_focus_success"]
        )
        right_disadvantage = (
            f3h["error_right_wrist_pelvis_local_rot_focus_success"]
            - f3["error_right_wrist_pelvis_local_rot_focus_success"]
        )
        worst_disadvantage = max(left_disadvantage, right_disadvantage)
        decision = {
            "rule": "choose F3h if its worst wrist-rotation disadvantage vs F3 is < 1 degree",
            "left_wrist_disadvantage_deg": left_disadvantage,
            "right_wrist_disadvantage_deg": right_disadvantage,
            "worst_wrist_disadvantage_deg": worst_disadvantage,
            "winner": "F3h" if worst_disadvantage < 1.0 else None,
            "requires_F3_5": worst_disadvantage >= 1.0,
        }

    summary = {"rows": rows, "model_20000_decision": decision}
    (args.output_dir / "fixed_checkpoint_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    fieldnames = list(rows[0])
    with (args.output_dir / "fixed_checkpoint_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_dir / "fixed_checkpoint_summary.json")


if __name__ == "__main__":
    main()
