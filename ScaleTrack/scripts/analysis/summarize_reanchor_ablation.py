#!/usr/bin/env python3
"""Summarize static acceptance and MuJoCo root-drift re-anchor runs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-name", default="static")
    parser.add_argument("--static-threshold-cm", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in sorted(glob.glob(args.results_glob)):
        path = Path(value)
        report = json.loads(path.read_text())
        ablation = report["reanchor_ablation"]
        drift_cm = float(ablation["injected_initial_root_drift_xyz_m"][0] * 100.0)
        row = {
            "run": path.stem,
            "reanchor_mode": ablation["mode"],
            "drift_cm": drift_cm,
            "success_rate_frames": report["success_rate_frames"],
        }
        for wrist in WRISTS:
            stats = report["active_link_pose_vs_reference"][wrist]["position_m"]
            row[f"{wrist}_global_mean_cm"] = stats["mean"] * 100.0
            row[f"{wrist}_global_p95_cm"] = stats["p95"] * 100.0
        rows.append(row)

    static_rows = [row for row in rows if args.static_name in row["run"]]
    static_acceptance = {"status": "not_found"}
    if static_rows:
        static = static_rows[0]
        worst_mean = max(static[f"{wrist}_global_mean_cm"] for wrist in WRISTS)
        static_acceptance = {
            "run": static["run"],
            "threshold_cm": args.static_threshold_cm,
            "worst_wrist_world_mean_cm": worst_mean,
            "passed": worst_mean <= args.static_threshold_cm,
        }
    report = {"rows": rows, "static_acceptance": static_acceptance}
    (args.output_dir / "reanchor_ablation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if rows:
        with (args.output_dir / "reanchor_ablation_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(args.output_dir / "reanchor_ablation_summary.json")


if __name__ == "__main__":
    main()
