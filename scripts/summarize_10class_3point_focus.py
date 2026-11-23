#!/usr/bin/env python3
"""Summarize ten-class ScaleBFM 3-point MuJoCo FOCUS rollouts."""

import csv
import json
from pathlib import Path

import numpy as np


RESULT_DIR = Path(
    "/root/shengyin/ScaleBFM/ScaleTrack/action_error_results/10class_3point"
)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def short_name(stem):
    return (
        stem.replace("G1_WBT_Brainco_", "")
        .replace("G1_WBT_Inspire_", "")
        .replace("_ep_0000", "")
    )


def main():
    rows = []
    pooled = {}
    for report_path in sorted(RESULT_DIR.glob("G1_WBT_*_ep_0000.json")):
        stem = report_path.stem
        report = json.loads(report_path.read_text())
        trace = np.load(RESULT_DIR / f"{stem}_trace.npz")
        focus = trace["focus_mask"].astype(bool)
        names = trace["active_link_names"].tolist()
        left = names.index("left_wrist_yaw_link")
        right = names.index("right_wrist_yaw_link")

        local_rot = trace["active_pelvis_local_rotation_deg"]
        local_pos = trace["active_pelvis_local_position_m"]
        global_rot = trace["active_global_rotation_deg"]
        global_pos = trace["active_global_position_m"]
        item = {
            "motion": short_name(stem),
            "stem": stem,
            "frames": int(len(focus)),
            "duration_s": float(report["duration_sec"]),
            "focus_frames": int(focus.sum()),
            "focus_fraction": float(focus.mean()),
            "success_rate_frames": float(report["success_rate_frames"]),
            "first_failure_frame": report["first_failure_frame"],
            "full_local_wrist_rotation_deg": {
                "left": stats(local_rot[:, left]),
                "right": stats(local_rot[:, right]),
            },
            "focus_local_wrist_rotation_deg": {
                "left": stats(local_rot[focus, left]),
                "right": stats(local_rot[focus, right]),
            },
            "focus_local_wrist_position_m": {
                "left": stats(local_pos[focus, left]),
                "right": stats(local_pos[focus, right]),
            },
            "focus_global_wrist_rotation_deg": {
                "left": stats(global_rot[focus, left]),
                "right": stats(global_rot[focus, right]),
            },
            "focus_global_wrist_position_m": {
                "left": stats(global_pos[focus, left]),
                "right": stats(global_pos[focus, right]),
            },
            "focus_head_global_position_m": stats(trace["head_global_position_m"][focus]),
            "focus_head_global_rotation_deg": stats(trace["head_global_rotation_deg"][focus]),
            "focus_head_local_position_m": stats(
                trace["head_pelvis_local_position_m"][focus]
            ),
            "focus_head_local_rotation_deg": stats(
                trace["head_pelvis_local_rotation_deg"][focus]
            ),
        }
        for side in ("left", "right"):
            item[f"{side}_focus_minus_full_local_rotation_mean_deg"] = (
                item["focus_local_wrist_rotation_deg"][side]["mean"]
                - item["full_local_wrist_rotation_deg"][side]["mean"]
            )
        rows.append(item)

        arrays = {
            "local_rot_left_full": local_rot[:, left],
            "local_rot_right_full": local_rot[:, right],
            "local_rot_left_focus": local_rot[focus, left],
            "local_rot_right_focus": local_rot[focus, right],
            "local_pos_left_focus": local_pos[focus, left],
            "local_pos_right_focus": local_pos[focus, right],
            "global_rot_left_focus": global_rot[focus, left],
            "global_rot_right_focus": global_rot[focus, right],
            "global_pos_left_focus": global_pos[focus, left],
            "global_pos_right_focus": global_pos[focus, right],
            "head_global_pos_focus": trace["head_global_position_m"][focus],
            "head_global_rot_focus": trace["head_global_rotation_deg"][focus],
            "head_local_pos_focus": trace["head_pelvis_local_position_m"][focus],
            "head_local_rot_focus": trace["head_pelvis_local_rotation_deg"][focus],
        }
        for key, values in arrays.items():
            pooled.setdefault(key, []).append(values)

    pooled_stats = {
        key: stats(np.concatenate(values)) for key, values in pooled.items()
    }
    output = {
        "evaluation": "ScaleBFM VR-3 (pelvis + two wrists), MuJoCo TWIST asset",
        "checkpoint": "model_22200.pt",
        "motions": rows,
        "pooled_frame_statistics": pooled_stats,
    }
    json_path = RESULT_DIR / "summary_10class_3point_focus.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n")

    csv_path = RESULT_DIR / "summary_10class_3point_focus.csv"
    fields = [
        "motion",
        "duration_s",
        "focus_fraction",
        "success_rate_frames",
        "first_failure_frame",
        "left_full_local_rot_mean_deg",
        "left_focus_local_rot_mean_deg",
        "left_focus_local_rot_p95_deg",
        "right_full_local_rot_mean_deg",
        "right_focus_local_rot_mean_deg",
        "right_focus_local_rot_p95_deg",
        "left_focus_local_pos_mean_cm",
        "right_focus_local_pos_mean_cm",
        "head_focus_local_pos_mean_cm",
        "head_focus_local_rot_mean_deg",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "motion": item["motion"],
                    "duration_s": item["duration_s"],
                    "focus_fraction": item["focus_fraction"],
                    "success_rate_frames": item["success_rate_frames"],
                    "first_failure_frame": item["first_failure_frame"],
                    "left_full_local_rot_mean_deg": item[
                        "full_local_wrist_rotation_deg"
                    ]["left"]["mean"],
                    "left_focus_local_rot_mean_deg": item[
                        "focus_local_wrist_rotation_deg"
                    ]["left"]["mean"],
                    "left_focus_local_rot_p95_deg": item[
                        "focus_local_wrist_rotation_deg"
                    ]["left"]["p95"],
                    "right_full_local_rot_mean_deg": item[
                        "full_local_wrist_rotation_deg"
                    ]["right"]["mean"],
                    "right_focus_local_rot_mean_deg": item[
                        "focus_local_wrist_rotation_deg"
                    ]["right"]["mean"],
                    "right_focus_local_rot_p95_deg": item[
                        "focus_local_wrist_rotation_deg"
                    ]["right"]["p95"],
                    "left_focus_local_pos_mean_cm": 100
                    * item["focus_local_wrist_position_m"]["left"]["mean"],
                    "right_focus_local_pos_mean_cm": 100
                    * item["focus_local_wrist_position_m"]["right"]["mean"],
                    "head_focus_local_pos_mean_cm": 100
                    * item["focus_head_local_position_m"]["mean"],
                    "head_focus_local_rot_mean_deg": item[
                        "focus_head_local_rotation_deg"
                    ]["mean"],
                }
            )
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
