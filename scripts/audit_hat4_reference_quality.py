#!/usr/bin/env python3
"""Compare left/right wrist reference quality on a prepared HAT-4 manifest."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
ROBOT_CFG = (
    REPO
    / "ScaleTrack"
    / "source"
    / "scaletrack"
    / "scaletrack"
    / "robots"
    / "g1_29dof.py"
)


def joint_names() -> list[str]:
    tree = ast.parse(ROBOT_CFG.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name)
                and target.id == "G1_29DOF_JOINT_NAMES"
                for target in targets
            ):
                return ast.literal_eval(node.value)
    raise KeyError("G1_29DOF_JOINT_NAMES")


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(q0 * q1, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"frames": 0, "mean": None, "p95": None}
    return {
        "frames": int(values.size),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    body_names = ["pelvis"] + [
        name.replace("_joint", "_link") for name in joint_names()
    ]
    body_names[body_names.index("waist_pitch_link")] = "torso_link"
    indices = {
        side: body_names.index(f"{side}_wrist_yaw_link")
        for side in ("left", "right")
    }
    pooled = {
        side: {"full": [], "focus": []} for side in ("left", "right")
    }
    per_motion = {}
    for name, path_value in sorted(manifest.items()):
        path = Path(path_value)
        with np.load(path) as data:
            fps = float(data["fps"])
            focus = np.asarray(data["focus_mask"], dtype=bool)
            row = {}
            for side, index in indices.items():
                quaternion = np.asarray(data["body_quat_w"][:, index])
                speed = np.rad2deg(
                    quat_angle(quaternion[:-1], quaternion[1:]) * fps
                )
                speed_focus = focus[1:] | focus[:-1]
                pooled[side]["full"].append(speed)
                pooled[side]["focus"].append(speed[speed_focus])
                row[side] = {
                    "angular_speed_deg_s_full": stats(speed),
                    "angular_speed_deg_s_focus": stats(speed[speed_focus]),
                }
            per_motion[name] = row
    summary = {
        side: {
            phase: stats(np.concatenate(values))
            for phase, values in phases.items()
        }
        for side, phases in pooled.items()
    }
    report = {
        "manifest": str(args.manifest.resolve()),
        "metric": (
            "world-frame wrist quaternion inter-frame geodesic angular speed; "
            "used to detect left/right reference-motion imbalance, not tracking error"
        ),
        "pooled": summary,
        "focus_mean_right_over_left": (
            summary["right"]["focus"]["mean"] / summary["left"]["focus"]["mean"]
            if summary["left"]["focus"]["mean"]
            else None
        ),
        "per_motion": per_motion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
