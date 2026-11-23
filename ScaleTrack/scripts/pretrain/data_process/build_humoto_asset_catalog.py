#!/usr/bin/env python3
"""Index every HuMoTo task URDF and its task-local object trajectory.

This deliberately does not invent a shared table.  A table, shelf, chair, or
other support is an ordinary task asset and is retained exactly when the task
pickle names it.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _has_inertial(urdf: Path) -> bool:
    return "<inertial" in urdf.read_text(encoding="utf-8")


def _humoto_pose_to_dexhand(value: np.ndarray) -> list[float]:
    """Map HuMoTo's Z-up world basis into the hand-rollout world basis.

    The hand rollout generation applies a -90 degree world-Z rotation:
    ``(x, y, z) -> (y, -x, z)``.  Quaternions are wxyz and receive the same
    left-multiplied basis rotation.
    """

    qw, qx, qy, qz, x, y, z = (float(component) for component in value)
    root_half = np.sqrt(0.5)
    # quaternion_multiply([cos(-pi/4), 0, 0, sin(-pi/4)], source)
    transformed_quaternion = np.asarray(
        (
            root_half * (qw + qz),
            root_half * (qx + qy),
            root_half * (qy - qx),
            root_half * (qz - qw),
        ),
        dtype=np.float64,
    )
    transformed_quaternion /= max(
        float(np.linalg.norm(transformed_quaternion)), 1.0e-12
    )
    return [
        y,
        -x,
        z,
        *transformed_quaternion.tolist(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humoto-root", type=Path, required=True)
    parser.add_argument("--usd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--motion-epsilon",
        type=float,
        default=1.0e-5,
        help="Maximum pose delta above which an asset is classified as moving",
    )
    args = parser.parse_args()

    root = args.humoto_root.expanduser().resolve()
    usd_root = args.usd_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    tasks = []
    all_urdfs = set(root.glob("*/*.urdf"))
    referenced_urdfs: set[Path] = set()
    missing_urdfs = []
    missing_usds = []
    moving_count = 0
    static_count = 0

    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pickles = sorted(task_dir.glob("*.pkl"))
        if len(pickles) != 1:
            raise ValueError(f"{task_dir}: expected exactly one task pickle, found {len(pickles)}")
        pickle_path = pickles[0]
        with pickle_path.open("rb") as stream:
            payload = pickle.load(stream)
        trajectories = payload.get("objects")
        if not isinstance(trajectories, dict):
            raise ValueError(f"{pickle_path}: 'objects' must be a dictionary")

        assets = []
        frame_counts = set()
        for name, raw_trajectory in trajectories.items():
            trajectory = np.asarray(raw_trajectory, dtype=np.float64)
            if trajectory.ndim != 2 or trajectory.shape[1] != 7 or len(trajectory) == 0:
                raise ValueError(f"{pickle_path}: {name!r} must have a non-empty (T, 7) trajectory")
            if not np.isfinite(trajectory).all():
                raise ValueError(f"{pickle_path}: {name!r} trajectory contains non-finite values")
            frame_counts.add(len(trajectory))

            urdf = task_dir / f"{name}.urdf"
            if not urdf.is_file():
                missing_urdfs.append(str(urdf))
                continue
            referenced_urdfs.add(urdf)
            relative = urdf.relative_to(root)
            usd = (usd_root / relative).with_suffix(".usd")
            if not usd.is_file():
                missing_usds.append(str(usd))

            moving = bool(np.max(np.abs(trajectory - trajectory[0])) > args.motion_epsilon)
            moving_count += int(moving)
            static_count += int(not moving)
            initial_state = _humoto_pose_to_dexhand(trajectory[0])
            assets.append(
                {
                    "name": name,
                    "role": "tracked_dynamic" if moving else "tracked_static",
                    "urdf_path": str(urdf),
                    "usd_path": str(usd),
                    "usd_exists": usd.is_file(),
                    "has_urdf_inertial": _has_inertial(urdf),
                    "trajectory_key": name,
                    "trajectory_frames": len(trajectory),
                    "initial_root_state_wxyz": initial_state,
                }
            )
        if len(frame_counts) > 1:
            raise ValueError(f"{pickle_path}: task objects have inconsistent frame counts {frame_counts}")
        tasks.append(
            {
                "name": task_dir.name,
                "pickle_path": str(pickle_path),
                "trajectory_layout": ["qw", "qx", "qy", "qz", "x", "y", "z"],
                "frames": next(iter(frame_counts), 0),
                "assets": assets,
            }
        )

    extra_urdfs = sorted(str(path) for path in all_urdfs - referenced_urdfs)
    catalog = {
        "schema_version": 1,
        "source": "humoto",
        "asset_policy": {
            "shared_dexmanip_table": False,
            "task_assets": "all URDFs named by each task pickle, including task-specific supports",
            "missing_inertial": "reported; convert with the HuMoTo mode's 200 kg/m^3 fallback density",
        },
        "tasks": tasks,
        "coverage": {
            "task_count": len(tasks),
            "urdf_count": len(all_urdfs),
            "referenced_asset_count": sum(len(task["assets"]) for task in tasks),
            "moving_asset_count": moving_count,
            "static_asset_count": static_count,
            "missing_urdfs": missing_urdfs,
            "extra_urdfs": extra_urdfs,
            "missing_usds": missing_usds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"tasks={len(tasks)} assets={sum(len(task['assets']) for task in tasks)} "
        f"moving={moving_count} static={static_count} missing_urdf={len(missing_urdfs)} "
        f"extra_urdf={len(extra_urdfs)} missing_usd={len(missing_usds)} output={args.output}"
    )


if __name__ == "__main__":
    main()
