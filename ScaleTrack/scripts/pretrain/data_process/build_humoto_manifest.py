#!/usr/bin/env python3
"""Bind HuMoTo hand caches to their task-local USD assets.

The hand rollout stores one manipulated-object reference for each hand, while a
HuMoTo task may contain many task-local props and supports.  The manipulated
asset is identified against every frame of the source task trajectory after the
known -90 degree world-Z basis conversion.  Remaining task assets are retained
as kinematic collision scene objects at the matched task frame.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_FRICTION = 2.0
MATCH_THRESHOLD_M = 0.05


def _metadata(cache: Path) -> dict:
    with np.load(cache, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"]))


def _pose_to_state(value: np.ndarray) -> list[float]:
    qw, qx, qy, qz, x, y, z = (float(component) for component in value)
    root_half = np.sqrt(0.5)
    quaternion = np.asarray(
        (
            root_half * (qw + qz),
            root_half * (qx + qy),
            root_half * (qy - qx),
            root_half * (qz - qw),
        ),
        dtype=np.float64,
    )
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    return [
        y,
        -x,
        z,
        *quaternion.tolist(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def _trajectory_positions(trajectory: np.ndarray) -> np.ndarray:
    return np.stack(
        (trajectory[:, 5], -trajectory[:, 4], trajectory[:, 6]), axis=-1
    )


def _match_asset(
    target_position: np.ndarray, trajectories: dict[str, np.ndarray]
) -> tuple[str, int, float]:
    best = (float("inf"), "", -1)
    for name, trajectory in trajectories.items():
        distances = np.linalg.norm(
            _trajectory_positions(trajectory) - target_position[None, :], axis=-1
        )
        frame = int(np.argmin(distances))
        candidate = (float(distances[frame]), name, frame)
        if candidate < best:
            best = candidate
    distance, name, frame = best
    if distance > MATCH_THRESHOLD_M:
        raise ValueError(
            f"closest HuMoTo asset is {name!r} at {distance:.4f} m "
            f"(threshold {MATCH_THRESHOLD_M:.3f} m)"
        )
    return name, frame, distance


def _assign_group_balanced_weights(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            groups[row["task_group"]].append(row)
    for group in groups.values():
        for row in group:
            row["sampling_weight"] = 1.0 / len(group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--heldout-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--asset-blacklist", type=Path, default=None)
    args = parser.parse_args()

    catalog_raw = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog = {task["name"]: task for task in catalog_raw["tasks"]}
    blacklist = set()
    if args.asset_blacklist and args.asset_blacklist.is_file():
        blacklist = {
            str(Path(value).expanduser().resolve())
            for value in json.loads(
                args.asset_blacklist.read_text(encoding="utf-8")
            ).get("usd_paths", [])
        }

    rows = []
    excluded = []
    degraded_scene = []
    match_distances = []
    match_assets = Counter()
    source_cache_count = 0
    for cache in sorted(args.cache_dir.glob("*.npz")):
        source_cache_count += 1
        metadata = _metadata(cache)
        name = cache.stem
        if metadata.get("free_air_only"):
            rows.append(
                {
                    "name": name,
                    "cache_path": str(cache.resolve()),
                    "split": "train",
                    "source": "humoto",
                    "task_group": "humoto:free_air",
                    "primitive": "free_air_capture",
                    "hand_type": "bih",
                    "sampling_weight": 1.0,
                    "object_rh_usd": None,
                    "object_lh_usd": None,
                    "object_rh_scale": 1.0,
                    "object_lh_scale": 1.0,
                    "object_rh_mass": None,
                    "object_lh_mass": None,
                    "object_rh_friction": DEFAULT_FRICTION,
                    "object_lh_friction": DEFAULT_FRICTION,
                    "table_height": None,
                    "table_center_x": 0.0,
                    "scene_objects": [],
                    "free_air_only": True,
                    "humoto_capture": metadata.get("humoto_capture"),
                }
            )
            continue

        task_name = metadata.get("humoto_task")
        if task_name not in catalog:
            excluded.append({"motion": name, "reason": f"missing catalog task {task_name!r}"})
            continue
        task = catalog[task_name]
        with Path(task["pickle_path"]).open("rb") as stream:
            payload = pickle.load(stream)
        trajectories = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in payload["objects"].items()
        }
        assets = {value["name"]: value for value in task["assets"]}
        if set(trajectories) != set(assets):
            excluded.append(
                {
                    "motion": name,
                    "reason": "task pickle asset names differ from the asset catalog",
                }
            )
            continue

        with np.load(cache, allow_pickle=False) as data:
            target_rh = np.asarray(data["state_manip_obj_rh"][0, :3], dtype=np.float64)
            target_lh = np.asarray(data["state_manip_obj_lh"][0, :3], dtype=np.float64)
        try:
            rh_name, rh_frame, rh_distance = _match_asset(target_rh, trajectories)
            lh_name, lh_frame, lh_distance = _match_asset(target_lh, trajectories)
        except ValueError as exc:
            excluded.append({"motion": name, "reason": str(exc)})
            continue

        matched_rh_usd = str(Path(assets[rh_name]["usd_path"]).resolve())
        matched_lh_usd = str(Path(assets[lh_name]["usd_path"]).resolve())
        rh_dynamic = assets[rh_name]["role"] == "tracked_dynamic"
        lh_dynamic = assets[lh_name]["role"] == "tracked_dynamic"
        rh_usd = matched_rh_usd if rh_dynamic else None
        lh_usd = matched_lh_usd if lh_dynamic else None
        if (rh_usd and rh_usd in blacklist) or (lh_usd and lh_usd in blacklist):
            excluded.append(
                {
                    "motion": name,
                    "reason": "manipulated HuMoTo asset is blacklisted",
                    "object_rh_usd": matched_rh_usd,
                    "object_lh_usd": matched_lh_usd,
                }
            )
            continue

        # Both rollout object tracks normally name the same task object.  When
        # they differ, use the earlier matched task frame for the shared scene.
        task_frame = min(rh_frame, lh_frame)
        # A hand rollout's single object channel sometimes tracks a static
        # support (table/shelf/container) rather than something held by a hand.
        # Keep that exact asset in the scene, but never turn it into a
        # gravity-enabled drop target merely to inflate object-bound coverage.
        main_assets = set()
        if rh_dynamic:
            main_assets.add(rh_name)
        if lh_dynamic:
            main_assets.add(lh_name)
        scene_objects = []
        for asset_name, asset in assets.items():
            if asset_name in main_assets:
                continue
            usd_path = str(Path(asset["usd_path"]).resolve())
            if usd_path in blacklist:
                degraded_scene.append(
                    {
                        "motion": name,
                        "asset": asset_name,
                        "usd_path": usd_path,
                        "reason": "blacklisted secondary task asset",
                    }
                )
                continue
            trajectory = trajectories[asset_name]
            frame = min(task_frame, len(trajectory) - 1)
            scene_objects.append(
                {
                    "shape": "usd",
                    "name": asset_name,
                    "role": asset["role"],
                    "usd_path": usd_path,
                    "state": _pose_to_state(trajectory[frame]),
                    # Only the one/two HDF5 object tracks have a physically
                    # synchronized reference. Secondary assets are collision
                    # geometry fixed at the matched source-task frame.
                    "kinematic": True,
                    "disable_gravity": True,
                }
            )

        match_distances.extend((rh_distance, lh_distance))
        match_assets[(task_name, rh_name, lh_name)] += 1
        rows.append(
            {
                "name": name,
                "cache_path": str(cache.resolve()),
                "split": "train",
                "source": "humoto",
                "task_group": f"humoto:{task_name}",
                "primitive": task_name,
                "hand_type": "bih",
                "sampling_weight": 1.0,
                "object_rh_usd": rh_usd,
                "object_lh_usd": lh_usd,
                "object_rh_scale": 1.0,
                "object_lh_scale": 1.0,
                "object_rh_mass": None,
                "object_lh_mass": None,
                "object_rh_friction": DEFAULT_FRICTION,
                "object_lh_friction": DEFAULT_FRICTION,
                "table_height": None,
                "table_center_x": 0.0,
                "scene_objects": scene_objects,
                "free_air_only": False,
                "physical_object_evaluable": bool(rh_usd or lh_usd),
                "humoto_task": task_name,
                "humoto_main_asset_rh": rh_name,
                "humoto_main_asset_lh": lh_name,
                "humoto_match_frame": task_frame,
                "humoto_match_distance_rh_m": rh_distance,
                "humoto_match_distance_lh_m": lh_distance,
            }
        )

    if not rows:
        raise SystemExit("no eligible HuMoTo caches found")

    # Optional per-source heldout is retained for standalone use.  The migration
    # pipeline passes zero here and freezes one global cross-source heldout later.
    rng = np.random.default_rng(args.seed)
    candidates: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        candidates[row["task_group"]].append(row)
    for values in candidates.values():
        rng.shuffle(values)
    heldout_names = {
        row["name"]
        for row in [
            candidates[key][0]
            for key in sorted(candidates)
            if candidates[key]
        ][: args.heldout_count]
    }
    for row in rows:
        if row["name"] in heldout_names:
            row["split"] = "heldout"
    _assign_group_balanced_weights(rows)

    manifest = {
        "schema_version": 1,
        "fps": 50,
        "seed": args.seed,
        "motions": rows,
        "asset_policy": {
            "shared_dexmanip_table": False,
            "main_object": "nearest HuMoTo task trajectory to HDF5 object state",
            "secondary_task_assets": "exact task-local USD, kinematic at matched task frame",
            "default_friction": DEFAULT_FRICTION,
        },
        "coverage": {
            "cache_count": source_cache_count,
            "included_count": len(rows),
            "task_asset_bound_count": sum(not row["free_air_only"] for row in rows),
            "physical_object_count": sum(
                bool(row.get("object_rh_usd") or row.get("object_lh_usd"))
                for row in rows
            ),
            "scene_only_task_count": sum(
                not row["free_air_only"]
                and not (row.get("object_rh_usd") or row.get("object_lh_usd"))
                for row in rows
            ),
            "free_air_count": sum(row["free_air_only"] for row in rows),
            "excluded": excluded,
            "degraded_scene_assets": degraded_scene,
            "blacklist_count": len(blacklist),
            "max_match_distance_m": max(match_distances, default=None),
            "mean_match_distance_m": (
                float(np.mean(match_distances)) if match_distances else None
            ),
            "main_asset_assignments": [
                {
                    "task": task,
                    "rh": rh,
                    "lh": lh,
                    "rollout_count": count,
                }
                for (task, rh, lh), count in sorted(match_assets.items())
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    heldout = {
        "schema_version": 1,
        "seed": args.seed,
        "count": len(heldout_names),
        "motions": sorted(heldout_names),
    }
    args.heldout.parent.mkdir(parents=True, exist_ok=True)
    args.heldout.write_text(
        json.dumps(heldout, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"included={len(rows)} object_bound="
        f"{sum(bool(row.get('object_rh_usd') or row.get('object_lh_usd')) for row in rows)} "
        f"task_asset_bound={sum(not row['free_air_only'] for row in rows)} free_air="
        f"{sum(row['free_air_only'] for row in rows)} excluded={len(excluded)} "
        f"manifest={args.output}"
    )


if __name__ == "__main__":
    main()
