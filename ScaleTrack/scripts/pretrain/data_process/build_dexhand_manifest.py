#!/usr/bin/env python3
"""Build a weighted train/held-out manifest and an honest asset coverage report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_DENSITY = 200.0
DEFAULT_FRICTION = 2.0
DEXMANIP_TABLE_SURFACE_Z = 0.415
DEXMANIP_TABLE_CENTER_X = -0.1
SCENE_COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "purple": (1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
}
OAKINK_MASS = {
    "O02@0015@00002": 0.101,
    "O02@0015@00001": 0.027,
    "C12001": 0.114,
    "O02@0030@00002": 0.0144,
    "O02@0033@00002": 0.014,
    "O02@0011@00003": 0.12,
    "O02@0206@00002": 0.163,
}
OAKINK_SCALE = {
    "O02@0206@00001": 1.15,
    "O02@0206@00002": 0.95,
    "O02@0029@00011": 0.9,
    "O02@0029@00012": 1.2,
    "O02@0015@00020": 0.98,
    "O02@0015@00019": 1.02,
}


def _matrix_to_wxyz(matrix: np.ndarray) -> list[float]:
    """Convert one rotation matrix to a normalized wxyz quaternion."""

    m = np.asarray(matrix, dtype=np.float64)
    values = np.empty(4, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        values[:] = (
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            values[:] = (
                (m[2, 1] - m[1, 2]) / scale,
                0.25 * scale,
                (m[0, 1] + m[1, 0]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            values[:] = (
                (m[0, 2] - m[2, 0]) / scale,
                (m[0, 1] + m[1, 0]) / scale,
                0.25 * scale,
                (m[1, 2] + m[2, 1]) / scale,
            )
        else:
            scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            values[:] = (
                (m[1, 0] - m[0, 1]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
                (m[1, 2] + m[2, 1]) / scale,
                0.25 * scale,
            )
    values /= max(float(np.linalg.norm(values)), 1.0e-8)
    return values.astype(np.float32).tolist()


def _favor_scene_objects(metadata: dict) -> list[dict]:
    """Reproduce only the FAVOR asset transform logic from DexManipSH."""

    scene_objects = metadata.get("scene_objects") or []
    if len(scene_objects) > 5:
        raise ValueError(f"FAVOR sequence has {len(scene_objects)} scene objects; expected at most five")
    rx = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    rz = np.asarray(
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ rx
    transform[2, 3] = DEXMANIP_TABLE_SURFACE_Z
    normalized = []
    for value in scene_objects:
        kind = value["type"]
        source_size = tuple(float(component) for component in value["size"])
        pose = np.asarray(value["pose"], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"invalid FAVOR scene object pose: {pose.shape}")
        if kind == "cube":
            if len(source_size) != 3:
                raise ValueError(f"cube size must contain three values: {source_size}")
            offset = np.eye(4, dtype=np.float64)
            offset[:3, 3] = np.asarray(source_size) * 0.5
            pose = pose @ offset
            size = source_size
        elif kind == "cylinder":
            if len(source_size) != 2:
                raise ValueError(f"cylinder size must contain radius and height: {source_size}")
            # DexManipSH deliberately represented cylinders as boxes.
            size = (2.0 * source_size[0], 2.0 * source_size[0], source_size[1])
        else:
            raise ValueError(f"unsupported FAVOR scene object type {kind!r}")
        pose = transform @ pose
        color = value.get("color", "white")
        normalized.append(
            {
                "shape": "cuboid",
                "size": list(size),
                "color": list(SCENE_COLORS.get(color, (1.0, 1.0, 1.0))),
                "state": [
                    *pose[:3, 3].astype(np.float32).tolist(),
                    *_matrix_to_wxyz(pose[:3, :3]),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )
    return normalized


def _load_metadata(cache: Path) -> dict:
    with np.load(cache, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"]))


def _object_usd(metadata: dict, hand: str, object_root: Path, usd_root: Path) -> tuple[str | None, str | None]:
    oid = metadata.get(f"oid_{hand}")
    relative = metadata.get(f"obj_{hand}_path")
    if not oid or not relative:
        return None, None
    urdf = object_root / Path(relative).relative_to("ObjURDF")
    try:
        asset_relative = urdf.relative_to(object_root).with_suffix(".usd")
    except ValueError:
        return None, str(urdf)
    usd = usd_root / asset_relative
    return (str(usd.resolve()) if usd.is_file() else None), str(urdf)


def _stratified_heldout(rows: list[dict], count: int, seed: int) -> set[str]:
    """Round-robin across source/primitive strata before filling randomly."""

    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["primitive"])].append(row)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected: set[str] = set()
    depth = 0
    while len(selected) < min(count, len(rows)):
        made_progress = False
        for key in keys:
            if depth < len(groups[key]):
                selected.add(groups[key][depth]["name"])
                made_progress = True
                if len(selected) == min(count, len(rows)):
                    break
        if not made_progress:
            break
        depth += 1
    return selected


def _assign_group_balanced_weights(rows: list[dict]) -> None:
    """Give every task group equal mass, then divide within its rollouts."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            grouped[row["task_group"]].append(row)
    for values in grouped.values():
        per_motion = 1.0 / len(values)
        for row in values:
            row["sampling_weight"] = per_motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True, help="OakInk ObjURDF directory")
    parser.add_argument("--usd-root", type=Path, required=True, help="Output root mirroring ObjURDF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--heldout-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--source",
        default="oakink",
        help="Source label for this invocation. Merge manifests for HuMoTo/FAVOR.",
    )
    parser.add_argument(
        "--free-air-on-missing-object",
        action="store_true",
        help="Keep missing-asset samples for BC with object fields disabled",
    )
    parser.add_argument(
        "--asset-blacklist",
        type=Path,
        default=None,
        help="Stability blacklist JSON; referenced bad USDs follow the missing-object policy",
    )
    args = parser.parse_args()
    blacklist = set()
    if args.asset_blacklist and args.asset_blacklist.is_file():
        blacklist = {
            str(Path(value).expanduser().resolve())
            for value in json.loads(
                args.asset_blacklist.read_text(encoding="utf-8")
            ).get("usd_paths", [])
        }
    source_key = args.source.strip().lower()
    uses_dexmanip_scene = source_key in {
        "dexmanipnet",
        "dexmanipnet_favor",
        "dexmanipnet_oakinkv2",
        "favor",
        "oakink",
        "oakinkv2",
    }

    rows = []
    missing = []
    defaults = Counter()
    for cache in sorted(args.cache_dir.glob("*.npz")):
        metadata = _load_metadata(cache)
        name = cache.stem
        primitive = metadata.get("primitive") or "unknown"
        rh_usd, rh_urdf = _object_usd(metadata, "rh", args.object_root, args.usd_root)
        lh_usd, lh_urdf = _object_usd(metadata, "lh", args.object_root, args.usd_root)
        blacklisted_hands = []
        if rh_usd in blacklist:
            blacklisted_hands.append("rh")
            rh_usd = None
        if lh_usd in blacklist:
            blacklisted_hands.append("lh")
            lh_usd = None
        for hand, oid, urdf, usd in (
            ("rh", metadata.get("oid_rh"), rh_urdf, rh_usd),
            ("lh", metadata.get("oid_lh"), lh_urdf, lh_usd),
        ):
            if oid and not usd:
                missing.append({"motion": name, "hand": hand, "oid": oid, "urdf": urdf})
        if (metadata.get("oid_rh") and not rh_usd) or (metadata.get("oid_lh") and not lh_usd):
            if not args.free_air_on_missing_object:
                continue
            rh_usd = None
            lh_usd = None
        oid_rh = metadata.get("oid_rh")
        oid_lh = metadata.get("oid_lh")
        if oid_rh and oid_rh not in OAKINK_MASS:
            defaults["mass_rh"] += 1
        if oid_lh and oid_lh not in OAKINK_MASS:
            defaults["mass_lh"] += 1
        rows.append(
            {
                "name": name,
                "cache_path": str(cache.resolve()),
                "split": "train",
                "source": args.source,
                "task_group": f"{args.source}:{primitive}",
                "primitive": primitive,
                "hand_type": metadata.get("hand_type") or "bih",
                "sampling_weight": 1.0,
                "object_rh_usd": rh_usd,
                "object_lh_usd": lh_usd,
                "object_rh_scale": OAKINK_SCALE.get(oid_rh, 1.0),
                "object_lh_scale": OAKINK_SCALE.get(oid_lh, 1.0),
                "object_rh_mass": OAKINK_MASS.get(oid_rh),
                "object_lh_mass": OAKINK_MASS.get(oid_lh),
                "object_rh_friction": DEFAULT_FRICTION,
                "object_lh_friction": DEFAULT_FRICTION,
                # Only DexManipNet/FAVOR owns this common table. HuMoTo tables
                # remain task URDFs and must never be replaced by this box.
                "table_height": DEXMANIP_TABLE_SURFACE_Z if uses_dexmanip_scene else None,
                "table_center_x": DEXMANIP_TABLE_CENTER_X if uses_dexmanip_scene else 0.0,
                "scene_objects": _favor_scene_objects(metadata) if uses_dexmanip_scene else [],
                "blacklisted_object_hands": blacklisted_hands,
            }
        )
    if not rows:
        raise SystemExit("no eligible caches found")

    heldout_names = _stratified_heldout(rows, args.heldout_count, args.seed)
    for row in rows:
        if row["name"] in heldout_names:
            row["split"] = "heldout"
    _assign_group_balanced_weights(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "fps": 50,
        "seed": args.seed,
        "motions": rows,
        "asset_policy": {
            "missing_object": "free_air" if args.free_air_on_missing_object else "exclude",
            "fallback_density_kg_m3": DEFAULT_DENSITY,
            "default_friction": DEFAULT_FRICTION,
            "dexmanip_table": (
                {
                    "size_m": [1.0, 1.6, 0.03],
                    "center_xyz_m": [DEXMANIP_TABLE_CENTER_X, 0.0, 0.4],
                    "surface_z_m": DEXMANIP_TABLE_SURFACE_Z,
                }
                if uses_dexmanip_scene
                else None
            ),
        },
        "coverage": {
            "cache_count": len(list(args.cache_dir.glob("*.npz"))),
            "included_count": len(rows),
            "with_any_object_usd": sum(bool(row["object_rh_usd"] or row["object_lh_usd"]) for row in rows),
            "missing_references": missing,
            "blacklist_count": len(blacklist),
            "rows_touching_blacklist": sum(
                bool(row["blacklisted_object_hands"]) for row in rows
            ),
            "default_property_counts": dict(defaults),
        },
    }
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    heldout = {
        "schema_version": 1,
        "seed": args.seed,
        "count": len(heldout_names),
        "motions": sorted(heldout_names),
        "strata": {
            name: {
                "source": row["source"],
                "primitive": row["primitive"],
                "task_group": row["task_group"],
            }
            for row in rows
            for name in (row["name"],)
            if name in heldout_names
        },
    }
    args.heldout.parent.mkdir(parents=True, exist_ok=True)
    args.heldout.write_text(json.dumps(heldout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"included={len(rows)} heldout={len(heldout_names)} missing_asset_refs={len(missing)} "
        f"manifest={args.output}"
    )


if __name__ == "__main__":
    main()
