"""Pure-Python data contracts shared by conversion, BC, and the online task.

This module intentionally has no Isaac Sim dependency.  Keeping the cache schema
and deterministic env assignment here lets the offline tools test the exact
layout used by the simulator without launching Kit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CACHE_SCHEMA_VERSION = 1
FPS = 50
NUM_HAND_JOINTS = 12
NUM_ACTIONS = 24
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# TWIST's body_link_names ordering.  joint_state is (T, 18 * 13).
TIP_BODY_INDICES = (17, 3, 6, 12, 9)
FUTURE_STEPS = (1, *tuple(range(5, 100, 5)))  # TWIST: 1, 5, 10, ..., 95
PROPRIO_HISTORY = 11


@dataclass(frozen=True)
class MotionEntry:
    """Normalized manifest entry used by the environment and tools."""

    name: str
    cache_path: str
    split: str = "train"
    source: str = "unknown"
    task_group: str = "unknown"
    primitive: str = "unknown"
    hand_type: str = "bih"
    sampling_weight: float = 1.0
    object_rh_usd: str | None = None
    object_lh_usd: str | None = None
    object_rh_scale: float = 1.0
    object_lh_scale: float = 1.0
    object_rh_mass: float | None = None
    object_lh_mass: float | None = None
    object_rh_friction: float = 2.0
    object_lh_friction: float = 2.0
    table_height: float | None = None
    table_center_x: float = -0.1
    scene_objects: tuple[dict[str, Any], ...] = ()

    @property
    def has_rh_object(self) -> bool:
        return bool(self.object_rh_usd)

    @property
    def has_lh_object(self) -> bool:
        # A bimanual sequence commonly names the same object for both hands.
        # It must be spawned once, not duplicated in the same pose.
        return bool(self.object_lh_usd) and self.object_lh_usd != self.object_rh_usd

    @property
    def uses_rh(self) -> bool:
        return self.hand_type in ("rh", "bih")

    @property
    def uses_lh(self) -> bool:
        return self.hand_type in ("lh", "bih")


def _resolve(base: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def load_manifest(path: str | Path, split: str | None = None) -> tuple[dict[str, Any], list[MotionEntry]]:
    """Load and validate a dexhand manifest.

    Relative cache/USD paths are resolved relative to the manifest, which makes
    experiment bundles relocatable.
    """

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if raw.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path}: expected schema_version={CACHE_SCHEMA_VERSION}, "
            f"got {raw.get('schema_version')!r}"
        )
    rows = raw.get("motions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest_path}: 'motions' must be a non-empty list")

    entries: list[MotionEntry] = []
    names: set[str] = set()
    for row in rows:
        if split is not None and row.get("split", "train") != split:
            continue
        name = str(row["name"])
        if name in names:
            raise ValueError(f"{manifest_path}: duplicate motion name {name!r}")
        names.add(name)
        normalized = dict(row)
        for field in ("cache_path", "object_rh_usd", "object_lh_usd"):
            normalized[field] = _resolve(manifest_path.parent, normalized.get(field))
        scene_objects = []
        for value in normalized.get("scene_objects", ()):
            scene_object = dict(value)
            if scene_object.get("shape") == "usd":
                scene_object["usd_path"] = _resolve(
                    manifest_path.parent, scene_object.get("usd_path")
                )
            scene_objects.append(scene_object)
        normalized["scene_objects"] = tuple(scene_objects)
        entry = MotionEntry(**{key: value for key, value in normalized.items() if key in MotionEntry.__dataclass_fields__})
        if entry.sampling_weight <= 0 or not math.isfinite(entry.sampling_weight):
            raise ValueError(f"{name}: sampling_weight must be finite and positive")
        for hand in ("rh", "lh"):
            mass = getattr(entry, f"object_{hand}_mass")
            friction = getattr(entry, f"object_{hand}_friction")
            if mass is not None and (mass <= 0 or not math.isfinite(mass)):
                raise ValueError(f"{name}: object_{hand}_mass must be positive or null")
            if friction <= 0 or not math.isfinite(friction):
                raise ValueError(f"{name}: object_{hand}_friction must be finite and positive")
        if len(entry.scene_objects) > 12:
            raise ValueError(f"{name}: at most twelve task scene objects are supported")
        for index, scene_object in enumerate(entry.scene_objects):
            shape = scene_object.get("shape")
            if shape not in ("cuboid", "usd"):
                raise ValueError(
                    f"{name}: scene object {index} must be a cuboid or USD"
                )
            if len(scene_object.get("state", ())) != 13:
                raise ValueError(f"{name}: invalid scene object {index} state")
            if shape == "cuboid" and len(scene_object.get("size", ())) != 3:
                raise ValueError(f"{name}: invalid scene object {index} cuboid size")
            if shape == "usd":
                usd_path = scene_object.get("usd_path")
                if not usd_path or not Path(usd_path).is_file():
                    raise FileNotFoundError(
                        f"{name}: scene object {index} USD does not exist: {usd_path}"
                    )
        if not Path(entry.cache_path).is_file():
            raise FileNotFoundError(f"{name}: cache does not exist: {entry.cache_path}")
        entries.append(entry)
    if not entries:
        raise ValueError(f"{manifest_path}: split={split!r} selected no motions")
    return raw, entries


def select_manifest_shard(
    entries: Iterable[MotionEntry],
    shard_count: int = 1,
    shard_index: int = 0,
    seed: int = 20260728,
) -> list[MotionEntry]:
    """Return one deterministic, balanced partition of manifest entries.

    Static per-environment assets cannot be swapped at reset. This partitioning
    lets a memory-limited run cycle through disjoint scene shards without
    silently dropping motions that do not fit in one Isaac Sim process.
    """

    rows = list(entries)
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    if shard_count > len(rows):
        raise ValueError(
            f"shard_count={shard_count} exceeds entry count={len(rows)}"
        )
    if shard_count == 1:
        return rows
    order = np.random.default_rng(seed).permutation(len(rows))
    selected = np.array_split(order, shard_count)[shard_index]
    return [rows[int(index)] for index in selected]


def deterministic_motion_assignment(
    entries: Iterable[MotionEntry],
    num_envs: int,
    seed: int = 0,
    exact_repeats: bool = False,
) -> np.ndarray:
    """Assign one immutable motion/asset to each environment.

    When enough environments are available every motion is covered once, then
    remaining slots follow the declared weights.  With fewer environments,
    systematic weighted sampling is used.  A seeded permutation avoids source
    blocks while remaining exactly reproducible for the scene spawner and
    command term.
    """

    rows = list(entries)
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not rows:
        raise ValueError("entries must not be empty")
    weights = np.asarray([entry.sampling_weight for entry in rows], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)

    if exact_repeats:
        if num_envs % len(rows):
            raise ValueError(
                "exact_repeats requires num_envs to be an integer multiple of "
                f"the motion count ({num_envs} vs {len(rows)})"
            )
        assignment = np.repeat(
            np.arange(len(rows), dtype=np.int64), num_envs // len(rows)
        )
        rng.shuffle(assignment)
        return assignment

    if num_envs >= len(rows):
        result = list(range(len(rows)))
        remaining = num_envs - len(rows)
        if remaining:
            quotas = weights * remaining
            counts = np.floor(quotas).astype(np.int64)
            residual = remaining - int(counts.sum())
            if residual:
                order = np.argsort(-(quotas - counts), kind="stable")
                counts[order[:residual]] += 1
            for motion_id, count in enumerate(counts):
                result.extend([motion_id] * int(count))
        assignment = np.asarray(result, dtype=np.int64)
        rng.shuffle(assignment)
        return assignment

    # Systematic samples have lower coverage variance than independent draws.
    cdf = np.cumsum(weights)
    offset = rng.random() / num_envs
    points = offset + np.arange(num_envs, dtype=np.float64) / num_envs
    return np.searchsorted(cdf, points, side="right").astype(np.int64)


def xyzw_to_wxyz(array: np.ndarray) -> np.ndarray:
    """Convert quaternion arrays from Isaac Gym to Isaac Lab convention."""

    return np.asarray(array)[..., (3, 0, 1, 2)]


def wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Vectorized unit-quaternion to rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-8)
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty((*q.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Stable vectorized rotation-matrix to wxyz quaternion conversion."""

    m = np.asarray(matrix, dtype=np.float32)
    shape = m.shape[:-2]
    flat = m.reshape(-1, 3, 3)
    out = np.empty((len(flat), 4), dtype=np.float32)
    for index, value in enumerate(flat):
        trace = float(np.trace(value))
        if trace > 0:
            scale = math.sqrt(trace + 1.0) * 2
            out[index] = (
                0.25 * scale,
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
            )
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                scale = math.sqrt(1 + value[0, 0] - value[1, 1] - value[2, 2]) * 2
                out[index] = (
                    (value[2, 1] - value[1, 2]) / scale,
                    0.25 * scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                )
            elif axis == 1:
                scale = math.sqrt(1 + value[1, 1] - value[0, 0] - value[2, 2]) * 2
                out[index] = (
                    (value[0, 2] - value[2, 0]) / scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    0.25 * scale,
                    (value[1, 2] + value[2, 1]) / scale,
                )
            else:
                scale = math.sqrt(1 + value[2, 2] - value[0, 0] - value[1, 1]) * 2
                out[index] = (
                    (value[1, 0] - value[0, 1]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                    (value[1, 2] + value[2, 1]) / scale,
                    0.25 * scale,
                )
    out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1.0e-8)
    return out.reshape(*shape, 4)


def mirror_state_wxyz(state: np.ndarray) -> np.ndarray:
    """Mirror an Isaac Lab 13-D root/body state across the world x-z plane."""

    mirrored = np.asarray(state, dtype=np.float32).copy()
    mirrored[..., 1] *= -1
    reflection = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    rotation = wxyz_to_matrix(mirrored[..., 3:7])
    rotation = reflection @ rotation @ reflection
    mirrored[..., 3:7] = matrix_to_wxyz(rotation)
    mirrored[..., 8] *= -1
    # Angular velocity is an axial vector: det(S) * S * omega.
    mirrored[..., 10] *= -1
    mirrored[..., 12] *= -1
    return mirrored
