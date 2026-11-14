#!/usr/bin/env python3
"""Create left/right mirrored dexhand cache variants.

Asymmetric meshes are *not* mirrored here.  The generated cache marks the
object asset as requiring a mirrored mesh; manifest construction must either
provide that USD or keep the sample free-air.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_PATH = REPO_ROOT / "source/scaletrack/scaletrack/tasks/dexhand_tracking/data.py"
_SPEC = importlib.util.spec_from_file_location("scaletrack_dexhand_data", _DATA_PATH)
_DATA = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DATA
_SPEC.loader.exec_module(_DATA)
mirror_state_wxyz = _DATA.mirror_state_wxyz


def _mirror_points(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    result[..., 1] *= -1
    return result


def _mirror_forces(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy().reshape(*value.shape[:-1], 5, 3)
    result[..., 1] *= -1
    return result.reshape(value.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--joint-signs",
        type=Path,
        default=None,
        help="Optional JSON with rh_to_lh and lh_to_rh 12-element sign vectors",
    )
    args = parser.parse_args()

    signs = {"rh_to_lh": [1.0] * 12, "lh_to_rh": [1.0] * 12}
    if args.joint_signs:
        signs.update(json.loads(args.joint_signs.read_text(encoding="utf-8")))
    rh_to_lh = np.asarray(signs["rh_to_lh"], dtype=np.float32)
    lh_to_rh = np.asarray(signs["lh_to_rh"], dtype=np.float32)
    if rh_to_lh.shape != (12,) or lh_to_rh.shape != (12,):
        raise ValueError("joint sign vectors must each contain 12 values")

    with np.load(args.input, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    data["q_rh"], data["q_lh"] = data["q_lh"] * lh_to_rh, data["q_rh"] * rh_to_lh
    data["dq_rh"], data["dq_lh"] = data["dq_lh"] * lh_to_rh, data["dq_rh"] * rh_to_lh
    for stem in ("state", "state_manip_obj"):
        rh, lh = data[f"{stem}_rh"], data[f"{stem}_lh"]
        data[f"{stem}_rh"], data[f"{stem}_lh"] = mirror_state_wxyz(lh), mirror_state_wxyz(rh)
    for stem in ("body_state",):
        rh, lh = data[f"{stem}_rh"], data[f"{stem}_lh"]
        data[f"{stem}_rh"], data[f"{stem}_lh"] = mirror_state_wxyz(lh), mirror_state_wxyz(rh)
    for stem in ("fingertip_pos",):
        rh, lh = data[f"{stem}_rh"], data[f"{stem}_lh"]
        data[f"{stem}_rh"], data[f"{stem}_lh"] = _mirror_points(lh), _mirror_points(rh)
    for stem in ("tip_force",):
        rh, lh = data[f"{stem}_rh"], data[f"{stem}_lh"]
        data[f"{stem}_rh"], data[f"{stem}_lh"] = _mirror_forces(lh), _mirror_forces(rh)

    q_next = np.concatenate((data["q_rh"][1:], data["q_rh"][-1:]), axis=0)
    q_next_lh = np.concatenate((data["q_lh"][1:], data["q_lh"][-1:]), axis=0)
    data["bc_action"] = 2.0 * np.concatenate((q_next, q_next_lh), axis=-1)
    metadata = json.loads(str(data["metadata_json"]))
    for stem in ("oid", "obj", "primitive"):
        rh_key = f"{stem}_rh" if stem != "obj" else "obj_rh_path"
        lh_key = f"{stem}_lh" if stem != "obj" else "obj_lh_path"
        metadata[rh_key], metadata[lh_key] = metadata.get(lh_key), metadata.get(rh_key)
    metadata["hand_type"] = {"rh": "lh", "lh": "rh"}.get(
        metadata.get("hand_type"), metadata.get("hand_type", "bih")
    )
    metadata.update(
        {
            "mirrored": True,
            "mirrored_from": str(args.input.resolve()),
            "requires_mirrored_object_mesh": True,
            "joint_signs": signs,
        }
    )
    data["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)


if __name__ == "__main__":
    main()
