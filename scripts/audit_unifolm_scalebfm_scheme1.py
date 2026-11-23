#!/usr/bin/env python3
"""Audit the data assumptions behind ScaleBFM adaptation scheme 1.

The audit treats the UnifoLM parquet ``robot_q_current`` as the source of truth
and checks its frame-by-frame relationship with:

* XQY FK pkl files;
* head-origin/egocentric human_policy HDF5 files;
* the optional TWIST Pickup Pillow body reference.

No dataset is modified.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import h5py
import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "DATASETS" / "UnifoLM_WBT"
DEFAULT_XQY_ROOT = REPO_ROOT / "DATASETS" / "XQY_PKL"
DEFAULT_EGO_ROOT = REPO_ROOT / "DATASETS" / "human_policy" / "convert_UnifoLM_WBT"
DEFAULT_TWIST_BODY = (
    REPO_ROOT
    / "TWIST"
    / "shengyin_checkpoints"
    / "HAT_output"
    / "unifolm_make_pickup"
    / "pickup_pillow_body_gmt.pkl"
)
DEFAULT_SCALETRACK_ROBOT = (
    REPO_ROOT
    / "ScaleBFM"
    / "ScaleTrack"
    / "source"
    / "scaletrack"
    / "scaletrack"
    / "robots"
    / "g1_29dof.py"
)

FAMILIES = (
    {
        "dataset": "G1_WBT_Brainco_Pickup_Pillow",
        "xqy_prefix": "G1_WBT_Brainco_Pickup_Pillow",
        "dataset_rel": "G1_WBT_Brainco_Pickup_Pillow",
    },
    {
        "dataset": "G1_WB_Dex5_Pickup_Pillow",
        "xqy_prefix": "G1_WB_Dex5_Pickup_Pillow",
        "dataset_rel": (
            "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly/"
            "G1_WB_Dex5_Pickup_Pillow"
        ),
    },
)

# This order is independently documented in the deployed SONIC MuJoCo interface.
UNIFOLM_MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def _rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 2, 3)
    a1, a2 = values[:, 0], values[:, 1]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / np.maximum(np.linalg.norm(a2_orth, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def _quat_error_deg(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    return np.rad2deg(
        (Rotation.from_quat(a_xyzw).inv() * Rotation.from_quat(b_xyzw)).magnitude()
    )


def _update_max(record: dict, key: str, value: float) -> None:
    record[key] = max(record.get(key, 0.0), float(value))


def _episode_rows(parquet_path: Path):
    parquet = pq.ParquetFile(parquet_path)
    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state.robot_q_current",
    ]
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
        for episode_index, episode in frame.groupby("episode_index", sort=True):
            episode = episode.sort_values("frame_index")
            yield int(episode_index), episode


def audit_family(
    family: dict,
    data_root: Path,
    xqy_root: Path,
    ego_root: Path,
    max_episodes: int | None,
) -> dict:
    dataset = data_root / family["dataset_rel"]
    info = json.loads((dataset / "meta" / "info.json").read_text())
    parquet_paths = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {dataset}")

    result = {
        "dataset": family["dataset"],
        "declared_fps": float(info["fps"]),
        "episodes_checked": 0,
        "frames_checked": 0,
        "missing_xqy": [],
        "missing_egocentric": [],
        "length_mismatches": [],
        "nonfinite_episodes": [],
    }

    stop = False
    for parquet_path in parquet_paths:
        for episode_index, episode in _episode_rows(parquet_path):
            if max_episodes is not None and result["episodes_checked"] >= max_episodes:
                stop = True
                break

            q = np.stack(episode["observation.state.robot_q_current"].to_numpy()).astype(
                np.float64
            )
            frame_index = episode["frame_index"].to_numpy()
            timestamp = episode["timestamp"].to_numpy(dtype=np.float64)
            n = len(q)
            tag = f"{family['xqy_prefix']}_ep_{episode_index:04d}"
            result["episodes_checked"] += 1
            result["frames_checked"] += n

            if q.shape != (n, 36) or not np.isfinite(q).all():
                result["nonfinite_episodes"].append(tag)
                continue
            _update_max(
                result,
                "raw_quaternion_norm_error_max",
                np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1.0)),
            )
            _update_max(
                result,
                "frame_index_step_error_max",
                np.max(np.abs(np.diff(frame_index) - 1)) if n > 1 else 0,
            )
            expected_dt = 1.0 / float(info["fps"])
            _update_max(
                result,
                "timestamp_step_error_sec_max",
                np.max(np.abs(np.diff(timestamp) - expected_dt)) if n > 1 else 0,
            )

            xqy_path = xqy_root / f"{tag}.pkl"
            if not xqy_path.exists():
                result["missing_xqy"].append(tag)
                continue
            xqy = joblib.load(xqy_path)
            if len(xqy["root_pos"]) != n:
                result["length_mismatches"].append(
                    {"episode": tag, "raw": n, "xqy": len(xqy["root_pos"])}
                )
                continue

            _update_max(result, "xqy_fps_error_max", abs(float(xqy["fps"]) - float(info["fps"])))
            _update_max(
                result, "raw_xqy_root_pos_abs_max", np.max(np.abs(q[:, 0:3] - xqy["root_pos"]))
            )
            _update_max(
                result,
                "raw_xqy_root_quat_abs_max",
                np.max(np.abs(q[:, [4, 5, 6, 3]] - xqy["root_rot"])),
            )
            _update_max(
                result, "raw_xqy_dof_abs_max", np.max(np.abs(q[:, 7:36] - xqy["dof_pos"]))
            )

            ego_path = ego_root / f"{tag}.hdf5"
            if not ego_path.exists():
                result["missing_egocentric"].append(tag)
                continue
            with h5py.File(ego_path, "r") as h5:
                state = h5["observation.state"][:]
                action = h5["action"][:]
                translation = np.asarray(
                    h5.attrs.get("head_origin_translation", np.zeros(3)), dtype=np.float64
                )
            if len(state) != n:
                result["length_mismatches"].append(
                    {"episode": tag, "raw": n, "egocentric": len(state)}
                )
                continue

            expected_q26 = q[:, :26].copy()
            expected_q26[:, :3] += translation
            _update_max(
                result,
                "raw_egocentric_qpos26_abs_max",
                np.max(np.abs(expected_q26 - state[:, 100:126])),
            )
            expected_action = np.concatenate((state[1:], state[-1:]), axis=0)
            _update_max(
                result,
                "egocentric_action_next_frame_abs_max",
                np.max(np.abs(action - expected_action)),
            )

            links = list(xqy["link_body_list"])
            for side, pos_slice, rot_slice, body_name, quat_key in (
                ("left", slice(80, 83), slice(83, 89), "L_hand_base_link", "left_wrist_rot"),
                ("right", slice(30, 33), slice(33, 39), "R_hand_base_link", "right_wrist_rot"),
            ):
                expected_pos = (
                    np.asarray(xqy["body_pos_w"])[:, links.index(body_name)] + translation
                )
                _update_max(
                    result,
                    f"xqy_egocentric_{side}_wrist_pos_abs_max",
                    np.max(np.abs(state[:, pos_slice] - expected_pos)),
                )
                state_quat = Rotation.from_matrix(
                    _rot6d_to_matrix(state[:, rot_slice])
                ).as_quat()
                _update_max(
                    result,
                    f"xqy_egocentric_{side}_wrist_rot_deg_max",
                    np.max(_quat_error_deg(state_quat, np.asarray(xqy[quat_key]))),
                )
        if stop:
            break
    return result


def audit_twist(body_path: Path, xqy_root: Path, ego_root: Path) -> dict:
    if not body_path.exists():
        return {"available": False, "path": str(body_path)}
    body = joblib.load(body_path)
    xqy = joblib.load(xqy_root / "G1_WBT_Brainco_Pickup_Pillow_ep_0000.pkl")
    ego_path = ego_root / "G1_WBT_Brainco_Pickup_Pillow_ep_0000.hdf5"
    with h5py.File(ego_path, "r") as h5:
        translation = np.asarray(h5.attrs["head_origin_translation"], dtype=np.float64)

    n = len(body["root_pos"])
    shifted = lambda values: np.concatenate((values[1:], values[-1:]), axis=0)
    result = {
        "available": True,
        "path": str(body_path),
        "frames": n,
        "fps": float(body["fps"]),
    }
    for side, quat_key in (("left", "left_wrist_rot"), ("right", "right_wrist_rot")):
        twist_quat = np.asarray(body[quat_key])[:, 0]
        xqy_quat = np.asarray(xqy[quat_key])
        result[f"{side}_wrist_same_frame_rot_deg_mean"] = float(
            np.mean(_quat_error_deg(twist_quat, xqy_quat))
        )
        result[f"{side}_wrist_next_frame_rot_deg_max"] = float(
            np.max(_quat_error_deg(twist_quat, shifted(xqy_quat)))
        )

    root_rotation = Rotation.from_quat(body["root_rot"])
    offset = np.asarray(body["coordinate_transform"]["translation_offset"])
    links_body = list(body["link_body_list"])
    links_xqy = list(xqy["link_body_list"])
    for side, body_name, xqy_name in (
        ("left", "left_wrist_yaw_link", "L_hand_base_link"),
        ("right", "right_wrist_yaw_link", "R_hand_base_link"),
    ):
        twist_world = np.asarray(body["root_pos"]) + root_rotation.apply(
            np.asarray(body["local_body_pos"])[:, links_body.index(body_name)]
        )
        xqy_world = (
            np.asarray(xqy["body_pos_w"])[:, links_xqy.index(xqy_name)]
            + translation
            + offset
        )
        result[f"{side}_wrist_next_frame_pos_abs_max"] = float(
            np.max(np.abs(twist_world - shifted(xqy_world)))
        )

    dof_delta = np.asarray(body["dof_pos"]) - np.asarray(xqy["dof_pos"])
    result["dof_vs_raw_rmse_rad"] = float(np.sqrt(np.mean(np.square(dof_delta))))
    result["dof_vs_raw_abs_max_rad"] = float(np.max(np.abs(dof_delta)))
    expected_root = np.asarray(xqy["root_pos"]) + translation + offset
    result["root_vs_raw_transformed_abs_max"] = float(
        np.max(np.abs(np.asarray(body["root_pos"]) - expected_root))
    )
    return result


def status_from_results(report: dict, tolerance: float) -> str:
    if not report["scaletrack_joint_order_is_identity"]:
        return "FAIL"
    for family in report["families"]:
        if (
            family["missing_xqy"]
            or family["missing_egocentric"]
            or family["length_mismatches"]
            or family["nonfinite_episodes"]
        ):
            return "FAIL"
        for key in (
            "raw_xqy_root_pos_abs_max",
            "raw_xqy_root_quat_abs_max",
            "raw_xqy_dof_abs_max",
            "raw_egocentric_qpos26_abs_max",
            "egocentric_action_next_frame_abs_max",
            "xqy_egocentric_left_wrist_pos_abs_max",
            "xqy_egocentric_right_wrist_pos_abs_max",
        ):
            if family.get(key, math.inf) > tolerance:
                return "FAIL"
    return "PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--xqy-root", type=Path, default=DEFAULT_XQY_ROOT)
    parser.add_argument("--ego-root", type=Path, default=DEFAULT_EGO_ROOT)
    parser.add_argument("--twist-body", type=Path, default=DEFAULT_TWIST_BODY)
    parser.add_argument("--max-episodes", type=int, default=None, help="limit per family")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaletrack_names = _literal_assignment(DEFAULT_SCALETRACK_ROBOT, "G1_29DOF_JOINT_NAMES")
    report = {
        "scheme": "ScaleBFM scheme 1 / raw robot_q_current conversion",
        "source_layout": "root xyz + root quaternion wxyz + 29 MuJoCo-order joints",
        "scaletrack_joint_order_is_identity": scaletrack_names
        == UNIFOLM_MUJOCO_JOINT_NAMES,
        "joint_reorder_indices": [
            UNIFOLM_MUJOCO_JOINT_NAMES.index(name) for name in scaletrack_names
        ],
        "families": [
            audit_family(
                family,
                args.data_root,
                args.xqy_root,
                args.ego_root,
                args.max_episodes,
            )
            for family in FAMILIES
        ],
        "twist_ep0000": audit_twist(args.twist_body, args.xqy_root, args.ego_root),
    }
    report["status"] = status_from_results(report, args.tolerance)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
