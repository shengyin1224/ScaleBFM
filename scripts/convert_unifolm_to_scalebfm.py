#!/usr/bin/env python3
"""Convert UnifoLM ``robot_q_current`` episodes to ScaleTrack input pkls."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from focus_definition import (
    ACTIVE_RANGE,
    FOCUS_EDGE_CLIP_S,
    FOCUS_PAD_S,
    GRASP_ENTER,
    GRASP_EXIT,
    HAND_FIELD,
    HAND_REDUCE,
    closure_channel_indices,
    focus_from_hand_state,
    infer_hand_type,
    time_to_next_focus_edge,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_CFG = (
    REPO_ROOT
    / "ScaleBFM"
    / "ScaleTrack"
    / "source"
    / "scaletrack"
    / "scaletrack"
    / "robots"
    / "g1_29dof.py"
)

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

def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episode",
        type=int,
        action="append",
        default=None,
        help="episode id to convert; repeat for multiple ids (default: all)",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--robot-cfg", type=Path, default=DEFAULT_ROBOT_CFG)
    parser.add_argument(
        "--hand-type",
        choices=("brainco", "inspire"),
        default=None,
        help="hand embodiment (default: infer from the full dataset path)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(dataset: Path):
    parquet_paths = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {dataset}/data")
    columns = [
        "episode_index",
        "frame_index",
        "observation.state.robot_q_current",
        HAND_FIELD,
    ]
    frames = [pq.read_table(path, columns=columns).to_pandas() for path in parquet_paths]
    # Avoid a pandas dependency in the script itself: pyarrow's conversion already
    # returns DataFrames, whose concat implementation is available through the type.
    if len(frames) == 1:
        return frames[0]
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    info = json.loads((dataset / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid fps={fps}")
    hand_type = args.hand_type or infer_hand_type(dataset)

    scaletrack_names = literal_assignment(args.robot_cfg.resolve(), "G1_29DOF_JOINT_NAMES")
    if scaletrack_names != UNIFOLM_MUJOCO_JOINT_NAMES:
        raise RuntimeError(
            "ScaleTrack joint order no longer matches UnifoLM MuJoCo order; "
            "define and validate an explicit name-based reorder before conversion"
        )

    rows = load_rows(dataset)
    episode_ids = sorted(int(value) for value in rows["episode_index"].unique())
    if args.episode is not None:
        requested = set(args.episode)
        missing = sorted(requested.difference(episode_ids))
        if missing:
            raise ValueError(f"Episode ids not present in dataset: {missing}")
        episode_ids = [value for value in episode_ids if value in requested]
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = dataset.name
    written = 0
    for episode_id in episode_ids:
        episode = rows[rows["episode_index"] == episode_id].sort_values("frame_index")
        frame_index = episode["frame_index"].to_numpy()
        if len(frame_index) < 2 or not np.array_equal(
            frame_index, np.arange(len(frame_index), dtype=frame_index.dtype)
        ):
            raise ValueError(f"Episode {episode_id}: frame_index is not contiguous from zero")
        q = np.stack(episode["observation.state.robot_q_current"].to_numpy()).astype(
            np.float32
        )
        hand = np.stack(episode[HAND_FIELD].to_numpy()).astype(np.float32)
        if q.shape != (len(episode), 36) or not np.isfinite(q).all():
            raise ValueError(f"Episode {episode_id}: invalid robot_q_current shape/data {q.shape}")
        quat_norm_error = float(np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1.0)))
        if quat_norm_error > 1e-5:
            raise ValueError(
                f"Episode {episode_id}: root quaternion norm error {quat_norm_error:.3g}"
            )
        if hand.shape[0] != len(q) or hand.ndim != 2 or not np.isfinite(hand).all():
            raise ValueError(f"Episode {episode_id}: invalid {HAND_FIELD} shape/data {hand.shape}")
        focus_mask, grasp, signal, active_indices = focus_from_hand_state(
            hand, fps, hand_type
        )
        focus_time_to_next_edge_s = time_to_next_focus_edge(focus_mask, fps)

        output_path = args.output_dir / f"{prefix}_ep_{episode_id:04d}.pkl"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        joblib.dump(
            {
                "fps": fps,
                "root_pos": q[:, 0:3].copy(),
                "root_rot": q[:, [4, 5, 6, 3]].copy(),  # wxyz -> xyzw
                "dof_pos": q[:, 7:36].copy(),
                "focus_mask": focus_mask,
                "focus_time_to_next_edge_s": focus_time_to_next_edge_s,
                "focus_valid": np.ones(len(q), dtype=bool),
                "focus_metadata": {
                    "definition_version": "focus_v2_physical_hand_closure",
                    "source": HAND_FIELD,
                    "hand_type": hand_type,
                    "hand_reduce": HAND_REDUCE,
                    "closure_channel_indices": list(closure_channel_indices(hand_type)),
                    "active_channel_indices": active_indices.tolist(),
                    "active_range": ACTIVE_RANGE,
                    "active_channel_rule": "must_cross_grasp_hysteresis_band",
                    "physical_scale": "brainco raw/0.8" if hand_type == "brainco" else "inspire 1-raw",
                    "grasp_enter": GRASP_ENTER,
                    "grasp_exit": GRASP_EXIT,
                    "focus_pad_s": FOCUS_PAD_S,
                    "edge_clip_s": FOCUS_EDGE_CLIP_S,
                    "mode": "grasp_endpoint_windows",
                    "hands_processed_independently": True,
                    "episode_boundaries_censored": True,
                },
            },
            output_path,
        )
        written += 1
        print(
            f"{output_path}: frames={len(q)} fps={fps:g} "
            f"focus={int(focus_mask.sum())} ({100.0 * focus_mask.mean():.1f}%) "
            f"grasp={int(grasp.sum())} signal_max={float(signal.max()):.3f}"
        )
    print(f"converted={written} joint_reorder=identity")


if __name__ == "__main__":
    main()
