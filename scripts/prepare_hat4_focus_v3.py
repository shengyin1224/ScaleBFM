#!/usr/bin/env python3
"""Build focus-v3 training NPZs, retaining every original body array and split.

Only focus_mask and focus_time_to_next_edge_s change. No body simulation,
motion retargeting, diagnostic reports or visualizations are generated.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from focus_definition import (
    HAND_FIELD, FOCUS_ONSET_THRESHOLD, FOCUS_ONSET_LEAD_S,
    focus_from_hand_state_v3, infer_hand_type, time_to_next_focus_edge,
)
from prepare_hat4_focus_v2 import FAMILIES, nearest_source_indices

REPO = Path(__file__).resolve().parents[1]


def build(dataset_root, source_root, output_root, onset_threshold=0.10, lead_s=0.07, resume=False):
    dataset_root, source_root, output_root = map(Path, (dataset_root, source_root, output_root))
    if source_root.resolve() == output_root.resolve():
        raise ValueError("v3 output must be separate from the original corpus")
    if not 0 <= lead_s <= 0.10 or not 0 < onset_threshold < 0.5:
        raise ValueError("Require lead_s in [0,0.10] and onset_threshold in (0,0.5)")
    manifests = {}
    for name in ("unifolm_hat4_train.yaml", "unifolm_hat4_heldout.yaml", "unifolm_hat4_all.yaml"):
        manifests[name] = yaml.safe_load((source_root / "manifests" / name).read_text())
    expected_names = set(manifests["unifolm_hat4_all.yaml"])
    train = set(manifests["unifolm_hat4_train.yaml"])
    heldout = set(manifests["unifolm_hat4_heldout.yaml"])
    if train & heldout or train | heldout != expected_names:
        raise ValueError("Original train/heldout splits are inconsistent")
    output_dir = output_root / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = set()
    for family in FAMILIES:
        dataset = dataset_root / family
        info = json.loads((dataset / "meta/info.json").read_text())
        fps = float(info["fps"])
        paths = sorted((dataset / "data").glob("chunk-*/*.parquet"))
        if not paths:
            raise FileNotFoundError(dataset / "data")
        import pandas as pd
        rows = pd.concat(
            [pq.read_table(path, columns=["episode_index", "frame_index", HAND_FIELD]).to_pandas()
             for path in paths], ignore_index=True,
        )
        count = changed = 0
        for episode_id, episode in rows.groupby("episode_index", sort=True):
            name = f"{family}_ep_{int(episode_id):04d}"
            if name not in expected_names:
                continue
            episode = episode.sort_values("frame_index")
            indexes = episode["frame_index"].to_numpy()
            if not np.array_equal(indexes, np.arange(len(indexes))):
                raise ValueError(f"{name}: noncontiguous source frames")
            hand = np.stack(episode[HAND_FIELD].to_numpy()).astype(np.float32)
            mask, _, _, _ = focus_from_hand_state_v3(
                hand, fps, infer_hand_type(dataset), onset_threshold=onset_threshold, lead_s=lead_s,
            )
            nearest = nearest_source_indices(len(mask), fps, 50.0)
            with np.load(manifests["unifolm_hat4_all.yaml"][name]) as source:
                arrays = {field: source[field] for field in source.files}
            if float(arrays["fps"]) != 50 or len(arrays["joint_pos"]) != len(nearest):
                raise ValueError(f"{name}: source/processed timeline mismatch")
            new_mask = mask[nearest]
            changed += int(not np.array_equal(arrays["focus_mask"], new_mask))
            arrays["focus_mask"] = new_mask
            arrays["focus_time_to_next_edge_s"] = time_to_next_focus_edge(new_mask, 50.0)
            output = output_dir / f"{name}.npz"
            if output.exists():
                if not resume:
                    raise FileExistsError(f"{output} exists; use --resume to validate and reuse")
            else:
                temporary = output.with_name(f".{output.name}.tmp.npz")
                np.savez(temporary, **arrays)
                os.replace(temporary, output)
            # Validate persisted labels and all unchanged source fields.
            with np.load(output) as written:
                if set(written.files) != set(arrays):
                    raise ValueError(f"{output}: field mismatch")
                for field, expected in arrays.items():
                    if not np.array_equal(written[field], expected):
                        raise ValueError(f"{output}: persisted/resumed {field} mismatch")
            count += 1
            completed.add(name)
        print(f"[v3] {family}: {count} episodes, {changed} changed masks; all body arrays verified", flush=True)
    if completed != expected_names:
        raise ValueError(f"Missing episodes: {sorted(expected_names - completed)[:5]}")
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, old in manifests.items():
        updated = {key: str((output_dir / f"{key}.npz").resolve()) for key in old}
        (manifest_dir / name).write_text(yaml.safe_dump(updated, sort_keys=True))
    print(f"[v3] Ready: {len(completed)} episodes, splits unchanged, {manifest_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nvme0n1/DATASETS/UnifoLM_WBT"))
    parser.add_argument("--source-root", type=Path, default=REPO / "ScaleTrack/data/hat4_unifolm_focus_v2")
    parser.add_argument("--output-root", type=Path, default=REPO / "ScaleTrack/data/hat4_unifolm_focus_v3")
    parser.add_argument("--onset-threshold", type=float, default=FOCUS_ONSET_THRESHOLD)
    parser.add_argument("--lead-s", type=float, default=FOCUS_ONSET_LEAD_S)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build(args.dataset_root, args.source_root, args.output_root, args.onset_threshold, args.lead_s, args.resume)


if __name__ == "__main__":
    main()
