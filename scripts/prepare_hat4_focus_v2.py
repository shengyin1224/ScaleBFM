#!/usr/bin/env python3
"""Build HAT4 with corrected FOCUS labels without rerunning body simulation.

The body motion arrays are copied bit-for-bit from the validated legacy HAT4
package.  Only the three FOCUS arrays are replaced after exact nearest-neighbor
resampling of newly converted 30 Hz labels to the existing 50 Hz timeline.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess

import joblib
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
CONVERTER = REPO / "scripts" / "convert_unifolm_to_scalebfm.py"
PYTHON = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
DEFAULT_DATASET_ROOT = Path("/mnt/nvme0n1/DATASETS/UnifoLM_WBT")
DEFAULT_LEGACY_ROOT = REPO / "ScaleTrack" / "data" / "hat4_unifolm"
DEFAULT_OUTPUT_ROOT = REPO / "ScaleTrack" / "data" / "hat4_unifolm_focus_v2"
FOCUS_FIELDS = ("focus_mask", "focus_time_to_next_edge_s", "focus_valid")
BODY_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "fps",
)
FAMILIES = (
    "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
    "G1_WBT_Brainco_Make_The_Bed",
    "G1_WBT_Brainco_Pick_Up_Medicine",
    "G1_WBT_Brainco_Pickup_Pillow",
    "G1_WBT_Inspire_Clean_The_Living_Room",
    "G1_WBT_Inspire_Pick_Up_Drinks",
    "G1_WBT_Inspire_Put_Clothes_Into_Basket",
    "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine",
    "G1_WBT_Inspire_Put_Drinks_Into_Fridge",
    "G1_WBT_Inspire_Put_Vegetables_Into_Basket",
)


def family_counts(dataset_root: Path) -> dict[str, int]:
    counts = {}
    for family in FAMILIES:
        info = json.loads((dataset_root / family / "meta" / "info.json").read_text())
        if float(info["fps"]) != 30.0:
            raise ValueError(f"{family}: expected 30 Hz, got {info['fps']}")
        counts[family] = int(info["total_episodes"])
    return counts


def convert(dataset_root: Path, output_root: Path, overwrite: bool) -> None:
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for family, count in family_counts(dataset_root).items():
        existing = list(raw_dir.glob(f"{family}_ep_*.pkl"))
        if len(existing) == count and not overwrite:
            print(f"[skip] {family}: {count} v2 raw files", flush=True)
            continue
        command = [
            str(PYTHON),
            str(CONVERTER),
            "--dataset",
            str(dataset_root / family),
            "--output-dir",
            str(raw_dir),
        ]
        if overwrite or existing:
            command.append("--overwrite")
        print("[run] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def nearest_source_indices(input_frames: int, input_fps: float, output_fps: float) -> np.ndarray:
    duration = (input_frames - 1) / float(input_fps)
    times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float32)
    phase = times / duration
    return np.clip(
        np.rint(phase * (input_frames - 1)).astype(np.int64),
        0,
        input_frames - 1,
    )


def relabel_one(task: tuple[str, str, str, bool]) -> dict[str, object]:
    raw_name, source_name, output_name, resume = task
    raw_path = Path(raw_name)
    source_path = Path(source_name)
    output_path = Path(output_name)
    raw = joblib.load(raw_path)
    source_frames = len(raw["focus_mask"])
    nearest = nearest_source_indices(source_frames, float(raw["fps"]), 50.0)
    expected = {
        field: np.asarray(raw[field])[nearest]
        for field in FOCUS_FIELDS
    }

    if output_path.is_file() and resume:
        with np.load(output_path) as existing:
            if all(np.array_equal(existing[field], expected[field]) for field in FOCUS_FIELDS):
                return {
                    "file": output_path.name,
                    "frames": len(nearest),
                    "focus": int(expected["focus_mask"].sum()),
                    "skipped": True,
                }
        raise ValueError(f"Existing output failed resume validation: {output_path}")
    if output_path.exists():
        raise FileExistsError(f"{output_path} exists; use --resume or --overwrite")

    with np.load(source_path) as source:
        if set(source.files) != set(BODY_FIELDS + FOCUS_FIELDS):
            raise ValueError(f"Unexpected NPZ fields in {source_path}: {source.files}")
        arrays = {field: np.asarray(source[field]) for field in BODY_FIELDS}
        if arrays["joint_pos"].shape[0] != len(nearest):
            raise ValueError(
                f"Timeline mismatch for {source_path}: "
                f"processed={arrays['joint_pos'].shape[0]} expected={len(nearest)}"
            )
    arrays.update(expected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, output_path)
    return {
        "file": output_path.name,
        "frames": len(nearest),
        "focus": int(expected["focus_mask"].sum()),
        "skipped": False,
    }


def relabel(
    legacy_root: Path,
    output_root: Path,
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    raw_dir = output_root / "raw"
    source_dir = legacy_root / "processed"
    output_dir = output_root / "processed"
    raw_paths = sorted(raw_dir.glob("*.pkl"))
    source_paths = sorted(source_dir.glob("*.npz"))
    if len(raw_paths) != 3000 or len(source_paths) != 3000:
        raise RuntimeError(
            f"Expected 3000 raw and processed files, found {len(raw_paths)} and {len(source_paths)}"
        )
    if overwrite:
        for output_path in output_dir.glob("*.npz"):
            output_path.unlink()
    tasks = []
    for raw_path in raw_paths:
        source_path = source_dir / f"{raw_path.stem}.npz"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        tasks.append((str(raw_path), str(source_path), str(output_dir / source_path.name), resume))

    total_frames = total_focus = completed = skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(relabel_one, tasks, chunksize=4):
            completed += 1
            skipped += int(result["skipped"])
            total_frames += int(result["frames"])
            total_focus += int(result["focus"])
            if completed % 100 == 0 or completed == len(tasks):
                print(
                    f"[relabel] {completed}/{len(tasks)} skipped={skipped} "
                    f"focus={total_focus}/{total_frames}",
                    flush=True,
                )


def manifests(legacy_root: Path, output_root: Path) -> None:
    source_manifest_dir = legacy_root / "manifests"
    output_manifest_dir = output_root / "manifests"
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_root / "processed"
    for name in (
        "unifolm_hat4_train.yaml",
        "unifolm_hat4_heldout.yaml",
        "unifolm_hat4_all.yaml",
    ):
        source = yaml.safe_load((source_manifest_dir / name).read_text())
        replaced = {
            key: str((processed_dir / f"{key}.npz").resolve())
            for key in source
        }
        missing = [path for path in replaced.values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: missing {len(missing)} files; first={missing[0]}")
        (output_manifest_dir / name).write_text(yaml.safe_dump(replaced, sort_keys=True))


def validate(legacy_root: Path, output_root: Path) -> None:
    raw_dir = output_root / "raw"
    source_dir = legacy_root / "processed"
    output_dir = output_root / "processed"
    family_report = {}
    totals = {"episodes": 0, "frames_50hz": 0, "focus_frames_50hz": 0}
    for family in FAMILIES:
        paths = sorted(output_dir.glob(f"{family}_ep_*.npz"))
        family_frames = family_focus = 0
        for output_path in paths:
            raw = joblib.load(raw_dir / f"{output_path.stem}.pkl")
            nearest = nearest_source_indices(len(raw["focus_mask"]), float(raw["fps"]), 50.0)
            with np.load(source_dir / output_path.name) as source, np.load(output_path) as output:
                for field in BODY_FIELDS:
                    if not np.array_equal(source[field], output[field]):
                        raise AssertionError(f"Body field changed: {output_path} {field}")
                for field in FOCUS_FIELDS:
                    expected = np.asarray(raw[field])[nearest]
                    if not np.array_equal(output[field], expected):
                        raise AssertionError(f"FOCUS mismatch: {output_path} {field}")
                frames = int(output["joint_pos"].shape[0])
                focus = int(np.asarray(output["focus_mask"], dtype=bool).sum())
            family_frames += frames
            family_focus += focus
        family_report[family] = {
            "episodes": len(paths),
            "frames_50hz": family_frames,
            "focus_frames_50hz": family_focus,
            "focus_fraction": family_focus / family_frames,
        }
        totals["episodes"] += len(paths)
        totals["frames_50hz"] += family_frames
        totals["focus_frames_50hz"] += family_focus
        print(
            f"[validate] {family}: episodes={len(paths)} "
            f"focus={family_focus / family_frames:.4f}",
            flush=True,
        )
    totals["focus_fraction"] = totals["focus_frames_50hz"] / totals["frames_50hz"]
    audit = {
        "definition": "focus_v2_physical_hand_closure",
        "body_arrays": "bit-identical to hat4_unifolm/processed",
        "focus_resampling": "same 30 Hz -> 50 Hz nearest-neighbor timeline as package_motions.py",
        "families": family_report,
        "totals": totals,
    }
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "focus_v2_processed_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    (output_root / "README.md").write_text(
        "# HAT4 FOCUS v2\n\n"
        "This is the 3000-episode HAT4 corpus with embodiment-aware FOCUS labels.\n"
        "BrainCo uses physical closure channels `[0,2,3,4,5]` per hand scaled by "
        "the 0.8 hardware close value; Inspire uses `1-state` on channels "
        "`[0,1,2,3,4]` per hand. Thumb side/yaw controls are excluded, channels "
        "must cross the hysteresis band, hands are processed independently, and "
        "episode boundaries are censored. Body arrays are bit-identical to the "
        "legacy HAT4 package.\n"
    )
    print(json.dumps(totals, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("convert", "relabel", "manifests", "validate", "all"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.stage in ("convert", "all"):
        convert(args.dataset_root, args.output_root, args.overwrite)
    if args.stage in ("relabel", "all"):
        relabel(args.legacy_root, args.output_root, args.workers, args.resume, args.overwrite)
    if args.stage in ("manifests", "all"):
        manifests(args.legacy_root, args.output_root)
    if args.stage in ("validate", "all"):
        validate(args.legacy_root, args.output_root)


if __name__ == "__main__":
    main()
