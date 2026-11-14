#!/usr/bin/env python3
"""Prepare the exact ten-family UnifoLM HAT-4 finetuning corpus.

The workflow is intentionally split into explicit stages because packaging
launches Isaac Sim:

  prepare_hat4_dataset.py convert
  prepare_hat4_dataset.py package
  prepare_hat4_dataset.py manifests

``all`` runs the three stages in that order.  The output manifests use an
episode-level, family-stratified deterministic 80/20 split.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml
import joblib


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
DATASET_ROOT = WORKSPACE / "DATASETS" / "UnifoLM_WBT"
RAW_DIR = REPO / "ScaleTrack" / "data" / "hat4_unifolm" / "raw"
PROCESSED_DIR = REPO / "ScaleTrack" / "data" / "hat4_unifolm" / "processed"
MANIFEST_DIR = REPO / "ScaleTrack" / "data" / "hat4_unifolm" / "manifests"
ISAAC_PYTHON = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
CONVERTER = REPO / "scripts" / "convert_unifolm_to_scalebfm.py"
PACKAGER = REPO / "ScaleTrack" / "scripts" / "pretrain" / "data_process" / "package_motions.py"

# This is the ten-class set used by the 0726 zero-shot table.  Nested Dex5
# exports and the Dex1 washing-machine duplicate are deliberately excluded.
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


def read_total_episodes(family: str) -> int:
    info_path = DATASET_ROOT / family / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text())
    if float(info["fps"]) != 30.0:
        raise ValueError(f"{family}: expected 30 Hz source, got {info['fps']}")
    return int(info["total_episodes"])


def convert(overwrite: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        expected = read_total_episodes(family)
        existing = list(RAW_DIR.glob(f"{family}_ep_*.pkl"))
        if len(existing) == expected and not overwrite:
            print(f"[skip] {family}: {expected} raw motions already exist")
            continue
        command = [
            str(ISAAC_PYTHON),
            str(CONVERTER),
            "--dataset",
            str(DATASET_ROOT / family),
            "--output-dir",
            str(RAW_DIR),
        ]
        if overwrite or existing:
            command.append("--overwrite")
        print("[run] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def package(
    num_envs: int,
    loader_workers: int,
    overwrite: bool,
    resume: bool,
) -> None:
    expected = sum(read_total_episodes(family) for family in FAMILIES)
    raw = list(RAW_DIR.glob("*.pkl"))
    if len(raw) != expected:
        raise RuntimeError(
            f"Raw corpus incomplete: found {len(raw)}, expected {expected}; run convert first"
        )
    existing = list(PROCESSED_DIR.glob("*.npz"))
    if existing and not overwrite and not resume:
        if len(existing) == expected:
            print(f"[skip] all {expected} packaged motions already exist")
            return
        raise RuntimeError(
            f"{PROCESSED_DIR} contains {len(existing)} partial outputs; "
            "pass --overwrite-package after reviewing them"
        )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(ISAAC_PYTHON),
        str(PACKAGER),
        "--data_dir",
        str(RAW_DIR),
        "--output_dir",
        str(PROCESSED_DIR),
        "--output_fps",
        "50",
        "--num_envs",
        str(num_envs),
        "--loader_workers",
        str(loader_workers),
        "--headless",
    ]
    if resume:
        command.append("--skip_existing")
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO / "ScaleTrack", check=True)


def episode_path(family: str, episode: int) -> Path:
    return PROCESSED_DIR / f"{family}_ep_{episode:04d}.npz"


def write_manifest(path: Path, motions: list[Path]) -> None:
    payload = {motion.stem: str(motion.resolve()) for motion in motions}
    path.write_text(yaml.safe_dump(payload, sort_keys=True))


def validate_processed(path: Path) -> tuple[int, int, float]:
    with np.load(path) as data:
        frames = int(data["joint_pos"].shape[0])
        required = (
            "focus_mask",
            "focus_time_to_next_edge_s",
            "focus_valid",
        )
        for key in required:
            if key not in data.files or data[key].shape != (frames,):
                raise ValueError(f"{path}: missing/misaligned {key}")
        if not np.asarray(data["focus_valid"]).all():
            raise ValueError(f"{path}: source UnifoLM motion unexpectedly has invalid FOCUS")
        packaged_focus_frames = int(
            np.asarray(data["focus_mask"], dtype=bool).sum()
        )
        packaged_root = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
    raw = joblib.load(RAW_DIR / f"{path.stem}.pkl")
    source_root = np.asarray(raw["root_pos"], dtype=np.float64)
    duration = (len(source_root) - 1) / float(raw["fps"])
    times = np.arange(0.0, duration, 1.0 / 50.0, dtype=np.float32)
    phase = times / duration
    index0 = np.floor(phase * (len(source_root) - 1)).astype(np.int64)
    index1 = np.minimum(index0 + 1, len(source_root) - 1)
    blend = phase * (len(source_root) - 1) - index0
    expected_root = (
        source_root[index0] * (1.0 - blend[:, None])
        + source_root[index1] * blend[:, None]
    )
    canonical_error = float(
        np.max(np.linalg.norm(packaged_root - expected_root, axis=-1))
    )
    if canonical_error > 1.0e-4:
        raise ValueError(
            f"{path}: packaged root is not canonical; max source error "
            f"{canonical_error:.6f} m (likely leaked scene.env_origins)"
        )
    return (
        frames,
        packaged_focus_frames,
        canonical_error,
    )


def manifests(seed: int) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    train: list[Path] = []
    heldout: list[Path] = []
    family_report = {}
    total_frames = total_focus = 0
    canonical_root_max_error_m = 0.0
    for family_index, family in enumerate(FAMILIES):
        count = read_total_episodes(family)
        ids = list(range(count))
        random.Random(seed + family_index).shuffle(ids)
        train_count = int(0.8 * count)
        family_train = sorted(ids[:train_count])
        family_heldout = sorted(ids[train_count:])
        paths = [episode_path(family, episode) for episode in range(count)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{family}: {len(missing)} packaged episodes missing; first={missing[0]}"
            )
        for path in paths:
            frames, focus, canonical_error = validate_processed(path)
            total_frames += frames
            total_focus += focus
            canonical_root_max_error_m = max(
                canonical_root_max_error_m, canonical_error
            )
        train.extend(episode_path(family, episode) for episode in family_train)
        heldout.extend(episode_path(family, episode) for episode in family_heldout)
        family_report[family] = {
            "episodes": count,
            "train": len(family_train),
            "heldout": len(family_heldout),
            "train_episode_ids": family_train,
            "heldout_episode_ids": family_heldout,
        }

    if set(train) & set(heldout):
        raise AssertionError("train/heldout overlap")
    write_manifest(MANIFEST_DIR / "unifolm_hat4_train.yaml", train)
    write_manifest(MANIFEST_DIR / "unifolm_hat4_heldout.yaml", heldout)
    write_manifest(MANIFEST_DIR / "unifolm_hat4_all.yaml", train + heldout)
    report = {
        "seed": seed,
        "split": "family-stratified episode-level 80/20",
        "families": family_report,
        "train_episodes": len(train),
        "heldout_episodes": len(heldout),
        "total_episodes": len(train) + len(heldout),
        "total_frames_50hz": total_frames,
        "focus_frames_50hz": total_focus,
        "focus_fraction": total_focus / total_frames,
        "canonical_root_max_error_m": canonical_root_max_error_m,
    }
    (MANIFEST_DIR / "split_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "families"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("convert", "package", "manifests", "all")
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--loader-workers", type=int, default=32)
    parser.add_argument("--overwrite-convert", action="store_true")
    parser.add_argument("--overwrite-package", action="store_true")
    parser.add_argument("--resume-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ("convert", "all"):
        convert(args.overwrite_convert)
    if args.stage in ("package", "all"):
        if args.overwrite_package and args.resume_package:
            raise ValueError(
                "--overwrite-package and --resume-package are mutually exclusive"
            )
        package(
            args.num_envs,
            args.loader_workers,
            args.overwrite_package,
            args.resume_package,
        )
    if args.stage in ("manifests", "all"):
        manifests(args.seed)


if __name__ == "__main__":
    main()
