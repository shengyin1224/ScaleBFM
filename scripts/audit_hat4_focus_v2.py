#!/usr/bin/env python3
"""Audit legacy and embodiment-aware FOCUS labels across the HAT4 corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from focus_definition import (
    ACTIVE_RANGE,
    focus_from_hand_state,
    focus_window,
    hysteresis,
    infer_hand_type,
    true_segments,
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

HAND_FIELD = "observation.state.hand_state"
COMMAND_FIELD = "action.hand_cmd"


def legacy_signal(hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(np.ptp(hand, axis=0) > ACTIVE_RANGE)
    if len(active) == 0:
        return np.zeros(len(hand), dtype=np.float32), active
    norm = hand[:, active] / np.maximum(hand[:, active].max(axis=0), 1.0e-6)
    return norm.max(axis=1).astype(np.float32), active


def load_family(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_parts = []
    hand_parts = []
    command_parts = []
    for path in sorted((dataset / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", HAND_FIELD, COMMAND_FIELD])
        episode_parts.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
        hand_parts.append(np.asarray(table[HAND_FIELD].to_pylist(), dtype=np.float32))
        command_parts.append(np.asarray(table[COMMAND_FIELD].to_pylist(), dtype=np.float32))
    if not episode_parts:
        raise FileNotFoundError(f"No parquet data under {dataset}")
    return (
        np.concatenate(episode_parts),
        np.concatenate(hand_parts),
        np.concatenate(command_parts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/nvme0n1/DATASETS/UnifoLM_WBT"),
    )
    parser.add_argument(
        "--legacy-raw-dir",
        type=Path,
        default=Path("/root/shengyin/ScaleBFM/ScaleTrack/data/hat4_unifolm/raw"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "definition": "focus_v2_physical_hand_closure",
        "dataset_root": str(args.dataset_root.resolve()),
        "families": {},
    }
    totals: dict[str, int] = {
        key: 0
        for key in (
            "episodes",
            "source_frames_30hz",
            "legacy_focus_frames_30hz",
            "v2_focus_frames_30hz",
            "focus_disagreement_frames_30hz",
            "legacy_grasp_starts_at_zero",
            "legacy_grasp_ends_at_last",
            "legacy_full_episode_grasp",
            "v2_grasp_starts_at_zero",
            "v2_grasp_ends_at_last",
            "v2_full_episode_grasp",
            "v2_zero_focus_episodes",
            "v2_grasp_segments",
            "v2_observed_endpoints",
            "legacy_start_argmax_nonclosure_thumb",
        )
    }
    command_abs_errors = []
    command_focus_disagreement = 0

    for family in FAMILIES:
        dataset = args.dataset_root / family
        hand_type = infer_hand_type(dataset)
        episode_index, hands, commands = load_family(dataset)
        family_count = int(episode_index.max()) + 1
        stats = {key: 0 for key in totals}
        stats["episodes"] = family_count
        stats["source_frames_30hz"] = len(hands)
        family_command_error = np.abs(hands - commands)
        command_abs_errors.append(family_command_error.reshape(-1))

        for episode_id in range(family_count):
            selector = episode_index == episode_id
            hand = hands[selector]
            command = commands[selector]
            old_signal, old_active = legacy_signal(hand)
            old_grasp = hysteresis(old_signal)
            old_focus = focus_window(
                old_grasp, 30.0, censor_episode_boundaries=False
            )
            new_focus, new_grasp, _, _ = focus_from_hand_state(
                hand, 30.0, hand_type
            )
            command_focus, _, _, _ = focus_from_hand_state(
                command, 30.0, hand_type
            )

            raw_path = args.legacy_raw_dir / f"{family}_ep_{episode_id:04d}.pkl"
            if raw_path.is_file():
                saved = np.asarray(joblib.load(raw_path)["focus_mask"], dtype=bool)
                if not np.array_equal(saved, old_focus):
                    raise AssertionError(f"Legacy raw mismatch: {raw_path}")

            stats["legacy_focus_frames_30hz"] += int(old_focus.sum())
            stats["v2_focus_frames_30hz"] += int(new_focus.sum())
            stats["focus_disagreement_frames_30hz"] += int(
                np.count_nonzero(old_focus != new_focus)
            )
            command_focus_disagreement += int(np.count_nonzero(new_focus != command_focus))

            old_segments = true_segments(old_grasp)
            new_segments = true_segments(new_grasp)
            old_start = bool(old_grasp[0])
            old_end = bool(old_grasp[-1])
            new_start = bool(new_grasp[0])
            new_end = bool(new_grasp[-1])
            stats["legacy_grasp_starts_at_zero"] += int(old_start)
            stats["legacy_grasp_ends_at_last"] += int(old_end)
            stats["legacy_full_episode_grasp"] += int(old_grasp.all())
            stats["v2_grasp_starts_at_zero"] += int(new_start)
            stats["v2_grasp_ends_at_last"] += int(new_end)
            stats["v2_full_episode_grasp"] += int(new_grasp.all())
            stats["v2_zero_focus_episodes"] += int(not new_focus.any())
            stats["v2_grasp_segments"] += len(new_segments)
            stats["v2_observed_endpoints"] += sum(
                int(start > 0) + int(end < len(new_grasp) - 1)
                for start, end in new_segments
            )
            if hand_type == "brainco" and old_start and len(old_active):
                normalized_start = hand[0, old_active] / np.maximum(
                    hand[:, old_active].max(axis=0), 1.0e-6
                )
                source_argmax = int(old_active[int(np.argmax(normalized_start))])
                stats["legacy_start_argmax_nonclosure_thumb"] += int(
                    source_argmax in (1, 7)
                )

        stats["legacy_focus_fraction"] = (
            stats["legacy_focus_frames_30hz"] / stats["source_frames_30hz"]
        )
        stats["v2_focus_fraction"] = (
            stats["v2_focus_frames_30hz"] / stats["source_frames_30hz"]
        )
        stats["focus_mask_disagreement_fraction"] = (
            stats["focus_disagreement_frames_30hz"] / stats["source_frames_30hz"]
        )
        stats["hand_type"] = hand_type
        report["families"][family] = stats
        for key in totals:
            totals[key] += int(stats[key])
        print(
            f"{family}: legacy={stats['legacy_focus_fraction']:.4f} "
            f"v2={stats['v2_focus_fraction']:.4f} "
            f"disagree={stats['focus_mask_disagreement_fraction']:.4f} "
            f"full_grasp={stats['legacy_full_episode_grasp']}->{stats['v2_full_episode_grasp']} "
            f"zero_focus={stats['v2_zero_focus_episodes']}"
        )

    all_errors = np.concatenate(command_abs_errors)
    totals["legacy_focus_fraction"] = (
        totals["legacy_focus_frames_30hz"] / totals["source_frames_30hz"]
    )
    totals["v2_focus_fraction"] = (
        totals["v2_focus_frames_30hz"] / totals["source_frames_30hz"]
    )
    totals["focus_mask_disagreement_fraction"] = (
        totals["focus_disagreement_frames_30hz"] / totals["source_frames_30hz"]
    )
    totals["hand_state_vs_command_median_abs"] = float(np.median(all_errors))
    totals["hand_state_vs_command_p99_abs"] = float(np.quantile(all_errors, 0.99))
    totals["hand_state_vs_command_focus_disagreement_fraction"] = (
        command_focus_disagreement / totals["source_frames_30hz"]
    )
    report["totals"] = totals
    report["manual_video_validation"] = {
        "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher_ep_0000": {
            "v2_grasp_frames": [242, 493],
            "result": "close/hold/release aligns with plate pickup and rack placement",
        },
        "G1_WBT_Brainco_Make_The_Bed_ep_0000": {
            "v2_grasp_frames": [359, 454],
            "result": "close/hold/release aligns with bedding manipulation",
        },
        "G1_WBT_Brainco_Pick_Up_Medicine_ep_0000": {
            "v2_grasp_segments": [[265, 358], [547, 722]],
            "result": "two events align with drawer manipulation and medicine pickup",
        },
        "G1_WBT_Brainco_Pickup_Pillow_ep_0000": {
            "v2_grasp_frames": [223, 404],
            "previous_manual_reference": [220, 405],
        },
        "G1_WBT_Brainco_Pickup_Pillow_ep_0149": {
            "v2_grasp_frames": [244, 466],
            "result": "max reduction retains visually valid partial-finger pillow grasp",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
