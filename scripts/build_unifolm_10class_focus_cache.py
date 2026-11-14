#!/usr/bin/env python3
"""Build TWIST-compatible FOCUS masks for ep0000 of the ten WBT families."""

from pathlib import Path
import sys

import numpy as np
import torch


TWIST_GATE_DIR = Path(
    "/root/shengyin/TWIST/shengyin_train_unifolm/grasp_still_gate_viz"
)
sys.path.insert(0, str(TWIST_GATE_DIR))
import detect_gates as dg  # noqa: E402


FAMILIES = [
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
]
OUT = Path(
    "/root/shengyin/ScaleBFM/ScaleTrack/action_error_results/"
    "unifolm_10class_ep0000_focus_cache.pt"
)


def main() -> None:
    focuses = []
    valids = []
    phases = []
    starts = []
    lengths = []
    files = []
    source_parquets = []
    cursor = 0

    for family in FAMILIES:
        stem = f"{family}_ep_0000"
        hand, parquet, _ = dg.load_source_hand(stem)
        signal, active_columns = dg.hand_signal(hand, dg.HAND_REDUCE)
        grasp = dg.hysteresis(signal, dg.GRASP_ENTER, dg.GRASP_EXIT)
        focus = dg.focus_window(grasp, fps=30, pad_s=dg.FOCUS_PAD_S)
        phase = dg.focus_phase(grasp, fps=30, pad_s=dg.FOCUS_PAD_S)

        starts.append(cursor)
        lengths.append(len(focus))
        files.append(f"{stem}.pkl")
        source_parquets.append(str(parquet))
        focuses.append(focus)
        valids.append(np.ones(len(focus), dtype=bool))
        phases.append(phase)
        cursor += len(focus)
        print(
            f"{stem}: frames={len(focus)}, focus={int(focus.sum())} "
            f"({100.0 * focus.mean():.1f}%), active_hand_cols={len(active_columns)}"
        )

    payload = {
        "focus": torch.from_numpy(np.concatenate(focuses)),
        "valid": torch.from_numpy(np.concatenate(valids)),
        "phase": torch.from_numpy(np.concatenate(phases)),
        "motion_start_idx": torch.tensor(starts, dtype=torch.long),
        "motion_num_frames": torch.tensor(lengths, dtype=torch.long),
        "motion_files": files,
        "source_parquet_files": source_parquets,
        "grasp_enter": dg.GRASP_ENTER,
        "grasp_exit": dg.GRASP_EXIT,
        "hand_reduce": dg.HAND_REDUCE,
        "focus_pad_s": dg.FOCUS_PAD_S,
        "focus_mode": "grasp_endpoint_windows",
        "source_fps": 30,
        "generation": "TWIST detect_gates.py exact algorithm",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
