#!/usr/bin/env python3
"""Build a repeated-pose static reference from a packaged ScaleTrack NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, default=5)
    parser.add_argument("--frames", type=int, default=251)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.source) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    source_frames = int(payload["joint_pos"].shape[0])
    if not 0 <= args.source_frame < source_frames:
        raise ValueError(args.source_frame)
    output = {}
    for key, value in payload.items():
        if value.ndim > 0 and value.shape[0] == source_frames:
            output[key] = np.repeat(value[args.source_frame : args.source_frame + 1], args.frames, axis=0)
        else:
            output[key] = value
    for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        output[key] = np.zeros_like(output[key])
    output["focus_mask"] = np.zeros(args.frames, dtype=bool)
    output["focus_valid"] = np.ones(args.frames, dtype=bool)
    output["focus_time_to_next_edge_s"] = np.full(args.frames, 2.0, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(args.output)


if __name__ == "__main__":
    main()
