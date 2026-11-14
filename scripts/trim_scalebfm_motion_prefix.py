#!/usr/bin/env python3
"""Copy a ScaleBFM raw motion while dropping known-corrupt prefix frames."""

import argparse
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=1)
    args = parser.parse_args()

    motion = joblib.load(args.input)
    result = dict(motion)
    lengths = {
        len(np.asarray(value))
        for key, value in motion.items()
        if key != "fps" and np.asarray(value).ndim > 0
    }
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent motion lengths: {sorted(lengths)}")
    original_length = lengths.pop()
    if not 0 < args.frames < original_length:
        raise ValueError("--frames must leave at least one motion frame")
    for key, value in motion.items():
        array = np.asarray(value)
        if key != "fps" and array.ndim > 0 and len(array) == original_length:
            result[key] = array[args.frames :].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, args.output)
    print(f"{args.input}: {original_length} -> {original_length - args.frames} frames")
    print(args.output)


if __name__ == "__main__":
    main()
