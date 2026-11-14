#!/usr/bin/env python3
"""Trim initial reference discontinuities and write a repaired manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--motion-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-prefix-frames", type=int, default=50)
    parser.add_argument("--root-step-threshold-m", type=float, default=0.10)
    parser.add_argument("--body-step-threshold-m", type=float, default=0.15)
    return parser.parse_args()


def discontinuities(
    body_pos: np.ndarray,
    root_threshold: float,
    body_threshold: float,
) -> np.ndarray:
    step = np.linalg.norm(np.diff(body_pos.astype(np.float64), axis=0), axis=-1)
    return np.flatnonzero(
        (step[:, 0] > root_threshold) | (step.max(axis=1) > body_threshold)
    ) + 1


def main() -> None:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    selected = set(filter(None, args.motion_list.read_text().splitlines()))
    unknown = selected - set(manifest)
    if unknown:
        raise ValueError(f"Unknown motions in list: {sorted(unknown)[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    repaired_manifest = {}
    records = []
    for name, source_value in manifest.items():
        source = Path(source_value).resolve()
        if name not in selected:
            repaired_manifest[name] = str(source)
            continue
        with np.load(source) as motion:
            payload = {key: np.asarray(motion[key]) for key in motion.files}
        frame_count = int(payload["joint_pos"].shape[0])
        bad = discontinuities(
            payload["body_pos_w"],
            args.root_step_threshold_m,
            args.body_step_threshold_m,
        )
        if not len(bad):
            repaired_manifest[name] = str(source)
            records.append(
                {"motion": name, "status": "selected_but_no_discontinuity", "source": str(source)}
            )
            continue
        if int(bad[-1]) > args.max_prefix_frames:
            repaired_manifest[name] = str(source)
            records.append(
                {
                    "motion": name,
                    "status": "not_repaired_non_prefix_discontinuity",
                    "source": str(source),
                    "discontinuity_transitions": bad.tolist(),
                }
            )
            continue

        start = int(bad[-1])
        repaired = {}
        for key, value in payload.items():
            if value.ndim > 0 and value.shape[0] == frame_count:
                repaired[key] = value[start:]
            else:
                repaired[key] = value
        remaining = discontinuities(
            repaired["body_pos_w"],
            args.root_step_threshold_m,
            args.body_step_threshold_m,
        )
        if len(remaining):
            raise RuntimeError(f"Repair left discontinuities in {name}: {remaining.tolist()}")
        target = (args.output_dir / f"{name}.npz").resolve()
        np.savez_compressed(target, **repaired)
        repaired_manifest[name] = str(target)
        records.append(
            {
                "motion": name,
                "status": "repaired",
                "source": str(source),
                "target": str(target),
                "removed_prefix_frames": start,
                "original_frames": frame_count,
                "repaired_frames": int(repaired["joint_pos"].shape[0]),
                "original_discontinuity_transitions": bad.tolist(),
            }
        )

    args.output_manifest.write_text(yaml.safe_dump(repaired_manifest, sort_keys=False))
    summary = {
        "source_manifest": str(args.manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "selected_motions": len(selected),
        "repaired_motions": sum(record["status"] == "repaired" for record in records),
        "unchanged_selected_motions": sum(record["status"] != "repaired" for record in records),
        "thresholds": {
            "max_prefix_frames": args.max_prefix_frames,
            "root_step_m": args.root_step_threshold_m,
            "body_step_m": args.body_step_threshold_m,
        },
        "records": records,
    }
    report = args.output_manifest.with_suffix(".repair.json")
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
