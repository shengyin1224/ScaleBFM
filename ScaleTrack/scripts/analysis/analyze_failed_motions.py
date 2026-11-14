#!/usr/bin/env python3
"""Analyze overlap and reference quality for ScaleBFM failed-motion lists."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


CATEGORIES = (
    "Collect_Plates_Into_Dishwasher",
    "Make_The_Bed",
    "Pick_Up_Medicine",
    "Pickup_Pillow",
    "Clean_The_Living_Room",
    "Pick_Up_Drinks",
    "Put_Clothes_Into_Basket",
    "Put_Clothes_into_Washing_Machine",
    "Put_Drinks_Into_Fridge",
    "Put_Vegetables_Into_Basket",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representatives", type=int, default=10)
    return parser.parse_args()


def category(name: str) -> str:
    matches = [value for value in CATEGORIES if value in name]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely categorize {name}: {matches}")
    return matches[0]


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def reference_audit(path: Path) -> dict:
    with np.load(path) as motion:
        fps = float(motion["fps"])
        body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        body_lin_vel = np.asarray(motion["body_lin_vel_w"], dtype=np.float64)
        body_ang_vel = np.asarray(motion["body_ang_vel_w"], dtype=np.float64)

    body_step = np.linalg.norm(np.diff(body_pos, axis=0), axis=-1)
    root_step = body_step[:, 0]
    joint_step = np.abs(np.diff(joint_pos, axis=0))
    lin_speed = np.linalg.norm(body_lin_vel, axis=-1)
    ang_speed = np.linalg.norm(body_ang_vel, axis=-1)
    max_root_step = float(root_step.max())
    max_body_step = float(body_step.max())
    max_joint_step = float(joint_step.max())
    max_lin_speed = float(lin_speed.max())
    max_ang_speed = float(ang_speed.max())
    body_step_per_transition = body_step.max(axis=1)
    discontinuity = (root_step > 0.10) | (body_step_per_transition > 0.15)
    discontinuity_steps = np.flatnonzero(discontinuity) + 1
    reasons = []
    if max_root_step > 0.10:
        reasons.append("root_step_gt_10cm")
    if max_body_step > 0.15:
        reasons.append("body_step_gt_15cm")
    if max_joint_step > 1.0:
        reasons.append("joint_step_gt_1rad")
    if max_lin_speed > 5.0:
        reasons.append("body_speed_gt_5mps")
    if max_ang_speed > 20.0:
        reasons.append("body_ang_speed_gt_20radps")
    return {
        "path": str(path.resolve()),
        "frames": int(len(joint_pos)),
        "fps": fps,
        "duration_s": float((len(joint_pos) - 1) / fps),
        "root_step_m": summarize(root_step),
        "body_step_m": summarize(body_step.reshape(-1)),
        "joint_step_rad": summarize(joint_step.reshape(-1)),
        "body_linear_speed_mps": summarize(lin_speed.reshape(-1)),
        "body_angular_speed_radps": summarize(ang_speed.reshape(-1)),
        "max_root_step_transition": int(np.argmax(root_step) + 1),
        "max_body_step_transition": int(np.argmax(body_step_per_transition) + 1),
        "discontinuity_transition_count": int(len(discontinuity_steps)),
        "discontinuity_transitions": discontinuity_steps.tolist(),
        "first_discontinuity_transition": (
            int(discontinuity_steps[0]) if len(discontinuity_steps) else None
        ),
        "last_discontinuity_transition": (
            int(discontinuity_steps[-1]) if len(discontinuity_steps) else None
        ),
        "prefix_trim_candidate": bool(
            len(discontinuity_steps) and int(discontinuity_steps[-1]) <= int(fps)
        ),
        "suspicious_reference": bool(reasons),
        "suspicious_reasons": reasons,
    }


def variant_name(path: Path) -> str:
    name = path.parent.name
    return name.removesuffix("_runtime")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load(args.manifest.read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"Invalid manifest: {args.manifest}")

    failure_paths = sorted(args.results_dir.glob("*_runtime/failed_train_motions.txt"))
    if not failure_paths:
        raise FileNotFoundError(f"No failed lists under {args.results_dir}")
    failed = {
        variant_name(path): set(filter(None, path.read_text().splitlines()))
        for path in failure_paths
    }
    all_names = set(manifest)
    for variant, names in failed.items():
        unknown = names - all_names
        if unknown:
            raise ValueError(f"{variant} has unknown motions: {sorted(unknown)[:3]}")

    common = set.intersection(*failed.values())
    union = set.union(*failed.values())
    frequency = Counter(name for names in failed.values() for name in names)
    variable = union - common
    pairwise = []
    for left, right in itertools.combinations(sorted(failed), 2):
        intersection = failed[left] & failed[right]
        pair_union = failed[left] | failed[right]
        pairwise.append(
            {
                "left": left,
                "right": right,
                "intersection": len(intersection),
                "union": len(pair_union),
                "jaccard": len(intersection) / len(pair_union),
            }
        )

    category_total = Counter(category(name) for name in all_names)
    common_by_category = Counter(category(name) for name in common)
    union_by_category = Counter(category(name) for name in union)
    per_variant_category = {
        variant: Counter(category(name) for name in names)
        for variant, names in failed.items()
    }

    audits = {}
    all_audits = {}
    for name in sorted(all_names):
        audit = reference_audit(Path(manifest[name]))
        all_audits[name] = audit
        if name in common:
            audits[name] = audit
    suspicious = [name for name, audit in audits.items() if audit["suspicious_reference"]]
    suspicious_all = {
        name for name, audit in all_audits.items() if audit["suspicious_reference"]
    }
    never_failed = all_names - union
    reference_quality_contingency = {
        "common_failures": {
            "suspicious": len(common & suspicious_all),
            "total": len(common),
        },
        "variable_failures": {
            "suspicious": len(variable & suspicious_all),
            "total": len(variable),
        },
        "never_failed": {
            "suspicious": len(never_failed & suspicious_all),
            "total": len(never_failed),
        },
        "all_motions": {
            "suspicious": len(suspicious_all),
            "total": len(all_names),
        },
        "common_failure_recall": len(common & suspicious_all) / len(common),
        "common_failure_precision": len(common & suspicious_all) / len(suspicious_all),
    }

    # Cover categories before adding further examples from high-failure buckets.
    representatives = []
    by_category = defaultdict(list)
    for name in sorted(common):
        by_category[category(name)].append(name)
    ordered_categories = sorted(
        CATEGORIES, key=lambda value: (-common_by_category[value], value)
    )
    for value in ordered_categories:
        candidates = [name for name in by_category[value] if name in suspicious]
        if not candidates:
            candidates = by_category[value]
        if candidates and len(representatives) < args.representatives:
            representatives.append(candidates[0])
    while len(representatives) < args.representatives:
        added = False
        for value in ordered_categories:
            candidates = [name for name in by_category[value] if name not in representatives]
            if candidates:
                representatives.append(candidates[0])
                added = True
                if len(representatives) >= args.representatives:
                    break
        if not added:
            break

    report = {
        "manifest": str(args.manifest.resolve()),
        "motions": len(manifest),
        "variants": {name: len(values) for name, values in sorted(failed.items())},
        "variant_count": len(failed),
        "common_failure_count": len(common),
        "union_failure_count": len(union),
        "variable_failure_count": len(variable),
        "common_fraction_of_union": len(common) / len(union),
        "pairwise_overlap": pairwise,
        "category_distribution": {
            value: {
                "total": category_total[value],
                "common_failed": common_by_category[value],
                "union_failed": union_by_category[value],
                "common_failure_rate": common_by_category[value] / category_total[value],
                "per_variant_failed": {
                    variant: counts[value]
                    for variant, counts in sorted(per_variant_category.items())
                },
            }
            for value in CATEGORIES
        },
        "common_failures": sorted(common),
        "variable_failures": [
            {
                "motion": name,
                "failed_variant_count": frequency[name],
                "failed_variants": sorted(
                    variant for variant, names in failed.items() if name in names
                ),
            }
            for name in sorted(variable)
        ],
        "reference_quality_thresholds": {
            "root_step_m": 0.10,
            "body_step_m": 0.15,
            "joint_step_rad": 1.0,
            "body_linear_speed_mps": 5.0,
            "body_angular_speed_radps": 20.0,
        },
        "suspicious_common_reference_count": len(suspicious),
        "suspicious_common_references": suspicious,
        "reference_quality_contingency": reference_quality_contingency,
        "prefix_trim_candidate_count_all": sum(
            audit["prefix_trim_candidate"] for audit in all_audits.values()
        ),
        "prefix_trim_candidate_count_common": sum(
            audit["prefix_trim_candidate"] for audit in audits.values()
        ),
        "reference_audits": audits,
        "representative_motions": representatives,
    }
    (args.output_dir / "failure_overlap_and_reference_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    with (args.output_dir / "category_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["category", "total", "common_failed", "union_failed", "common_failure_rate"]
        )
        for value in CATEGORIES:
            writer.writerow(
                [
                    value,
                    category_total[value],
                    common_by_category[value],
                    union_by_category[value],
                    common_by_category[value] / category_total[value],
                ]
            )

    (args.output_dir / "common_failures.txt").write_text(
        "\n".join(sorted(common)) + "\n"
    )
    (args.output_dir / "representative_motions.yaml").write_text(
        yaml.safe_dump(
            {name: str(Path(manifest[name]).resolve()) for name in representatives},
            sort_keys=False,
        )
    )
    print(args.output_dir / "failure_overlap_and_reference_audit.json")


if __name__ == "__main__":
    main()
