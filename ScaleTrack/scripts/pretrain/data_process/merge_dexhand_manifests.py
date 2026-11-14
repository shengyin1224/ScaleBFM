#!/usr/bin/env python3
"""Merge per-source manifests and re-normalize source/task sampling mass."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    seen = set()
    coverages = {}
    for path in args.inputs:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported schema")
        coverages[path.stem] = raw.get("coverage", {})
        for row in raw["motions"]:
            name = row["name"]
            if name in seen:
                raise ValueError(f"duplicate motion name: {name}")
            seen.add(name)
            rows.append(row)
    train_groups = defaultdict(list)
    for row in rows:
        if row.get("split", "train") == "train":
            train_groups[row["task_group"]].append(row)
    for group in train_groups.values():
        for row in group:
            row["sampling_weight"] = 1.0 / len(group)
    result = {
        "schema_version": 1,
        "fps": 50,
        "motions": rows,
        "coverage_by_input": coverages,
        "weighting": "equal task_group mass; uniform within group",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

