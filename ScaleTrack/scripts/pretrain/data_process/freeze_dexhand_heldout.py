#!/usr/bin/env python3
"""Select, audit, and freeze one global object-bound dexhand held-out set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _is_object_bound(row: dict) -> bool:
    return bool(row.get("object_rh_usd") or row.get("object_lh_usd"))


def _requirement_matches(row: dict, requirement: str) -> bool:
    source, separator, pattern = requirement.partition(":")
    if not separator:
        raise ValueError(
            f"required category {requirement!r} must use SOURCE:TEXT"
        )
    if row["source"].lower() != source.lower():
        return False
    pattern = pattern.lower()
    name = str(row.get("name", "")).lower()
    primitive = str(row.get("primitive", "")).lower()
    task_group = str(row.get("task_group", "")).lower()
    return (
        pattern == name
        or pattern == primitive
        or pattern == task_group
        or task_group.endswith(f":{pattern}")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="SOURCE:TEXT",
        help="Force one object-bound motion matching source and primitive/task text",
    )
    parser.add_argument(
        "--min-source",
        action="append",
        default=[],
        metavar="SOURCE=COUNT",
        help="Guarantee at least COUNT held-out motions from a source",
    )
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = raw["motions"]
    for row in rows:
        row["split"] = "train"
    eligible = [row for row in rows if _is_object_bound(row)]
    if len(eligible) < args.count:
        raise ValueError(
            f"requested {args.count} object-bound heldout motions, only "
            f"{len(eligible)} are eligible"
        )

    rng = np.random.default_rng(args.seed)
    selected: dict[str, str] = {}
    requirement_hits = {}
    for requirement in args.require:
        candidates = [
            row
            for row in eligible
            if row["name"] not in selected
            and _requirement_matches(row, requirement)
        ]
        if not candidates:
            raise ValueError(
                f"required heldout category has no unused object-bound match: "
                f"{requirement}"
            )
        candidates.sort(key=lambda row: row["name"])
        chosen = candidates[int(rng.integers(len(candidates)))]
        selected[chosen["name"]] = f"required:{requirement}"
        requirement_hits[requirement] = chosen["name"]

    minimum_sources = {}
    for value in args.min_source:
        source, separator, raw_count = value.partition("=")
        if not separator or not source or not raw_count.isdigit():
            raise ValueError(f"invalid --min-source value {value!r}; expected SOURCE=COUNT")
        minimum_sources[source] = int(raw_count)
    for source, minimum in minimum_sources.items():
        existing = sum(
            row_by_source.lower() == source.lower()
            for name in selected
            for row_by_source in (
                next(row["source"] for row in eligible if row["name"] == name),
            )
        )
        candidates = [
            row
            for row in eligible
            if row["source"].lower() == source.lower()
            and row["name"] not in selected
        ]
        rng.shuffle(candidates)
        needed = max(0, minimum - existing)
        if len(candidates) < needed:
            raise ValueError(
                f"source {source!r} needs {needed} additional heldout motions, "
                f"only {len(candidates)} are available"
            )
        for row in candidates[:needed]:
            selected[row["name"]] = f"minimum_source:{source}"

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        if row["name"] not in selected:
            groups[(row["source"], row["task_group"])].append(row)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    depth = 0
    while len(selected) < args.count:
        made_progress = False
        for key in keys:
            if depth < len(groups[key]):
                row = groups[key][depth]
                if row["name"] not in selected:
                    selected[row["name"]] = "stratified:source+task_group"
                    made_progress = True
                    if len(selected) == args.count:
                        break
        if not made_progress:
            raise RuntimeError("heldout stratification exhausted unexpectedly")
        depth += 1

    for row in rows:
        if row["name"] in selected:
            row["split"] = "heldout"

    # Recompute train weights after removal. Every task group has equal total
    # mass and heldout weights remain positive solely for standalone evaluation.
    train_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            train_groups[row["task_group"]].append(row)
    for values in train_groups.values():
        for row in values:
            row["sampling_weight"] = 1.0 / len(values)

    details = []
    row_by_name = {row["name"]: row for row in rows}
    for name in sorted(selected):
        row = row_by_name[name]
        details.append(
            {
                "name": name,
                "source": row["source"],
                "primitive": row["primitive"],
                "task_group": row["task_group"],
                "selection_reason": selected[name],
                "object_rh_usd": row.get("object_rh_usd"),
                "object_lh_usd": row.get("object_lh_usd"),
            }
        )
    digest_payload = json.dumps(
        [(value["source"], value["name"]) for value in details],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(digest_payload).hexdigest()
    source_counts = Counter(value["source"] for value in details)

    result_manifest = dict(raw)
    result_manifest["motions"] = rows
    result_manifest["heldout_freeze"] = {
        "frozen": True,
        "count": len(details),
        "seed": args.seed,
        "selection_sha256": digest,
        "object_bound_only": True,
        "required_categories": requirement_hits,
        "minimum_source_counts": minimum_sources,
        "source_counts": dict(source_counts),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(result_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    heldout = {
        "schema_version": 1,
        "frozen": True,
        "seed": args.seed,
        "count": len(details),
        "selection_sha256": digest,
        "object_bound_only": True,
        "required_categories": requirement_hits,
        "minimum_source_counts": minimum_sources,
        "source_counts": dict(source_counts),
        "motions": details,
    }
    args.heldout.parent.mkdir(parents=True, exist_ok=True)
    args.heldout.write_text(
        json.dumps(heldout, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"heldout={len(details)} sources={dict(source_counts)} "
        f"sha256={digest} manifest={args.output_manifest}"
    )


if __name__ == "__main__":
    main()
