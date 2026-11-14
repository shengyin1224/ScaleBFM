#!/usr/bin/env python3
"""Instantiate HAT-4 env configs and print pairwise structural differences."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def leaf(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                result.append(
                    {"path": child, "left": "<MISSING>", "right": normalize(right[key])}
                )
            elif key not in right:
                result.append(
                    {"path": child, "left": normalize(left[key]), "right": "<MISSING>"}
                )
            else:
                result.extend(differences(left[key], right[key], child))
        return result
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return [{"path": path, "left": normalize(left), "right": normalize(right)}]
        result = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result.extend(differences(left_item, right_item, f"{path}[{index}]"))
        return result
    left_value = leaf(left)
    right_value = leaf(right)
    if left_value != right_value:
        return [{"path": path, "left": left_value, "right": right_value}]
    return []


def normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [normalize(item) for item in value]
    return leaf(value)


def main() -> None:
    args = parse_args()
    simulation_app = AppLauncher(headless=True).app
    try:
        from scaletrack.tasks.tracking.config.g1_29dof.hat4_finetune_env_cfg import (
            G1HAT4FinetuneEnvCfg,
            G1HAT4FocusFinetuneEnvCfg,
            G1HAT4FocusHeadFinetuneEnvCfg,
            G1HAT4FocusRewardFinetuneEnvCfg,
            G1HAT4PhaseObsOnlyFinetuneEnvCfg,
        )

        configs = {
            "F1": G1HAT4FinetuneEnvCfg().to_dict(),
            "F2": G1HAT4FocusRewardFinetuneEnvCfg().to_dict(),
            "F2o": G1HAT4PhaseObsOnlyFinetuneEnvCfg().to_dict(),
            "F3": G1HAT4FocusFinetuneEnvCfg().to_dict(),
            "F3h": G1HAT4FocusHeadFinetuneEnvCfg().to_dict(),
        }
        pairs = (
            ("F1", "F2"),
            ("F1", "F2o"),
            ("F2", "F3"),
            ("F2o", "F3"),
            ("F3", "F3h"),
        )
        expected_paths = {
            "F1_vs_F2": {
                "focus_weight",
                "phase_weighted_body_names",
                "rewards.motion_body_pos.params.focus_weight",
                "rewards.motion_body_pos.params.phase_weighted_body_names",
                "rewards.motion_body_rot.params.focus_weight",
                "rewards.motion_body_rot.params.phase_weighted_body_names",
            },
            "F1_vs_F2o": {
                "observations.critic_task.focus_phase",
                "observations.mode_mapping.mode_mapping.params.extra_unmasked_dims",
                "observations.policy_task.focus_phase",
            },
            "F2_vs_F3": {
                "observations.critic_task.focus_phase",
                "observations.mode_mapping.mode_mapping.params.extra_unmasked_dims",
                "observations.policy_task.focus_phase",
            },
            "F2o_vs_F3": {
                "focus_weight",
                "phase_weighted_body_names",
                "rewards.motion_body_pos.params.focus_weight",
                "rewards.motion_body_pos.params.phase_weighted_body_names",
                "rewards.motion_body_rot.params.focus_weight",
                "rewards.motion_body_rot.params.phase_weighted_body_names",
            },
            "F3_vs_F3h": {
                "phase_weighted_body_names",
                "rewards.motion_body_pos.params.phase_weighted_body_names",
                "rewards.motion_body_rot.params.phase_weighted_body_names",
            },
        }
        pair_differences = {
            f"{left}_vs_{right}": differences(configs[left], configs[right])
            for left, right in pairs
        }
        checks = {}
        for pair_name, rows in pair_differences.items():
            actual = {row["path"] for row in rows}
            expected = expected_paths[pair_name]
            checks[pair_name] = {
                "passed": actual == expected,
                "unexpected_paths": sorted(actual - expected),
                "missing_paths": sorted(expected - actual),
            }
        report = {
            "comparison": "fully instantiated IsaacLab env config to_dict()",
            "all_checks_passed": all(row["passed"] for row in checks.values()),
            "checks": checks,
            "pairs": pair_differences,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["all_checks_passed"]:
            raise RuntimeError("Unexpected HAT-4 config difference; refusing launch")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
