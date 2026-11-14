#!/usr/bin/env python3
"""Convert TWIST successful HDF5 rollouts into the dexhand GMT cache schema.

This tool does not launch Isaac Sim.  Run it in an environment with numpy and
h5py (the existing ``twist`` and ``isaaclab_sonic`` conda envs both qualify).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_PATH = (
    REPO_ROOT
    / "source"
    / "scaletrack"
    / "scaletrack"
    / "tasks"
    / "dexhand_tracking"
    / "data.py"
)
_SPEC = importlib.util.spec_from_file_location("scaletrack_dexhand_data", _DATA_PATH)
_DATA = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DATA
_SPEC.loader.exec_module(_DATA)
CACHE_SCHEMA_VERSION = _DATA.CACHE_SCHEMA_VERSION
FPS = _DATA.FPS
NUM_HAND_JOINTS = _DATA.NUM_HAND_JOINTS
TIP_BODY_INDICES = _DATA.TIP_BODY_INDICES
xyzw_to_wxyz = _DATA.xyzw_to_wxyz


REQUIRED_FIELDS = (
    "q_rh",
    "q_lh",
    "dq_rh",
    "dq_lh",
    "state_rh",
    "state_lh",
    "joint_state_rh",
    "joint_state_lh",
    "state_manip_obj_rh",
    "state_manip_obj_lh",
    "tip_force_rh",
    "tip_force_lh",
    "actions",
)


class NoSuccessfulRollouts(ValueError):
    """The file is valid but contains no successful samples."""


def _score_rollout(group: h5py.Group) -> float:
    if "reward" not in group:
        return float("-inf")
    reward = np.asarray(group["reward"], dtype=np.float64)
    return float(np.nansum(reward))


def _successful_rollouts(handle: h5py.File) -> list[tuple[str, h5py.Group, float]]:
    try:
        successful = handle["rollouts/successful"]
    except KeyError as exc:
        raise ValueError("missing /rollouts/successful group") from exc
    if len(successful) == 0:
        raise NoSuccessfulRollouts("0 successful rollouts")

    candidates = []
    incomplete = {}
    for name, group in successful.items():
        missing = [field for field in REQUIRED_FIELDS if field not in group]
        if missing:
            incomplete[name] = missing
        else:
            candidates.append((name, group, _score_rollout(group)))
    if incomplete:
        summary = "; ".join(f"{name}: {fields}" for name, fields in incomplete.items())
        raise ValueError(f"successful rollout fields missing: {summary}")
    return sorted(candidates, key=lambda item: _rollout_index(item[0]))


def _best_successful_rollout(handle: h5py.File) -> tuple[str, h5py.Group, float]:
    return max(_successful_rollouts(handle), key=lambda item: item[2])


def _rollout_index(name: str) -> tuple[int, str]:
    match = re.match(r"rollout_(\d+)(?:_|$)", name)
    return (int(match.group(1)), name) if match else (sys.maxsize, name)


def _rollout_suffix(name: str, ordinal: int) -> int:
    number, _ = _rollout_index(name)
    return number if number != sys.maxsize else ordinal


def _state_to_wxyz(value: np.ndarray) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32).copy()
    if state.ndim != 2 or state.shape[1] != 13:
        raise ValueError(f"state must be (T, 13), got {state.shape}")
    state[:, 3:7] = xyzw_to_wxyz(state[:, 3:7])
    return state


def _body_state_and_tips(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(value, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] % 13:
        raise ValueError(f"joint_state must be (T, B*13), got {raw.shape}")
    body = raw.reshape(raw.shape[0], -1, 13).copy()
    if body.shape[1] <= max(TIP_BODY_INDICES):
        raise ValueError(f"joint_state has only {body.shape[1]} bodies")
    body[..., 3:7] = xyzw_to_wxyz(body[..., 3:7])
    return body, body[:, TIP_BODY_INDICES, :3].copy()


def _audit_actions(actions: np.ndarray, q_rh: np.ndarray, q_lh: np.ndarray) -> dict:
    """Report whether legacy actions contain a recognizable 24-D hand target.

    Existing OakInk caches contain 36-D actions whose origin is not documented
    and which do not numerically match the current hand PD target convention.
    The converter therefore preserves them but always creates ``bc_action`` from
    q_ref.  This audit prevents accidentally treating an opaque block as labels.
    """

    result = {
        "input_dim": int(actions.shape[1]),
        "expected_policy_dim": 24,
        "layout": "opaque",
        "best_mse_to_qref_action": None,
        "best_block_start": None,
        "best_hand_order": None,
    }
    if actions.shape[1] < 24 or len(actions) < 2:
        return result
    targets = {
        "rh_lh": 2.0 * np.concatenate((q_rh, q_lh), axis=-1),
        "lh_rh": 2.0 * np.concatenate((q_lh, q_rh), axis=-1),
    }
    best = (float("inf"), None, None)
    for start in range(actions.shape[1] - 24 + 1):
        block = actions[:-1, start : start + 24]
        for order, target in targets.items():
            mse = float(np.mean(np.square(block - target[1:])))
            if mse < best[0]:
                best = (mse, start, order)
    result["best_mse_to_qref_action"] = best[0]
    result["best_block_start"] = best[1]
    result["best_hand_order"] = best[2]
    # The threshold is intentionally strict: labels are expected to be the same
    # target, not merely correlated with it.
    if best[0] < 1.0e-5:
        result["layout"] = f"qref_action[{best[1]}:{best[1] + 24}]/{best[2]}"
    return result


def convert_one(
    input_path: Path,
    output_path: Path,
    metadata: dict,
    rollout_name: str | None = None,
) -> dict:
    with h5py.File(input_path, "r") as handle:
        candidates = _successful_rollouts(handle)
        if rollout_name is None:
            selected_name, group, score = max(candidates, key=lambda item: item[2])
        else:
            selected = [item for item in candidates if item[0] == rollout_name]
            if not selected:
                raise ValueError(f"successful rollout {rollout_name!r} does not exist")
            selected_name, group, score = selected[0]
        arrays = {field: np.asarray(group[field], dtype=np.float32) for field in REQUIRED_FIELDS}

    lengths = {field: len(value) for field, value in arrays.items()}
    shortest = min(lengths.values())
    longest = max(lengths.values())
    if longest - shortest > 1:
        raise ValueError(f"inconsistent rollout lengths: {lengths}")
    if longest != shortest:
        arrays = {field: value[:shortest] for field, value in arrays.items()}
    length = shortest
    for field in ("q_rh", "q_lh", "dq_rh", "dq_lh"):
        if arrays[field].shape != (length, NUM_HAND_JOINTS):
            raise ValueError(f"{field}: expected {(length, NUM_HAND_JOINTS)}, got {arrays[field].shape}")

    state_rh = _state_to_wxyz(arrays["state_rh"])
    state_lh = _state_to_wxyz(arrays["state_lh"])
    object_rh = _state_to_wxyz(arrays["state_manip_obj_rh"])
    object_lh = _state_to_wxyz(arrays["state_manip_obj_lh"])
    body_rh, fingertips_rh = _body_state_and_tips(arrays["joint_state_rh"])
    body_lh, fingertips_lh = _body_state_and_tips(arrays["joint_state_lh"])
    action_audit = _audit_actions(arrays["actions"], arrays["q_rh"], arrays["q_lh"])

    # q_target = default + 0.5 * action, and the Inspire defaults are zero.
    # The next frame is the supervised target; repeat the final target so every
    # online frame has a well-defined anchor label.
    q_next = np.concatenate(
        (
            np.concatenate((arrays["q_rh"][1:], arrays["q_lh"][1:]), axis=-1),
            np.concatenate((arrays["q_rh"][-1:], arrays["q_lh"][-1:]), axis=-1),
        ),
        axis=0,
    )
    bc_action = 2.0 * q_next

    cache_metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fps": FPS,
        "source_hdf5": str(input_path.resolve()),
        "selected_rollout": selected_name,
        "selected_rollout_reward_sum": score,
        "original_field_lengths": lengths,
        "trimmed_to_length": length,
        "quat_convention": "wxyz",
        "tip_body_indices": list(TIP_BODY_INDICES),
        "action_audit": action_audit,
        **metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.int32(CACHE_SCHEMA_VERSION),
        fps=np.int32(FPS),
        q_rh=arrays["q_rh"],
        q_lh=arrays["q_lh"],
        dq_rh=arrays["dq_rh"],
        dq_lh=arrays["dq_lh"],
        state_rh=state_rh,
        state_lh=state_lh,
        body_state_rh=body_rh,
        body_state_lh=body_lh,
        fingertip_pos_rh=fingertips_rh,
        fingertip_pos_lh=fingertips_lh,
        state_manip_obj_rh=object_rh,
        state_manip_obj_lh=object_lh,
        tip_force_rh=arrays["tip_force_rh"],
        tip_force_lh=arrays["tip_force_lh"],
        legacy_actions=arrays["actions"],
        bc_action=bc_action.astype(np.float32),
        metadata_json=np.asarray(json.dumps(cache_metadata, sort_keys=True)),
    )
    return cache_metadata


def _sequence_metadata(hdf5_path: Path) -> dict:
    info_path = hdf5_path.with_name("seq_info.json")
    # OakInk's released successful-rollout subset is flattened as
    # ``rollouts/<sequence>@<index>.hdf5`` while the object/primitive metadata
    # remains in ``sequences/<sequence>@<index>_<hand>/seq_info.json``.
    # Resolve that companion explicitly; otherwise every cache converts but is
    # silently unusable for object-bound simulation.
    if not info_path.is_file() and hdf5_path.name != "rollouts.hdf5":
        sequences_root = hdf5_path.parent.parent / "sequences"
        matches = sorted(sequences_root.glob(f"{hdf5_path.stem}_*/seq_info.json"))
        if len(matches) > 1:
            raise ValueError(
                f"{hdf5_path}: multiple sequence metadata matches: "
                + ", ".join(str(path) for path in matches)
            )
        if matches:
            info_path = matches[0]
    if not info_path.is_file():
        return {}
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    return {
        "hand_type": info.get("type"),
        "primitive": info.get("primitive"),
        "primitive_rh": info.get("primitive_rh"),
        "primitive_lh": info.get("primitive_lh"),
        "interaction_mode": info.get("interaction_mode"),
        "oid_rh": info.get("oid_rh"),
        "oid_lh": info.get("oid_lh"),
        "obj_rh_path": info.get("obj_rh_path"),
        "obj_lh_path": info.get("obj_lh_path"),
        "description": info.get("description"),
        # FAVOR uses up to five fixed colored primitives in addition to the
        # common DexManipNet table.  Preserve their exact source transforms.
        "scene_objects": info.get("scene_objects", []),
    }


def _load_humoto_mapping(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    lookup = {}
    for task, values in mapping.get("task_to_rollouts", {}).items():
        for value in values:
            directory = value["dir"]
            if directory in lookup:
                raise ValueError(f"{path}: duplicate HuMoTo rollout directory {directory!r}")
            lookup[directory] = {
                "source": "humoto",
                "humoto_task": task,
                "task_group": f"humoto:{task}",
                "free_air_only": False,
            }
    for value in mapping.get("take_captures_free_air_only", []):
        directory = value["dir"]
        if directory in lookup:
            raise ValueError(f"{path}: duplicate HuMoTo rollout directory {directory!r}")
        capture_match = re.search(r"(Take_\d+)", directory)
        lookup[directory] = {
            "source": "humoto",
            "humoto_task": None,
            "humoto_capture": capture_match.group(1) if capture_match else directory,
            "task_group": "humoto:free_air",
            "free_air_only": True,
        }
    return lookup


def _input_name(input_path: Path) -> str:
    """Use sequence directories for canonical files and stems for flat HDF5s."""

    return input_path.parent.name if input_path.name == "rollouts.hdf5" else input_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="rollouts.hdf5 files or directories containing them")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--all-successful",
        action="store_true",
        help="Convert every successful rollout and append its numeric __r<k> suffix",
    )
    parser.add_argument(
        "--humoto-mapping",
        type=Path,
        default=None,
        help="HuMoTo directory-to-task mapping (defaults to the sibling mapping JSON with --all-successful)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs: list[Path] = []
    for value in args.inputs:
        if value.is_dir():
            inputs.extend(sorted(value.rglob("rollouts.hdf5")))
        else:
            inputs.append(value)
    inputs = [path.expanduser().resolve() for path in inputs]
    if not inputs:
        raise SystemExit("no rollouts.hdf5 inputs found")

    # Detect output collisions before writing anything. This specifically guards
    # flat OakInk files, for which using parent.name collapsed every input to
    # rollouts.npz.
    names: dict[str, Path] = {}
    for input_path in inputs:
        name = _input_name(input_path)
        if name in names and names[name] != input_path:
            raise ValueError(
                f"duplicate output basename {name!r}: {names[name]} and {input_path}"
            )
        names[name] = input_path

    mapping_path = args.humoto_mapping
    if mapping_path is None and args.all_successful:
        candidate = Path(__file__).with_name("humoto_hand_rollout_mapping.json")
        if candidate.is_file():
            mapping_path = candidate
    humoto_lookup = _load_humoto_mapping(mapping_path.resolve()) if mapping_path else {}

    report = {"schema_version": CACHE_SCHEMA_VERSION, "converted": [], "skipped": [], "failed": []}
    for input_path in inputs:
        name = _input_name(input_path)
        metadata = _sequence_metadata(input_path)
        if humoto_lookup:
            humoto_metadata = humoto_lookup.get(input_path.parent.name)
            if humoto_metadata is None:
                report["failed"].append(
                    {
                        "input": str(input_path),
                        "error": "HuMoTo mapping has no entry for this rollout directory",
                    }
                )
                print(f"[FAIL] {input_path}: missing HuMoTo mapping", file=sys.stderr)
                continue
            metadata.update(humoto_metadata)
        try:
            with h5py.File(input_path, "r") as handle:
                candidates = _successful_rollouts(handle)
        except NoSuccessfulRollouts as exc:
            report["skipped"].append(
                {"input": str(input_path), "reason": str(exc), "successful_rollouts": 0}
            )
            print(f"[SKIP] {input_path}: {exc}")
            continue
        except Exception as exc:
            report["failed"].append({"input": str(input_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {input_path}: {exc}", file=sys.stderr)
            continue

        selected = candidates if args.all_successful else [max(candidates, key=lambda item: item[2])]
        if args.all_successful:
            suffixes = [
                _rollout_suffix(rollout_name, ordinal)
                for ordinal, (rollout_name, _, _) in enumerate(selected)
            ]
            if len(suffixes) != len(set(suffixes)):
                report["failed"].append(
                    {
                        "input": str(input_path),
                        "error": f"duplicate all-successful output suffixes: {suffixes}",
                    }
                )
                print(f"[FAIL] {input_path}: duplicate rollout output suffix", file=sys.stderr)
                continue
        for ordinal, (rollout_name, _, _) in enumerate(selected):
            if args.all_successful:
                suffix = _rollout_suffix(rollout_name, ordinal)
                output_path = args.output_dir / f"{name}__r{suffix}.npz"
            else:
                output_path = args.output_dir / f"{name}.npz"
            if output_path.exists() and not args.overwrite:
                report["skipped"].append(
                    {
                        "input": str(input_path),
                        "rollout": rollout_name,
                        "output": str(output_path),
                        "reason": "output exists",
                    }
                )
                continue
            try:
                converted_metadata = convert_one(
                    input_path,
                    output_path,
                    metadata,
                    rollout_name=rollout_name,
                )
                report["converted"].append(
                    {
                        "input": str(input_path),
                        "rollout": rollout_name,
                        "output": str(output_path),
                        "metadata": converted_metadata,
                    }
                )
                print(f"[OK] {input_path}:{rollout_name} -> {output_path}")
            except Exception as exc:
                report["failed"].append(
                    {
                        "input": str(input_path),
                        "rollout": rollout_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[FAIL] {input_path}:{rollout_name}: {exc}", file=sys.stderr)

    report_path = args.report or args.output_dir / "conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        f"converted={len(report['converted'])} skipped={len(report['skipped'])} "
        f"failed={len(report['failed'])}; report={report_path}"
    )
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
