#!/usr/bin/env python3
"""Audit HAT world-to-pelvis re-anchoring without hiding coordinate assumptions.

With ``--static-trace``, the file must contain measured hardware/simulator data:
``target_world_pos[T,K,3]``, ``endpoint_world_pos[T,K,3]``,
``measured_pelvis_pos[T,3]``, and ``measured_pelvis_quat_wxyz[T,4]``.
The script reports the real endpoint landing error and verifies that each
re-anchored local target reconstructs the requested world target.

The always-run synthetic section injects 5/10 cm root translation into a
prepared ScaleTrack reference and compares stale-local (off) with per-cycle
re-anchoring (on).  It is a coordinate-chain unit test, not a policy or robot
performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).copy()
    result[..., 1:] *= -1
    return result


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., 1:]
    return v + 2.0 * (
        q[..., :1] * np.cross(qv, v) + np.cross(qv, np.cross(qv, v))
    )


def world_to_local(
    pelvis_pos: np.ndarray, pelvis_quat: np.ndarray, world_pos: np.ndarray
) -> np.ndarray:
    return quat_apply(
        quat_conjugate(pelvis_quat)[..., None, :],
        world_pos - pelvis_pos[..., None, :],
    )


def local_to_world(
    pelvis_pos: np.ndarray, pelvis_quat: np.ndarray, local_pos: np.ndarray
) -> np.ndarray:
    return pelvis_pos[..., None, :] + quat_apply(
        pelvis_quat[..., None, :], local_pos
    )


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_m": float(values.mean()),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(values.max()),
    }


def validate_quaternions(q: np.ndarray, label: str) -> None:
    error = float(np.max(np.abs(np.linalg.norm(q, axis=-1) - 1.0)))
    if error > 1e-5:
        raise ValueError(f"{label} quaternion norm max error={error}")


def synthetic(reference_path: Path, drift_cm: list[float]) -> dict:
    with np.load(reference_path) as ref:
        root_pos = np.asarray(ref["body_pos_w"][:, 0], dtype=np.float64)
        root_quat = np.asarray(ref["body_quat_w"][:, 0], dtype=np.float64)
        # Full body points are useful here: the identity must not depend on a
        # hand-picked endpoint.
        targets = np.asarray(ref["body_pos_w"], dtype=np.float64)
    validate_quaternions(root_quat, "reference pelvis")
    nominal_local = world_to_local(root_pos, root_quat, targets)
    rows = {}
    direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    for cm in drift_cm:
        measured_pos = root_pos + direction * (cm / 100.0)
        stale_world = local_to_world(measured_pos, root_quat, nominal_local)
        reanchored_local = world_to_local(measured_pos, root_quat, targets)
        reanchored_world = local_to_world(
            measured_pos, root_quat, reanchored_local
        )
        rows[f"{cm:g}cm"] = {
            "reanchor_off_endpoint_error": summary(
                np.linalg.norm(stale_world - targets, axis=-1)
            ),
            "reanchor_on_reconstruction_error": summary(
                np.linalg.norm(reanchored_world - targets, axis=-1)
            ),
        }
    return {
        "claim_scope": (
            "coordinate-chain synthetic test only; no dynamics, policy, or "
            "hardware tracking is represented"
        ),
        "reference": str(reference_path.resolve()),
        "translation_direction_xyz": direction.tolist(),
        "drifts": rows,
    }


def static_trace(path: Path) -> dict:
    required = (
        "target_world_pos",
        "endpoint_world_pos",
        "measured_pelvis_pos",
        "measured_pelvis_quat_wxyz",
    )
    with np.load(path) as trace:
        missing = [key for key in required if key not in trace.files]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        target = np.asarray(trace["target_world_pos"], dtype=np.float64)
        endpoint = np.asarray(trace["endpoint_world_pos"], dtype=np.float64)
        pelvis_pos = np.asarray(trace["measured_pelvis_pos"], dtype=np.float64)
        pelvis_quat = np.asarray(
            trace["measured_pelvis_quat_wxyz"], dtype=np.float64
        )
    if target.shape != endpoint.shape or target.ndim != 3:
        raise ValueError("target/endpoint must both have shape [T,K,3]")
    if pelvis_pos.shape != (len(target), 3) or pelvis_quat.shape != (
        len(target),
        4,
    ):
        raise ValueError("measured pelvis arrays are not frame-aligned")
    validate_quaternions(pelvis_quat, "measured pelvis")
    local = world_to_local(pelvis_pos, pelvis_quat, target)
    reconstructed = local_to_world(pelvis_pos, pelvis_quat, local)
    landing = np.linalg.norm(endpoint - target, axis=-1)
    reconstruction = np.linalg.norm(reconstructed - target, axis=-1)
    return {
        "claim_scope": "measured trace",
        "trace": str(path.resolve()),
        "frames": len(target),
        "points_per_frame": target.shape[1],
        "endpoint_world_landing_error": summary(landing),
        "reanchor_transform_roundtrip_error": summary(reconstruction),
        "acceptance_threshold_m": 0.025,
        "accepted": bool(float(np.percentile(landing, 95)) <= 0.025),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--static-trace", type=Path)
    parser.add_argument("--drift-cm", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "synthetic": synthetic(args.reference, args.drift_cm),
        "static_acceptance": (
            static_trace(args.static_trace)
            if args.static_trace is not None
            else {
                "status": "NOT RUN: --static-trace was not supplied",
                "required_schema": [
                    "target_world_pos[T,K,3]",
                    "endpoint_world_pos[T,K,3]",
                    "measured_pelvis_pos[T,3]",
                    "measured_pelvis_quat_wxyz[T,4]",
                ],
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
