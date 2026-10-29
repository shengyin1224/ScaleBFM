#!/usr/bin/env python3
"""Replay recorded online HAT chunks through the production seam matcher.

This is intentionally read-only: it reuses the aligned chunks recorded during
real-world runs and reports whether the current matcher can switch between them
without discontinuity, backtracking, or the old frame-99 endpoint lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.protocol import validate_hat_chunk


LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def _event_time(log_path: Path, marker: str) -> float | None:
    for line in log_path.read_text(errors="replace").splitlines():
        if marker not in line:
            continue
        match = LOG_TIME.match(line)
        if match:
            return datetime.fromisoformat(match.group(1)).timestamp()
    return None


def _make_env() -> HAT4OnlineMotionTrackingEnv:
    env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
    env.dt = 0.02
    env.state_sequence_id = 0
    env.state_timestamp_history = []
    env.chunk_phase_alignment = True
    env.phase_rotation_weight = 0.08
    env.phase_velocity_weight = 0.03
    # Measured link poses were not included in early chunk recordings.  The
    # production matcher treats them only as a weak consistency term; replay
    # therefore disables that term instead of fabricating measurements.
    env.phase_measured_weight = 0.0
    env.phase_max_root_jump_m = 0.20
    env.future_offsets_s = np.arange(6, dtype=np.float64) / 50.0
    env.last_reference_sample = None
    env.last_published_robot_state = None
    env.reference_play_gate = False
    env.reference_playing = True
    env.localization_paused = False
    env.awaiting_fresh_chunk = False
    env.held_sample = None
    env.previous_chunk = None
    env.previous_chunk_started_ns = None
    env.blend_started_ns = None
    env.cfg = {"chunk_blend_s": 0.22}
    return env


def replay(run_dir: Path) -> dict:
    log_path = run_dir / "run.log"
    chunks_path = run_dir / "online_hat_chunks.jsonl"
    start_time = _event_time(log_path, "R1 pressed: starting closed loop")
    if start_time is None:
        raise RuntimeError("no closed-loop R1 event")
    # Robot-state freeze is an external communications failure, not a chunk
    # alignment result.  Stop before it so the two failure classes stay apart.
    stop_time = _event_time(log_path, "Robot state frozen")

    records = []
    with chunks_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "hat_chunk":
                continue
            if stop_time is not None and float(record["time"]) >= stop_time:
                break
            records.append(record)
    before = [record for record in records if float(record["time"]) < start_time]
    after = [record for record in records if float(record["time"]) >= start_time]
    if not before or not after:
        raise RuntimeError("recording does not straddle the R1 event")

    seed_record = before[-1]["aligned"]
    seed_state = seed_record.get("execution_alignment", {})
    seed = validate_hat_chunk(seed_record)
    seed_frame = int(seed_state.get("selected_frame", 0))
    first_now_ns = int(round(float(after[0]["monotonic"]) * 1e9))
    start_monotonic_ns = first_now_ns - int(
        (float(after[0]["time"]) - start_time) * 1e9
    )

    env = _make_env()
    env.chunk = seed
    env.chunk_started_ns = start_monotonic_ns - int(
        seed_frame / float(seed["fps"]) * 1e9
    )

    accepted = 0
    deferred = 0
    longest_deferred = 0
    deferred_streak = 0
    endpoint_candidates = 0
    endpoint_accepted = 0
    selected_frames = []
    root_jumps = []
    backtracks = []
    rows = []
    output_roots = []
    output_link_positions = []
    output_link_quaternions = []
    max_switch_step_m = 0.0
    sample_ns = start_monotonic_ns

    def sample_until(end_ns: int) -> None:
        nonlocal sample_ns
        while sample_ns <= end_ns:
            sample = env._sample_reference(sample_ns)
            output_roots.append(np.asarray(sample["root_pos"][0], dtype=np.float64))
            output_link_positions.append(np.stack([
                sample[key][0] for key in (
                    "root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"
                )
            ]).astype(np.float64))
            output_link_quaternions.append(np.stack([
                sample[key][0] for key in (
                    "root_quat_wxyz", "head_quat_wxyz",
                    "left_wrist_quat_wxyz", "right_wrist_quat_wxyz"
                )
            ]).astype(np.float64))
            sample_ns += 20_000_000

    for record in after:
        candidate_record = record["aligned"]
        candidate = validate_hat_chunk(candidate_record)
        raw = record["raw"]
        now_ns = int(round(float(record["monotonic"]) * 1e9))
        source_id = int(raw["source_state_sequence_id"])
        generated_ns = int(raw.get("generated_at_ns", now_ns))
        old_state = candidate_record.get("execution_alignment", {})
        source_age_s = old_state.get("source_age_s")
        source_ns = (
            now_ns - int(float(source_age_s) * 1e9)
            if source_age_s is not None
            else generated_ns
        )
        env.state_sequence_id = source_id + 1
        env.state_timestamp_history = [(source_id, source_ns)]

        sample_until(now_ns)
        before_switch = env._sample_reference(now_ns)["root_pos"][0]
        selected, state = env._phase_match_start_frame(candidate, now_ns)
        outgoing_frame = float(state.get("outgoing_frame", 0.0))
        at_endpoint = outgoing_frame >= float(env.chunk["frame_count"] - 1)
        endpoint_candidates += int(at_endpoint)
        valid = bool(state["transition_valid"])
        rows.append((int(raw["sequence_id"]), selected, outgoing_frame, valid))
        if valid:
            accepted += 1
            endpoint_accepted += int(at_endpoint)
            deferred_streak = 0
            selected_frames.append(selected)
            root_jumps.append(float(state.get("root_jump_m", 0.0)))
            backtracks.append(float(state.get("backtrack_m", 0.0)))
            env.previous_chunk = env.chunk
            env.previous_chunk_started_ns = env.chunk_started_ns
            env.chunk = candidate
            env.chunk_started_ns = source_ns
            env.blend_started_ns = now_ns
            after_switch = env._sample_reference(now_ns)["root_pos"][0]
            max_switch_step_m = max(
                max_switch_step_m,
                float(np.linalg.norm(after_switch - before_switch)),
            )
        else:
            deferred += 1
            deferred_streak += 1
            longest_deferred = max(longest_deferred, deferred_streak)

    sample_until(sample_ns + 300_000_000)
    roots = np.asarray(output_roots, dtype=np.float64)
    steps = np.linalg.norm(np.diff(roots, axis=0), axis=1)
    link_positions = np.asarray(output_link_positions, dtype=np.float64)
    link_steps = np.linalg.norm(np.diff(link_positions, axis=0), axis=-1)
    link_quats = np.asarray(output_link_quaternions, dtype=np.float64)
    quat_dots = np.abs(np.sum(link_quats[1:] * link_quats[:-1], axis=-1))
    quat_steps_deg = np.degrees(2.0 * np.arccos(np.clip(quat_dots, 0.0, 1.0)))
    return {
        "run": str(run_dir),
        "candidates": len(after),
        "accepted": accepted,
        "deferred": deferred,
        "longest_deferred": longest_deferred,
        "endpoint_candidates": endpoint_candidates,
        "endpoint_accepted": endpoint_accepted,
        "selected_min": min(selected_frames) if selected_frames else None,
        "selected_max": max(selected_frames) if selected_frames else None,
        "max_root_jump_m": max(root_jumps) if root_jumps else None,
        "min_backtrack_m": min(backtracks) if backtracks else None,
        "max_switch_step_m": max_switch_step_m,
        "output_max_20ms_step_m": float(np.max(steps)) if len(steps) else None,
        "output_p99_20ms_step_m": float(np.quantile(steps, 0.99)) if len(steps) else None,
        "all_links_max_20ms_step_m": (
            float(np.max(link_steps)) if len(link_steps) else None
        ),
        "all_links_p99_20ms_step_m": (
            float(np.quantile(link_steps, 0.99)) if len(link_steps) else None
        ),
        "all_links_max_20ms_rotation_deg": (
            float(np.max(quat_steps_deg)) if len(quat_steps_deg) else None
        ),
        "output_net_xy_m": (
            (roots[-1, :2] - roots[0, :2]).tolist() if len(roots) else None
        ),
        "last_rows": rows[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for run_dir in args.run_dirs:
        try:
            result = replay(run_dir)
        except Exception as exc:
            failed = True
            print(json.dumps({"run": str(run_dir), "error": str(exc)}))
        else:
            print(json.dumps(result))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
