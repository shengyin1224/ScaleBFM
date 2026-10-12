"""Ordered raw wrist-history matching, separate from legacy pose continuity."""

from __future__ import annotations

import numpy as np

from scalebridge.online.reference import _sample_linear, _sample_quat


def continuity_mode(value):
    """Keep the existing bool switch; add one explicit opt-in string value."""
    if value is True or (isinstance(value, str) and value.lower() == "true"):
        return "legacy"
    if value is False or (isinstance(value, str) and value.lower() == "false"):
        return "off"
    if value == "trajectory":
        return "trajectory"
    raise ValueError("chunk_start_reference_continuity expects false, true, or trajectory")


def match_wrist_trajectory(old, new, start_frame, end_frame, min_end_frame,
                          *, side="right", reserve_s=.5, samples=12,
                          max_position_error_m=.03, max_rotation_error_deg=10.):
    """Find the END of the best same-duration ordered raw wrist segment.

    Position RMS + 0.1s * velocity RMS + 0.02m/rad * rotation RMS matches
    the offline prototype. No X-monotonicity rule is imposed. The old poses
    must be pre-RTC/pre-rebase; otherwise an already flattened seam would
    erase the prediction progress that this matcher is meant to preserve.
    """
    old_fps, new_fps = float(old["fps"]), float(new["fps"])
    if (not np.isfinite([old_fps, new_fps, start_frame, end_frame, reserve_s]).all()
            or min(old_fps, new_fps) <= 0 or reserve_s < 0 or samples < 2):
        raise ValueError("Invalid trajectory continuity clock or sample count")
    pkey, qkey = f"{side}_wrist_pos", f"{side}_wrist_quat_wxyz"
    op, oq = np.asarray(old[pkey]), np.asarray(old[qkey])
    np_, nq = np.asarray(new[pkey]), np.asarray(new[qkey])
    for p, q in ((op, oq), (np_, nq)):
        if (p.ndim != 2 or p.shape[1] != 3 or q.shape != (len(p), 4)
                or not np.isfinite(p).all() or not np.isfinite(q).all()
                or np.any(np.linalg.norm(q, axis=-1) < 1e-6)):
            raise ValueError("Invalid raw wrist trajectory")
    end_frame = min(float(end_frame), len(op) - 1.)
    start_frame = max(0., float(start_frame))
    duration = (end_frame - start_frame) / old_fps
    if duration < max(1. / old_fps, 1. / new_fps):
        return None
    lower = max(int(np.ceil(duration * new_fps)), int(min_end_frame))
    upper = int(np.floor(len(np_) - 1 - reserve_s * new_fps))
    if lower > upper:
        return None
    ends = np.arange(lower, upper + 1, dtype=np.int64)
    old_frames = np.linspace(start_frame, end_frame, int(samples))
    new_frames = ends[:, None] + np.linspace(-duration * new_fps, 0., int(samples))
    old_pos = _sample_linear(op, old_frames)
    new_pos = _sample_linear(np_, new_frames)
    old_rot = _sample_quat(oq, old_frames)
    new_rot = _sample_quat(nq, new_frames)
    position = np.sqrt(np.mean(np.sum((new_pos - old_pos) ** 2, axis=-1), axis=-1))
    sample_dt = duration / (samples - 1)
    velocity = np.sqrt(np.mean(np.sum(
        ((np.diff(new_pos, axis=1) - np.diff(old_pos, axis=0)) / sample_dt) ** 2,
        axis=-1), axis=-1))
    dot = np.abs(np.sum(new_rot * old_rot, axis=-1))
    norm = np.linalg.norm(new_rot, axis=-1) * np.linalg.norm(old_rot, axis=-1)
    angles = 2. * np.arccos(np.clip(dot / norm, 0., 1.))
    rotation = np.sqrt(np.mean(angles ** 2, axis=-1))
    costs = position + .1 * velocity + .02 * rotation
    index = int(np.argmin(costs))
    # A poor correspondence is not evidence of shared progress. Let the
    # caller fall back to the untouched legacy selector for this swap.
    if (position[index] > max_position_error_m
            or np.degrees(rotation[index]) > max_rotation_error_deg):
        return None
    return dict(
        selected_frame=int(ends[index]),
        matched_start_frame=float(ends[index] - duration * new_fps),
        history_start_frame=start_frame, history_end_frame=end_frame,
        history_duration_s=duration, search_range=[lower, upper], side=side,
        position_error_m=float(position[index]),
        velocity_error_mps=float(velocity[index]),
        rotation_error_rad=float(rotation[index]), cost=float(costs[index]),
    )
