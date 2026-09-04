"""Online adaptation of the canonical grasp-derived FOCUS definition."""

from __future__ import annotations

import numpy as np


def focus_phase_from_mask(focus_mask, fps, clip_s=2.0):
    """Return `[focus_flag, signed time to next focus edge]` per frame.

    This matches the packaged ScaleTrack observation: time is positive outside
    FOCUS, negative inside FOCUS, and clipped to two seconds.  The sequence end
    is the final edge when no later state transition is available.
    """
    mask = np.asarray(focus_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"focus_mask must be one-dimensional, got {mask.shape}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if len(mask) == 0:
        return np.empty((0, 2), dtype=np.float32)

    edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    frame = np.arange(len(mask), dtype=np.int64)
    edge_slot = np.searchsorted(edges, frame, side="right")
    edge_with_end = np.r_[edges, len(mask)]
    seconds = (edge_with_end[edge_slot] - frame) / float(fps)
    seconds = np.where(mask, -seconds, seconds)
    seconds = np.clip(seconds, -float(clip_s), float(clip_s))
    return np.stack((mask.astype(np.float32), seconds), axis=-1).astype(np.float32)


def derive_chunk_focus_phase(
    hand_history,
    predicted_hand,
    fps,
    focus_from_hand_state,
    *,
    focus_pad_s=1.0,
    phase_clip_s=2.0,
):
    """Apply the canonical definition with causal history and future padding.

    A HAT chunk is not a complete motion.  History preserves a grasp endpoint
    that occurred shortly before frame zero.  Repeating the last predicted hand
    state beyond the chunk prevents its artificial boundary from being treated
    as a grasp-release endpoint by the offline full-motion definition.
    """
    history = np.asarray(hand_history, dtype=np.float32)
    predicted = np.asarray(predicted_hand, dtype=np.float32)
    if history.ndim != 2 or predicted.ndim != 2:
        raise ValueError(
            f"hand history/prediction must be 2-D, got {history.shape}/{predicted.shape}"
        )
    if history.shape[1] != predicted.shape[1]:
        raise ValueError(
            f"hand history/prediction widths differ: {history.shape[1]} vs {predicted.shape[1]}"
        )
    if len(predicted) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if not np.isfinite(history).all() or not np.isfinite(predicted).all():
        raise ValueError("hand history/prediction contains NaN or infinity")

    extension_frames = int(
        round((float(focus_pad_s) + float(phase_clip_s)) * float(fps))
    ) + 2
    extension = np.repeat(predicted[-1:], extension_frames, axis=0)
    context = np.concatenate((history, predicted, extension), axis=0)
    focus_mask, _ = focus_from_hand_state(
        context, float(fps), focus_pad_s=float(focus_pad_s)
    )
    phase = focus_phase_from_mask(focus_mask, float(fps), float(phase_clip_s))
    start = len(history)
    return phase[start : start + len(predicted)]
