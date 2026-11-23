"""Online adaptation of the canonical grasp-derived FOCUS definition."""

from __future__ import annotations

import numpy as np


# HAT motor order: index, middle, pinky, ring, thumb rotation, thumb bend.
HAND_BEND_COLUMNS = np.array([0, 1, 2, 3, 5])


def derive_hand_focus_events(history, predicted, fps, *, pad_s=1.0,
                             entry_mode="legacy"):
    """Physical, per-hand intentions, relative to the measured source hand.

    Unlike a full-motion training label, an online event must not normalize a
    small twitch to a complete grasp, or invent an enter edge at history start.
    Rotation is deliberately not a closing channel. Frame zero is valid when
    several predicted fingers actually request motion from the measured hand.

    "v3" derives exactly the same events as "legacy"; it filters them later,
    in the handoff, using the wrist travel the chunk still predicts before the
    grasp frame. Keeping that gate out of here leaves this function a pure
    description of the prediction.
    """
    predicted = np.asarray(predicted, dtype=np.float32)
    history = np.asarray(history, dtype=np.float32)
    if entry_mode not in ("legacy", "v2", "v3"):
        raise ValueError(f"Invalid focus_handoff_version {entry_mode!r}")
    if predicted.ndim != 2 or predicted.shape[1] != 6 or not len(predicted):
        raise ValueError("Expected a nonempty (frames, 6) hand prediction")
    if (history.ndim != 2 or history.shape[1] != 6 or not np.isfinite(fps) or fps <= 0
            or not np.isfinite(history).all() or not np.isfinite(predicted).all()):
        raise ValueError("Invalid hand history or fps")
    bend = np.clip(predicted[:, HAND_BEND_COLUMNS], 0., 1.)
    actual = (np.clip(history[-1, HAND_BEND_COLUMNS], 0., 1.)
              if len(history) else np.zeros(5, dtype=np.float32))
    close = np.sum((bend >= .1) & (bend - actual >= .05), axis=1) >= 2
    opening = np.sum(actual - bend >= .1, axis=1) >= 2
    if entry_mode == "v2":
        # Values are closure = 1 - motor action. Only frame-zero intent
        # qualifies; a future closing edge cannot arm this version.
        horizon = max(1, int(round(.3 * fps)))
        intent = np.sum((bend[0] >= .1) & (bend[0] - actual >= .03)) >= 2
        progress = (horizon < len(bend)
                    and np.sum(bend[horizon] - bend[0] >= .05) >= 2)
        close[:] = intent and progress
        # A qualified close takes priority over opposing finger motion.
        opening[close] = False
    # A frame can request opposing individual fingers. Prefer the larger
    # coordinated displacement rather than allowing two simultaneous events.
    both = close & opening
    prefer_close = np.sum(bend - actual, axis=1) >= 0
    close[both & ~prefer_close] = False
    opening[both & prefer_close] = False
    events = []
    mask = np.zeros(len(bend), dtype=bool)
    pad = max(0, int(round(pad_s * fps)))
    for kind, motion in (("close", close), ("open", opening)):
        edges = np.flatnonzero(np.diff(np.r_[False, motion, False].astype(np.int8)))
        for start, stop in edges.reshape(-1, 2):
            if stop - start < 3:
                continue
            # Release retains its pre-window. Close starts at real intent .1.
            entry = int(start if kind == "close" else max(0, start - pad))
            confirmed = np.sum(bend[start:stop] >= .5 if kind == "close"
                               else bend[start:stop] < .45, axis=1) >= 2
            hit = np.flatnonzero(confirmed)
            endpoint = int(start + hit[0]) if len(hit) else int(stop - 1)
            end = min(len(bend) - 1, endpoint + pad)
            goal_samples = bend[start:min(stop, end + 1)]
            goal = (goal_samples.max(axis=0) if kind == "close"
                    else goal_samples.min(axis=0))
            events.append(dict(kind=kind, start=entry, motion_start=int(start),
                               end=end, goal=goal.copy()))
            mask[entry:end + 1] = True
    events.sort(key=lambda event: event["start"])
    return dict(phase=focus_phase_from_mask(mask, fps), events=events,
                closure=predicted.copy(), actual=actual.copy())


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
