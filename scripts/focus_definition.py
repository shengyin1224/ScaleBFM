#!/usr/bin/env python3
"""Embodiment-aware grasp endpoint FOCUS labels for UnifoLM hand state.

FOCUS is intentionally defined around observed hand-close/hand-open transitions,
not across the full interval in which an object is held.  BrainCo and Inspire
encode finger closure with different polarity and include one non-closure thumb
channel per hand, so their raw 12-channel vectors must not be normalized as an
undifferentiated per-episode vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


HAND_FIELD = "observation.state.hand_state"
HAND_REDUCE = "max_active_physical_closure"
ACTIVE_RANGE = 0.05
GRASP_ENTER = 0.5
GRASP_EXIT = 0.45
FOCUS_PAD_S = 1.0
FOCUS_EDGE_CLIP_S = 2.0

# BrainCo SDK order per hand:
#   [thumb_open, thumb_tilt, index, middle, ring, little]
# thumb_tilt is an opposition/side-swing control, not finger closure.  Recorded
# closure commands are capped at approximately 0.8, with 0=open and 0.8=closed.
_BRAINCO_GROUPS = ((0, 2, 3, 4, 5), (6, 8, 9, 10, 11))
_BRAINCO_CLOSED_VALUE = 0.8

# Inspire SDK order per hand:
#   [index, middle, ring, little, thumb_pitch, thumb_yaw]
# thumb_yaw is not an open/close channel.  The first five controls use the
# opposite polarity from physical closure: approximately 0.9=open, 0.2=closed.
_INSPIRE_GROUPS = ((0, 1, 2, 3, 4), (6, 7, 8, 9, 10))


def infer_hand_type(dataset: str | Path) -> str:
    """Infer the hand embodiment from the full source dataset path."""
    name = str(dataset).lower()
    if "brainco" in name:
        return "brainco"
    if "inspire" in name or "dex5" in name:
        return "inspire"
    raise ValueError(
        f"Cannot infer hand type from {dataset!s}; pass --hand-type explicitly"
    )


def closure_channel_indices(hand_type: str) -> tuple[int, ...]:
    """Return the ten raw indices that represent physical finger closure."""
    if hand_type == "brainco":
        groups = _BRAINCO_GROUPS
    elif hand_type == "inspire":
        groups = _INSPIRE_GROUPS
    else:
        raise ValueError(f"Unsupported hand_type={hand_type!r}")
    return tuple(index for group in groups for index in group)


def physical_closure(hand: np.ndarray, hand_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Map raw hand state to ten [0, 1] physical-closure channels.

    Returns ``(closure, source_indices)``.  The two five-channel groups retain
    left/right hand order.  Values outside the nominal hardware range are
    clipped instead of changing the scale on an episode-by-episode basis.
    """
    hand = np.asarray(hand, dtype=np.float32)
    if hand.ndim != 2 or hand.shape[1] != 12:
        raise ValueError(f"hand must have shape (frames, 12), got {hand.shape}")
    if not np.isfinite(hand).all():
        raise ValueError("hand contains non-finite values")

    indices = np.asarray(closure_channel_indices(hand_type), dtype=np.int64)
    selected = hand[:, indices]
    if hand_type == "brainco":
        closure = selected / _BRAINCO_CLOSED_VALUE
    else:
        closure = 1.0 - selected
    return np.clip(closure, 0.0, 1.0).astype(np.float32), indices


def hand_signal(
    hand: np.ndarray,
    hand_type: str,
    *,
    active_range: float = ACTIVE_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce active physical finger-closure channels to one grasp signal.

    Inactive channels are ignored so an unused hand parked in a closed pose does
    not label the full episode as grasping.  Importantly, active channels keep
    their hardware-defined physical scale; small motion is never amplified to a
    synthetic full-range closure.
    """
    signals, active = hand_signals(hand, hand_type, active_range=active_range)
    return signals.max(axis=1), active


def hand_signals(
    hand: np.ndarray,
    hand_type: str,
    *,
    active_range: float = ACTIVE_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent left/right closure signals and used raw indices.

    A channel must cross the complete hysteresis band somewhere in the episode.
    This rejects a parked finger that remains closed while drifting slightly.
    Hands are kept independent so one hand holding an object cannot hide the
    other hand's enter/release transitions.
    """
    closure, source_indices = physical_closure(hand, hand_type)
    signals = np.zeros((len(closure), 2), dtype=np.float32)
    active_parts = []
    for hand_index, start in enumerate((0, 5)):
        values = closure[:, start : start + 5]
        eligible = (
            (np.ptp(values, axis=0) > active_range)
            & (values.min(axis=0) < GRASP_EXIT)
            & (values.max(axis=0) >= GRASP_ENTER)
        )
        active_local = np.flatnonzero(eligible)
        if len(active_local):
            signals[:, hand_index] = values[:, active_local].max(axis=1)
            active_parts.append(source_indices[start + active_local])
    active = (
        np.concatenate(active_parts).astype(np.int64)
        if active_parts
        else source_indices[:0]
    )
    return signals, active


def hysteresis(
    signal: np.ndarray,
    enter: float = GRASP_ENTER,
    exit_: float = GRASP_EXIT,
) -> np.ndarray:
    """Declare grasp at ``enter`` and retain it until below ``exit_``."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError(f"signal must be a finite vector, got {signal.shape}")
    grasp = np.zeros(len(signal), dtype=bool)
    state = False
    for frame, value in enumerate(signal):
        if not state and value >= enter:
            state = True
        elif state and value < exit_:
            state = False
        grasp[frame] = state
    return grasp


def true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` bounds for true segments."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"mask must be a vector, got {mask.shape}")
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(start), int(stop - 1)) for start, stop in edges.reshape(-1, 2)]


def focus_window(
    grasp: np.ndarray,
    fps: float,
    pad_s: float = FOCUS_PAD_S,
    *,
    censor_episode_boundaries: bool = True,
) -> np.ndarray:
    """Mark observed grasp-enter/release endpoints +/- ``pad_s`` as FOCUS.

    A grasp segment already active at frame zero has no observed enter event;
    likewise a segment still active at the final frame has no observed release.
    Boundary censoring prevents either dataset cut from becoming a fake event.
    """
    grasp = np.asarray(grasp, dtype=bool)
    if grasp.ndim != 1:
        raise ValueError(f"grasp must be a vector, got {grasp.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    pad = int(round(pad_s * fps))
    focus = np.zeros(len(grasp), dtype=bool)
    for start, end in true_segments(grasp):
        if not censor_episode_boundaries or start > 0:
            focus[max(0, start - pad) : min(len(grasp), start + pad + 1)] = True
        if not censor_episode_boundaries or end < len(grasp) - 1:
            focus[max(0, end - pad) : min(len(grasp), end + pad + 1)] = True
    return focus


def time_to_next_focus_edge(
    focus: np.ndarray,
    fps: float,
    clip_s: float = FOCUS_EDGE_CLIP_S,
) -> np.ndarray:
    """Signed seconds to next FOCUS boundary; negative means in-window."""
    focus = np.asarray(focus, dtype=bool)
    transitions = np.flatnonzero(focus[1:] != focus[:-1]) + 1
    transition_set = set(int(value) for value in transitions)
    output = np.full(len(focus), clip_s, dtype=np.float32)
    next_edge = len(focus)
    for frame in range(len(focus) - 1, -1, -1):
        seconds = min(max(next_edge - frame, 0) / float(fps), clip_s)
        output[frame] = -seconds if focus[frame] else seconds
        if frame in transition_set:
            next_edge = frame
    return output


def focus_from_hand_state(
    hand: np.ndarray,
    fps: float,
    hand_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(focus, grasp, signal, active_source_indices)``."""
    signals, active = hand_signals(hand, hand_type)
    hand_grasps = np.stack(
        [hysteresis(signals[:, hand_index]) for hand_index in range(2)], axis=1
    )
    hand_focus = np.stack(
        [focus_window(hand_grasps[:, hand_index], fps) for hand_index in range(2)],
        axis=1,
    )
    return (
        hand_focus.any(axis=1),
        hand_grasps.any(axis=1),
        signals.max(axis=1),
        active,
    )


# v3 keeps the original grasp/release events and their end boundaries. Only
# the pre-grasp onset is anchored to the lower closure threshold.
FOCUS_ONSET_THRESHOLD = 0.10
FOCUS_ONSET_LEAD_S = 0.07


def onset_focus_window(
    signal: np.ndarray,
    grasp: np.ndarray,
    fps: float,
    *,
    onset_threshold: float = FOCUS_ONSET_THRESHOLD,
    lead_s: float = FOCUS_ONSET_LEAD_S,
    post_pad_s: float = FOCUS_PAD_S,
) -> np.ndarray:
    """v3: onset >=0.1 minus <=0.07s; retain v2 enter+1s/release windows.

    For each observed 0.5 grasp entry, locate the start of its uninterrupted
    >=0.1 closure run, without reaching back into the previous grasp. A small
    aborted close that never reaches 0.5 does not create a grasp event.
    Episode cuts still do not fabricate entry/release events.
    """
    signal = np.asarray(signal, dtype=np.float32)
    grasp = np.asarray(grasp, dtype=bool)
    if signal.ndim != 1 or signal.shape != grasp.shape or not np.isfinite(signal).all():
        raise ValueError("signal and grasp must be aligned finite vectors")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    if not 0 <= lead_s <= 0.10 or not np.isfinite(lead_s):
        raise ValueError("onset advance must be in [0, 0.10] seconds")
    if not 0 < onset_threshold < GRASP_ENTER or post_pad_s < 0:
        raise ValueError("Require 0 < onset threshold < grasp entry and post_pad_s >= 0")
    # Floor ensures the source-frame advance does not exceed the configured lead.
    lead_frames = int(np.floor(lead_s * fps + 1e-8))
    pad_frames = int(round(post_pad_s * fps))
    focus = np.zeros(len(grasp), dtype=bool)
    previous_end = -1
    for start, end in true_segments(grasp):
        if start > 0:
            onset = start
            while onset > previous_end + 1 and signal[onset - 1] >= onset_threshold - 1e-6:
                onset -= 1
            begin = max(0, onset - lead_frames)
            focus[begin : min(len(grasp), start + pad_frames + 1)] = True
        # Preserve the original release window, including boundary censoring.
        if end < len(grasp) - 1:
            focus[max(0, end - pad_frames) : min(len(grasp), end + pad_frames + 1)] = True
        previous_end = end
    return focus


def focus_from_hand_state_v3(
    hand: np.ndarray,
    fps: float,
    hand_type: str,
    *,
    onset_threshold: float = FOCUS_ONSET_THRESHOLD,
    lead_s: float = FOCUS_ONSET_LEAD_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use independent hands and the unchanged embodiment-aware grasp detector."""
    signals, active = hand_signals(hand, hand_type)
    hand_grasps = np.stack(
        [hysteresis(signals[:, index]) for index in range(2)], axis=1
    )
    # For onset only, calibrate Inspire's nominal 0.9=open, 0.2=closed
    # hardware range to [0,1]. Legacy 1-state has a ~0.1 open baseline;
    # comparing it directly with 0.1 would incorrectly start at frame zero.
    # The original signals/grasp remain untouched so end boundaries stay exact.
    onset_signals = (
        np.clip((signals - 0.10) / 0.70, 0.0, 1.0)
        if hand_type == "inspire" else signals
    )
    hand_focus = np.stack([
        onset_focus_window(onset_signals[:, index], hand_grasps[:, index], fps,
                           onset_threshold=onset_threshold, lead_s=lead_s)
        for index in range(2)
    ], axis=1)
    return hand_focus.any(axis=1), hand_grasps.any(axis=1), signals.max(axis=1), active
