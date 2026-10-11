#!/usr/bin/env python3
"""Canonical grasp-derived FOCUS definition used by the TWIST data pipeline.

FOCUS starts around closing intent, before the grasp-enter threshold, and
keeps the existing grasp-enter/release window ends. It does not cover the
entire interval during which an object is held.
"""

from __future__ import annotations

import numpy as np


HAND_REDUCE = "max"
GRASP_ENTER = 0.5
GRASP_EXIT = 0.45
FOCUS_ENTER = 0.1
FOCUS_PAD_S = 1.0
FOCUS_PRE_PAD_S = 0.0


def hand_signal(hand: np.ndarray, reduce: str = HAND_REDUCE) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-finger hand state to a normalized scalar closure signal.

    Channels that vary by at most 0.05 over the motion are ignored.  Every
    remaining channel is normalized by its own motion-wide maximum.  The
    canonical ``max`` reduction therefore triggers on any active finger.
    """
    hand = np.asarray(hand, dtype=np.float32)
    if hand.ndim != 2:
        raise ValueError(f"hand must have shape (frames, channels), got {hand.shape}")

    active = np.flatnonzero(np.ptp(hand, axis=0) > 0.05)
    if len(active) == 0:
        return np.zeros(len(hand), dtype=np.float32), active

    norm = hand[:, active] / np.maximum(hand[:, active].max(axis=0, keepdims=True), 1e-6)
    if reduce == "max":
        signal = norm.max(axis=1)
    elif reduce == "mean":
        signal = norm.mean(axis=1)
    elif reduce == "min":
        signal = norm.min(axis=1)
    elif reduce == "frac_gt_05":
        signal = (norm > 0.5).mean(axis=1)
    else:
        raise ValueError(f"unknown hand reduction: {reduce}")
    return signal.astype(np.float32), active


def hysteresis(signal: np.ndarray, enter: float = GRASP_ENTER, exit_: float = GRASP_EXIT) -> np.ndarray:
    """Declare grasp at >= enter and retain it until the signal falls below exit."""
    signal = np.asarray(signal)
    grasp = np.zeros(len(signal), dtype=bool)
    is_grasping = False
    for frame, value in enumerate(signal):
        if not is_grasping and value >= enter:
            is_grasping = True
        elif is_grasping and value < exit_:
            is_grasping = False
        grasp[frame] = is_grasping
    return grasp


def true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` bounds for every true segment."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(start), int(stop - 1)) for start, stop in edges.reshape(-1, 2)]


def focus_window(grasp: np.ndarray, fps: float, pad_s: float = FOCUS_PAD_S) -> np.ndarray:
    """Mark grasp-enter/release endpoints +/- ``pad_s`` as FOCUS."""
    grasp = np.asarray(grasp, dtype=bool)
    pad = int(round(pad_s * fps))
    focus = np.zeros(len(grasp), dtype=bool)
    for start, end in true_segments(grasp):
        focus[max(0, start - pad) : min(len(grasp), start + pad + 1)] = True
        focus[max(0, end - pad) : min(len(grasp), end + pad + 1)] = True
    return focus


def focus_from_hand_state(
    hand: np.ndarray,
    fps: float,
    *,
    grasp_enter: float = GRASP_ENTER,
    grasp_exit: float = GRASP_EXIT,
    focus_pad_s: float = FOCUS_PAD_S,
    focus_enter: float = FOCUS_ENTER,
    focus_pre_pad_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closing-intent FOCUS and the unchanged grasp-state mask.

    The lower ``focus_enter`` threshold advances only the grasp FOCUS start.
    A confirmed grasp still ends its enter window at ``grasp_start + pad``;
    release windows retain the original definition. An attempt that returns
    below ``focus_enter`` without becoming a grasp ends after the same pad.
    Closing FOCUS starts at intent by default; release padding is unchanged.
    """
    if not 0.0 < focus_enter <= grasp_enter:
        raise ValueError("focus_enter must satisfy 0 < focus_enter <= grasp_enter")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    pre_pad_s = FOCUS_PRE_PAD_S if focus_pre_pad_s is None else focus_pre_pad_s
    if not np.isfinite(focus_pad_s) or focus_pad_s < 0:
        raise ValueError("focus_pad_s must be non-negative")
    if not np.isfinite(pre_pad_s) or pre_pad_s < 0:
        raise ValueError("focus_pre_pad_s must be non-negative")
    signal, _ = hand_signal(hand, HAND_REDUCE)
    grasp = hysteresis(signal, grasp_enter, grasp_exit)
    if focus_enter == grasp_enter and pre_pad_s == focus_pad_s:
        return focus_window(grasp, fps, focus_pad_s), grasp
    pad = int(round(focus_pad_s * fps))
    pre_pad = int(round(pre_pad_s * fps))
    focus = np.zeros(len(grasp), dtype=bool)
    grasp_segments = true_segments(grasp)
    intent_segments = true_segments(signal >= focus_enter)
    intent_starts = np.asarray([start for start, _ in intent_segments], dtype=int)
    intent_ends = np.asarray([end for _, end in intent_segments], dtype=int)
    confirmed_intents = set()
    previous_end = -1

    for start, end in grasp_segments:
        intent_index = int(np.searchsorted(intent_starts, start, side="right") - 1)
        onset = start
        if intent_index >= 0 and start <= intent_ends[intent_index]:
            # Do not reuse the beginning of an earlier grasp if the hand only
            # partially opened before grasping again.
            intent_start = int(intent_starts[intent_index])
            onset = max(intent_start, previous_end + 1)
            confirmed_intents.add(intent_index)
        focus[max(0, onset - pre_pad) : min(len(focus), start + pad + 1)] = True
        focus[max(0, end - pad) : min(len(focus), end + pad + 1)] = True
        previous_end = end

    for index, (start, end) in enumerate(intent_segments):
        if index not in confirmed_intents and not np.any(grasp[start : end + 1]):
            # A low-amplitude closing attempt also deserves attention, even
            # when a later abort means the grasp threshold is never reached.
            focus[max(0, start - pre_pad) : min(len(focus), end + pad + 1)] = True
    return focus, grasp
