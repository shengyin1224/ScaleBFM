"""Standalone, training-free wrist horizon selection (no robot or model I/O).

PACE-inspired speed valleys plus AAC's minimum-motion idea. This is NOT AAC
entropy estimation: one prediction cannot supply same-observation uncertainty.
All frame limits refer to additional frames after the accepted playback cursor.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


SIDES = ("left", "right")


@dataclass(frozen=True)
class Horizon:
    frame: float
    reason: str
    travel_m: float


def closing_boundary(chunk, side, start=0., hand=None):
    """Conservative first close boundary; prefer same-chunk physical IK events.

    Geometry is only a fallback: stop before sustained fingertip contraction,
    or an already closed fingertip configuration. It is not contact detection.
    """
    n = int(chunk["frame_count"])
    begin = int(np.clip(np.floor(start), 0, n - 1))
    if hand is not None:
        events = [e for e in hand.get("events", ())
                  if e["kind"] == "close" and int(e["end"]) >= begin]
        if events:
            return max(begin, min(int(e["start"]) for e in events))
    tips = np.asarray(chunk[f"{side}_fingertip_local"], dtype=float)
    spread = np.linalg.norm(tips[:, 1:] - tips[:, :1], axis=-1)
    # v3's closed-spread cue; two or more contracting fingers also fence a
    # close that has not yet reached that absolute spread.
    closed = np.all(spread <= .11, axis=1)
    shrinking = np.sum(spread[begin] - spread >= .005, axis=1) >= 2
    for frame in range(begin, n):
        if closed[frame] or (frame + 3 <= n and shrinking[frame:frame + 3].all()):
            return frame
    return n - 1


def select_horizon(chunk, start, *, near=False, minimum=25, maximum=50,
                   near_minimum=8, near_maximum=12, min_travel_m=.05,
                   boundary=None):
    """Select an absolute switch cursor; boundaries override the motion floor.

    No credible valley: use the bounded maximum. If 5cm is unattainable in
    that window, report it rather than waiting forever or crossing an event.
    """
    n = int(chunk["frame_count"])
    start = float(np.clip(start, 0, n - 1))
    lo, hi = (near_minimum, near_maximum) if near else (minimum, maximum)
    end = min(n - 1, int(np.floor(start + hi)))
    if boundary is not None:
        end = min(end, max(int(np.ceil(start)), int(boundary)))
    first = min(end, int(np.ceil(start + lo)))
    paths = np.stack([np.asarray(chunk[f"{s}_wrist_pos"], float) for s in SIDES])
    if not np.isfinite(paths).all():
        raise ValueError("Adaptive execution requires finite wrist paths")
    steps = np.linalg.norm(np.diff(paths, axis=1), axis=-1)
    cumulative = np.concatenate((np.zeros((2, 1)), np.cumsum(steps, axis=1)), axis=1)
    origin = np.array([np.interp(start, np.arange(n), c) for c in cumulative])
    travel = np.max(cumulative - origin[:, None], axis=0)
    speed = np.max(steps, axis=0) * float(chunk["fps"])
    speed = np.r_[speed, speed[-1]] if n > 1 else np.zeros(1)
    smooth = np.convolve(np.pad(speed, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    for frame in range(max(1, first), min(end, n - 2) + 1):
        if not near and travel[frame] < min_travel_m:
            continue
        peak = np.max(smooth[max(0, frame - 8):min(n, frame + 9)])
        if (smooth[frame] <= smooth[frame - 1]
                and smooth[frame] < smooth[frame + 1]
                and peak > 1e-6 and smooth[frame] <= .65 * peak):
            return Horizon(float(frame), "speed_valley", float(travel[frame]))
    reason = ("event_boundary" if boundary is not None and end == boundary
              else "motion_floor_unmet" if not near and travel[end] < min_travel_m
              else "bounded_fallback")
    return Horizon(float(end), reason, float(travel[end]))


def advance_on_path(path, cursor, distance_m, boundary):
    """Interpolate forward by arc length without passing a known boundary."""
    path = np.asarray(path, float)
    cursor = float(np.clip(cursor, 0, len(path) - 1))
    at = lambda f: np.array([np.interp(f, np.arange(len(path)), path[:, k]) for k in range(3)])
    value = at(cursor)
    stop = max(cursor, min(float(boundary), len(path) - 1))
    for frame in range(int(np.floor(cursor)) + 1, int(np.ceil(stop)) + 1):
        target = at(min(frame, stop))
        length = float(np.linalg.norm(target - value))
        if length > distance_m:
            return value + (target - value) * max(0., distance_m) / length
        distance_m -= length
        value = target
    return value


class StallLookahead:
    """Time-stamped measured history; repeated sampling never integrates state."""

    def __init__(self, frames=10, motion_m=.005, increment_m=.005, maximum_m=.05):
        self.frames, self.motion_m = int(frames), float(motion_m)
        self.increment_m, self.maximum_m = float(increment_m), float(maximum_m)
        self.reset()

    def reset(self):
        self.history = {s: deque() for s in SIDES}
        self.budget = dict.fromkeys(SIDES, 0.)
        self.last_stamp = None

    def update(self, stamp, measured, fps, blocked=()):
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return
        self.last_stamp = stamp
        duration = self.frames / float(fps)
        for side in SIDES:
            history = self.history[side]
            point = measured.get(side)
            if side in blocked or point is None or not np.isfinite(point).all():
                history.clear()
                self.budget[side] = 0.
                continue
            history.append((stamp, np.asarray(point, float).copy()))
            while len(history) > 1 and (stamp - history[1][0]) * 1e-9 >= duration:
                history.popleft()
            if (stamp - history[0][0]) * 1e-9 < duration:
                continue
            # Path length catches out-and-back motion; endpoint distance alone
            # would incorrectly classify a moving wrist as stationary.
            motion = np.linalg.norm(np.diff([p for _, p in history], axis=0), axis=1).sum()
            if motion < self.motion_m:
                self.budget[side] = min(self.maximum_m, self.budget[side] + self.increment_m)
            history.clear()
            history.append((stamp, np.asarray(point, float).copy()))
