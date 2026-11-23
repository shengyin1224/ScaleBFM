"""Opt-in, world-space wrist advancement towards live finger intent.

No transport, IK, finger commands or FOCUS decisions live here. Sampling is
pure; only update() integrates time, so repeated FOCUS/RTC queries cannot
advance the wrist twice in one policy step.
"""

from collections import deque

import numpy as np

from scalebridge.online.focus import HAND_BEND_COLUMNS


def finger_intent_entry(prediction, side, actual, elapsed_s):
    """First sustained opposite bend intent on the still-live prediction tail.

Use the worker's physical closure (1 = closed), excluding thumb rotation.
Re-evaluate against NOW's feedback, not the worker's older source hand.
Opening uses motion onset, rather than FOCUS's padded pre-window.
"""
    hand = prediction.get("hand_focus", {}).get(side)
    if hand is None:
        return None
    closure = np.asarray(hand.get("closure"), dtype=float)
    actual = np.asarray(actual, dtype=float)
    wrist = np.asarray(prediction[f"{side}_wrist_pos"], dtype=float)
    if (closure.shape != (len(wrist), 6) or actual.shape != (5,)
            or not np.isfinite(closure).all() or not np.isfinite(actual).all()
            or not np.isfinite(wrist).all()):
        return None
    bend = np.clip(closure[:, HAND_BEND_COLUMNS], 0., 1.)
    kind = "open" if np.median(actual) >= .5 else "close"
    motion = (np.sum(actual - bend >= .1, axis=1) >= 2 if kind == "open"
              else np.sum((bend >= .1) & (bend - actual >= .05), axis=1) >= 2)
    cursor = max(0, int(np.ceil(max(0., elapsed_s) * prediction["fps"])))
    for start in range(cursor, len(motion) - 2):
        if motion[start:start + 3].all():
            return dict(kind=kind, frame=start, target=wrist[start].copy(),
                        sequence_id=int(prediction["sequence_id"]))
    return None


class StallWristAssist:
    def __init__(self, *, sides=("right",), stall_s=2., motion_m=.005,
                 speed_m_s=.005, max_offset_m=.15, max_tracking_error_m=.05):
        self.sides = tuple(sides)
        if (not self.sides or len(set(self.sides)) != len(self.sides)
                or any(s not in ("left", "right") for s in self.sides)):
            raise ValueError("stall_wrist_assist_sides must contain left and/or right")
        values = (stall_s, motion_m, speed_m_s, max_offset_m, max_tracking_error_m)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Stall wrist assist limits must be finite and positive")
        self.stall_s, self.motion_m, self.speed_m_s = map(float, values[:3])
        self.max_offset_m = float(max_offset_m)
        self.max_tracking_error_m = float(max_tracking_error_m)
        self.reset()

    def reset(self):
        self.last_ns = None
        self.states = {side: dict(mode="watch", history=deque(),
                                 offset=np.zeros(3), direction=None, floor=None,
                                 target=None, kind=None, frame=None, sequence_id=None,
                                 reason="watch") for side in self.sides}

    def suspend(self):
        # Retain displacement across a temporary hold, but never integrate
        # held time or mistake missing feedback for a stationary wrist.
        self.last_ns = None
        for state in self.states.values():
            state["history"].clear()

    def update(self, now_ns, measured, commands, intents, focus_sides=(), *, running=True):
        if not running:
            self.suspend()
            return
        dt = (0. if self.last_ns is None else
              min(.1, max(0., (now_ns - self.last_ns) * 1e-9)))
        self.last_ns = int(now_ns)
        for side, state in self.states.items():
            command = commands.get(side)
            if command is None or not np.isfinite(command).all():
                state["history"].clear()
                continue
            command = np.asarray(command, dtype=float)
            if side in focus_sides:
                if state["mode"] == "advance":
                    # Handoff already rebases its fresh playback to the
                    # assisted outgoing target. Do NOT add the offset again.
                    state["mode"] = "guard"
                    state["floor"] = float(np.dot(command, state["direction"]))
                    state["offset"][:] = 0.
                state["history"].clear()
                state["reason"] = "focus"
                if state["mode"] == "guard":
                    state["floor"] = max(state["floor"], float(np.dot(command, state["direction"])))
                continue
            if state["mode"] == "guard":
                state["floor"] = max(state["floor"], float(np.dot(command, state["direction"])))
                state["reason"] = "no_backtrack"
                continue
            position = measured.get(side)
            if position is None or not np.isfinite(position).all():
                state["history"].clear()
                state["reason"] = "missing_feedback"
                continue
            position = np.asarray(position, dtype=float)
            history = state["history"]
            history.append((int(now_ns), position.copy()))
            cutoff = now_ns - int(self.stall_s * 1e9)
            while len(history) > 1 and history[1][0] <= cutoff:
                history.popleft()
            stalled = (history[0][0] <= cutoff and
                       np.max(np.linalg.norm(np.stack([p for _, p in history]) - history[0][1], axis=1))
                       <= self.motion_m)
            intent = intents.get(side)
            if intent is None:
                state["reason"] = "no_live_intent"
                continue
            target = np.asarray(intent["target"], dtype=float)
            state.update(target=target.copy(), kind=intent["kind"], frame=intent["frame"],
                         sequence_id=intent["sequence_id"])
            if state["mode"] == "watch" and not stalled:
                continue
            delta = target - position
            distance = np.linalg.norm(delta)
            if distance < 1e-6:
                state["reason"] = "at_target"
                continue
            direction = delta / distance
            state["direction"] = direction
            if state["mode"] == "watch":
                state["mode"] = "advance"
            if np.linalg.norm(command - position) > self.max_tracking_error_m:
                state["reason"] = "tracking_lag"
                continue
            # Arrival belongs to the measured wrist, not the commanded
            # reference: tracking lag can require the reference to lead past
            # the predicted entry. The tracking-error gate above bounds that
            # lead, retaining the accumulated offset while the wrist catches up.
            step = direction * min(self.speed_m_s * dt, distance)
            proposed = state["offset"] + step
            # Solve the step length at the spherical offset limit; don't
            # rescale the whole accumulated vector when direction changes.
            if np.linalg.norm(proposed) > self.max_offset_m:
                along = float(np.dot(state["offset"], direction))
                room = max(0., -along + np.sqrt(max(0., along * along +
                    self.max_offset_m ** 2 - float(np.dot(state["offset"], state["offset"])))))
                proposed = state["offset"] + direction * min(np.linalg.norm(step), room)
                state["reason"] = "offset_limit"
            else:
                state["reason"] = "advance"
            state["offset"] = proposed

    def apply(self, sample, focus_sides=()):
        """Apply equal translation to all wrist future tokens; never mutate input."""
        result = dict(sample)
        for side, state in self.states.items():
            key = f"{side}_wrist_pos"
            if state["mode"] == "advance" and side not in focus_sides:
                result[key] = (np.asarray(sample[key]) + state["offset"]).astype(np.float32)
            elif state["mode"] == "guard":
                direction = state["direction"]
                positions = np.asarray(sample[key])
                # Clamp signed progress, so negative X/Y/Z approaches work.
                missing = np.maximum(0., state["floor"] - positions @ direction)
                result[key] = (positions + missing[:, None] * direction).astype(np.float32)
        return result

    def release_caught_up(self, raw_sample, focus_sides=(), *, playback_sample=None):
        for side, state in self.states.items():
            if state["mode"] != "guard" or side in focus_sides:
                continue
            # Only ordinary playback may retire the guard; freshly rebased
            # FOCUS playback includes our offset and is not raw catch-up.
            p = raw_sample[f"{side}_wrist_pos"][0]
            playing = (p if playback_sample is None
                       else playback_sample[f"{side}_wrist_pos"][0])
            if (float(np.dot(p, state["direction"])) >= state["floor"] - 1e-6
                    and float(np.dot(playing, state["direction"])) >= state["floor"] - 1e-6):
                state.update(mode="watch", floor=None, direction=None, reason="caught_up")
                state["history"].clear()

    def snapshot(self):
        return {side: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in state.items() if k != "history"}
                for side, state in self.states.items()}
