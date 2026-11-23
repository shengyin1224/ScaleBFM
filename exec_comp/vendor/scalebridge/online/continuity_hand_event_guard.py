"""Optional search and per-hand playback bounds; never decides FOCUS.

Consumes the existing worker's *same-chunk* closure prediction. Closure is
0 open / 1 closed, in HAT motor order. A small pre-bend is not an event:
several bending channels must change substantially from the latency pose,
still disagree with current feedback, and sustain that change. Routine swaps
wait for same-chunk IK while the outgoing reference continues. The search's
missing-data fallback remains available for startup/recovery handoffs.
"""

from dataclasses import dataclass

import numpy as np

from scalebridge.online.focus import HAND_BEND_COLUMNS
from scalebridge.online.reference import sample_chunk, quat_slerp


@dataclass(frozen=True)
class ContinuityHandEventGuard:
    # Experimental execution-conflict thresholds, not grasp-success labels.
    min_feedback_delta: float = 0.25
    min_interval_delta: float = 0.10
    sustain_s: float = 0.15
    min_fingers: int = 2

    def __post_init__(self):
        for value in (self.min_feedback_delta, self.min_interval_delta):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Continuity hand guard deltas must be in (0, 1]")
        if not np.isfinite(self.sustain_s) or self.sustain_s <= 0:
            raise ValueError("Continuity hand guard sustain_s must be positive")
        if self.min_fingers != int(self.min_fingers) or not 2 <= self.min_fingers <= 5:
            raise ValueError("Continuity hand guard needs 2..5 bending fingers")

    @classmethod
    def from_config(cls, config):
        return cls(
            min_feedback_delta=float(config.get("continuity_hand_event_feedback_delta", .25)),
            min_interval_delta=float(config.get("continuity_hand_event_interval_delta", .10)),
            sustain_s=float(config.get("continuity_hand_event_sustain_s", .15)),
        )

    def search_limit(self, chunk, latency_frame, locked_sides, prediction, actual):
        """Return inclusive maximum start frame and serializable diagnostics.

        Only a newly crossed change is guarded. An already changed latency
        pose cannot be repaired by restricting continuity and is left alone.
        The first sustained conflict is a fence: later returning to the same
        finger pose must not make jumping over the intervening event legal.
        """
        lower = int(latency_frame)
        count = int(chunk["frame_count"])
        limit = count - 1
        diagnostics = {"reason": "no_conflict", "locked_sides": list(locked_sides),
                       "events": []}
        if not locked_sides:
            diagnostics["reason"] = "hands_unlocked"
            return limit, diagnostics
        # Never use a previous sequence's asynchronous result for this chunk.
        if (prediction is None
                or prediction.get("sequence_id") != chunk["sequence_id"]
                or prediction.get("source_state_sequence_id") != chunk["source_state_sequence_id"]
                or prediction.get("frame_count") != count
                or prediction.get("fps") != chunk["fps"]
                or not isinstance(prediction.get("hand_focus"), dict)):
            diagnostics["reason"] = "prediction_pending"
            return lower, diagnostics
        sustain = max(3, int(np.ceil(self.sustain_s * float(chunk["fps"]))))
        for side in locked_sides:
            hand = prediction["hand_focus"].get(side, {})
            try:
                closure = np.asarray(hand.get("closure"), dtype=np.float64)
                feedback = np.asarray(actual.get(side), dtype=np.float64)
                if (closure.shape != (count, 6) or feedback.shape != (5,)
                        or not np.isfinite(closure).all() or not np.isfinite(feedback).all()):
                    raise ValueError("Missing/invalid hand data")
            except (TypeError, ValueError):
                diagnostics.update(reason="hand_data_unavailable", side=side)
                return lower, diagnostics
            bend = np.clip(closure[:, HAND_BEND_COLUMNS], 0., 1.)
            feedback = np.clip(feedback, 0., 1.)
            from_latency = bend - bend[lower]
            from_feedback = bend - feedback
            for kind, direction in (("close", 1.), ("open", -1.)):
                coordinated = np.sum(
                    (direction * from_latency >= self.min_interval_delta)
                    & (direction * from_feedback >= self.min_feedback_delta), axis=1
                ) >= self.min_fingers
                # Values at/before the latency frame cannot create a new fence.
                coordinated[:lower + 1] = False
                edges = np.flatnonzero(np.diff(
                    np.r_[False, coordinated, False].astype(np.int8)))
                for start, stop in edges.reshape(-1, 2):
                    if stop - start >= sustain:
                        limit = min(limit, int(start) - 1)
                        diagnostics["events"].append(
                            {"side": side, "kind": kind, "conflict_frame": int(start)})
                        break
        if diagnostics["events"]:
            diagnostics["reason"] = "unexecuted_hand_change"
        return limit, diagnostics

    def limit_for_env(self, env, chunk, latency_frame, now_ns):
        """Small adapter; the controller/worker and their lifecycle are read-only."""
        locked = tuple(side for side, controller in env.focus_handoff.controllers.items()
                       if not controller.bypass_continuity)
        prediction = env._focus_prediction_chunks.get(chunk["sequence_id"])
        actual = env._handoff_measured_bends(now_ns) if locked else {}
        limit, state = self.search_limit(chunk, latency_frame, locked, prediction, actual)
        if state["reason"] == "unexecuted_hand_change":
            events = [event for event in state["events"]
                      if event["kind"] != env.focus_handoff.controllers[event["side"]].last_kind]
            limit = min((event["conflict_frame"] - 1 for event in events),
                        default=int(chunk["frame_count"]) - 1)
            state = dict(state, events=events,
                         reason="unexecuted_hand_change" if events else "no_conflict")
        return limit, state


class GuardChunkPreflight:
    """Inspect a pinned candidate while the outgoing chunk keeps playing.

    New arrivals stay in the env's latest-pending slot; they cannot replace
    the inspected candidate every tick and starve its IK result. A timeout
    drops the candidate without resetting the accepted-stream watchdog.
    """

    def __init__(self, detector, timeout_s=.4):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("continuity_hand_event_check_timeout_s must be positive")
        self.detector = detector
        self.timeout_s = float(timeout_s)
        self.reset()

    def reset(self):
        self.waiting = None
        self.sequence = None
        self.started_ns = None
        self.last_wait_s = 0.

    @staticmethod
    def applies(env):
        handoff = getattr(env, "focus_handoff", None)
        return (handoff is not None and hasattr(handoff, "controllers")
                and not handoff.bypass_continuity
                and env.reference_playing and not env.awaiting_fresh_chunk
                and not env.localization_paused and not env.requires_restart
                and env.held_sample is None and env.chunk is not None
                and env.chunk_started_ns is not None
                and getattr(env, "chunk_start_reference_continuity", False))

    def take(self, env, latest):
        waiting = self.waiting
        self.waiting = None
        if waiting is None:
            return latest
        if not self.applies(env):
            return latest if latest is not None else waiting
        if latest is not None and latest["sequence_id"] > waiting["sequence_id"]:
            env._pending_raw_chunk = latest
        return waiting

    def check(self, env, candidate, aligned, now_ns):
        if not self.applies(env):
            self.reset()
            return "bypass"
        sequence = int(candidate["sequence_id"])
        lower, _, _ = env._latency_start_frame(aligned, now_ns)
        _, diagnostic = self.detector.limit_for_env(env, aligned, lower, now_ns)
        if diagnostic["reason"] not in ("prediction_pending", "hand_data_unavailable"):
            self.last_wait_s = (max(0., (now_ns - self.started_ns) * 1e-9)
                                if self.sequence == sequence and self.started_ns is not None else 0.)
            self.sequence = None
            self.started_ns = None
            return "ready"
        if self.sequence != sequence:
            self.sequence = sequence
            self.started_ns = int(now_ns)
            preview = env._focus_prediction_chunks.get(sequence)
            if preview is None:
                env._preview_chunk_focus(candidate, env._source_state_timestamp_ns(
                    candidate["source_state_sequence_id"]), now_ns)
            else:
                # A latest-only worker may have dropped an earlier preview.
                # Re-submit this pinned candidate once, not on every tick.
                env._submit_chunk_focus(preview, now_ns)
        if (now_ns - self.started_ns) * 1e-9 >= self.timeout_s:
            self.reset()
            return "timeout"
        self.waiting = candidate
        return "waiting"


class HandEventPlaybackBoundary:
    """Cap locked-hand cursors/horizons and latch reached boundaries across replans.

    Source/body clocks still run, so localization, staleness and the other hand
    retain their timing. Only the affected wrist pose/fingertip channels stop.
    Originals are retained before RTC; workers always inspect those originals.
    """

    def __init__(self, detector):
        self.detector = detector
        self.reset()

    def reset(self):
        self.originals = {}
        self.holds = {}
        self.armed = {}
        self.known = set()
        self.diagnostics = {}

    @staticmethod
    def keys(side):
        return (f"{side}_wrist_pos", f"{side}_wrist_quat_wxyz",
                f"{side}_fingertip_local")

    def register(self, chunk, latency_frame):
        sequence = int(chunk["sequence_id"])
        self.originals[sequence] = (chunk, int(latency_frame))
        while len(self.originals) > 32:
            removed = next(iter(self.originals))
            self.originals.pop(removed)
            self.known = {item for item in self.known if item[0] != removed}
            self.armed = {key: value for key, value in self.armed.items() if key[0] != removed}

    def bounds(self, env, chunk, now_ns):
        sequence = int(chunk["sequence_id"])
        original, lower = self.originals.get(sequence, (chunk, int(
            chunk.get("execution_alignment", {}).get("expected_frame", 0))))
        actual = env._handoff_measured_bends(now_ns)
        prediction = env._focus_prediction_chunks.get(sequence)
        result = {}
        for side, controller in env.focus_handoff.controllers.items():
            if controller.bypass_continuity:
                self.holds.pop(side, None)
                self.diagnostics.pop(side, None)
                self.armed = {key: value for key, value in self.armed.items() if key[1] != side}
                continue
            _, state = self.detector.search_limit(
                original, lower, (side,), prediction, actual)
            # Pending IK is not evidence of a physical event. It only cancels
            # continuity's forward search; it must not freeze ordinary approach.
            events = [event for event in state["events"]
                      if event["kind"] != controller.last_kind]
            if events:
                event = min(events, key=lambda item: item["conflict_frame"])
                existing = self.armed.get((sequence, side))
                if existing is None or event["conflict_frame"] - 1 < existing["max_frame"]:
                    self.armed[(sequence, side)] = dict(
                        event, max_frame=event["conflict_frame"] - 1, sequence_id=sequence)
            retained = self.armed.get((sequence, side))
            if retained is not None and retained["kind"] != controller.last_kind:
                result[side] = retained
            elif side not in self.holds:
                self.diagnostics.pop(side, None)
        return result

    def clip_chunk(self, env, chunk, now_ns):
        """Copy only guarded arrays, including old/new RTC inputs and tails."""
        bounds = self.bounds(env, chunk, now_ns)
        clipped = chunk
        for side, controller in env.focus_handoff.controllers.items():
            if controller.bypass_continuity:
                continue
            hold = self.holds.get(side)
            event = bounds.get(side)
            if hold is None and event is None:
                continue
            if clipped is chunk:
                clipped = dict(chunk)
            for key in self.keys(side):
                values = np.asarray(chunk[key]).copy()
                if hold is not None:
                    values[:] = hold[key]
                else:
                    frame = event["max_frame"]
                    values[frame + 1:] = values[frame]
                clipped[key] = values
        return clipped

    def apply(self, env, sample, now_ns, offsets):
        """Final fence after RTC, overlap blending and optional wrist assistance."""
        chunk = env.chunk
        current_bounds = self.bounds(env, chunk, now_ns)
        previous = getattr(env, "previous_chunk", None)
        previous_start = getattr(env, "previous_chunk_started_ns", None)
        blend_start = getattr(env, "blend_started_ns", None)
        blend_s = max(float(env.cfg.get("chunk_blend_s", .22)), 1e-6)
        alpha = (min(max((now_ns - blend_start) * 1e-9 / blend_s, 0.), 1.)
                 if blend_start is not None else 1.)
        previous_bounds = (self.bounds(env, previous, now_ns)
                           if previous is not None and previous_start is not None and alpha < 1.
                           else {})
        affected = set(current_bounds) | set(previous_bounds) | set(self.holds)
        if not affected:
            return sample
        clipped = self.clip_chunk(env, chunk, now_ns)
        elapsed = env._elapsed(now_ns, env.chunk_started_ns)
        safe = sample_chunk(clipped, elapsed, offsets)
        if previous_bounds or (previous is not None and previous_start is not None and alpha < 1.):
            old = sample_chunk(self.clip_chunk(env, previous, now_ns),
                               env._elapsed(now_ns, previous_start), offsets)
            for side in affected:
                for key in self.keys(side):
                    if "quat" in key:
                        safe[key] = quat_slerp(old[key], safe[key], np.full(len(offsets), alpha)).astype(np.float32)
                    else:
                        safe[key] = old[key] * (1. - alpha) + safe[key] * alpha
        output = dict(sample)
        for side in affected:
            if env.focus_handoff.controllers[side].bypass_continuity:
                continue
            event = current_bounds.get(side) or previous_bounds.get(side)
            for candidate, cursor, fps in (
                    (current_bounds.get(side), elapsed, chunk["fps"]),
                    (previous_bounds.get(side), env._elapsed(now_ns, previous_start),
                     previous["fps"] if previous is not None else chunk["fps"])):
                if candidate is None:
                    continue
                token = (candidate["sequence_id"], side)
                newly_known = token not in self.known
                self.known.add(token)
                if cursor * fps >= candidate["max_frame"] and side not in self.holds:
                    # A delayed result cannot undo a command already sent. Hold
                    # its last pose instead of jumping backwards to a past edge.
                    last = getattr(env, "last_reference_sample", None)
                    anchor = (last if newly_known and cursor * fps > candidate["max_frame"]
                              and last is not None else safe)
                    self.holds[side] = {key: np.asarray(anchor[key][0]).copy()
                                        for key in self.keys(side)}
                    self.diagnostics[side] = dict(candidate, held=True)
            for key in self.keys(side):
                if side in self.holds:
                    output[key] = np.repeat(self.holds[side][key][None], len(offsets), axis=0)
                else:
                    output[key] = safe[key]
            if event is not None and side not in self.holds:
                self.diagnostics[side] = dict(event, held=False)
        return output
