"""Measured-pose FOCUS entry and continuous wrist chunk handoff.

This controller never advances finger time to match an outgoing wrist target.
It waits for measured wrists, then requests a fresh state-conditioned chunk.
"""

from collections import deque

import numpy as np

from scalebridge.online.reference import quat_inv, quat_mul, sample_chunk


def rotation_error_deg(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(norm) or norm < 1e-12:
        return float("inf")
    dot = abs(float(np.dot(a, b))) / norm
    return float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))


def closure_approach_distance(chunk, side, *, spread_m):
    """Wrist travel this chunk still predicts before its own grasp frame.

    The grasp frame is the first one whose four non-thumb fingertips sit
    within ``spread_m`` of the thumb tip. The returned distance is the
    straight-line wrist displacement from frame zero to that frame, so it is
    invariant to the rigid alignment applied to a chunk.

    A prediction that asks to close while still several centimetres short of
    its own grasp pose is describing a future grasp, not the present one.
    Returns ``None`` when the chunk never reaches the closed spread.
    """
    tips = np.asarray(chunk[f"{side}_fingertip_local"], dtype=np.float64)
    wrist = np.asarray(chunk[f"{side}_wrist_pos"], dtype=np.float64)
    if (tips.ndim != 3 or tips.shape[1:] != (5, 3) or wrist.shape != (len(tips), 3)
            or not len(tips) or not np.isfinite(tips).all() or not np.isfinite(wrist).all()):
        return None
    spread = np.linalg.norm(tips[:, 1:] - tips[:, :1], axis=2).mean(axis=1)
    closed = np.flatnonzero(spread < spread_m)
    if not len(closed):
        return None
    return float(np.linalg.norm(wrist[int(closed[0])] - wrist[0]))


def rebase_wrist_chunk(chunk, elapsed_s, outgoing, sides, *, positions_only=False):
    """Apply one fixed pose offset per chunk, preserving its wrist increments."""
    rebased = dict(chunk)
    origin = sample_chunk(chunk, elapsed_s, np.array([0.0]))
    for side in sides:
        p, q = f"{side}_wrist_pos", f"{side}_wrist_quat_wxyz"
        rebased[p] = (np.asarray(chunk[p]) + outgoing[p][0] - origin[p][0]).astype(np.float32)
        if not positions_only:
            delta = quat_mul(outgoing[q][0], quat_inv(origin[q][0]))
            rebased[q] = quat_mul(delta, np.asarray(chunk[q])).astype(np.float32)
    return rebased


class FocusHandoff:
    def __init__(self, *, sides=("right",), position_tolerance_m=0.03,
                 rotation_tolerance_deg=10.0, consensus_chunks=2):
        self.sides = tuple(sides)
        if not self.sides or len(set(self.sides)) != len(self.sides) or any(
            side not in ("left", "right") for side in self.sides
        ):
            raise ValueError("focus_handoff_sides must contain left and/or right")
        if not (np.isfinite(position_tolerance_m) and position_tolerance_m > 0
                and np.isfinite(rotation_tolerance_deg) and 0 < rotation_tolerance_deg <= 180
                and int(consensus_chunks) == consensus_chunks and consensus_chunks >= 1):
            raise ValueError("Invalid FOCUS handoff tolerances or consensus count")
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.consensus_chunks = int(consensus_chunks)
        self.reset()

    def reset(self):
        self.state = "approach"
        self.predictions = deque(maxlen=self.consensus_chunks)
        self.entry = None
        self.source_fence = None
        self.hold = None
        self.seen_focus = False
        self.completed_sequence = -1
        self.last_prediction_sequence = -1

    @property
    def active(self):
        return self.state == "focus"

    @property
    def bypass_continuity(self):
        return self.state in ("await_chunk", "focus")

    def observe_prediction(self, chunk, phase):
        """Retain raw world-space entry poses, including late worker results."""
        sequence = int(chunk["sequence_id"])
        if sequence <= self.last_prediction_sequence or self.bypass_continuity:
            return
        self.last_prediction_sequence = sequence
        mask = np.asarray(phase)[:, 0] >= 0.5
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        if self.completed_sequence >= 0:
            # Sliding history can create a one-frame FOCUS at frame zero for
            # a continuously held grasp. It is not a new release/grasp event.
            starts = starts[starts > 0]
        if self.state == "cooldown":
            # A chunk still inside the completed event must not unlock again.
            if sequence <= self.completed_sequence or (mask[0] and not len(starts)):
                return
            self.state = "approach"
        if not len(starts):
            if mask[0] and self.completed_sequence >= 0:
                # The previously predicted next event may now be at frame 0;
                # retain its spatial entry until the actual wrist arrives.
                return
            self.predictions.clear()
            self.entry = None
            return
        index = int(starts[0])
        entry = {key: np.asarray(chunk[key][index], dtype=np.float32).copy()
                 for side in self.sides
                 for key in (f"{side}_wrist_pos", f"{side}_wrist_quat_wxyz")}
        # Do not average predictions of different spatial events.
        if self.predictions and any(
            np.linalg.norm(entry[f"{s}_wrist_pos"] - self.predictions[-1][f"{s}_wrist_pos"])
            > self.position_tolerance_m * 2
            or rotation_error_deg(entry[f"{s}_wrist_quat_wxyz"],
                                  self.predictions[-1][f"{s}_wrist_quat_wxyz"])
            > self.rotation_tolerance_deg * 2 for s in self.sides
        ):
            self.predictions.clear()
        self.predictions.append(entry)
        self.entry = None
        if len(self.predictions) >= self.consensus_chunks:
            self.entry = dict(entry)
            for side in self.sides:
                key = f"{side}_wrist_pos"
                self.entry[key] = np.median([x[key] for x in self.predictions], axis=0)

    def arrive(self, measured, source_fence, outgoing):
        if self.state != "approach" or self.entry is None or outgoing is None:
            return False
        for side in self.sides:
            pose = measured.get(f"{side}_wrist_yaw_link")
            if pose is None:
                return False
            p, q = pose
            if not np.isfinite(p).all() or not np.isfinite(q).all():
                return False
            if (np.linalg.norm(p - self.entry[f"{side}_wrist_pos"]) > self.position_tolerance_m
                    or rotation_error_deg(q, self.entry[f"{side}_wrist_quat_wxyz"])
                    > self.rotation_tolerance_deg):
                return False
        self.state = "await_chunk"
        self.source_fence = int(source_fence)
        self.hold = {k: np.asarray(v).copy() for k, v in outgoing.items()}
        return True

    def accepts(self, chunk):
        return self.state != "await_chunk" or int(chunk["source_state_sequence_id"]) >= self.source_fence

    def start(self):
        if self.state == "await_chunk":
            self.state = "focus"
            self.seen_focus = False

    def update_phase(self, focused, sequence):
        if not self.active:
            return False
        if focused:
            self.seen_focus = True
        elif self.seen_focus:
            self.state = "cooldown"
            self.completed_sequence = int(sequence)
            self.predictions.clear()
            self.entry = None
            self.hold = None
            return True
        return False


class FocusAdaptiveExecution:
    """Thirty-frame approach, ten-frame near-focus execution and motor gating.

    Uses canonical binary FOCUS (whose grasp-intent threshold is 0.1), not a
    second threshold on the phase flag. Prediction clocks remain source clocks.
    """

    def __init__(self, *, approach_frames=30, near_frames=10,
                 near_distance_m=0.045, consensus_chunks=3, sides=("right",)):
        if (any(int(n) != n or n <= 0 for n in (approach_frames, near_frames))
                or near_frames > approach_frames
                or not np.isfinite(near_distance_m) or near_distance_m <= 0):
            raise ValueError("Invalid adaptive FOCUS execution frames or distance")
        self.approach_frames = int(approach_frames)
        self.near_frames = int(near_frames)
        self.near_distance_m = float(near_distance_m)
        self.estimator = FocusHandoff(
            sides=sides, position_tolerance_m=near_distance_m / 2,
            rotation_tolerance_deg=180, consensus_chunks=consensus_chunks,
        )
        self.reset()

    def reset(self):
        self.estimator.reset()
        self.fast = False
        self.arrived = False
        self.active = False
        self.was_focused = False
        self.prediction = None
        self.finger_sample = None
        self.last_sequence = -1

    @property
    def execution_frames(self):
        return self.near_frames if self.fast else self.approach_frames

    def observe_prediction(self, chunk, phase):
        sequence = int(chunk["sequence_id"])
        if sequence <= self.last_sequence:
            return
        if np.asarray(phase).shape != (chunk["frame_count"], 2):
            raise ValueError("Invalid adaptive FOCUS prediction shape")
        self.last_sequence = sequence
        self.prediction = dict(chunk, focus_phase=phase)
        if not self.active:
            self.estimator.observe_prediction(chunk, phase)

    def update_distance(self, measured):
        if self.active:
            return
        self.arrived = False
        if self.estimator.entry is None:
            return
        for side in self.estimator.sides:
            pose = measured.get(f"{side}_wrist_yaw_link")
            if pose is None or not np.isfinite(pose[0]).all():
                return
            if np.linalg.norm(pose[0] - self.estimator.entry[f"{side}_wrist_pos"]) > self.near_distance_m:
                return
        self.fast = True
        self.arrived = True

    def update(self, now_ns, measured, *, running):
        if running:
            self.update_distance(measured)
        self.finger_sample = None
        focused = False
        prediction = self.prediction
        if prediction is not None:
            elapsed = (now_ns - prediction["_focus_source_started_ns"]) * 1e-9
            if 0 <= elapsed < prediction["frame_count"] / prediction["fps"]:
                self.finger_sample = sample_chunk(prediction, elapsed, [0.0])
                focused = bool(self.finger_sample["focus_phase"][0, 0] >= .5)
        if self.was_focused and not focused:
            self.estimator.state = "cooldown"
            self.estimator.completed_sequence = self.last_sequence
            self.estimator.predictions.clear()
            self.estimator.entry = None
            self.arrived = False
        self.was_focused = focused
        self.active = bool(running and self.arrived and focused)


class GraspFingerChunkControl:
    """Grasp-only chunk stitching in DDS space (0 closed, 1 open).

    Continuity anchors the first solved sample of each chunk to the last
    applied command, then preserves that chunk's increments. Progress hold
    instead rejects opening on each bending motor. Thumb rotation is neither
    ordered nor rebased. State belongs to one grasp of one hand only.

    "probe" discards the prediction for the whole grasp and ramps every
    bending motor shut over ``probe_close_s`` from the pose held at entry.
    The aim is a measurement rather than a firmer grip: against a bottle the
    fingers stall at its radius, in free air they reach the commanded close.
    No other mode can show this, because completion freezes the command as
    soon as two motors pass halfway, before anything is actually loaded.
    """

    MODES = ("off", "continuity", "hold_progress", "probe")

    def __init__(self, mode="off", probe_close_s=0.5):
        if mode not in self.MODES:
            raise ValueError(f"Invalid focus_handoff_finger_mode {mode!r}; expected {self.MODES}")
        self.mode = mode
        self.probe_close_s = float(probe_close_s)
        if not (np.isfinite(self.probe_close_s) and self.probe_close_s > 0):
            raise ValueError("focus_probe_close_s must be positive")
        self.states = {}

    def reset(self, side=None):
        if side is None:
            self.states.clear()
        else:
            self.states.pop(side, None)

    def apply(self, side, event_id, chunk_id, proposed, previous, now_ns=None):
        proposed = np.clip(np.asarray(proposed, dtype=np.float64), 0., 1.).copy()
        previous = np.clip(np.asarray(previous, dtype=np.float64), 0., 1.)
        if self.mode == "off":
            return proposed
        state = self.states.get(side)
        if state is None or state["event_id"] != event_id:
            state = dict(event_id=event_id, chunk_id=None, offset=np.zeros(5),
                         probe_start=previous[:5].copy())
            self.states[side] = state
        if self.mode == "probe":
            # event_id is this grasp's started_ns, so the ramp needs no extra
            # clock. Chunk identity is ignored on purpose: a replan that
            # reopens its first frames is exactly what this mode removes.
            if now_ns is None or event_id is None:
                return previous.copy()
            elapsed = max(0., (int(now_ns) - int(event_id)) * 1e-9)
            remaining = max(0., 1. - elapsed / self.probe_close_s)
            proposed[:5] = np.clip(state["probe_start"] * remaining, 0., 1.)
            state["chunk_id"] = chunk_id
            return proposed
        if state["chunk_id"] is not None and chunk_id < state["chunk_id"]:
            return previous.copy()
        if self.mode == "continuity":
            if chunk_id != state["chunk_id"]:
                state["offset"] = previous[:5] - proposed[:5]
            proposed[:5] = np.clip(proposed[:5] + state["offset"], 0., 1.)
        else:
            proposed[:5] = np.minimum(proposed[:5], previous[:5])
        state["chunk_id"] = chunk_id
        return proposed


class HandEventHandoff(FocusHandoff):
    """One hand's measured-pose entry and feedback-confirmed completion."""

    def __init__(self, *, side, timeout_s=5.0, completion_hold_s=1.0,
                 handoff_version="legacy", v3_closed_spread_m=0.11,
                 v3_max_approach_m=0.02, probe_s=0.0, probe_empty_closure=0.85,
                 probe_release_on_empty=True, **kwargs):
        if handoff_version not in ("legacy", "v2", "v3"):
            raise ValueError(f"Invalid focus_handoff_version {handoff_version!r}")
        self.handoff_version = handoff_version
        self.v3_closed_spread_m = float(v3_closed_spread_m)
        self.v3_max_approach_m = float(v3_max_approach_m)
        if not (np.isfinite(self.v3_closed_spread_m) and self.v3_closed_spread_m > 0
                and np.isfinite(self.v3_max_approach_m) and self.v3_max_approach_m >= 0):
            raise ValueError("Invalid v3 closure spread or approach distance")
        # probe_s == 0 disables the probe and leaves completion untouched.
        self.probe_s = float(probe_s)
        self.probe_empty_closure = float(probe_empty_closure)
        self.probe_release_on_empty = bool(probe_release_on_empty)
        if not (np.isfinite(self.probe_s) and self.probe_s >= 0
                and np.isfinite(self.probe_empty_closure)
                and 0 < self.probe_empty_closure <= 1):
            raise ValueError("Invalid FOCUS probe duration or empty-grasp closure")
        # An unreachable probe would abort every grasp on the timeout instead.
        if self.probe_s >= float(timeout_s):
            raise ValueError("focus_probe_decide_s must be below focus_handoff_timeout_s")
        if handoff_version == "v2":
            kwargs["consensus_chunks"] = max(2, kwargs.get("consensus_chunks", 2))
        self.side = side
        self.timeout_s = float(timeout_s)
        self.completion_hold_s = float(completion_hold_s)
        if not (np.isfinite(self.timeout_s) and self.timeout_s > 0
                and np.isfinite(self.completion_hold_s)
                and 0 <= self.completion_hold_s < self.timeout_s):
            raise ValueError("Invalid handoff completion/timeout duration")
        super().__init__(sides=(side,), **kwargs)

    def reset(self):
        super().reset()
        self.event = None
        self.last_kind = None
        self.baseline = None
        self.started_ns = None
        self.completion_since_ns = None
        self.end_reason = None
        self.latest_prediction = None
        # Last v3 gate measurement, for logging only.
        self.v3_last_approach = None
        # Mean measured closure the probe read, and the request it raises when
        # that reading says the hand shut on nothing.
        self.probe_closure = None
        self.reopen_request = False
        self.playback = None
        self.playback_started_ns = None
        # Wall-clock twin of playback_started_ns, used when the reference clock
        # is slowed but the fingers must keep the trained rate.
        self.playback_started_real_ns = None

    def observe_event(self, chunk, hand):
        sequence = int(chunk["sequence_id"])
        if self.latest_prediction is not None and sequence <= self.latest_prediction[0]["sequence_id"]:
            return
        previous_sequence = (None if self.latest_prediction is None
                             else int(self.latest_prediction[0]["sequence_id"]))
        self.latest_prediction = (chunk, hand)
        if self.bypass_continuity:
            return
        # Close and release alternate: after a completed close only an open
        # event may arm, and vice versa. This alternation is also the only
        # path out of "cooldown", because super().observe_prediction below is
        # what clears that state and it is unreachable when events is empty.
        events = [event for event in hand["events"]
                  if self.last_kind is None or event["kind"] != self.last_kind]
        if self.handoff_version == "v2":
            # The worker tests finger intent; use the unrebased prediction's
            # wrist displacement here, before measured-pose arrival gating.
            horizon = max(1, int(round(.3 * chunk["fps"])))
            wrist = np.asarray(chunk[f"{self.side}_wrist_pos"])
            steady = (horizon < len(wrist) and np.isfinite(wrist[[0, horizon]]).all()
                      and np.linalg.norm(wrist[horizon] - wrist[0]) <= .012)
            events = [event for event in events
                      if event["kind"] != "close" or (event["start"] == 0 and steady)]
            if (events and events[0]["kind"] == "close"
                    and previous_sequence is not None and sequence != previous_sequence + 1):
                self.predictions.clear()
                self.entry = None
        elif self.handoff_version == "v3":
            # Legacy arms on any closing edge in the chunk, including one
            # several seconds and centimetres ahead, then drives the wrist to
            # that stale pose. Keep only predictions whose own grasp frame is
            # already here. Rejecting a close the chunk cannot locate is
            # deliberate: an unmeasurable entry is what v3 exists to remove.
            self.v3_last_approach = None
            kept = []
            for event in events:
                if event["kind"] != "close":
                    kept.append(event)
                    continue
                travel = closure_approach_distance(
                    chunk, self.side, spread_m=self.v3_closed_spread_m)
                self.v3_last_approach = travel
                if travel is not None and travel <= self.v3_max_approach_m:
                    kept.append(event)
            events = kept
        if not events:
            self.entry = None
            self.predictions.clear()
            return
        event = events[0]
        if self.event is not None and self.event["kind"] != event["kind"]:
            self.predictions.clear()
        self.event = event
        phase = np.zeros((chunk["frame_count"], 2), dtype=np.float32)
        phase[event["start"]:event["end"] + 1, 0] = 1
        # Frame-zero events have already passed the physical motion test. They
        # are valid even after a previous event; do not discard them globally.
        self.completed_sequence = -1
        super().observe_prediction(chunk, phase)

    def arrive(self, measured, source_fence, outgoing, actual):
        if actual is None or not np.isfinite(actual).all():
            return False
        if not super().arrive(measured, source_fence, outgoing):
            return False
        self.baseline = np.asarray(actual, dtype=np.float32).copy()
        return True

    def request_external(self, kind, source_fence, outgoing, actual):
        """Enter the existing fresh-chunk handoff from an external supervisor.

        GPT has already decided that the measured hand is at the semantic
        grasp/release point, so this deliberately bypasses only the keypoint
        event detector and measured-entry proximity test.  The established
        wrist hold, source-state fence, fresh-chunk rebase, finger unlock and
        continuity rejoin paths remain unchanged.
        """
        if kind not in ("close", "open"):
            raise ValueError(f"external hand event must be close/open, got {kind!r}")
        if self.state in ("await_chunk", "focus") or outgoing is None:
            return False
        actual = np.asarray(actual, dtype=np.float32)
        if actual.shape != (5,) or not np.isfinite(actual).all():
            return False
        self.event = {
            "kind": kind,
            "start": 0,
            "end": 0,
            "goal": np.ones_like(actual) if kind == "close" else np.zeros_like(actual),
            "source": "gpt",
        }
        self.state = "await_chunk"
        self.source_fence = int(source_fence)
        self.hold = {key: np.asarray(value).copy() for key, value in outgoing.items()}
        self.baseline = actual.copy()
        self.started_ns = None
        self.completion_since_ns = None
        self.end_reason = None
        self.entry = None
        self.predictions.clear()
        return True

    def start(self, now_ns):
        super().start()
        self.started_ns = int(now_ns)
        self.completion_since_ns = None
        self.end_reason = None

    def finish(self, reason):
        self.state = "cooldown"
        # A timed-out attempt did not complete; allow a new consensus to retry.
        # "closed_empty" is likewise not a completed grasp, so it must leave
        # last_kind clear and let the next close arm without first waiting for
        # HAT to predict a release it has no reason to predict.
        self.last_kind = (self.event["kind"]
                          if reason in ("feedback_complete", "grasp_confirmed")
                          else None)
        self.end_reason = reason
        self.entry = None
        self.predictions.clear()
        self.hold = None

    def _finish_probe(self, now_ns, actual):
        """Classify the grasp once the commanded close has had its window.

        Returns True once the probe has decided. Until then it withholds the
        ordinary completion tests, whose edge-based pass would end the window
        before the fingers ever reach the object.
        """
        if (now_ns - self.started_ns) * 1e-9 < self.probe_s:
            return None
        closure = float(np.mean(actual))
        self.probe_closure = closure
        if closure >= self.probe_empty_closure:
            # Commanded shut and nothing stopped the fingers.
            self.finish("closed_empty")
            if self.probe_release_on_empty:
                self.reopen_request = True
        else:
            self.finish("grasp_confirmed")
        return True

    def update_actual(self, now_ns, actual):
        if not self.active:
            return False
        if (now_ns - self.started_ns) * 1e-9 >= self.timeout_s:
            self.finish("timeout")
            return True
        if actual is None or not np.isfinite(actual).all():
            self.completion_since_ns = None
            return False
        actual = np.asarray(actual, dtype=np.float32)
        closing = self.event["kind"] == "close"
        if closing and self.probe_s > 0:
            # A release still completes on its own feedback; only the grasp is
            # decided by the commanded probe.
            return bool(self._finish_probe(now_ns, actual))
        direction = 1.0 if closing else -1.0
        moved = direction * (actual - self.baseline) >= .05
        confirmed = np.sum(moved & (actual >= .5 if closing else actual < .45)) >= 2
        requested = direction * (self.event["goal"] - self.baseline) >= .05
        reached = np.sum(moved & requested &
                         (direction * (self.event["goal"] - actual) <= .12)) >= 2
        # Prediction edges alone NEVER complete a grasp/release. Keep the
        # state across replans until fresh physical feedback shows progress.
        # v3 changes the entry gate only, so it keeps the legacy exit intact
        # and stays a clean A/B against legacy.
        if confirmed or (self.handoff_version in ("legacy", "v3") and reached):
            if self.completion_since_ns is None:
                self.completion_since_ns = int(now_ns)
            if (now_ns - self.completion_since_ns) * 1e-9 >= self.completion_hold_s:
                self.finish("feedback_complete")
                return True
        else:
            self.completion_since_ns = None
        return False


class PerHandFocusHandoff:
    """Independent left/right gates sharing only the HAT chunk transport."""

    def __init__(self, *, sides=("left", "right"), **kwargs):
        sides = tuple(sides)
        if not sides or len(set(sides)) != len(sides) or any(
                side not in ("left", "right") for side in sides):
            raise ValueError("Invalid handoff sides")
        self.sides = sides
        self.controllers = {side: HandEventHandoff(side=side, **kwargs) for side in sides}

    def reset(self):
        for controller in self.controllers.values():
            controller.reset()

    @property
    def active_sides(self):
        return tuple(side for side, c in self.controllers.items() if c.active)

    @property
    def bypass_sides(self):
        return tuple(side for side, c in self.controllers.items() if c.bypass_continuity)

    @property
    def active(self):
        return bool(self.active_sides)

    @property
    def bypass_continuity(self):
        return bool(self.bypass_sides)

    @property
    def state(self):
        states = {c.state for c in self.controllers.values()}
        return next(iter(states)) if len(states) == 1 else "mixed"

    @property
    def entry(self):
        entries = {key: value for c in self.controllers.values() if c.entry is not None
                   for key, value in c.entry.items()}
        return entries or None

    def accepts(self, chunk):
        # One hand waiting for its fence must not block fresh replans driving
        # the other. start_fresh independently checks each hand's source fence.
        return True

    def observe_prediction(self, chunk, payload):
        for side, controller in self.controllers.items():
            controller.observe_event(chunk, payload["hands"][side])

    def start_fresh(self, chunk, now_ns):
        started = []
        for side, c in self.controllers.items():
            if c.state == "await_chunk" and chunk["source_state_sequence_id"] >= c.source_fence:
                c.start(now_ns)
                started.append(side)
        return started

    def request_external(self, side, kind, source_fence, outgoing, actual):
        if side not in self.controllers:
            return False
        return self.controllers[side].request_external(
            kind, source_fence, outgoing, actual,
        )

    def rebase(self, chunk, elapsed_s, outgoing):
        for side, c in self.controllers.items():
            if not c.bypass_continuity:
                continue
            anchor = c.hold if c.state == "await_chunk" else outgoing
            chunk = rebase_wrist_chunk(chunk, elapsed_s, anchor, (side,))
        return chunk
