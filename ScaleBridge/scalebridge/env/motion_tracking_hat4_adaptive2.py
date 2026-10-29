"""Adaptive execution with short RTC and handoff-independent grasp holding."""

from time import monotonic_ns

import numpy as np
from loguru import logger

from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from scalebridge.online.adaptive_chunk_execution import SIDES
from scalebridge.online.focus_handoff import PerHandFocusHandoff


class HAT4Adaptive2MotionTrackingEnv(HAT4AdaptiveMotionTrackingEnv):
    """A conservative revision of adaptive; the original environment is intact."""

    def _setup_metadata(self):
        self._adaptive2_hand_focus = {}
        self._adaptive2_grasp = dict.fromkeys(SIDES)
        self._adaptive2_hold_suppressed = dict.fromkeys(SIDES, False)
        self._adaptive2_last_hand_motor = dict.fromkeys(SIDES)
        self._adaptive2_grasp_serial = 0
        super()._setup_metadata()

        self._adaptive2_hold_enabled = bool(
            self.cfg.get("adaptive2_independent_hold_progress", True)
        )
        timeout = float(self.cfg.get("adaptive2_hold_progress_timeout_s", 15.0))
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("adaptive2_hold_progress_timeout_s must be finite and positive")
        self._adaptive2_hold_timeout_s = timeout
        carry = float(self.cfg.get("adaptive2_stall_carry_max_m", .02))
        cosine = float(self.cfg.get("adaptive2_stall_carry_min_cosine", .5))
        if not np.isfinite(carry) or carry < 0:
            raise ValueError("adaptive2_stall_carry_max_m must be finite and non-negative")
        if not np.isfinite(cosine) or not -1 <= cosine <= 1:
            raise ValueError("adaptive2_stall_carry_min_cosine must be in [-1, 1]")
        if self._adaptive2_hold_enabled:
            if self.focus_handoff is not None:
                raise ValueError(
                    "adaptive2 independent hold_progress cannot share full FOCUS handoff"
                )
            if self._handoff_finger_control.mode != "hold_progress":
                raise ValueError(
                    "adaptive2 independent hold requires focus_handoff_finger_mode=hold_progress"
                )
            logger.info(
                "[Adaptive2 HAT] Independent hold_progress enabled; CLOSE/OPEN "
                f"comes from per-hand FOCUS metadata, safety timeout={timeout:.1f}s."
            )

    def _on_chunk_focus_result(self, sequence_id, payload):
        if not isinstance(payload, dict) or "hands" not in payload:
            return
        self._adaptive2_hand_focus[int(sequence_id)] = payload["hands"]
        while len(self._adaptive2_hand_focus) > 32:
            self._adaptive2_hand_focus.pop(next(iter(self._adaptive2_hand_focus)))

    def _adaptive_hand(self, side):
        hand = super()._adaptive_hand(side)
        if hand is not None or self.chunk is None:
            return hand
        return self._adaptive2_hand_focus.get(
            int(self.chunk["sequence_id"]), {}
        ).get(side)

    def _adaptive_schedule_boundary(self, boundary, start):
        # Full FOCUS handoff still owns its historical boundary semantics. With
        # no handoff, CLOSE only caps stall lookahead in _sample_reference; it
        # must not force an early chunk switch by itself.
        if isinstance(getattr(self, "focus_handoff", None), PerHandFocusHandoff):
            return super()._adaptive_schedule_boundary(boundary, start)
        return None

    @staticmethod
    def _path_direction(chunk, side, cursor, *, skip_frames=0, span_frames=5):
        path = np.asarray(chunk[f"{side}_wrist_pos"], dtype=np.float64)
        start = float(np.clip(cursor + skip_frames, 0, len(path) - 1))
        stop = float(np.clip(start + span_frames, 0, len(path) - 1))
        if stop <= start:
            return None
        frames = np.arange(len(path), dtype=np.float64)
        at = lambda frame: np.array(
            [np.interp(frame, frames, path[:, axis]) for axis in range(3)]
        )
        delta = at(stop) - at(start)
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(norm) or norm < 1e-3:
            return None
        return delta / norm

    def _adaptive2_carried_stall_budget(
        self, old_chunk, old_cursor, new_chunk, new_cursor, old_budget
    ):
        result = dict.fromkeys(SIDES, 0.0)
        if old_chunk is None or new_chunk is None:
            return result
        cap = float(self.cfg.get("adaptive2_stall_carry_max_m", .02))
        threshold = float(self.cfg.get("adaptive2_stall_carry_min_cosine", .5))
        skip = int(self.cfg.get("rtc_prefix_frames", 0)) if self.cfg.get("rtc_enabled", False) else 0
        for side in SIDES:
            budget = min(cap, max(0.0, float(old_budget.get(side, 0.0))))
            if budget <= 0:
                continue
            old_direction = self._path_direction(old_chunk, side, old_cursor)
            new_direction = self._path_direction(
                new_chunk, side, new_cursor, skip_frames=skip
            )
            if old_direction is None or new_direction is None:
                continue
            if float(np.dot(old_direction, new_direction)) >= threshold:
                result[side] = budget
        return result

    def _accept_latest_chunk(self, now_ns):
        old_chunk = self.chunk
        old_cursor = (
            self._elapsed(now_ns, self.chunk_started_ns) * float(self.chunk["fps"])
            if self.chunk is not None and self.chunk_started_ns is not None else 0.0
        )
        old_budget = dict(self._adaptive_stall.budget)
        accepted = super()._accept_latest_chunk(now_ns)
        if not accepted:
            return False
        new_cursor = (
            self._elapsed(now_ns, self.chunk_started_ns) * float(self.chunk["fps"])
            if self.chunk is not None and self.chunk_started_ns is not None else 0.0
        )
        carried = self._adaptive2_carried_stall_budget(
            old_chunk, old_cursor, self.chunk, new_cursor, old_budget
        )
        self._adaptive_stall.budget.update(carried)
        if any(value > 0 for value in carried.values()):
            logger.info(
                f"[Adaptive2 HAT] Stall budget carried across chunk: {carried}."
            )
        return True

    def _adaptive2_event_sides(self, sequence_id, cursor):
        hands = self._adaptive2_hand_focus.get(int(sequence_id), {})
        close, opening = set(), set()
        for side in SIDES:
            for event in hands.get(side, {}).get("events", ()):
                if int(event["start"]) <= cursor <= int(event["end"]):
                    (close if event["kind"] == "close" else opening).add(side)
        # An explicit release wins if noisy windows overlap.
        close.difference_update(opening)
        return tuple(sorted(close)), tuple(sorted(opening))

    def _raw_hand_ik_context(self):
        if (not getattr(self, "_adaptive2_hold_enabled", False)
                or not self.reference_playing or self.chunk is None
                or self.chunk_started_ns is None):
            return None
        now_ns = monotonic_ns()
        cursor = self._elapsed(now_ns, self.chunk_started_ns) * float(self.chunk["fps"])
        sequence_id = int(self.chunk["sequence_id"])
        close, opening = self._adaptive2_event_sides(sequence_id, cursor)
        return {
            "adaptive2": {
                "sequence_id": sequence_id,
                "close": close,
                "open": opening,
                "timestamp_ns": now_ns,
            }
        }

    def _reset_adaptive2_hold(self):
        if not hasattr(self, "_adaptive2_grasp"):
            return
        self._adaptive2_grasp = dict.fromkeys(SIDES)
        self._adaptive2_hold_suppressed = dict.fromkeys(SIDES, False)
        self._adaptive2_last_hand_motor = dict.fromkeys(SIDES)
        self._handoff_finger_control.reset()

    def _filter_raw_hand_targets(self, targets, solved):
        context = solved.get("_context", {}) or {}
        context = context.get("adaptive2")
        output = [np.asarray(target, dtype=np.float64).copy() for target in targets]
        if not getattr(self, "_adaptive2_hold_enabled", False) or context is None:
            for side, target in zip(SIDES, output):
                self._adaptive2_last_hand_motor[side] = np.asarray(
                    self.hand_dof12_to_motor(target), dtype=np.float64
                ).copy()
            return tuple(output)

        close = set(context["close"])
        opening = set(context["open"])
        now_ns = monotonic_ns()
        for index, side in enumerate(SIDES):
            proposed = np.asarray(
                self.hand_dof12_to_motor(output[index]), dtype=np.float64
            )
            previous = self._adaptive2_last_hand_motor[side]
            if previous is None:
                previous = proposed.copy()

            if side in opening:
                self._adaptive2_grasp[side] = None
                self._adaptive2_hold_suppressed[side] = False
                self._handoff_finger_control.reset(side)

            state = self._adaptive2_grasp[side]
            if state is not None and (
                now_ns - state["started_ns"]
            ) * 1e-9 >= self._adaptive2_hold_timeout_s:
                logger.warning(
                    f"[Adaptive2 HAT] {side} independent hold_progress timed out "
                    f"after {self._adaptive2_hold_timeout_s:.1f}s; releasing until OPEN."
                )
                self._adaptive2_grasp[side] = None
                self._adaptive2_hold_suppressed[side] = True
                self._handoff_finger_control.reset(side)
                state = None

            if (side in close and state is None
                    and not self._adaptive2_hold_suppressed[side]):
                self._adaptive2_grasp_serial += 1
                state = {
                    "event_id": self._adaptive2_grasp_serial,
                    "started_ns": now_ns,
                }
                self._adaptive2_grasp[side] = state
                logger.info(
                    f"[Adaptive2 HAT] {side} independent hold_progress entered "
                    f"on chunk {context['sequence_id']}."
                )

            if state is not None:
                proposed = self._handoff_finger_control.apply(
                    side,
                    state["event_id"],
                    int(context["sequence_id"]),
                    proposed,
                    previous,
                    now_ns=now_ns,
                )
                output[index] = self.hand_motor_to_dof12(proposed)
            self._adaptive2_last_hand_motor[side] = proposed.copy()
        return tuple(output)

    def _send_hand_target(self):
        if self.reference_play_gate and not self.reference_playing:
            self._reset_adaptive2_hold()
        super()._send_hand_target()
