"""Opt-in adaptive execution; per-hand FOCUS handoff is optional."""

import numpy as np
from loguru import logger

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.adaptive_chunk_execution import (
    SIDES, StallLookahead, advance_on_path, closing_boundary, select_horizon,
)
from scalebridge.online.focus_handoff import PerHandFocusHandoff


class HAT4AdaptiveMotionTrackingEnv(HAT4OnlineMotionTrackingEnv):
    def _setup_metadata(self):
        # Reject incompatible experiments before the parent starts hand workers.
        if self.cfg.get("focus_adaptive_enabled", False) or self.cfg.get("stall_wrist_assist", False):
            raise ValueError("Adaptive execution cannot share another horizon/stall controller")
        if self.cfg.get("finger_chunk_source", "playback") != "playback":
            raise ValueError("Adaptive close protection requires finger_chunk_source=playback")
        if (self.cfg.get("focus_handoff_enabled", False)
                and self.cfg.get("focus_handoff_source", "keypoints") != "keypoints"):
            raise ValueError("Adaptive execution currently requires keypoint handoff")
        for prefix in ("far", "near"):
            low = int(self.cfg.get(f"adaptive_{prefix}_min_frames", 25 if prefix == "far" else 8))
            high = int(self.cfg.get(f"adaptive_{prefix}_max_frames", 50 if prefix == "far" else 12))
            if not 1 <= low <= high:
                raise ValueError(f"Invalid adaptive {prefix} frame window")
        for name, default in (("near_distance_m", .05), ("min_wrist_travel_m", .05),
                              ("stall_motion_m", .005), ("stall_increment_m", .005),
                              ("stall_max_lookahead_m", .05), ("stall_frames", 10)):
            value = float(self.cfg.get(f"adaptive_{name}", default))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"adaptive_{name} must be finite and positive")
        self._adaptive_plan = None
        self._adaptive_key = None
        self._adaptive_stall = StallLookahead(
            frames=int(self.cfg.get("adaptive_stall_frames", 10)),
            motion_m=float(self.cfg.get("adaptive_stall_motion_m", .005)),
            increment_m=float(self.cfg.get("adaptive_stall_increment_m", .005)),
            maximum_m=float(self.cfg.get("adaptive_stall_max_lookahead_m", .05)),
        )
        super()._setup_metadata()
        if (self.cfg.get("focus_handoff_enabled", False)
                and not isinstance(self.focus_handoff, PerHandFocusHandoff)):
            raise ValueError("Adaptive execution requires independent hand handoff")

    def _adaptive_controllers(self):
        handoff = getattr(self, "focus_handoff", None)
        return handoff.controllers if handoff is not None else {}

    def _adaptive_running(self):
        return (self.reference_playing and not self.localization_paused
                and not self.awaiting_fresh_chunk and not self.requires_restart
                and self.held_sample is None and self.chunk is not None
                and self.chunk_started_ns is not None)

    def _adaptive_hand(self, side):
        # Never use an IK event index from a different prediction's clock.
        prediction = self._focus_prediction_chunks.get(self.chunk["sequence_id"], {})
        return prediction.get("hand_focus", {}).get(side)

    def _adaptive_near(self, now_ns):
        poses = getattr(self.current_pose_fk, "last_body_poses", {}) or {}
        for side, c in self._adaptive_controllers().items():
            if c.bypass_continuity:
                return True
            if c.entry is not None and f"{side}_wrist_yaw_link" in poses:
                distance = np.linalg.norm(np.asarray(poses[f"{side}_wrist_yaw_link"][0])
                                          - c.entry[f"{side}_wrist_pos"])
                if distance <= float(self.cfg.get("adaptive_near_distance_m", .05)):
                    return True
        # v3 intentionally rejects far-away grasp events. Inspect raw predicted
        # close geometry too, so that rejection does not hide the near phase.
        start = self._elapsed(now_ns, self.chunk_started_ns) * float(self.chunk["fps"])
        for side in SIDES:
            event = closing_boundary(self.chunk, side, start, self._adaptive_hand(side))
            wrist = np.asarray(self.chunk[f"{side}_wrist_pos"])
            begin = min(int(start), len(wrist) - 1)
            measured = poses.get(f"{side}_wrist_yaw_link")
            current = np.asarray(measured[0]) if measured is not None else wrist[begin]
            distance = np.linalg.norm(wrist[event] - current)
            # A chunk with no close is not 'near' merely because it is static.
            if event < len(wrist) - 1 and distance <= float(self.cfg.get("adaptive_near_distance_m", .05)):
                return True
        return False

    def _adaptive_schedule_boundary(self, boundary, start):
        """CLOSE scheduling policy; adaptive2 overrides only this hook."""
        return boundary if boundary > start else None

    def _adaptive_horizon(self, now_ns):
        # _play_now_ns integrates a monotonic clock; asking it for an older
        # timestamp returns NOW, not the historical phase. Use the parent's
        # saved acceptance stamp so the horizon never slides forward each tick.
        accepted_play_ns = getattr(self, "_last_chunk_received_play_ns", None)
        if accepted_play_ns is None:
            accepted_play_ns = self.last_chunk_received_ns
        start = max(0., (accepted_play_ns - self.chunk_started_ns) * 1e-9) * float(self.chunk["fps"])
        near = self._adaptive_near(now_ns)
        key = (self.chunk["sequence_id"], self.chunk_started_ns, near,
               tuple(id(self._adaptive_hand(s)) for s in SIDES))
        if self._adaptive_key != key:
            boundary = min(closing_boundary(self.chunk, s, start, self._adaptive_hand(s)) for s in SIDES)
            self._adaptive_plan = select_horizon(
                self.chunk, start, near=near,
                minimum=int(self.cfg.get("adaptive_far_min_frames", 25)),
                maximum=int(self.cfg.get("adaptive_far_max_frames", 50)),
                near_minimum=int(self.cfg.get("adaptive_near_min_frames", 8)),
                near_maximum=int(self.cfg.get("adaptive_near_max_frames", 12)),
                min_travel_m=float(self.cfg.get("adaptive_min_wrist_travel_m", .05)),
                # When the event is already here, near cadence + handoff owns
                # it. A future event always limits coarse execution.
                boundary=self._adaptive_schedule_boundary(boundary, start),
            )
            self._adaptive_key = key
            logger.info(f"[Adaptive HAT] chunk={key[0]} near={near} "
                        f"start={start:.1f} switch={self._adaptive_plan.frame:.1f} "
                        f"travel={self._adaptive_plan.travel_m:.3f}m reason={self._adaptive_plan.reason}")
        return self._adaptive_plan

    def _chunk_execution_fraction(self, chunk=None):
        # Scheduling lives in _min_execution_hold_active; no fixed floor.
        return 0.

    def _min_execution_hold_active(self, now_ns):
        if not self._adaptive_running() or self.last_chunk_received_ns is None:
            return False
        handoff = getattr(self, "focus_handoff", None)
        if handoff is not None and handoff.bypass_continuity:
            return False  # fresh-arrival fence retains its original semantics
        cursor = self._elapsed(now_ns, self.chunk_started_ns) * float(self.chunk["fps"])
        margin = float(self.cfg.get("chunk_end_margin_s", .5)) * float(self.chunk["fps"])
        if cursor >= int(self.chunk["frame_count"]) - margin:
            return False
        return cursor < self._adaptive_horizon(now_ns).frame

    def _accept_latest_chunk(self, now_ns):
        # The parent can still accept replans for the body/other hand. Preserve
        # each active CLOSE overlay and both of its clocks across that operation.
        pinned = {}
        if self._adaptive_running():
            for side, c in self._adaptive_controllers().items():
                if c.active and c.event is not None and c.event["kind"] == "close" and c.playback is not None:
                    pinned[side] = (c.playback, c.playback_started_ns, c.playback_started_real_ns)
        try:
            accepted = super()._accept_latest_chunk(now_ns)
        finally:
            for side, playback in pinned.items():
                c = self._adaptive_controllers().get(side)
                if c is not None and c.active and c.event is not None and c.event["kind"] == "close":
                    c.playback, c.playback_started_ns, c.playback_started_real_ns = playback
        if accepted:
            self._adaptive_key = None
            # Stationary measured history spans replans. Only path-relative
            # rescue distance belongs to the replaced chunk.
            self._adaptive_stall.budget = dict.fromkeys(SIDES, 0.)
        return accepted

    def _adaptive_cap_close(self):
        """A pinned close may not run on into its predicted release suffix."""
        for side, c in self._adaptive_controllers().items():
            if not (c.active and c.event is not None and c.event["kind"] == "close"
                    and c.playback is not None):
                continue
            playback = c.playback
            if playback.get("_adaptive_close_side") == side:
                continue
            prediction = self._focus_prediction_chunks.get(playback["sequence_id"], {})
            hand = prediction.get("hand_focus", {}).get(side)
            if hand is None:
                continue
            events = [e for e in hand.get("events", ()) if e["kind"] == "close"]
            if not events:
                continue
            end = min(int(events[0]["end"]), int(playback["frame_count"]) - 1)
            capped = dict(playback, _adaptive_close_side=side)
            for suffix in ("wrist_pos", "wrist_quat_wxyz", "fingertip_local"):
                key = f"{side}_{suffix}"
                values = np.asarray(playback[key]).copy()
                values[end + 1:] = values[end]
                capped[key] = values
            c.playback = capped

    def _publish_robot_state(self):
        super()._publish_robot_state()
        if not hasattr(self, "_adaptive_stall"):
            return
        message = self.last_published_robot_state
        if not self._adaptive_running() or message is None:
            self._adaptive_stall.reset()
            self._adaptive_key = None
            return
        measured = {s: np.asarray(message[f"{s}_wrist_pos_world"]) for s in SIDES}
        handoff = getattr(self, "focus_handoff", None)
        self._adaptive_stall.update(message["timestamp_ns"], measured, self.chunk["fps"],
                                    blocked=handoff.bypass_sides if handoff is not None else ())

    def _sample_reference(self, now_ns, *, apply_stall_assist=True):
        if hasattr(self, "_adaptive_stall") and self._adaptive_running():
            self._adaptive_cap_close()
        sample = super()._sample_reference(now_ns, apply_stall_assist=apply_stall_assist)
        if not apply_stall_assist or not hasattr(self, "_adaptive_stall") or not self._adaptive_running():
            return sample
        # Do not overwrite an ongoing seam blend with raw path samples.
        if self.blend_started_ns is not None and (now_ns - self.blend_started_ns) * 1e-9 < float(self.cfg.get("chunk_blend_s", .22)):
            return sample
        cursor = self._elapsed(now_ns, self.chunk_started_ns) * float(self.chunk["fps"])
        offsets = self._playback_future_offsets_s() * float(self.chunk["fps"])
        handoff = getattr(self, "focus_handoff", None)
        for side in SIDES:
            budget = self._adaptive_stall.budget[side]
            if budget <= 0 or (handoff is not None and side in handoff.bypass_sides):
                continue
            hand = self._adaptive_hand(side)
            # With handoff, retain its same-chunk IK gate. Without handoff,
            # use the raw chunk's conservative fingertip contraction boundary;
            # do not silently disable adaptive rescue for lack of FOCUS state.
            if hand is None and handoff is not None:
                continue
            boundary = closing_boundary(self.chunk, side, cursor, hand)
            c = self._adaptive_controllers().get(side)
            if c is not None and c.entry is not None:
                path = np.asarray(self.chunk[f"{side}_wrist_pos"])
                begin = min(int(np.ceil(cursor)), len(path) - 1)
                goal = c.entry[f"{side}_wrist_pos"]
                boundary = min(boundary, begin + int(np.argmin(np.linalg.norm(path[begin:] - goal, axis=1))))
            key = f"{side}_wrist_pos"
            positions = sample[key].copy()
            for i, offset in enumerate(offsets):
                frame = cursor + float(offset)
                if frame < boundary:
                    positions[i] = advance_on_path(self.chunk[key], frame, budget, boundary)
            sample[key] = positions
        return sample
