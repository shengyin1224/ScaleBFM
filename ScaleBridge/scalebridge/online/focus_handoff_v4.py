"""Opt-in v4: the complete v3 gate OR the legacy measured gate at 2.5 cm.

Each branch keeps its own prediction consensus. Neither branch changes wrist
execution; arrival still uses the existing fresh-source fence and lifecycle.
"""

import numpy as np

from scalebridge.online.focus_handoff import HandEventHandoff


class HandEventHandoffV4(HandEventHandoff):
    def __init__(self, *, v1_position_tolerance_m=0.025, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("handoff_version", None)
        if not np.isfinite(v1_position_tolerance_m) or v1_position_tolerance_m <= 0:
            raise ValueError("v4 legacy position tolerance must be positive")
        super().__init__(handoff_version="v3", **kwargs)
        self.handoff_version = "v4"
        legacy_kwargs = dict(kwargs, position_tolerance_m=float(v1_position_tolerance_m))
        self._entry_branches = {
            "v3": HandEventHandoff(handoff_version="v3", **kwargs),
            "v1": HandEventHandoff(handoff_version="legacy", **legacy_kwargs),
        }

    def reset(self):
        super().reset()
        self.entry_branch = None
        for branch in getattr(self, "_entry_branches", {}).values():
            branch.reset()

    def observe_event(self, chunk, hand):
        if (self.latest_prediction is not None
                and int(chunk["sequence_id"]) <= int(self.latest_prediction[0]["sequence_id"])):
            return
        self.latest_prediction = (chunk, hand)
        if self.bypass_continuity:
            return
        for branch in self._entry_branches.values():
            branch.last_kind = self.last_kind
            branch.state = self.state
            branch.completed_sequence = self.completed_sequence
            branch.observe_event(chunk, hand)
        self.v3_last_approach = self._entry_branches["v3"].v3_last_approach
        # Prefer the unchanged v3 route; the independent legacy route remains
        # available even when v3 clears its own predictions after rejection.
        selected = next((b for b in self._entry_branches.values()
                         if b.entry is not None), None)
        self.entry = None if selected is None else selected.entry
        if selected is not None:
            self.event = selected.event
            self.state = selected.state

    def arrive(self, measured, source_fence, outgoing, actual):
        if self.state != "approach":
            return False
        saved_entry, saved_event = self.entry, self.event
        saved_tolerance = self.position_tolerance_m
        try:
            for name, branch in self._entry_branches.items():
                if branch.entry is None:
                    continue
                self.entry, self.event = branch.entry, branch.event
                self.position_tolerance_m = branch.position_tolerance_m
                if super().arrive(measured, source_fence, outgoing, actual):
                    self.entry_branch = name
                    return True
            self.entry, self.event = saved_entry, saved_event
            return False
        finally:
            self.position_tolerance_m = saved_tolerance

    def finish(self, reason):
        super().finish(reason)
        for branch in self._entry_branches.values():
            branch.reset()

    def update_actual(self, now_ns, actual):
        # v4 changes entry only. Reuse v3/legacy's exact completion condition,
        # including the goal-reached alternative, without editing that code.
        self.handoff_version = "v3"
        try:
            return super().update_actual(now_ns, actual)
        finally:
            self.handoff_version = "v4"
