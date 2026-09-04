"""Spawn-isolated HAT chunk hand IK for ScaleTrack FOCUS features."""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
import os
import sys

import numpy as np

from scalebridge.online.focus import derive_chunk_focus_phase
from scalebridge.online.protocol import INSPIRE_DDS_TO_HAT_QPOS


IK_TO_TWIST2 = np.array(
    [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5], dtype=np.int64
)


def _replace_queue_item(queue, item):
    """Best-effort latest-only put without ever blocking a control loop."""
    try:
        queue.put_nowait(item)
        return True
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            # multiprocessing.Queue's feeder can briefly report Full before
            # the item is visible to get_nowait.  Dropping this update is safer
            # than blocking the 50 Hz policy loop; the next chunk supersedes it.
            return False
        try:
            queue.put_nowait(item)
            return True
        except Full:
            return False


def _focus_worker_main(task_queue, result_queue, worker_config):
    """Child entry point; imports Pinocchio without sharing the CUDA process."""
    try:
        os.environ["SHENGYIN_SKIP_ISAACGYM_IMPORT"] = "1"
        module_root = worker_config["module_root"]
        mapping_path = worker_config["mapping_path"]
        for path in (module_root, mapping_path):
            if path not in sys.path:
                sys.path.insert(0, path)

        from human_policy.twist_hand_gmt_bridge import _PinocchioInspireIK
        from dofpos2cmd import dofpos12_to_q6
        from focus_definition import focus_from_hand_state

        solver_kwargs = worker_config["solver_kwargs"]
        solvers = {
            side: _PinocchioInspireIK(side, **solver_kwargs)
            for side in ("lh", "rh")
        }
    except Exception as exc:
        _replace_queue_item(result_queue, ("error", -1, repr(exc)))
        return

    def solve_closure(targets, side):
        q_previous = np.zeros(12, dtype=np.float32)
        # Converge onto frame zero before emitting any samples.  Otherwise the
        # arbitrary zero IK seed becomes an artificial finger-closing motion at
        # every action-chunk boundary and canonical FOCUS marks a false event.
        for _ in range(8):
            q_solved, _ = solvers[side].solve_frame(targets[0], q_previous)
            delta = float(np.max(np.abs(q_solved - q_previous)))
            q_previous = q_solved
            if delta < 1e-5:
                break

        def to_closure(q):
            motor_open = np.asarray(
                dofpos12_to_q6(q[IK_TO_TWIST2]), dtype=np.float32
            )
            # History is recorded in HAT's training order.  Keep the predicted
            # columns in that same order so a chunk boundary cannot compare one
            # physical finger against another.
            motor_open = motor_open[INSPIRE_DDS_TO_HAT_QPOS]
            return 1.0 - np.clip(motor_open, 0.0, 1.0)

        closure = [to_closure(q_previous)]
        for target in targets[1:]:
            q_previous, _ = solvers[side].solve_frame(target, q_previous)
            closure.append(to_closure(q_previous))
        return np.stack(closure, axis=0)

    while True:
        task = task_queue.get()
        if task is None:
            return
        # If inference produced several chunks while IK was busy, only compute
        # the newest one.  Older chunks can no longer drive the live reference.
        while True:
            try:
                newer = task_queue.get_nowait()
            except Empty:
                break
            if newer is None:
                return
            task = newer

        sequence, left, right, history, fps = task
        try:
            predicted = np.concatenate(
                (solve_closure(left, "lh"), solve_closure(right, "rh")), axis=1
            )
            phase = derive_chunk_focus_phase(
                history,
                predicted,
                fps,
                focus_from_hand_state,
                focus_pad_s=worker_config["focus_pad_s"],
            )
            _replace_queue_item(result_queue, ("result", sequence, phase))
        except Exception as exc:
            _replace_queue_item(result_queue, ("error", sequence, repr(exc)))


class LatestFocusProcess:
    """Latest-only FOCUS worker in a spawn process, safe beside CUDA/50 Hz."""

    def __init__(self, *, module_root, mapping_path, solver_kwargs, focus_pad_s):
        context = mp.get_context("spawn")
        self._task_queue = context.Queue(maxsize=2)
        self._result_queue = context.Queue(maxsize=2)
        worker_config = {
            "module_root": str(module_root),
            "mapping_path": str(mapping_path),
            "solver_kwargs": dict(solver_kwargs),
            "focus_pad_s": float(focus_pad_s),
        }
        self._process = context.Process(
            target=_focus_worker_main,
            args=(self._task_queue, self._result_queue, worker_config),
            name="hat-chunk-focus",
            daemon=True,
        )
        self._process.start()
        self._closed = False

    def submit(self, sequence_id, left_tips, right_tips, hand_history, fps):
        task = (
            int(sequence_id),
            np.asarray(left_tips, dtype=np.float32).copy(),
            np.asarray(right_tips, dtype=np.float32).copy(),
            np.asarray(hand_history, dtype=np.float32).copy(),
            float(fps),
        )
        _replace_queue_item(self._task_queue, task)

    def latest(self):
        latest = None
        while True:
            try:
                latest = self._result_queue.get_nowait()
            except Empty:
                break
        if latest is not None:
            kind, sequence, payload = latest
            if kind == "error":
                raise RuntimeError(
                    f"Online FOCUS worker failed for chunk {sequence}: {payload}"
                )
            return sequence, payload
        if not self._process.is_alive() and self._process.exitcode is not None:
            raise RuntimeError(
                f"Online FOCUS worker exited with code {self._process.exitcode}"
            )
        return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        _replace_queue_item(self._task_queue, None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        for queue in (self._task_queue, self._result_queue):
            queue.cancel_join_thread()
            queue.close()
