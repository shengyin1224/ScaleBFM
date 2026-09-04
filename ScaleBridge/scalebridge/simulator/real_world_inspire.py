"""Opt-in ScaleBridge real-world backend with synchronized Inspire-hand IK."""

import atexit
import json
import os
import select
import subprocess
import threading
import time

import numpy as np
from loguru import logger

from scalebridge.simulator.real_world import RealWorld


class InspireRealWorld(RealWorld):
    """The stock 29-DOF real backend plus a separate Inspire DDS worker."""

    def _setup_asset(self):
        super()._setup_asset()
        self._hand_process = None
        self._hand_stderr_thread = None
        self._hand_state_lock = threading.Lock()
        self._hand_left_state = None
        self._hand_right_state = None
        self._hand_log_enabled = bool(self.cfg.get("deployment_hardening", False))
        self._hand_log_lock = threading.Lock()
        self._hand_log_file = None
        self._start_hand_worker()

    def _record_hand(self, record_type, left, right):
        """Append one hand command/state sample to hand_control.jsonl."""
        if not self._hand_log_enabled:
            return
        with self._hand_log_lock:
            if self._hand_log_file is None:
                try:
                    from hydra.core.hydra_config import HydraConfig
                    output_dir = HydraConfig.get().runtime.output_dir
                except (ImportError, ValueError):
                    output_dir = os.getcwd()
                path = os.path.abspath(os.path.join(output_dir, "hand_control.jsonl"))
                self._hand_log_file = open(path, "w", encoding="utf-8", buffering=1)
                logger.info("[Simulator] Recording Inspire hand data to {}".format(path))
            self._hand_log_file.write(json.dumps({
                "type": record_type,
                "time": time.time(),
                "left": np.asarray(left, dtype=np.float64).round(5).tolist(),
                "right": np.asarray(right, dtype=np.float64).round(5).tolist(),
            }, separators=(",", ":")) + "\n")

    def _start_hand_worker(self):
        worker = os.path.join(os.path.dirname(__file__), "inspire_hand_worker.py")
        command = [
            str(self.cfg.hand_bridge_python),
            "-u",
            worker,
            "--net",
            str(self.cfg.hand_network),
            "--sdk-path",
            str(self.cfg.hand_sdk_path),
            "--mapping-path",
            str(self.cfg.hand_mapping_path),
            "--max-step",
            str(float(self.cfg.hand_max_step)),
            "--state-timeout",
            str(float(self.cfg.hand_state_timeout)),
        ]
        self._hand_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + float(self.cfg.hand_state_timeout) + 2.0
        startup_lines = []
        while time.monotonic() < deadline:
            if self._hand_process.poll() is not None:
                break
            ready, _, _ = select.select(
                [self._hand_process.stdout], [], [], min(0.2, deadline - time.monotonic())
            )
            if not ready:
                continue
            line = self._hand_process.stdout.readline().strip()
            if line == "READY":
                break
            if line:
                startup_lines.append(line)
        else:
            line = ""
        if line != "READY":
            detail = " | ".join(startup_lines[-5:]) or "no diagnostic output"
            self.close_hand_worker()
            raise RuntimeError("Inspire DDS worker failed during startup: {}".format(detail))

        self._hand_stderr_thread = threading.Thread(
            target=self._forward_hand_worker_output, daemon=True
        )
        self._hand_stderr_thread.start()
        atexit.register(self.close_hand_worker)
        logger.info(
            "[Simulator] Inspire DDS ready on {}. Hand commands remain idle until "
            "real-world calibration is complete.".format(self.cfg.hand_network)
        )

    def _forward_hand_worker_output(self):
        process = self._hand_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("STATE "):
                try:
                    state = json.loads(line[6:])
                    left = np.asarray(state["left"], dtype=np.float32)
                    right = np.asarray(state["right"], dtype=np.float32)
                    if left.shape != (6,) or right.shape != (6,):
                        raise ValueError("hand state must contain six values per side")
                    with self._hand_state_lock:
                        self._hand_left_state = left
                        self._hand_right_state = right
                    self._record_hand("state", left, right)
                except Exception as exc:
                    logger.warning("[Inspire worker] Invalid state feedback: {}".format(exc))
            elif line:
                logger.warning("[Inspire worker] {}".format(line))

    def get_hand_motor_state(self):
        """Return measured left/right six-motor positions in standard semantics."""
        with self._hand_state_lock:
            if self._hand_left_state is None or self._hand_right_state is None:
                return np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)
            return self._hand_left_state.copy(), self._hand_right_state.copy()

    def set_hand_ik_target(self, left_dof_pos, right_dof_pos):
        """Send one left/right 12-DOF IK sample to the rate-limited DDS worker."""
        process = self._hand_process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("Inspire DDS worker is not running")
        left = np.asarray(left_dof_pos, dtype=np.float64)
        right = np.asarray(right_dof_pos, dtype=np.float64)
        if left.shape != (12,) or right.shape != (12,):
            raise ValueError(
                "Inspire IK targets must be (12,), got left={} right={}".format(
                    left.shape, right.shape
                )
            )
        payload = json.dumps(
            {"left": left.tolist(), "right": right.tolist()}, separators=(",", ":")
        )
        self._record_hand("cmd12", left, right)
        try:
            process.stdin.write(payload + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Lost the Inspire DDS worker") from exc

    def close_hand_worker(self):
        with self._hand_log_lock:
            stream, self._hand_log_file = self._hand_log_file, None
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        process = getattr(self, "_hand_process", None)
        if process is None:
            return
        self._hand_process = None
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
        except OSError:
            pass
