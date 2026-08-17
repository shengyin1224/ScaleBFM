from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from scalebridge.utils.vive_tracker.receiver import ViveTrackerReceiver
from scalebridge.utils.vive_tracker.processor import ViveTrackerProcessor
from scalebridge.utils.vive_tracker.data_buffer import ViveTrackerDataBuffer

class ViveTrackerOnlineClient:

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        max_buffer_frames: int = 500, # 1s
        poll_interval: float = 0.002,
        robust_tracking: bool = False,
        stale_timeout: float = 0.25,
        recovery_valid_frames: int = 10,
        calibration_frames: int = 50,
        max_position_step: float = 0.08,
        max_position_speed: float = 3.0,
        position_smoothing_alpha: float = 0.2,
        calibration_max_position_std: float = 0.01,
    ):
        self.receiver = ViveTrackerReceiver(host=host, port=port)
        self.processor = ViveTrackerProcessor()
        self.buffer = ViveTrackerDataBuffer(max_frames=max_buffer_frames)
        self.poll_interval = float(poll_interval)
        self.robust_tracking = bool(robust_tracking)
        self.stale_timeout = max(0.0, float(stale_timeout))
        self.recovery_valid_frames = max(1, int(recovery_valid_frames))
        self.calibration_frames = max(1, int(calibration_frames))
        self.max_position_step = max(0.0, float(max_position_step))
        self.max_position_speed = max(0.0, float(max_position_speed))
        self.position_smoothing_alpha = min(1.0, max(0.0, float(position_smoothing_alpha)))
        self.calibration_max_position_std = max(0.0, float(calibration_max_position_std))
        self._running = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._last_fc = 0
        self._last_process_warning_mono = 0.0
        self._record_stream = None
        self._record_path: Optional[str] = None
        self._record_count = 0
        self._last_record_flush_mono = 0.0
        self._record_lock = threading.Lock()
        self._tracking_healthy = False
        self._tracking_state_known = False
        self._tracking_valid_streak = 0
        self._candidate_pose: Optional[np.ndarray] = None
        self._candidate_streak = 0
        self._recovery_smoothing_remaining = 0
        self._last_source_timestamp: Optional[float] = None
        self._last_valid_mono: Optional[float] = None
        self._tracking_events = []
        self._tracking_event_lock = threading.Lock()
        # Floor/pelvis roles are latched at runtime from the raw height gap:
        # SteamVR may hand out tracker_1/tracker_2 in either order between
        # sessions, so the row order on the wire cannot be trusted.
        self._floor_device: Optional[str] = None
        self._role_probe: Dict[str, list] = {}
        atexit.register(self.close_recording)

    def listen(self) -> None:
        self.receiver.listen()

    def accept_blocking(self) -> tuple[str, int]:
        return self.receiver.accept_blocking()
    
    def start(self) -> None:
        self.receiver.start_receiving()
        self._running.set()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="ViveTrackerOnlinePoll")
        self._poll_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None
        self.receiver.stop_receiving()
        self.receiver.close()
        self.close_recording()

    def is_tracking_healthy(self) -> bool:
        """Whether a complete, recently valid two-tracker solution is usable."""
        if not self.robust_tracking:
            return len(self.buffer) > 0
        if not self._tracking_healthy or self._last_valid_mono is None:
            return False
        if time.monotonic() - self._last_valid_mono > self.stale_timeout:
            self._tracking_valid_streak = 0
            self._set_tracking_state(False)
            return False
        return True

    def consume_tracking_events(self):
        with self._tracking_event_lock:
            events, self._tracking_events = self._tracking_events, []
        return events

    def _emit_tracking_event(self, name: str) -> None:
        with self._tracking_event_lock:
            self._tracking_events.append((str(name), time.time()))
        if self._record_stream is not None:
            self.record_event(str(name))

    def _set_tracking_state(self, healthy: bool) -> None:
        healthy = bool(healthy)
        if healthy == self._tracking_healthy and self._tracking_state_known:
            return
        self._tracking_healthy = healthy
        # Do announce an initial loss, but do not call a normally connected
        # tracker a "reconnection" when the process first starts.
        event = "tracker_reconnected" if healthy and self._tracking_state_known else (
            "tracker_lost" if not healthy else None
        )
        self._tracking_state_known = True
        if event is not None:
            self._emit_tracking_event(event)

    def _assign_tracker_roles(self, frame: Dict[str, Any], data: np.ndarray) -> Optional[np.ndarray]:
        """Reorder a two-tracker frame so row 0 is always the floor tracker.

        The role of each device is latched once per session from the raw
        height gap (raw Y is up): the persistently lower device is the floor
        tracker, the higher one the pelvis tracker. Returns the (possibly
        reordered) data, or None while the roles are still being probed.
        Single-tracker frames are returned unchanged.
        """
        names = list(frame.get("device_names", []))
        if data.shape[0] != 2 or len(names) != 2:
            return data
        if self._floor_device is None:
            for name, row in zip(names, data):
                samples = self._role_probe.setdefault(name, [])
                samples.append(float(row[1]))
                if len(samples) > 100:
                    del samples[0]
            if len(self._role_probe) == 2 and min(len(v) for v in self._role_probe.values()) >= 25:
                heights = {n: float(np.median(v)) for n, v in self._role_probe.items()}
                (low_name, low_h), (high_name, high_h) = sorted(heights.items(), key=lambda kv: kv[1])
                if high_h - low_h < 0.30:
                    t_mono = time.monotonic()
                    if t_mono - self._last_process_warning_mono >= 1.0:
                        logger.warning(
                            "[Localization] Cannot assign tracker roles yet: raw height gap "
                            f"{high_h - low_h:.2f}m < 0.30m ({heights}). Keep the pelvis "
                            "tracker well above the floor tracker."
                        )
                        self._last_process_warning_mono = t_mono
                    return None
                self._floor_device = low_name
                self._role_probe.clear()
                logger.info(
                    f"[Localization] Tracker roles latched from height: floor={low_name} "
                    f"({low_h:.2f}m), pelvis={high_name} ({high_h:.2f}m)."
                )
                self._emit_tracking_event(f"tracker_roles_latched floor={low_name} pelvis={high_name}")
            else:
                return None
        if names[0] != self._floor_device:
            data = data[::-1].copy()
            frame["data"] = data
            frame["device_names"] = [names[1], names[0]]
        return data

    def _validate_robust_frame(self, frame, data: np.ndarray) -> None:
        if data.shape != (2, 14):
            raise ValueError(f"expected both trackers with shape (2, 14), got {data.shape}")
        names = tuple(frame.get("device_names", []))
        if len(names) != 2 or self._floor_device is None or names[0] != self._floor_device:
            raise ValueError(f"unexpected tracker order/devices: {names}")
        if not np.all(np.isfinite(data)):
            raise ValueError("tracker frame contains NaN or Inf")
        quaternion_norms = np.linalg.norm(data[:, 3:7], axis=1)
        if np.any(np.abs(quaternion_norms - 1.0) > 0.1):
            raise ValueError("tracker quaternion is invalid")
        source_timestamp = float(frame.get("source_timestamp", 0.0))
        if source_timestamp <= 0.0:
            raise ValueError("tracker source timestamp is missing")
        if self._last_source_timestamp is not None and source_timestamp <= self._last_source_timestamp:
            raise ValueError("tracker source timestamp did not advance")

    def start_recording(self, output_path: Optional[str] = None) -> str:
        """Start a JSONL recording of every raw tracker frame after the first R2."""
        if self._record_stream is not None:
            assert self._record_path is not None
            return self._record_path

        if output_path is None:
            try:
                from hydra.core.hydra_config import HydraConfig

                output_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError):
                output_dir = os.getcwd()
            output_path = os.path.join(output_dir, "vive_tracker_raw.jsonl")

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._record_stream = open(output_path, "w", encoding="utf-8")
        self._record_path = output_path
        self._record_count = 0
        self._last_record_flush_mono = time.monotonic()
        self._record_stream.write(
            json.dumps(
                {
                    "record_type": "metadata",
                    "record_schema_version": 1,
                    "started_at": time.time(),
                    "row_fields": [
                        "pos_x", "pos_y", "pos_z",
                        "quat_w", "quat_x", "quat_y", "quat_z",
                        "vel_x", "vel_y", "vel_z",
                        "ang_vel_x", "ang_vel_y", "ang_vel_z",
                        "sender_timestamp",
                    ],
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._record_stream.flush()
        logger.info(f"[Localization] Recording raw tracker frames to {output_path}")
        return output_path

    def record_event(self, name: str) -> None:
        """Write a controller-stage marker into the tracker timeline."""
        with self._record_lock:
            if self._record_stream is None:
                return
            self._record_stream.write(
                json.dumps(
                    {
                        "record_type": "event",
                        "name": str(name),
                        "time": time.time(),
                        "monotonic": time.monotonic(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._record_stream.flush()

    def close_recording(self) -> None:
        with self._record_lock:
            stream = self._record_stream
            self._record_stream = None
            if stream is None:
                return
            try:
                stream.flush()
                stream.close()
            except OSError:
                pass

    def _record_frame(
        self,
        frame: Dict[str, Any],
        receive_wall_time: float,
        receive_monotonic: float,
        processed_data: Optional[np.ndarray],
        error: Optional[Exception],
    ) -> None:
        tracking_healthy = self.is_tracking_healthy() if self.robust_tracking else None
        valid_streak = self._tracking_valid_streak if self.robust_tracking else None
        data_age = (
            None if self._last_valid_mono is None
            else max(0.0, receive_monotonic - self._last_valid_mono)
        )
        with self._record_lock:
            stream = self._record_stream
            if stream is None:
                return
            data = frame["data"]
            record = {
                "record_type": "frame",
                "frame_count": frame["frame_count"],
                "receive_time": receive_wall_time,
                "receive_monotonic": receive_monotonic,
                "source_timestamp": frame.get("source_timestamp", 0.0),
                "wire_schema_version": frame.get("schema_version", 1),
                "requested_device_names": frame.get("requested_device_names", []),
                "device_names": frame.get("device_names", []),
                "shape": list(data.shape),
                "rows": data.tolist(),
                "processed_root_pose": None if processed_data is None else processed_data.tolist(),
                "error": None if error is None else str(error),
                "tracking_healthy": tracking_healthy,
                "valid_streak": valid_streak,
                "data_age": data_age,
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._record_count += 1
            if receive_monotonic - self._last_record_flush_mono >= 1.0:
                stream.flush()
                self._last_record_flush_mono = receive_monotonic

    def wait_first_frame(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.receiver.frame_count >= 1:
                return True
            time.sleep(0.02)
        return False
    
    def _poll_loop(self) -> None:
        while self._running.is_set():
            frame = self.receiver.latest_frame()
            if frame is None or frame["frame_count"] == self._last_fc:
                if self.robust_tracking:
                    self.is_tracking_healthy()
                time.sleep(self.poll_interval)
                continue
            self._last_fc = frame["frame_count"]
            data = frame["data"]
            t_mono = time.monotonic()
            t_wall = time.time()
            processed_data = None
            process_error = None
            role_data = self._assign_tracker_roles(frame, data)
            if role_data is None:
                # Roles not latched yet: skip quietly without flagging a
                # tracking loss, so startup does not announce a false outage.
                self._record_frame(frame, t_wall, t_mono, None, None)
                continue
            data = role_data
            try:
                if self.robust_tracking:
                    self._validate_robust_frame(frame, data)
                processed_data = self.processor.process(data)
                if self.robust_tracking and (
                    self.max_position_step > 0 or self.max_position_speed > 0
                ) and len(self.buffer):
                    previous = self.buffer.latest()[1]
                    dt = max(t_mono - self.buffer.latest()[0], 1e-4)
                    jump = np.linalg.norm(processed_data[:3] - previous[:3])
                    speed = jump / dt
                    jump_invalid = self._recovery_smoothing_remaining == 0 and (
                        (self.max_position_step > 0 and jump > self.max_position_step) or (
                        self.max_position_speed > 0 and speed > self.max_position_speed
                        )
                    )
                    if jump_invalid:
                        # Keep the old valid pose while checking that the new
                        # pose is stable for several consecutive frames. Do not
                        # modify or re-calibrate the coordinate system.
                        if (
                            self._candidate_pose is not None
                            and np.linalg.norm(processed_data[:3] - self._candidate_pose[:3])
                            <= (self.max_position_step if self.max_position_step > 0 else 0.03)
                        ):
                            self._candidate_streak += 1
                        else:
                            self._candidate_pose = processed_data.copy()
                            self._candidate_streak = 1
                        if self._candidate_streak < self.recovery_valid_frames:
                            raise ValueError(
                                f"root position jump {jump:.3f}m ({speed:.2f}m/s); awaiting "
                                f"{self.recovery_valid_frames} stable recovery frames"
                            )
                        # The new pose has now been stable for the requested
                        # number of samples. Keep its real coordinates and
                        # blend the Global input over a bounded number of
                        # frames; no origin or calibration parameter changes.
                        self._recovery_smoothing_remaining = max(
                            self.recovery_valid_frames * 2, 5
                        )
                        self._tracking_valid_streak = self.recovery_valid_frames - 1
                        self._candidate_pose = None
                        self._candidate_streak = 0
                    else:
                        self._candidate_pose = None
                        self._candidate_streak = 0
                    alpha = self.position_smoothing_alpha
                    if alpha < 1.0:
                        processed_data[:3] = alpha * processed_data[:3] + (1.0 - alpha) * previous[:3]
                    if self._recovery_smoothing_remaining > 0:
                        self._recovery_smoothing_remaining -= 1
                self.buffer.append(t_mono, processed_data)
                self._last_source_timestamp = float(frame.get("source_timestamp", 0.0))
                self._last_valid_mono = t_mono
                self._tracking_valid_streak += 1
                if not self.robust_tracking or self._tracking_valid_streak >= self.recovery_valid_frames:
                    self._set_tracking_state(True)
            except Exception as e:
                process_error = e
                self._tracking_valid_streak = 0
                if (
                    self.robust_tracking
                    and (
                        self._last_valid_mono is None
                        or t_mono - self._last_valid_mono > self.stale_timeout
                    )
                ):
                    self._set_tracking_state(False)
                # WARNING is visible in the main run.py terminal (whose console
                # level defaults to INFO). Throttle it because tracker frames
                # arrive much faster than a human can read the terminal.
                if t_mono - self._last_process_warning_mono >= 1.0:
                    logger.warning(
                        f"[Localization] Invalid tracker frame (shape={data.shape}): "
                        f"{e}. Global root pose is stale until valid data resumes."
                    )
                    self._last_process_warning_mono = t_mono
            self._record_frame(frame, t_wall, t_mono, processed_data, process_error)

    def calibrate(self, robot_root_quat):
        if self.robust_tracking:
            # Use only fresh samples collected after the R2 calibration edge.
            self.buffer.clear()
            deadline = time.monotonic() + max(2.0, self.calibration_frames * 0.02)
            while len(self.buffer) < self.calibration_frames and time.monotonic() < deadline:
                time.sleep(0.01)
            samples = [sample[1] for sample in self.buffer.as_list()][-self.calibration_frames:]
            if not samples:
                raise RuntimeError("No valid tracker samples available for calibration")
            pose = np.mean(np.asarray(samples), axis=0)
            position_std = np.std(np.asarray(samples)[:, :3], axis=0)
            if np.max(position_std) > self.calibration_max_position_std:
                raise RuntimeError(
                    "Tracker calibration is not stationary: position std "
                    f"{np.round(position_std, 5).tolist()}m"
                )
            quat = np.mean(np.asarray(samples)[:, 3:7], axis=0)
            quat /= max(np.linalg.norm(quat), 1e-12)
            pose[3:7] = quat
            self.processor.calibrate(robot_root_quat, pose)
            # Discard pre-calibration processed samples and wait for one sample
            # expressed with the newly locked R2 calibration.
            self.buffer.clear()
            deadline = time.monotonic() + 1.0
            while len(self.buffer) == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            if len(self.buffer) == 0:
                raise RuntimeError("No valid tracker frame arrived after calibration")
            self._tracking_valid_streak = self.recovery_valid_frames
            self._tracking_healthy = True
            self._tracking_state_known = True
            self._last_valid_mono = time.monotonic()
            return
        self.processor.calibrate(robot_root_quat, self.buffer.latest()[-1])
    
    def get_root_pos(self):
        latest = self.buffer.latest()
        if latest is None:
            return np.zeros((3,), dtype=np.float32)
        # Keep the state tensor shape valid for the simulator while exposing
        # health separately. Robust mode never treats this stale value as a
        # valid Global measurement; MotionTrackingEnv switches to local hold.
        return latest[1][:3]
    
if __name__ == "__main__":
    client = ViveTrackerOnlineClient()
    client.listen()
    print("Waiting for client...")
    addr = client.accept_blocking()
    print(f"Client connected: {addr}")
    client.start()
    time.sleep(1.0)
    input()
    client.calibrate(robot_root_quat=np.array([1, 0, 0, 0], dtype=np.float64))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.stop()
