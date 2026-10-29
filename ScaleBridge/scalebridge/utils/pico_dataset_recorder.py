"""Proxy ScaleBFM control frames to SONIC's canonical LeRobot exporter.

ScaleBridge deliberately does not define a second dataset schema. A worker in
the GR00T-WholeBodyControl data-collection environment imports SONIC's feature
definition, RobotModel, camera client, hand telemetry decoder, and
Gr00tDataExporter directly. This small proxy keeps those heavier dependencies
out of the CUDA control process.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


class PicoDatasetRecorder:
    """Send ordered recording commands to the canonical SONIC writer process."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        fps: float = 50.0,
        task: str = "pico_scalebfm",
        *,
        body_joint_names: list[str],
        body_action_names: list[str],
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        self.fps = float(fps)
        self.task = str(task)
        self._recording = False
        self._episode_index = 0

        sonic_root = Path(
            os.environ.get(
                "PICO_SONIC_ROOT",
                "/home/nerv/qingyaoxu/GR00T-WholeBodyControl",
            )
        ).expanduser().absolute()
        worker = sonic_root / "integrations" / "pico_scalebfm" / "sonic_dataset_worker.py"
        data_python = Path(
            os.environ.get(
                "PICO_DATA_PYTHON",
                str(sonic_root / ".venv_data_collection" / "bin" / "python"),
            )
        ).expanduser().absolute()
        if not worker.is_file():
            raise FileNotFoundError(f"SONIC dataset worker not found: {worker}")
        if not data_python.is_file():
            raise FileNotFoundError(f"SONIC data-collection Python not found: {data_python}")

        ready_path = self.root / "meta" / ".scalebfm_sonic_ready"
        self._status_path = self.root / "meta" / ".scalebfm_sonic_status.json"
        error_path = self.root.parent / f".{self.root.name}.scalebfm_sonic_error"
        for marker in (ready_path, self._status_path, error_path):
            try:
                marker.unlink()
            except FileNotFoundError:
                pass

        command = [
            str(data_python), "-u", str(worker),
            "--root", str(self.root),
            "--fps", str(int(round(self.fps))),
            "--task", self.task,
            "--body-joint-names-json", json.dumps(list(body_joint_names), separators=(",", ":")),
            "--body-action-names-json", json.dumps(list(body_action_names), separators=(",", ":")),
            "--ready-path", str(ready_path),
            "--status-path", str(self._status_path),
            "--error-path", str(error_path),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(sonic_root), os.environ.get("PYTHONPATH", "")])
            ),
        }
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            # SONIC's canonical data path still contains repository-relative
            # assets such as gear_sonic/data/human/human_joints_info.pkl.
            cwd=str(sonic_root),
        )

        deadline = time.monotonic() + float(os.environ.get("PICO_DATASET_START_TIMEOUT", "60"))
        while time.monotonic() < deadline:
            if ready_path.is_file():
                try:
                    info = json.loads((self.root / "meta" / "info.json").read_text())
                    self._episode_index = int(info.get("total_episodes", 0))
                except (OSError, ValueError, json.JSONDecodeError):
                    self._episode_index = 0
                self._request_id = 0
                return
            if error_path.is_file():
                detail = error_path.read_text(encoding="utf-8", errors="replace").strip()
                self._terminate_worker()
                raise RuntimeError(f"SONIC dataset worker failed: {detail}")
            code = self._process.poll()
            if code is not None:
                raise RuntimeError(f"SONIC dataset worker exited during startup (code {code})")
            time.sleep(0.05)
        self._terminate_worker()
        raise TimeoutError(f"Timed out waiting for SONIC dataset worker: {ready_path}")

    @property
    def recording(self) -> bool:
        return self._recording

    def _send(self, command: dict[str, Any], *, wait: bool = False) -> dict[str, Any] | None:
        process = self._process
        if process.poll() is not None or process.stdin is None:
            raise RuntimeError(
                f"SONIC dataset worker is not running (code {process.returncode})"
            )
        request_id = None
        if wait:
            self._request_id += 1
            request_id = self._request_id
            command = {**command, "request_id": request_id}
        process.stdin.write(json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n")
        process.stdin.flush()
        if not wait:
            return None

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"SONIC dataset worker exited while handling a command (code {process.returncode})"
                )
            try:
                status = json.loads(self._status_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                time.sleep(0.02)
                continue
            if status.get("request_id") == request_id:
                if not status.get("ok", False):
                    raise RuntimeError(status.get("error", "SONIC dataset command failed"))
                return status
            time.sleep(0.02)
        raise TimeoutError(f"Timed out waiting for SONIC dataset command {command['command']}")

    def start(self) -> str:
        if not self._recording:
            self._send({"command": "start"}, wait=True)
            self._recording = True
        return self._episode_path()

    def _episode_path(self) -> str:
        return str(
            self.root / "data" / f"chunk-{self._episode_index // 1000:03d}"
            / f"episode_{self._episode_index:06d}.parquet"
        )

    def append(self, record: dict[str, Any]) -> None:
        if self._recording:
            self._send({"command": "append", "record": record})

    def save(self) -> str | None:
        if not self._recording:
            return None
        path = self._episode_path()
        status = self._send({"command": "save"}, wait=True)
        self._recording = False
        if status is not None and status.get("saved", False):
            self._episode_index = int(status["next_episode_index"])
            return path
        return None

    def discard(self) -> None:
        if not self._recording:
            return
        # SONIC keeps the same LeRobot episode index after a discarded take.
        status = self._send({"command": "discard"}, wait=True)
        self._recording = False
        if status is not None:
            self._episode_index = int(status["next_episode_index"])

    def _terminate_worker(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            self._send({"command": "close"})
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=60.0)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            self._terminate_worker()
        if process.returncode not in (0, None):
            raise RuntimeError(f"SONIC dataset worker exited with code {process.returncode}")
