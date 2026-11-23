"""Asynchronous vision supervisor for online HAT execution.

The worker is deliberately separated from the 50 Hz control loop.  The
control side submits latest-only numeric snapshots and polls a one-slot result
mailbox; camera I/O, image resizing and model requests happen on background
threads only.
"""

from __future__ import annotations

import base64
from collections import deque
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.request

import numpy as np
from loguru import logger


PHASES = (
    "PRE_GRASP", "FOCUS_GRASP", "HOLDING", "FOCUS_RELEASE", "POST_RELEASE",
)
STATES = (
    "normal", "ready_focus", "grasp_secured", "grasp_failed",
    "ready_release", "release_done", "release_failed",
)
ALLOWED_STATES = {
    "PRE_GRASP": {"normal", "ready_focus", "grasp_failed"},
    "FOCUS_GRASP": {"normal", "grasp_secured", "grasp_failed"},
    "HOLDING": {"normal", "grasp_failed", "ready_release"},
    "FOCUS_RELEASE": {"normal", "release_done", "release_failed"},
    "POST_RELEASE": {"normal"},
}
TRANSITIONS = {
    ("PRE_GRASP", "ready_focus"): ("begin_grasp", "FOCUS_GRASP"),
    ("FOCUS_GRASP", "grasp_secured"): ("finish_grasp", "HOLDING"),
    ("FOCUS_GRASP", "grasp_failed"): ("abort_grasp", "PRE_GRASP"),
    ("HOLDING", "grasp_failed"): ("lost_grasp", "PRE_GRASP"),
    ("HOLDING", "ready_release"): ("begin_release", "FOCUS_RELEASE"),
    ("FOCUS_RELEASE", "release_done"): ("finish_release", "POST_RELEASE"),
    ("FOCUS_RELEASE", "release_failed"): ("abort_release", "HOLDING"),
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": list(STATES)},
        "speed_level": {"type": "integer", "minimum": 0, "maximum": 2},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string", "maxLength": 160},
    },
    "required": ["state", "speed_level", "confidence", "evidence"],
    "additionalProperties": False,
}


def validate_model_decision(value):
    """Validate the model-owned part of a decision without trusting schema mode."""
    if not isinstance(value, dict):
        raise ValueError("decision must be a JSON object")
    if set(value) != {"state", "speed_level", "confidence", "evidence"}:
        raise ValueError("decision has missing or extra fields")
    state = value["state"]
    speed_level = value["speed_level"]
    confidence = value["confidence"]
    evidence = value["evidence"]
    if state not in STATES:
        raise ValueError(f"invalid state {state!r}")
    if isinstance(speed_level, bool) or not isinstance(speed_level, int):
        raise ValueError("speed_level must be an integer")
    if speed_level not in (0, 1, 2):
        raise ValueError(f"invalid speed_level {speed_level!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"invalid confidence {confidence!r}")
    if not isinstance(evidence, str):
        raise ValueError("evidence must be a string")
    if len(evidence.split()) > 40:
        raise ValueError("evidence exceeds 40 words")
    return {
        "state": state,
        "speed_level": speed_level,
        "confidence": confidence,
        "evidence": evidence,
    }


class SupervisorPhase:
    """Code-owned phase and decision validation; it never touches the robot."""

    def __init__(self, *, speed_levels=(1.0, 0.5, 0.25),
                 max_decision_age_s=5.0, trigger_max_drift_m=0.02,
                 min_focus_time_s=0.5, focus_timeout_s=10.0,
                 min_confidence_state=0.6, focus_side="right"):
        self.speed_levels = tuple(float(item) for item in speed_levels)
        if len(self.speed_levels) != 3 or any(
                not np.isfinite(item) or item <= 0.0 for item in self.speed_levels):
            raise ValueError("gpt_speed_levels must contain three positive values")
        if focus_side not in ("left", "right"):
            raise ValueError("gpt_focus_side must be left or right")
        self.max_decision_age_s = float(max_decision_age_s)
        self.trigger_max_drift_m = float(trigger_max_drift_m)
        self.min_focus_time_s = float(min_focus_time_s)
        self.focus_timeout_s = float(focus_timeout_s)
        self.min_confidence_state = float(min_confidence_state)
        self.focus_side = focus_side
        if not (
            self.max_decision_age_s > 0.0
            and self.trigger_max_drift_m >= 0.0
            and 0.0 <= self.min_focus_time_s < self.focus_timeout_s
            and 0.0 <= self.min_confidence_state <= 1.0
        ):
            raise ValueError("invalid GPT supervisor validation thresholds")
        self.reset(0)

    def reset(self, now_ns):
        self.phase = "PRE_GRASP"
        self.phase_entered_ns = int(now_ns)
        self.speed_level = 0
        self.recent_decisions = deque(maxlen=3)

    def force_phase(self, phase, now_ns):
        if phase not in PHASES:
            raise ValueError(f"invalid phase {phase!r}")
        if phase != self.phase:
            self.phase = phase
            self.phase_entered_ns = int(now_ns)

    def time_in_phase_s(self, now_ns):
        return max(0.0, (int(now_ns) - self.phase_entered_ns) * 1e-9)

    def evaluate(self, decision, now_ns, current_wrist):
        """Return independently validated state and speed proposals."""
        rejected = {}
        now_ns = int(now_ns)
        age_s = (now_ns - int(decision["t_obs_ns"])) * 1e-9
        result = {
            "obs_id": int(decision["obs_id"]),
            "phase": self.phase,
            "age_s": age_s,
            "state": None,
            "action": None,
            "next_phase": self.phase,
            "speed_level": None,
            "rejected": rejected,
        }
        if age_s < 0.0 or age_s > self.max_decision_age_s:
            rejected["decision"] = (
                f"age {age_s:.3f}s outside [0,{self.max_decision_age_s:.3f}]"
            )
            return result

        model = decision["model_decision"]
        result["speed_level"] = int(model["speed_level"])
        state = model["state"]
        phase_at_obs = decision["phase_at_obs"]
        if phase_at_obs != self.phase:
            rejected["state"] = (
                f"phase changed from {phase_at_obs} to {self.phase}"
            )
            return result
        if state not in ALLOWED_STATES[self.phase]:
            rejected["state"] = f"{state} is illegal in {self.phase}"
            return result
        if state != "normal" and model["confidence"] < self.min_confidence_state:
            rejected["state"] = (
                f"confidence {model['confidence']:.3f} below "
                f"{self.min_confidence_state:.3f}"
            )
            return result
        if self.phase.startswith("FOCUS_") and state != "normal" and (
            self.time_in_phase_s(now_ns) < self.min_focus_time_s
        ):
            rejected["state"] = (
                f"focus age {self.time_in_phase_s(now_ns):.3f}s below "
                f"{self.min_focus_time_s:.3f}s"
            )
            return result
        if state in ("ready_focus", "ready_release"):
            observed = np.asarray(decision["wrist_measured"], dtype=np.float64)
            current = np.asarray(current_wrist, dtype=np.float64)
            drift = float(np.linalg.norm(current - observed))
            result["trigger_drift_m"] = drift
            if not np.isfinite(drift) or drift >= self.trigger_max_drift_m:
                rejected["state"] = (
                    f"wrist drift {drift:.4f}m is not below "
                    f"{self.trigger_max_drift_m:.4f}m"
                )
                result["retry_immediately"] = True
                return result
        result["state"] = state
        transition = TRANSITIONS.get((self.phase, state))
        if transition is not None:
            result["action"], result["next_phase"] = transition
        return result

    def commit(self, evaluation, now_ns):
        if evaluation["speed_level"] is not None:
            self.speed_level = int(evaluation["speed_level"])
        state = evaluation["state"]
        if state is not None:
            self.recent_decisions.append({"phase": self.phase, "state": state})
        next_phase = evaluation["next_phase"]
        if next_phase != self.phase:
            self.phase = next_phase
            self.phase_entered_ns = int(now_ns)

    def timeout(self, now_ns):
        if (not self.phase.startswith("FOCUS_")
                or self.time_in_phase_s(now_ns) < self.focus_timeout_s):
            return None
        if self.phase == "FOCUS_GRASP":
            return {"action": "abort_grasp", "next_phase": "PRE_GRASP",
                    "state": "grasp_failed", "reason": "focus_timeout"}
        return {"action": "abort_release", "next_phase": "HOLDING",
                "state": "release_failed", "reason": "focus_timeout"}

    def commit_timeout(self, timeout, now_ns):
        self.recent_decisions.append(
            {"phase": self.phase, "state": timeout["state"]}
        )
        self.phase = timeout["next_phase"]
        self.phase_entered_ns = int(now_ns)


class ZedJpegSource:
    """Independent latest-frame subscriber for the robot's ZED JPEG stream."""

    def __init__(self, endpoint, topic=b"realsense"):
        import zmq

        self._zmq = zmq
        self._endpoint = str(endpoint)
        self._topic = topic if isinstance(topic, bytes) else str(topic).encode()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._jpeg = None
        self._received_ns = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="gpt-supervisor-camera", daemon=True,
        )
        self._thread.start()

    def _run(self):
        socket = self._zmq.Context.instance().socket(self._zmq.SUB)
        socket.setsockopt(self._zmq.SUBSCRIBE, b"")
        socket.setsockopt(self._zmq.RCVHWM, 2)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.connect(self._endpoint)
        try:
            while not self._stop.is_set():
                if socket.poll(100) == 0:
                    continue
                parts = socket.recv_multipart()
                if len(parts) < 3 or parts[0] != self._topic:
                    continue
                with self._lock:
                    self._jpeg = bytes(parts[2])
                    self._received_ns = time.monotonic_ns()
        except Exception as exc:
            with self._lock:
                self._error = exc
        finally:
            socket.close(linger=0)

    def latest(self, max_age_s):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"GPT camera receiver failed: {self._error}")
            jpeg, received_ns = self._jpeg, self._received_ns
        if jpeg is None:
            raise RuntimeError("GPT camera has not received a frame")
        age_s = (time.monotonic_ns() - received_ns) * 1e-9
        if age_s > float(max_age_s):
            raise RuntimeError(f"GPT camera frame is stale ({age_s:.3f}s)")
        return jpeg, received_ns

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


def build_prompt(snapshot, *, latency_s, horizon_s):
    allowed = sorted(ALLOWED_STATES[snapshot["phase"]])
    return f"""You supervise a humanoid robot doing a manipulation task. A high-level HAT
model predicts keypoint trajectories and a low-level tracker follows them with a
few centimetres of error. Continuity keeps the hand moving along HAT's trajectory.
You do not control motion. Report execution state and choose a speed level only.
Your answer is applied about {latency_s:.1f}s after the image was taken.

Task: {snapshot['task_instruction']!r}
Current phase (tracked by code): {snapshot['phase']}
Allowed state values in this phase: {allowed}

State meanings:
- normal: do not change execution state.
- ready_focus: the hand is positioned so closing now would capture the object;
  object is between fingers/palm, fingers are not blocked, and insertion is deep
  enough. Mere proximity is not enough.
- grasp_secured: fingers have closed and the object is visibly held.
- grasp_failed: the attempt clearly failed or the object was dropped.
- ready_release: the held object is above the intended place with enough clearance.
- release_done: the object was released and rests correctly.
- release_failed: opening did not place the object correctly.

Commanded closure or target arrival is not proof; use the image and compare target
with measured positions. If unsure choose normal. Choose speed for about
{horizon_s:.1f}s: 0 full speed in free space, 1 half speed while approaching,
2 quarter speed when contact/insertion is imminent or retrying nearby.
Keep evidence to one short English sentence of at most 25 words.
Do not use tools. Return only a JSON object with exactly these fields:
state, speed_level, confidence, evidence.

Numeric snapshot:
{json.dumps(snapshot, separators=(',', ':'), allow_nan=False)}
"""


class CodexDecisionClient:
    """One-shot, stateless Codex CLI adapter with strict JSON output."""

    _COMPLETION_EXIT_GRACE_S = 0.25
    _TERMINATE_GRACE_S = 0.5

    def __init__(self, *, commands=("codex2", "codex"), model="gpt-6.1-sol",
                 reasoning_effort="low", timeout_s=60.0, retries=1,
                 image_max_side=512, fast_mode=True):
        self.commands = tuple(str(item) for item in commands if str(item))
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.image_max_side = int(image_max_side)
        self.fast_mode = bool(fast_mode)
        self._process_lock = threading.Lock()
        self._process = None
        self._cancelled = threading.Event()
        if not self.commands:
            raise ValueError("at least one GPT supervisor command is required")
        if self.timeout_s <= 0.0 or self.retries < 0 or self.image_max_side <= 0:
            raise ValueError("invalid GPT client timeout/retry/image settings")

    def _command(self, executable, image_path, schema_path, output_path, workdir):
        command = [
            executable, "exec", "--ephemeral", "--ignore-rules",
            "--skip-git-repo-check", "--sandbox", "read-only",
            "--cd", str(workdir), "--model", self.model,
            "--config", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
        ]
        if self.fast_mode:
            command.extend(("--enable", "fast_mode"))
        command.extend((
            "--output-schema", str(schema_path), "--image", str(image_path),
            "--json", "--output-last-message", str(output_path), "-",
        ))
        return command

    @staticmethod
    def _event_payload(stdout):
        """Return the last agent JSON text and usage from Codex JSONL output."""
        message = None
        usage = None
        completed = False
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    message = item.get("text")
            elif event.get("type") == "turn.completed":
                completed = True
                usage = event.get("usage")
        return message, usage, completed

    def _run_process(self, command, prompt):
        """Run Codex until it exits, times out, or reports ``turn.completed``.

        Some Codex CLI builds keep the process alive after emitting a complete
        result.  Reading stdout incrementally lets the caller accept that result
        immediately instead of waiting for the process timeout and retrying it.
        """
        process = subprocess.Popen(
            command,
            text=True,
            bufsize=1,
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with self._process_lock:
            self._process = process

        stdout_lines = []
        stderr_lines = []
        turn_completed = threading.Event()

        def read_stdout():
            try:
                for line in process.stdout:
                    stdout_lines.append(line)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "turn.completed":
                        turn_completed.set()
            except (OSError, ValueError):
                # A forced process-group shutdown may close this pipe while a
                # reader is blocked. All complete lines are already retained.
                pass

        def read_stderr():
            try:
                stderr_lines.extend(process.stderr)
            except (OSError, ValueError):
                pass

        stdout_thread = threading.Thread(
            target=read_stdout, name="gpt-codex-stdout", daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_stderr, name="gpt-codex-stderr", daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        forced_exit_after_completion = False
        deadline = time.monotonic() + self.timeout_s
        try:
            process.stdin.write(prompt)
            process.stdin.close()
            # Avoid subprocess attempting to flush an already closed stream.
            process.stdin = None
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    timed_out = True
                    self._signal_process_group(process, signal.SIGKILL)
                    break
                if not turn_completed.wait(timeout=min(0.05, remaining)):
                    continue
                try:
                    process.wait(timeout=self._COMPLETION_EXIT_GRACE_S)
                except subprocess.TimeoutExpired:
                    forced_exit_after_completion = True
                    self._signal_process_group(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=self._TERMINATE_GRACE_S)
                    except subprocess.TimeoutExpired:
                        self._signal_process_group(process, signal.SIGKILL)
                break
            process.wait()
        finally:
            if process.poll() is None:
                self._signal_process_group(process, signal.SIGKILL)
                process.wait()
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            process.stdout.close()
            process.stderr.close()
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            with self._process_lock:
                if self._process is process:
                    self._process = None

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        _, usage, completed = self._event_payload(stdout)
        return {
            "process": process,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out and not completed,
            "turn_completed": completed,
            "forced_exit_after_completion": forced_exit_after_completion,
            "usage": usage,
        }

    @staticmethod
    def _signal_process_group(process, sig):
        """Signal the isolated CLI process group, including helper children."""
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        except (OSError, PermissionError):
            # Defensive fallback; start_new_session normally makes killpg safe.
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

    def call(self, snapshot, jpeg, record_dir, *, latency_s, horizon_s):
        from PIL import Image

        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        image_path = record_dir / "head.jpg"
        schema_path = record_dir / "schema.json"
        output_path = record_dir / "model_output.json"
        request_path = record_dir / "request.json"
        prompt = build_prompt(snapshot, latency_s=latency_s, horizon_s=horizon_s)
        (record_dir / "head_source.jpg").write_bytes(jpeg)
        with Image.open(BytesIO(jpeg)) as image:
            image = image.convert("RGB")
            image.thumbnail((self.image_max_side, self.image_max_side))
            image.save(image_path, format="JPEG", quality=92)
        schema_path.write_text(json.dumps(DECISION_SCHEMA, indent=2) + "\n")
        request_path.write_text(json.dumps({
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt": prompt,
            "image": image_path.name,
        }, ensure_ascii=False, indent=2) + "\n")

        failures = []
        started = time.monotonic_ns()
        attempts = self.retries + 1
        for attempt in range(attempts):
            if self._cancelled.is_set():
                raise RuntimeError("Codex decision request cancelled")
            executable = self.commands[min(attempt, len(self.commands) - 1)]
            resolved = shutil.which(executable)
            if resolved is None:
                failures.append(f"{executable}: command not found")
                continue
            output_path.unlink(missing_ok=True)
            try:
                run = self._run_process(
                    self._command(
                        resolved, image_path, schema_path, output_path, record_dir,
                    ),
                    prompt,
                )
            except OSError as exc:
                failures.append(f"{executable}: failed to start: {exc}")
                continue
            process = run["process"]
            stdout = run["stdout"]
            stderr = run["stderr"]
            (record_dir / f"codex_attempt_{attempt + 1}.json").write_text(
                json.dumps({
                    "command": executable,
                    "returncode": process.returncode,
                    "timeout": run["timed_out"],
                    "turn_completed": run["turn_completed"],
                    "forced_exit_after_completion": run[
                        "forced_exit_after_completion"
                    ],
                    "stdout": stdout,
                    "stderr": stderr,
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if self._cancelled.is_set():
                raise RuntimeError("Codex decision request cancelled")
            if run["timed_out"]:
                failures.append(f"{executable}: timeout after {self.timeout_s:.1f}s")
                continue
            if process.returncode != 0 and not run["turn_completed"]:
                failures.append(
                    f"{executable}: exit {process.returncode}: {stderr[-500:]}"
                )
                continue
            try:
                if output_path.exists():
                    raw_value = output_path.read_text(encoding="utf-8")
                else:
                    raw_value, _, _ = self._event_payload(stdout)
                    if raw_value is None:
                        raise ValueError(
                            "no output file or completed agent message"
                        )
                value = json.loads(raw_value)
                value = validate_model_decision(value)
            except Exception as exc:
                failures.append(f"{executable}: invalid response: {exc}")
                continue
            latency = (time.monotonic_ns() - started) * 1e-9
            response = {
                "model_decision": value,
                "api_latency_s": latency,
                "command": executable,
                "attempt": attempt + 1,
                "token_usage": run["usage"],
            }
            (record_dir / "response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n"
            )
            return response
        raise RuntimeError("; ".join(failures))

    def cancel(self):
        self._cancelled.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            self._signal_process_group(process, signal.SIGKILL)


class ResponsesDecisionClient:
    """Direct Responses API client using the configured codex2 provider."""

    def __init__(self, *, codex_home="~/.codex2", model="gpt-6.1-sol",
                 reasoning_effort="low", timeout_s=60.0, retries=1,
                 image_max_side=512, image_detail="low", max_output_tokens=256,
                 response_format="json_schema", base_url=None, api_key=None):
        self.codex_home = Path(codex_home).expanduser()
        loaded = self._load_codex_provider(self.codex_home)
        self.provider = loaded["provider"]
        self.base_url = str(base_url or loaded["base_url"]).rstrip("/")
        self.api_key = str(api_key or loaded["api_key"])
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.image_max_side = int(image_max_side)
        self.image_detail = str(image_detail)
        self.max_output_tokens = int(max_output_tokens)
        self.response_format = str(response_format)
        self._cancelled = threading.Event()
        if self.timeout_s <= 0.0 or self.retries < 0 or self.image_max_side <= 0:
            raise ValueError("invalid Responses API timeout/retry/image settings")
        if self.image_detail not in ("low", "high", "auto", "original"):
            raise ValueError("gpt_image_detail must be low, high, auto or original")
        if self.max_output_tokens < 16:
            raise ValueError("gpt_max_output_tokens must be at least 16")
        if self.response_format not in ("json_schema", "json_object"):
            raise ValueError(
                "gpt_api_response_format must be json_schema or json_object"
            )

    @staticmethod
    def _load_codex_provider(codex_home):
        config_path = codex_home / "config.toml"
        auth_path = codex_home / "auth.json"
        try:
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"cannot read codex2 provider config {config_path}: {exc}"
            ) from exc
        provider_name = config.get("model_provider")
        provider = config.get("model_providers", {}).get(provider_name, {})
        if provider.get("wire_api") != "responses":
            raise RuntimeError(
                f"codex2 provider {provider_name!r} does not use responses API"
            )
        base_url = provider.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith(
                ("http://", "https://")):
            raise RuntimeError(
                f"codex2 provider {provider_name!r} has no valid base_url"
            )
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"cannot read codex2 auth {auth_path}: {exc}"
            ) from exc
        api_key = auth.get("OPENAI_API_KEY")
        if not isinstance(api_key, str) or not api_key:
            raise RuntimeError(f"OPENAI_API_KEY is missing from {auth_path}")
        return {
            "provider": str(provider_name),
            "base_url": base_url,
            "api_key": api_key,
        }

    @staticmethod
    def _output_text(payload):
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct:
            return direct
        texts = []
        for item in payload.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and isinstance(
                        content.get("text"), str):
                    texts.append(content["text"])
        if not texts:
            raise ValueError("Responses API returned no output_text")
        return "".join(texts)

    def _text_format(self):
        if self.response_format == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "name": "gpt_supervisor_decision",
            "strict": True,
            "schema": DECISION_SCHEMA,
        }

    def call(self, snapshot, jpeg, record_dir, *, latency_s, horizon_s):
        from PIL import Image

        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        image_path = record_dir / "head.jpg"
        request_path = record_dir / "request.json"
        prompt = build_prompt(snapshot, latency_s=latency_s, horizon_s=horizon_s)
        (record_dir / "head_source.jpg").write_bytes(jpeg)
        with Image.open(BytesIO(jpeg)) as image:
            image = image.convert("RGB")
            image.thumbnail((self.image_max_side, self.image_max_side))
            image.save(image_path, format="JPEG", quality=92)
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        # Some Responses-compatible gateways reject Structured Outputs for
        # models that accept JSON mode. Local validation below remains the
        # authority before a decision can affect robot execution.
        text_format = self._text_format()
        payload = {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": self.image_detail,
                    },
                ],
            }],
            "text": {"format": text_format},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        request_path.write_text(json.dumps({
            "backend": "api",
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "response_format": self.response_format,
            "prompt": prompt,
            "image": image_path.name,
            "image_detail": self.image_detail,
            "max_output_tokens": self.max_output_tokens,
        }, ensure_ascii=False, indent=2) + "\n")

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        failures = []
        started = time.monotonic_ns()
        for attempt in range(self.retries + 1):
            if self._cancelled.is_set():
                raise RuntimeError("Responses API request cancelled")
            request = urllib.request.Request(
                f"{self.base_url}/responses",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "scalebridge-gpt-supervisor/1",
                },
                method="POST",
            )
            attempt_started = time.monotonic_ns()
            try:
                with urllib.request.urlopen(
                        request, timeout=self.timeout_s) as http_response:
                    response_body = http_response.read().decode("utf-8")
                    status_code = int(http_response.status)
                    request_id = http_response.headers.get("x-request-id")
            except urllib.error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                status_code = int(exc.code)
                request_id = exc.headers.get("x-request-id")
                error = f"HTTP {status_code}: {response_body[-1000:]}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                response_body = ""
                status_code = None
                request_id = None
                error = f"{type(exc).__name__}: {exc}"
            else:
                error = None

            attempt_payload = {
                "backend": "api",
                "provider": self.provider,
                "status_code": status_code,
                "request_id": request_id,
                "latency_s": (time.monotonic_ns() - attempt_started) * 1e-9,
                "error": error,
            }
            (record_dir / f"api_attempt_{attempt + 1}.json").write_text(
                json.dumps(attempt_payload, ensure_ascii=False, indent=2) + "\n"
            )
            if self._cancelled.is_set():
                raise RuntimeError("Responses API request cancelled")
            if error is not None:
                failures.append(error)
                if status_code is not None and (
                        status_code < 500
                        and status_code not in (408, 409, 429)
                ):
                    break
                continue
            try:
                raw_response = json.loads(response_body)
                (record_dir / "openai_response.json").write_text(
                    json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n"
                )
                if raw_response.get("error"):
                    raise ValueError(f"API error: {raw_response['error']}")
                if raw_response.get("status") not in (None, "completed"):
                    raise ValueError(
                        f"response status is {raw_response.get('status')!r}"
                    )
                value = validate_model_decision(
                    json.loads(self._output_text(raw_response))
                )
            except Exception as exc:
                failures.append(f"invalid API response: {exc}")
                continue
            response = {
                "model_decision": value,
                "api_latency_s": (time.monotonic_ns() - started) * 1e-9,
                "backend": "api",
                "provider": self.provider,
                "command": "responses_api",
                "attempt": attempt + 1,
                "response_id": raw_response.get("id"),
                "token_usage": raw_response.get("usage"),
            }
            (record_dir / "response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n"
            )
            return response
        raise RuntimeError("; ".join(failures))

    def cancel(self):
        # The blocking request is bounded by timeout_s. The flag prevents a
        # retry or result publication after supervisor shutdown.
        self._cancelled.set()


class GPTSupervisorWorker:
    """Latest-only request worker and latest-only result mailbox."""

    def __init__(self, *, camera, client, log_root, min_interval_s=0.5,
                 max_camera_age_s=1.0, latency_s=1.0, horizon_s=3.0):
        self.camera = camera
        self.client = client
        self.log_root = Path(log_root)
        self.min_interval_s = float(min_interval_s)
        self.max_camera_age_s = float(max_camera_age_s)
        self.latency_s = float(latency_s)
        self.horizon_s = float(horizon_s)
        if (self.min_interval_s < 0.0 or self.max_camera_age_s <= 0.0
                or self.latency_s < 0.0 or self.horizon_s <= 0.0):
            raise ValueError("invalid GPT supervisor worker timing")
        self._condition = threading.Condition()
        self._stop = False
        self._pending = None
        self._result = None
        self._error = None
        self._last_finished = 0.0
        self._thread = threading.Thread(
            target=self._run, name="gpt-supervisor", daemon=True,
        )
        self._thread.start()

    def submit(self, snapshot):
        with self._condition:
            self._pending = json.loads(json.dumps(snapshot, allow_nan=False))
            self._condition.notify()

    def poll(self):
        with self._condition:
            result, error = self._result, self._error
            self._result = None
            self._error = None
        return result, error

    def _run(self):
        while True:
            with self._condition:
                while not self._stop and self._pending is None:
                    self._condition.wait(timeout=0.2)
                if self._stop:
                    return
                wait_s = self.min_interval_s - (time.monotonic() - self._last_finished)
                urgent = bool(self._pending.get("_urgent", False))
                if wait_s > 0.0 and not urgent:
                    self._condition.wait(timeout=wait_s)
                    continue
                snapshot = self._pending
                self._pending = None
                snapshot.pop("_urgent", None)
            record_dir = (
                self.log_root / f"episode_{int(snapshot['episode_id']):04d}"
                / f"{int(snapshot['obs_id']):08d}"
            )
            try:
                jpeg, image_received_ns = self.camera.latest(self.max_camera_age_s)
                record_dir.mkdir(parents=True, exist_ok=True)
                (record_dir / "state.json").write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
                )
                response = self.client.call(
                    snapshot, jpeg, record_dir,
                    latency_s=self.latency_s, horizon_s=self.horizon_s,
                )
                result = {
                    **response,
                    "obs_id": int(snapshot["obs_id"]),
                    "t_obs_ns": int(snapshot["t_obs_ns"]),
                    "phase_at_obs": snapshot["phase"],
                    "wrist_measured": snapshot["wrist_measured"],
                    "image_received_ns": int(image_received_ns),
                    "record_dir": str(record_dir),
                }
                with self._condition:
                    self._result = result
                    self._error = None
            except Exception as exc:
                with self._condition:
                    self._result = None
                    self._error = {
                        "obs_id": int(snapshot["obs_id"]), "error": str(exc),
                        "record_dir": str(record_dir),
                    }
            finally:
                self._last_finished = time.monotonic()

    @staticmethod
    def record_application(decision, evaluation):
        record_dir = decision.get("record_dir")
        if not record_dir:
            return
        path = Path(record_dir) / "decision.json"
        payload = {
            "decision": decision["model_decision"],
            "phase_at_obs": decision["phase_at_obs"],
            "api_latency_s": decision.get("api_latency_s"),
            "token_usage": decision.get("token_usage"),
            "application": evaluation,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self.client.cancel()
        self._thread.join(timeout=2.0)
        self.camera.close()
        if self._thread.is_alive():
            logger.warning("[GPT supervisor] worker did not stop within 2 seconds")

