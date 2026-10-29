import unittest
from io import BytesIO
from pathlib import Path
import json
import sys
import tempfile
import time
from unittest import mock

import numpy as np
from PIL import Image

from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.focus_handoff import HandEventHandoff
from scalebridge.online.gpt_supervisor import (
    CodexDecisionClient,
    DECISION_SCHEMA,
    ResponsesDecisionClient,
    SupervisorPhase,
    validate_model_decision,
)


def decision(*, state="normal", speed=0, confidence=1.0, phase="PRE_GRASP",
             t_obs_ns=1_000_000_000, wrist=(0.0, 0.0, 0.0)):
    return {
        "obs_id": 7,
        "t_obs_ns": t_obs_ns,
        "phase_at_obs": phase,
        "wrist_measured": list(wrist),
        "model_decision": {
            "state": state,
            "speed_level": speed,
            "confidence": confidence,
            "evidence": "clear visual evidence",
        },
    }


class DecisionValidationTest(unittest.TestCase):
    def test_rejects_non_integer_speed(self):
        with self.assertRaisesRegex(ValueError, "speed_level"):
            validate_model_decision({
                "state": "normal", "speed_level": 1.0,
                "confidence": 1.0, "evidence": "ok",
            })

    def test_phase_change_rejects_state_but_keeps_speed(self):
        phase = SupervisorPhase()
        phase.force_phase("HOLDING", 1_500_000_000)
        result = phase.evaluate(
            decision(state="ready_focus", speed=2), 2_000_000_000,
            np.zeros(3),
        )
        self.assertIsNone(result["state"])
        self.assertEqual(result["speed_level"], 2)
        self.assertIn("phase changed", result["rejected"]["state"])

    def test_stale_decision_rejects_both_fields(self):
        phase = SupervisorPhase(max_decision_age_s=0.5)
        result = phase.evaluate(decision(), 2_000_000_000, np.zeros(3))
        self.assertIsNone(result["state"])
        self.assertIsNone(result["speed_level"])
        self.assertIn("decision", result["rejected"])

    def test_trigger_drift_forces_immediate_retry(self):
        phase = SupervisorPhase(trigger_max_drift_m=0.02)
        result = phase.evaluate(
            decision(state="ready_focus"), 1_100_000_000,
            np.array([0.02, 0.0, 0.0]),
        )
        self.assertIsNone(result["state"])
        self.assertTrue(result["retry_immediately"])

    def test_focus_exit_honours_minimum_time(self):
        phase = SupervisorPhase(min_focus_time_s=0.5)
        start = phase.evaluate(
            decision(state="ready_focus"), 1_100_000_000, np.zeros(3),
        )
        phase.commit(start, 1_100_000_000)
        early = phase.evaluate(
            decision(state="grasp_secured", phase="FOCUS_GRASP",
                     t_obs_ns=1_200_000_000),
            1_300_000_000, np.zeros(3),
        )
        self.assertIsNone(early["state"])
        accepted = phase.evaluate(
            decision(state="grasp_secured", phase="FOCUS_GRASP",
                     t_obs_ns=1_500_000_000),
            1_700_000_000, np.zeros(3),
        )
        self.assertEqual(accepted["action"], "finish_grasp")
        phase.commit(accepted, 1_700_000_000)
        self.assertEqual(phase.phase, "HOLDING")

    def test_focus_timeout_is_code_owned(self):
        phase = SupervisorPhase(focus_timeout_s=1.0, min_focus_time_s=0.1)
        phase.force_phase("FOCUS_RELEASE", 2_000_000_000)
        timeout = phase.timeout(3_100_000_000)
        self.assertEqual(timeout["action"], "abort_release")
        self.assertEqual(timeout["next_phase"], "HOLDING")


class CodexDecisionClientTest(unittest.TestCase):
    def test_turn_completed_is_success_even_if_cli_does_not_exit(self):
        model_decision = {
            "state": "normal",
            "speed_level": 1,
            "confidence": 0.9,
            "evidence": "The hand is still approaching the object.",
        }
        script = "\n".join((
            "import json, time",
            f"decision = {model_decision!r}",
            "print(json.dumps({'type': 'item.completed', 'item': "
            "{'type': 'agent_message', 'text': json.dumps(decision)}}), flush=True)",
            "print(json.dumps({'type': 'turn.completed', 'usage': "
            "{'input_tokens': 10, 'output_tokens': 5}}), flush=True)",
            "time.sleep(5)",
        ))
        jpeg_buffer = BytesIO()
        Image.new("RGB", (8, 8)).save(jpeg_buffer, format="JPEG")
        snapshot = {
            "episode_id": 0,
            "obs_id": 1,
            "t_obs_ns": 1,
            "task_instruction": "test",
            "phase": "PRE_GRASP",
            "wrist_measured": [0.0, 0.0, 0.0],
        }
        client = CodexDecisionClient(
            commands=(sys.executable,), timeout_s=2.0, retries=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            client, "_command", return_value=[sys.executable, "-c", script],
        ):
            started = time.monotonic()
            response = client.call(
                snapshot, jpeg_buffer.getvalue(), Path(tmpdir),
                latency_s=1.0, horizon_s=3.0,
            )
            elapsed = time.monotonic() - started
            attempt = json.loads(
                (Path(tmpdir) / "codex_attempt_1.json").read_text()
            )

        self.assertLess(elapsed, 1.5)
        self.assertEqual(response["attempt"], 1)
        self.assertEqual(response["model_decision"], model_decision)
        self.assertEqual(response["token_usage"]["input_tokens"], 10)
        self.assertTrue(attempt["turn_completed"])
        self.assertTrue(attempt["forced_exit_after_completion"])
        self.assertFalse(attempt["timeout"])

    def test_timeout_kills_helper_processes_holding_output_pipes(self):
        script = "\n".join((
            "import subprocess, sys, time",
            "subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(5)'])",
            "time.sleep(5)",
        ))
        client = CodexDecisionClient(
            commands=(sys.executable,), timeout_s=0.2, retries=0,
        )

        started = time.monotonic()
        run = client._run_process([sys.executable, "-c", script], "")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.5)
        self.assertTrue(run["timed_out"])
        self.assertFalse(run["turn_completed"])


class ResponsesDecisionClientTest(unittest.TestCase):
    @staticmethod
    def _jpeg():
        buffer = BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_uses_codex2_provider_and_auth_without_logging_key(self):
        model_decision = {
            "state": "normal",
            "speed_level": 1,
            "confidence": 0.9,
            "evidence": "The hand is approaching the bottle.",
        }
        api_payload = {
            "id": "resp_test",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": json.dumps(model_decision),
                }],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        class FakeResponse:
            status = 200
            headers = {"x-request-id": "request_test"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(api_payload).encode()

        snapshot = {
            "episode_id": 0,
            "obs_id": 1,
            "t_obs_ns": 1,
            "task_instruction": "test",
            "phase": "PRE_GRASP",
            "wrist_measured": [0.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            codex_home = root / "codex2"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                'model_provider = "test"\n'
                '[model_providers.test]\n'
                'base_url = "https://example.test/v1"\n'
                'wire_api = "responses"\n'
            )
            (codex_home / "auth.json").write_text(
                json.dumps({"OPENAI_API_KEY": "secret-test-key"})
            )
            client = ResponsesDecisionClient(
                codex_home=codex_home, retries=0,
            )
            record_dir = root / "record"
            with mock.patch(
                    "urllib.request.urlopen", return_value=FakeResponse()) as send:
                response = client.call(
                    snapshot, self._jpeg(), record_dir,
                    latency_s=1.0, horizon_s=3.0,
                )
            sent_request = send.call_args.args[0]
            sent_payload = json.loads(sent_request.data)
            request_log = (record_dir / "request.json").read_text()

        self.assertEqual(sent_request.full_url, "https://example.test/v1/responses")
        self.assertEqual(
            sent_request.get_header("Authorization"), "Bearer secret-test-key"
        )
        self.assertEqual(
            sent_payload["text"]["format"]["schema"],
            DECISION_SCHEMA,
        )
        self.assertTrue(
            sent_payload["input"][0]["content"][1]["image_url"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertNotIn("secret-test-key", request_log)
        self.assertEqual(response["model_decision"], model_decision)
        self.assertEqual(response["backend"], "api")

    def test_json_object_format_keeps_local_validation(self):
        client = ResponsesDecisionClient.__new__(ResponsesDecisionClient)
        client.response_format = "json_object"
        self.assertEqual(client._text_format(), {"type": "json_object"})
        self.assertEqual(
            validate_model_decision({
                "state": "normal",
                "speed_level": 1,
                "confidence": 0.9,
                "evidence": "The hand is approaching the bottle.",
            })["state"],
            "normal",
        )


class ExistingHandoffReuseTest(unittest.TestCase):
    def test_external_trigger_enters_existing_fresh_chunk_path(self):
        handoff = HandEventHandoff(side="right")
        outgoing = {
            "right_wrist_pos": np.zeros((1, 3), dtype=np.float32),
            "right_wrist_quat_wxyz": np.array(
                [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
            ),
        }
        self.assertTrue(handoff.request_external(
            "close", 42, outgoing, np.zeros(5, dtype=np.float32),
        ))
        self.assertEqual(handoff.state, "await_chunk")
        self.assertEqual(handoff.source_fence, 42)
        self.assertEqual(handoff.event["source"], "gpt")
        handoff.start(3_000_000_000)
        self.assertTrue(handoff.active)


class GPTPlaybackTest(unittest.TestCase):
    def test_speed_level_is_smoothed_outside_focus_and_bypassed_inside(self):
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.approach_slowdown = False
        env.focus_handoff_source = "gpt"
        env.gpt_phase = SupervisorPhase(speed_levels=(1.0, 0.5, 0.25))
        env.gpt_phase.speed_level = 2
        env.gpt_speed_slew_per_s = 0.5
        env.reference_playing = True
        env.localization_paused = False
        env._play_scale = 1.0
        env._play_scale_logged = 1.0
        env._play_scale_hold = {}
        env._gpt_speed_updated_ns = 0
        env._update_playback_scale(1_000_000_000)
        self.assertAlmostEqual(env._play_scale, 0.5)
        env.gpt_phase.force_phase("FOCUS_GRASP", 1_500_000_000)
        env._update_playback_scale(1_500_000_000)
        self.assertEqual(env._play_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
