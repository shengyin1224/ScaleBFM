"""Offline regressions for R2 open-hand startup; never connects to DDS."""

import os
import io
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ["SHENGYIN_SKIP_ISAACGYM_IMPORT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MyScaleBFM"))

from human_policy.twist_hand_gmt_bridge import _PinocchioCoupledMotorIK
from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.simulator.real_world_inspire import InspireRealWorld
from scalebridge.simulator import inspire_hand_worker


class CoupledMotorDirectionTest(unittest.TestCase):
    def solver(self, side):
        return _PinocchioCoupledMotorIK(
            side, iters=20, damping=0.001, step=0.7,
            smooth_w=0.01, reg_w=0.0001,
        )

    def test_open_target_stays_open_after_repeated_solves(self):
        for side in ("lh", "rh"):
            solver = self.solver(side)
            q = solver._q_from_motor(np.ones(6))
            target = solver._fk_tips(q)
            for _ in range(20):
                q, error = solver.solve_frame(target, q)
                motor = solver._q_to_motor(q[solver._twist2_from_ik])
                np.testing.assert_allclose(motor, 1.0, atol=1e-6)
                self.assertLess(error, 1e-6)

    def test_reachable_open_and_closed_targets_reduce_tip_error(self):
        for side in ("lh", "rh"):
            for start, end in ((1.0, 0.2), (0.2, 1.0)):
                solver = self.solver(side)
                q = solver._q_from_motor(np.full(6, start))
                target = solver._fk_tips(solver._q_from_motor(np.full(6, end)))
                initial_error = np.mean(np.linalg.norm(target - solver._fk_tips(q), axis=1))
                for _ in range(50):
                    q, error = solver.solve_frame(target, q)
                self.assertLess(error, initial_error * 0.05)
                motor = solver._q_to_motor(q[solver._twist2_from_ik])
                np.testing.assert_allclose(motor[:4], end, atol=0.02)

    def test_nonfinite_target_rejected(self):
        with self.assertRaises(ValueError):
            self.solver("lh").solve_frame(np.full((5, 3), np.nan))


class StartupHandGateTest(unittest.TestCase):
    def env(self):
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.cfg = {"startup_hand_timeout_s": 1.0}
        env.reference_play_gate = True
        env.reference_playing = False
        env.requires_restart = False
        env._latched_sample = None
        env.hand_ik_worker = Mock()
        env.open_hand_dof12 = np.zeros(12)
        env.last_reference_sample = {
            "left_fingertip_local": np.zeros((1, 5, 3)),
            "right_fingertip_local": np.zeros((1, 5, 3)),
        }
        env.simulator = SimpleNamespace(
            set_hand_ik_target=Mock(), get_hand_motor_state=Mock(),
        )
        return env

    def test_prestart_ignores_closed_async_ik_result(self):
        env = self.env()
        env.hand_ik_worker.latest.return_value = {"lh": np.ones(12), "rh": np.ones(12)}
        env._send_hand_target()
        left, right = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_array_equal(left, np.zeros(12))
        np.testing.assert_array_equal(right, np.zeros(12))
        env.hand_ik_worker.latest.assert_not_called()
        env.hand_ik_worker.submit.assert_not_called()

    def test_r1_releases_hand_to_live_ik(self):
        env = self.env()
        env.reference_playing = True
        env.hand_ik_worker.latest.return_value = {"lh": np.ones(12), "rh": np.ones(12)}
        env._send_hand_target()
        left, _ = env.simulator.set_hand_ik_target.call_args.args
        np.testing.assert_array_equal(left, np.ones(12))
        env.hand_ik_worker.submit.assert_called_once()

    def test_waits_for_both_hands_and_valid_feedback(self):
        env = self.env()
        env.simulator.get_hand_motor_state.side_effect = [
            (np.full(6, np.nan), np.ones(6)),
            (np.zeros(6), np.ones(6)),
            (np.ones(6), np.full(6, 0.8)),
            (np.full(6, 0.98), np.ones(6)),
        ]
        with patch("scalebridge.env.motion_tracking_hat4_online.time.sleep"):
            env._prepare_startup_hands()
        self.assertEqual(env.simulator.set_hand_ik_target.call_count, 4)

    def test_startup_timeout_does_not_accept_closed_hand(self):
        env = self.env()
        env.simulator.get_hand_motor_state.return_value = (np.zeros(6), np.zeros(6))
        with patch("scalebridge.env.motion_tracking_hat4_online.time.monotonic", side_effect=[0, 0, 2]), patch("scalebridge.env.motion_tracking_hat4_online.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "fresh open feedback"):
                env._prepare_startup_hands()

    def test_stale_hand_feedback_is_not_published_as_open(self):
        env = self.env()
        env._hand_state_valid = False
        env._measured_hand_state = Mock(return_value=(None, None, None, None))
        env.state_publisher = Mock()
        env._publish_robot_state()
        env.state_publisher.send.assert_not_called()


class MeasuredHandFeedbackTest(unittest.TestCase):
    def test_missing_or_stale_feedback_is_invalid_not_closed(self):
        sim = InspireRealWorld.__new__(InspireRealWorld)
        sim._hand_state_lock = threading.Lock()
        sim._hand_left_state = sim._hand_right_state = None
        sim._hand_state_at = None
        self.assertTrue(np.isnan(sim.get_hand_motor_state()).all())
        sim._hand_left_state = sim._hand_right_state = np.ones(6)
        sim._hand_state_at = time.monotonic() - 1.0
        self.assertTrue(np.isnan(sim.get_hand_motor_state()).all())
        sim._hand_state_at = time.monotonic()
        np.testing.assert_array_equal(sim.get_hand_motor_state(), np.ones((2, 6)))


class HandWorkerFeedbackTest(unittest.TestCase):
    def test_idle_feedback_updates_and_first_ramp_uses_latest_pose(self):
        callbacks = []
        published = []

        def state(q):
            return SimpleNamespace(states=[SimpleNamespace(q=q) for _ in range(12)])

        class Subscriber:
            def __init__(self, *args):
                pass

            def Init(self, callback):
                callbacks.append(callback)
                callback(state(0.2))  # Pose at worker startup.

        class Publisher:
            def __init__(self, *args):
                pass

            def Init(self):
                pass

            def Write(self, command):
                published.append([motor.q for motor in command.cmds])

        def commands():
            # A fresh DDS update while stdin has not supplied any command.
            callbacks[0](state(0.6))
            self.assertFalse(published)
            yield json.dumps({"left": [0.0] * 12, "right": [0.0] * 12})

        modules = {
            "unitree_sdk2py.core.channel": SimpleNamespace(
                ChannelFactoryInitialize=lambda *args: None,
                ChannelPublisher=Publisher, ChannelSubscriber=Subscriber,
            ),
            "unitree_sdk2py.idl.default": SimpleNamespace(
                unitree_go_msg_dds__MotorCmd_=lambda: SimpleNamespace(q=0.0),
            ),
            "unitree_sdk2py.idl.unitree_go.msg.dds_": SimpleNamespace(
                MotorCmds_=lambda cmds: SimpleNamespace(cmds=cmds), MotorStates_=object,
            ),
            "dofpos2cmd": SimpleNamespace(dofpos12_to_q6=lambda q: np.ones(6)),
        }
        args = SimpleNamespace(sdk_path="", mapping_path="", net="offline", state_timeout=1.0, max_step=0.02)
        output = io.StringIO()
        with patch.dict(sys.modules, modules), patch.object(inspire_hand_worker, "_parse_args", return_value=args), patch.object(sys, "stdin", commands()), patch.object(sys, "stdout", output):
            inspire_hand_worker.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "READY")
        feedback = json.loads(next(line[6:] for line in lines if line.startswith("STATE ")))
        np.testing.assert_allclose(feedback["left"], 0.6)
        self.assertIn("measured_at", feedback)
        np.testing.assert_allclose(published, np.full((1, 12), 0.62))


if __name__ == "__main__":
    unittest.main()
