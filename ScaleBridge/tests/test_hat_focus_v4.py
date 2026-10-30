"""Offline v4 entry/lifecycle and opt-in composition regressions; no robot I/O."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scalebridge.online.focus_handoff import HandEventHandoff, PerHandFocusHandoff
from scalebridge.online.focus_handoff_v4 import HandEventHandoffV4


def prediction(sequence, *, travel=.08, closed=True, kind="close", goal=.8):
    n = 40
    wrist = np.zeros((n, 3), dtype=np.float32)
    wrist[:, 0] = np.minimum(np.arange(n) / 10., 1.) * travel
    tips = np.zeros((n, 5, 3), dtype=np.float32)
    tips[:, 1:, 0] = .15
    if closed:
        tips[10:, 1:, 0] = .05
    chunk = dict(sequence_id=sequence, source_state_sequence_id=10,
                 frame_count=n, fps=20., right_wrist_pos=wrist,
                 right_wrist_quat_wxyz=np.tile([1., 0., 0., 0.], (n, 1)),
                 right_fingertip_local=tips)
    hand = dict(events=[dict(kind=kind, start=10, end=30,
                             goal=np.full(5, goal, dtype=np.float32))])
    return chunk, hand


def arrive(controller, *, gap=.024, angle=0., actual=None):
    p = np.array([.08 + gap, 0., 0.])
    q = np.array([np.cos(np.radians(angle) / 2), 0., 0., np.sin(np.radians(angle) / 2)])
    measured = dict(right_wrist_yaw_link=(p, q))
    outgoing = dict(right_wrist_pos=p[None].copy(), right_wrist_quat_wxyz=q[None].copy())
    return controller.arrive(measured, 25, outgoing,
                             np.zeros(5) if actual is None else actual)


class V4EntryTests(unittest.TestCase):
    def ready(self, **options):
        c = HandEventHandoffV4(side="right", **options)
        c.observe_event(*prediction(1))
        c.observe_event(*prediction(2))
        return c

    def test_legacy_branch_passes_when_v3_rejects_future_grasp(self):
        c = self.ready()
        self.assertIsNone(c._entry_branches["v3"].entry)
        self.assertTrue(arrive(c))
        self.assertEqual(c.entry_branch, "v1")
        self.assertEqual(c.state, "await_chunk")
        self.assertEqual(c.source_fence, 25)
        self.assertFalse(c.accepts(dict(source_state_sequence_id=24)))
        self.assertTrue(c.accepts(dict(source_state_sequence_id=25)))

    def test_legacy_boundary_is_2point5cm(self):
        self.assertTrue(arrive(self.ready(), gap=.025 - 1e-6))
        c = self.ready()
        self.assertFalse(arrive(c, gap=.025 + 1e-6))
        self.assertEqual(c.state, "approach")

    def test_v3_retains_3cm_arrival_route(self):
        c = HandEventHandoffV4(side="right")
        for seq in (1, 2):
            chunk, hand = prediction(seq, travel=.01)
            chunk["right_wrist_pos"][:, 0] += .07
            c.observe_event(chunk, hand)
        self.assertTrue(arrive(c, gap=.029))
        self.assertEqual(c.entry_branch, "v3")

    def test_both_routes_reject_rotation_and_missing_feedback(self):
        c = self.ready()
        self.assertFalse(arrive(c, angle=11.))
        self.assertFalse(arrive(c, actual=np.full(5, np.nan)))
        self.assertEqual(c.state, "approach")

    def test_both_branches_keep_consensus_and_sequence_ordering(self):
        c = HandEventHandoffV4(side="right")
        c.observe_event(*prediction(1))
        self.assertIsNone(c.entry)
        c.observe_event(*prediction(1))
        self.assertIsNone(c.entry)
        c.observe_event(*prediction(2))
        self.assertIsNotNone(c.entry)
        c.observe_event(*prediction(0, kind="open"))
        self.assertEqual(c.event["kind"], "close")

    def test_legacy_branch_does_not_require_v3_closed_spread(self):
        c = HandEventHandoffV4(side="right")
        for seq in (1, 2):
            c.observe_event(*prediction(seq, closed=False))
        self.assertTrue(arrive(c))
        self.assertEqual(c.entry_branch, "v1")

    def test_predictions_during_focus_do_not_change_armed_event(self):
        c = self.ready()
        self.assertTrue(arrive(c))
        c.start(0)
        c.observe_event(*prediction(3, kind="open"))
        self.assertEqual(c.event["kind"], "close")
        self.assertTrue(c.active)

    def test_goal_reached_exit_matches_v3_and_allows_release(self):
        c = HandEventHandoffV4(side="right", completion_hold_s=0.)
        for seq in (1, 2):
            c.observe_event(*prediction(seq, goal=.3))
        self.assertTrue(arrive(c))
        c.start(0)
        # Under .5 closure: only the existing goal-reached alternative passes.
        self.assertTrue(c.update_actual(100_000_000, np.full(5, .25)))
        self.assertEqual(c.end_reason, "feedback_complete")
        self.assertEqual(c.handoff_version, "v4")
        c.observe_event(*prediction(3))
        c.observe_event(*prediction(4))
        self.assertIsNone(c.entry)  # completed close cannot re-arm another close
        for seq in (5, 6):
            c.observe_event(*prediction(seq, kind="open"))
        self.assertTrue(arrive(c))
        self.assertEqual(c.event["kind"], "open")

    def test_timeout_retry_reset_and_inherited_transport(self):
        c = self.ready()
        self.assertTrue(arrive(c))
        c.start(0)
        self.assertTrue(c.update_actual(5_000_000_000, np.zeros(5)))
        self.assertIsNone(c.last_kind)
        for seq in (3, 4):
            c.observe_event(*prediction(seq))
        self.assertTrue(arrive(c))
        c.reset()
        self.assertIsNone(c.entry)
        self.assertIsNone(c.entry_branch)
        self.assertTrue(all(b.entry is None for b in c._entry_branches.values()))
        self.assertIs(HandEventHandoffV4.start, HandEventHandoff.start)


class V4IntegrationTests(unittest.TestCase):
    def test_env_choice_and_worker_setup_are_independent(self):
        from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
        from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
        from scalebridge.env.motion_tracking_hat4_v4 import environment_class
        from omegaconf import OmegaConf
        for name, base in (("motion_tracking_hat4_online", HAT4OnlineMotionTrackingEnv),
                           ("motion_tracking_hat4_adaptive", HAT4AdaptiveMotionTrackingEnv)):
            cls = environment_class(name)
            self.assertTrue(issubclass(cls, base))
            if name.endswith("online"):
                self.assertFalse(issubclass(cls, HAT4AdaptiveMotionTrackingEnv))
            e = cls.__new__(cls)
            e.cfg = OmegaConf.create(dict(focus_handoff_enabled=True,
                                         focus_handoff_version="v4"))
            def setup(instance):
                self.assertEqual(instance.cfg.focus_handoff_version, "v3")
                instance.focus_handoff = PerHandFocusHandoff()
            with patch.object(base, "_setup_metadata", setup):
                e._setup_metadata()
            self.assertEqual(e.cfg.focus_handoff_version, "v4")
            self.assertTrue(all(isinstance(c, HandEventHandoffV4)
                                for c in e.focus_handoff.controllers.values()))
            seen = []
            with patch.object(base, "_setup_hand_ik", lambda instance: seen.append(
                    instance.cfg.focus_handoff_version)):
                e._setup_hand_ik()
            self.assertEqual(seen, ["v3"])
            self.assertEqual(e.cfg.focus_handoff_version, "v4")

    def test_worker_setup_restores_config_on_exception(self):
        from scalebridge.env.motion_tracking_hat4_v4 import _legacy_worker_version
        config = dict(focus_handoff_version="v4")
        with self.assertRaises(RuntimeError), _legacy_worker_version(config):
            raise RuntimeError("worker setup failed")
        self.assertEqual(config["focus_handoff_version"], "v4")

    def test_hydra_plain_adaptive_and_a1_composition(self):
        from hydra import compose, initialize_config_dir
        root = Path(__file__).resolve().parents[1] / "scalebridge/config"
        for name in ("motion_tracking_hat4_online", "motion_tracking_hat4_adaptive"):
            for a1 in (False, True):
                overrides = [f"env={name}"] + (["+exec_comp=a1"] if a1 else [])
                with initialize_config_dir(version_base=None, config_dir=str(root)):
                    c = compose(config_name="hat4_online",
                                overrides=overrides + ["+focus_handoff=v4"])
                self.assertEqual(c.env._target_, "env.motion_tracking_hat4_v4.HAT4FocusV4Env")
                self.assertEqual(c.env.config.focus_handoff_version, "v4")
                self.assertEqual(c.env.config.focus_v4_v1_position_tolerance_m, .025)
                self.assertEqual(c.env.config.min_chunk_execution_fraction,
                                 0. if name.endswith("adaptive") else .66)

    def test_direct_true_and_v4_keep_plain_target_for_native_activation(self):
        from hydra import compose, initialize_config_dir
        root = Path(__file__).resolve().parents[1] / "scalebridge/config"
        for name in ("motion_tracking_hat4_online", "motion_tracking_hat4_adaptive"):
            with initialize_config_dir(version_base=None, config_dir=str(root)):
                config = compose(
                    config_name="hat4_online",
                    overrides=[
                        f"env={name}",
                        "env.config.focus_handoff_enabled=true",
                        "env.config.focus_handoff_version=v4",
                        "env.config.focus_handoff_source=keypoints",
                    ],
                )
            self.assertEqual(
                config.env._target_, f"env.{name}."
                + ("HAT4AdaptiveMotionTrackingEnv" if name.endswith("adaptive")
                   else "HAT4OnlineMotionTrackingEnv"),
            )
            self.assertTrue(config.env.config.focus_handoff_enabled)
            self.assertEqual(config.env.config.focus_handoff_version, "v4")

    def test_native_v4_uses_v3_event_detector(self):
        from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.cfg = dict(focus_handoff_enabled=True, focus_handoff_version="v4")
        self.assertEqual(env._focus_detector_version(), "v3")
        env.cfg["focus_handoff_enabled"] = False
        self.assertEqual(env._focus_detector_version(), "legacy")


if __name__ == "__main__":
    unittest.main()
