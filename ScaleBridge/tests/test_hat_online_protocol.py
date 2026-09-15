import numpy as np
import torch
import unittest

from scalebridge.online.protocol import (
    inspire_motor_to_hat_hand_q,
    make_hat_chunk,
    make_robot_state,
    validate_hat_chunk,
)
from scalebridge.online.reference import (
    align_chunk_to_root,
    alignment_transform,
    apply_alignment,
    sample_chunk,
)


def robot_state(sequence_id=3):
    return make_robot_state(
        sequence_id=sequence_id,
        timestamp_ns=100,
        root_pos_world=[1.0, 2.0, 0.8],
        root_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        head_pos_world=[1.0, 2.0, 1.2],
        head_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        left_wrist_pos_world=[1.1, 2.2, 1.0],
        left_wrist_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        right_wrist_pos_world=[1.1, 1.8, 1.0],
        right_wrist_quat_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        body_q19=np.arange(19) * 0.01,
        left_arm_q=np.arange(7),
        left_hand_q=np.arange(6),
        left_hand_keypoints_local=np.arange(18).reshape(6, 3) * 0.01,
        right_arm_q=np.arange(7) + 10,
        right_hand_q=np.arange(6) + 10,
        right_hand_keypoints_local=np.arange(18).reshape(6, 3) * -0.01,
    )


def chunk(n=64, sequence_id=1):
    t = np.arange(n, dtype=np.float32) / 30.0
    identity = np.zeros((n, 4), dtype=np.float32)
    identity[:, 0] = 1.0
    root = np.stack((t, np.zeros(n), np.full(n, 0.8)), axis=-1)
    head = root + [0.0, 0.0, 0.6]
    left = root + [0.0, 0.25, 0.3]
    right = root + [0.0, -0.25, 0.3]
    fingers = np.zeros((n, 5, 3), dtype=np.float32)
    fingers[..., 0] = np.linspace(0.02, 0.10, 5)
    return make_hat_chunk(
        sequence_id=sequence_id,
        source_state_sequence_id=3,
        generated_at_ns=200,
        fps=30.0,
        frame_count=n,
        root_pos=root,
        root_quat_wxyz=identity,
        head_pos=head,
        head_quat_wxyz=identity,
        left_wrist_pos=left,
        left_wrist_quat_wxyz=identity,
        right_wrist_pos=right,
        right_wrist_quat_wxyz=identity,
        left_fingertip_local=fingers,
        right_fingertip_local=fingers,
    )


class HATOnlineProtocolTest(unittest.TestCase):
    def test_hat_chunk_preserves_action_representation(self):
        original = validate_hat_chunk(chunk())
        self.assertEqual(original["action_representation"], "original")

        relative = chunk()
        relative["action_representation"] = "relative_chunk"
        self.assertEqual(
            validate_hat_chunk(relative)["action_representation"],
            "relative_chunk",
        )

        relative["action_representation"] = "unknown"
        with self.assertRaisesRegex(ValueError, "action_representation"):
            validate_hat_chunk(relative)

    def test_hat_chunk_preserves_optional_global_position_origin(self):
        raw = chunk(n=100, sequence_id=5)
        raw["position_origin_world"] = [-0.078, -0.001, 1.272]
        checked = validate_hat_chunk(raw)
        np.testing.assert_allclose(
            checked["position_origin_world"], [-0.078, -0.001, 1.272]
        )
        restored = apply_alignment(
            checked,
            checked["position_origin_world"],
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            restored["root_pos"][0],
            checked["root_pos"][0] + checked["position_origin_world"],
            atol=1e-6,
        )

    @staticmethod
    def _phase_env(checked, frame, source_timestamp_ns=1_000_000_000):
        from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv

        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.dt = 0.02
        env.state_sequence_id = 4
        env.state_timestamp_history = [(3, source_timestamp_ns)]
        env.chunk_phase_alignment = True
        env.phase_rotation_weight = 0.08
        env.phase_velocity_weight = 0.03
        env.phase_measured_weight = 0.15
        env.phase_max_root_jump_m = 0.20
        env.future_offsets_s = np.arange(6, dtype=np.float64) / 50.0
        env.last_reference_sample = None
        env.last_published_robot_state = {
            "root_pos_world": checked["root_pos"][frame],
            "root_quat_world_wxyz": checked["root_quat_wxyz"][frame],
            "head_pos_world": checked["head_pos"][frame],
            "head_quat_world_wxyz": checked["head_quat_wxyz"][frame],
            "left_wrist_pos_world": checked["left_wrist_pos"][frame],
            "left_wrist_quat_world_wxyz": checked["left_wrist_quat_wxyz"][frame],
            "right_wrist_pos_world": checked["right_wrist_pos"][frame],
            "right_wrist_quat_world_wxyz": checked["right_wrist_quat_wxyz"][frame],
        }
        return env

    def test_chunk_start_uses_timestamp_latency_as_its_only_clock(self):
        checked = validate_hat_chunk(chunk(n=100, sequence_id=2))
        env = self._phase_env(checked, frame=14)
        # 0.4 s at 30 Hz means frame 12 belongs to now.  Measured pose is frame
        # 14, but it must not create a second clock and fast-forward the chunk.
        now_ns = 1_400_000_000
        selected, state = env._phase_match_start_frame(checked, now_ns)
        self.assertEqual(state["expected_frame"], 12)
        self.assertEqual(selected, 12)
        self.assertEqual(state["search_range"], [12, 12])
        self.assertGreater(state["pose_cost"], 0.0)

        env.reference_play_gate = False
        env.reference_playing = True
        env.localization_paused = False
        env.held_sample = None
        env.requires_restart = False
        env._latched_sample = None
        env.chunk = checked
        env.chunk_started_ns = 1_000_000_000
        env.previous_chunk = None
        env.previous_chunk_started_ns = None
        env.blend_started_ns = None
        sample = env._sample_reference(now_ns)
        np.testing.assert_allclose(
            sample["root_pos"][0], checked["root_pos"][selected], atol=1e-6
        )

    def test_timestamp_clock_cannot_jump_to_distant_return_branch(self):
        checked = validate_hat_chunk(chunk(n=100, sequence_id=2))
        # The trajectory reaches x=0.3 at frames 10 and 80. A pose-based clock
        # would be ambiguous; the timestamp must select frame 10 exactly.
        x = np.concatenate((
            np.linspace(0.0, 0.6, 21),
            np.linspace(0.6, 0.3, 60),
            np.linspace(0.3, 0.0, 19),
        )).astype(np.float32)
        for key in ("root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"):
            offset = checked[key][:, 0] - checked["root_pos"][:, 0]
            checked[key][:, 0] = x + offset
        env = self._phase_env(checked, frame=80)
        now_ns = 1_333_333_333  # expected frame 10
        selected, state = env._phase_match_start_frame(checked, now_ns)
        self.assertEqual(state["expected_frame"], 10)
        self.assertEqual(selected, 10)
        self.assertEqual(state["search_range"], [10, 10])

    def test_slow_root_motion_does_not_override_timestamp_phase(self):
        checked = validate_hat_chunk(chunk(n=100, sequence_id=2))
        # Even when a later pose is geometrically closer to the measurement,
        # the source timestamp remains the only unambiguous execution clock.
        x = np.arange(100, dtype=np.float32) * 0.0015
        for key in ("root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"):
            offset = checked[key][:, 0] - checked["root_pos"][:, 0]
            checked[key][:, 0] = x + offset
        env = self._phase_env(checked, frame=10)
        now_ns = 1_066_666_667  # latency estimate is frame 2
        selected, state = env._phase_match_start_frame(checked, now_ns)
        self.assertEqual(state["expected_frame"], 2)
        self.assertEqual(selected, 2)

    def test_replan_uses_new_chunk_timestamp_not_old_frame_number(self):
        old = validate_hat_chunk(chunk(n=100, sequence_id=1))
        new = validate_hat_chunk(chunk(n=100, sequence_id=2))
        # The new receding-horizon chunk was captured when the old trajectory
        # had reached frame 20, so its frame zero is old frame 20 in world
        # space.  At now=0.7 s its one-frame stale prefix, not frame 21 copied
        # from the old chunk, represents the same absolute target time.
        for key in ("root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"):
            new[key][:, 0] += 20.0 / 30.0
        env = self._phase_env(new, frame=0, source_timestamp_ns=666_666_667)
        env.reference_playing = True
        env.localization_paused = False
        env.awaiting_fresh_chunk = False
        env.chunk = old
        env.chunk_started_ns = 0

        now_ns = 700_000_000
        selected, state = env._phase_match_start_frame(new, now_ns)
        self.assertEqual(state["mode"], "time_aligned_reference")
        self.assertAlmostEqual(state["outgoing_frame"], 21.0, places=5)
        self.assertEqual(state["expected_frame"], 1)
        self.assertEqual(state["search_center_frame"], 1)
        self.assertEqual(selected, 1)
        self.assertLess(state["root_jump_m"], 1e-6)
        self.assertGreaterEqual(state["backtrack_m"], -1e-6)
        self.assertTrue(state["transition_valid"])

    def test_endpoint_replan_uses_early_new_timestamp_frame(self):
        old = validate_hat_chunk(chunk(n=100, sequence_id=1))
        new = validate_hat_chunk(chunk(n=100, sequence_id=2))
        # Reproduce the endpoint failure shape: the outgoing command is old
        # frame 99 while the fresh chunk's timestamp says frame 1 belongs to
        # now. Inheriting old frame 99 would falsely hold forever.
        shift = (99 - 1) / 30.0
        for key in ("root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"):
            new[key][:, 0] += shift
        env = self._phase_env(new, frame=0, source_timestamp_ns=3_266_666_667)
        env.reference_playing = True
        env.localization_paused = False
        env.awaiting_fresh_chunk = False
        env.chunk = old
        env.chunk_started_ns = 0

        now_ns = 3_300_000_000
        selected, state = env._phase_match_start_frame(new, now_ns)
        self.assertAlmostEqual(state["outgoing_frame"], 99.0, places=5)
        self.assertEqual(state["search_center_frame"], 1)
        self.assertEqual(state["search_range"], [1, 1])
        self.assertEqual(selected, 1)
        self.assertLess(state["root_jump_m"], 1e-6)
        self.assertTrue(state["transition_valid"])

    def test_normal_reprediction_backtrack_is_blended_not_deadlocked(self):
        old = validate_hat_chunk(chunk(n=100, sequence_id=1))
        new = validate_hat_chunk(chunk(n=100, sequence_id=2))
        # At now=0.7 s old frame 21 is outgoing and new frame 1 is current.
        # Let the fresh prediction revise that target backward by 5 cm: this is
        # far above the obsolete 3 mm gate but is safe to overlap-blend.
        shift = 20.0 / 30.0 - 0.05
        for key in ("root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"):
            new[key][:, 0] += shift
        env = self._phase_env(new, frame=0, source_timestamp_ns=666_666_667)
        env.reference_playing = True
        env.localization_paused = False
        env.awaiting_fresh_chunk = False
        env.chunk = old
        env.chunk_started_ns = 0

        selected, state = env._phase_match_start_frame(new, 700_000_000)
        self.assertEqual(selected, 1)
        self.assertAlmostEqual(state["root_jump_m"], 0.05, places=5)
        self.assertLess(state["backtrack_m"], -0.049)
        self.assertTrue(state["transition_valid"])

    def test_corrupt_twenty_centimetre_seam_trips_emergency_guard(self):
        old = validate_hat_chunk(chunk(n=100, sequence_id=1))
        new = validate_hat_chunk(chunk(n=100, sequence_id=2))
        shift = 20.0 / 30.0 + 0.25
        for key in ("root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"):
            new[key][:, 0] += shift
        env = self._phase_env(new, frame=0, source_timestamp_ns=666_666_667)
        env.reference_playing = True
        env.localization_paused = False
        env.awaiting_fresh_chunk = False
        env.chunk = old
        env.chunk_started_ns = 0

        _, state = env._phase_match_start_frame(new, 700_000_000)
        self.assertGreater(state["root_jump_m"], 0.20)
        self.assertFalse(state["transition_valid"])

    def test_many_replans_never_accumulate_to_frame_99(self):
        active = validate_hat_chunk(chunk(n=100, sequence_id=1))
        env = self._phase_env(active, frame=0, source_timestamp_ns=0)
        env.reference_playing = True
        env.localization_paused = False
        env.awaiting_fresh_chunk = False
        env.chunk = active
        env.chunk_started_ns = 0

        for sequence_id in range(2, 202):
            # Fresh prediction every 0.34 s, received 0.02 s after its source
            # observation. Its positions are expressed on the same world-time
            # trajectory as all earlier predictions.
            source_ns = int((sequence_id - 1) * 0.34 * 1e9)
            now_ns = source_ns + 20_000_000
            fresh = validate_hat_chunk(chunk(n=100, sequence_id=sequence_id))
            source_s = source_ns * 1e-9
            for key in ("root_pos", "head_pos", "left_wrist_pos", "right_wrist_pos"):
                fresh[key][:, 0] += source_s
            env.state_timestamp_history = [(3, source_ns)]

            selected, state = env._phase_match_start_frame(fresh, now_ns)
            self.assertEqual(selected, 1)
            self.assertLess(state["outgoing_frame"], 12.0)
            self.assertTrue(state["transition_valid"])
            env.chunk = fresh
            env.chunk_started_ns = source_ns

    def test_online_gate_keeps_policy_active_then_blends_after_start(self):
        from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv

        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.reference_play_gate = True
        env.reference_playing = False
        env.takeover_hold_dof_pos = None
        env.takeover_started_ns = None
        env.cfg = {"takeover_blend_s": 1.0}
        requested = torch.tensor([[5.0, 10.0]])

        applied, alpha = env._select_applied_target(requested, 10_000_000_000)
        torch.testing.assert_close(applied, requested)
        self.assertEqual(alpha, 0.0)

        env.reference_playing = True
        env.takeover_hold_dof_pos = torch.tensor([[1.0, 2.0]])
        env.takeover_started_ns = 10_000_000_000
        applied, alpha = env._select_applied_target(requested, 10_500_000_000)
        torch.testing.assert_close(applied, torch.tensor([[3.0, 6.0]]))
        self.assertAlmostEqual(alpha, 0.5)

        applied, alpha = env._select_applied_target(requested, 11_500_000_000)
        torch.testing.assert_close(applied, requested)
        self.assertEqual(alpha, 1.0)

    def test_online_gate_repeats_one_latched_frame_across_future_tokens(self):
        from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv

        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.reference_play_gate = True
        env.reference_playing = False
        env.prestart_reference_sample = {
            "root_pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "focus_phase": np.array([[1.0, -0.2]], dtype=np.float32),
        }
        env.future_offsets_s = np.arange(6, dtype=np.float64) / 50.0
        env.chunk = {"unused": True}
        env.held_sample = None
        env.requires_restart = False
        env._latched_sample = None

        sample = env._sample_reference(123)
        self.assertEqual(sample["root_pos"].shape, (6, 3))
        np.testing.assert_allclose(sample["root_pos"], [[1.0, 2.0, 3.0]] * 6)
        np.testing.assert_allclose(sample["focus_phase"], [[1.0, -0.2]] * 6)

    def test_ready_reference_uses_measured_g1_raised_arms_and_open_fingers(self):
        from scalebridge.env.motion_tracking_hat4_online import (
            HAT4OnlineMotionTrackingEnv,
            RAISED_READY_ARM_Q,
        )

        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.prestart_arm_pose = "raised"
        joint_names = []
        for side in ("left", "right"):
            joint_names.extend([
                f"{side}_shoulder_pitch_joint",
                f"{side}_shoulder_roll_joint",
                f"{side}_shoulder_yaw_joint",
                f"{side}_elbow_joint",
                f"{side}_wrist_roll_joint",
                f"{side}_wrist_pitch_joint",
                f"{side}_wrist_yaw_joint",
            ])
        env.metadata_dict = {"joint_names": joint_names}
        env.state_buffer = {
            "dof_pos_buffer": torch.full((1, 1, len(joint_names)), 9.0)
        }

        class FakeFK:
            def compute(self, root_pos, root_quat, dof_pos):
                self.root_pos = np.asarray(root_pos)
                self.root_quat = np.asarray(root_quat)
                self.dof_pos = np.asarray(dof_pos)
                self.last_body_poses = {
                    "torso_link": (
                        np.array([3.0, -2.0, 1.0], dtype=np.float32),
                        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    )
                }
                return {
                    "head_pos_world": np.array([3.0, -2.0, 1.4], dtype=np.float32),
                    "head_quat_world_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "left_wrist_pos_world": np.array([3.3, -1.8, 1.2], dtype=np.float32),
                    "left_wrist_quat_world_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "right_wrist_pos_world": np.array([3.3, -2.2, 1.2], dtype=np.float32),
                    "right_wrist_quat_world_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                }

        env.current_pose_fk = FakeFK()
        env.open_hand_fingertips = {
            "lh": np.full((5, 3), 3.0, dtype=np.float32),
            "rh": np.full((5, 3), 4.0, dtype=np.float32),
        }
        live = {
            "left_fingertip_local": np.ones((1, 5, 3), dtype=np.float32),
            "right_fingertip_local": np.full((1, 5, 3), 2.0, dtype=np.float32),
        }
        # A pitched measured quat (~11.5 deg) must flow through unchanged: the
        # obs rotate targets by the FULL measured root quat, so a heading-only
        # reference would feed the policy a constant pitch error on the real
        # robot (Vive tilt) and tip it over.
        measured_quat = np.array([0.995, 0.0, 0.0998, 0.0], dtype=np.float32)
        ready = env._make_prestart_reference(
            np.array([3.0, -2.0, 0.9], dtype=np.float32),
            measured_quat,
            live,
        )
        np.testing.assert_allclose(ready["root_pos"][0], [3.0, -2.0, 0.9])
        np.testing.assert_allclose(ready["left_wrist_pos"][0], [3.3, -1.8, 1.2])
        np.testing.assert_allclose(ready["right_wrist_pos"][0], [3.3, -2.2, 1.2])
        np.testing.assert_allclose(env.current_pose_fk.root_quat, measured_quat)
        np.testing.assert_allclose(ready["root_quat_wxyz"][0], measured_quat)
        for side_index, side in enumerate(("left", "right")):
            np.testing.assert_allclose(
                env.current_pose_fk.dof_pos[side_index * 7:(side_index + 1) * 7],
                RAISED_READY_ARM_Q[side],
            )
        np.testing.assert_allclose(ready["left_fingertip_local"], 3.0)
        np.testing.assert_allclose(ready["right_fingertip_local"], 4.0)
        np.testing.assert_allclose(ready["focus_phase"], [[0.0, 2.0]])

    def test_ready_reference_supports_sonic_0904_whole_body_opening_pose(self):
        from scalebridge.env.motion_tracking_hat4_online import (
            HAT4OnlineMotionTrackingEnv,
            SONIC_0904_READY_ARM_Q,
            SONIC_0904_READY_LOWER_Q,
        )

        env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
        env.prestart_arm_pose = "sonic_0904"
        joint_names = list(SONIC_0904_READY_LOWER_Q)
        for side in ("left", "right"):
            joint_names.extend([
                f"{side}_shoulder_pitch_joint",
                f"{side}_shoulder_roll_joint",
                f"{side}_shoulder_yaw_joint",
                f"{side}_elbow_joint",
                f"{side}_wrist_roll_joint",
                f"{side}_wrist_pitch_joint",
                f"{side}_wrist_yaw_joint",
            ])
        env.metadata_dict = {"joint_names": joint_names}
        env.state_buffer = {
            "dof_pos_buffer": torch.full((1, 1, len(joint_names)), 9.0)
        }

        class FakeFK:
            def compute(self, root_pos, root_quat, dof_pos):
                self.dof_pos = np.asarray(dof_pos)
                self.last_body_poses = {
                    "torso_link": (
                        np.zeros(3, dtype=np.float32),
                        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    )
                }
                return {
                    "head_pos_world": np.zeros(3, dtype=np.float32),
                    "head_quat_world_wxyz": np.array(
                        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
                    ),
                    "left_wrist_pos_world": np.zeros(3, dtype=np.float32),
                    "left_wrist_quat_world_wxyz": np.array(
                        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
                    ),
                    "right_wrist_pos_world": np.zeros(3, dtype=np.float32),
                    "right_wrist_quat_world_wxyz": np.array(
                        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
                    ),
                }

        env.current_pose_fk = FakeFK()
        env.open_hand_fingertips = {
            "lh": np.zeros((5, 3), dtype=np.float32),
            "rh": np.zeros((5, 3), dtype=np.float32),
        }
        env._make_prestart_reference(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            {},
        )

        actual = dict(zip(joint_names, env.current_pose_fk.dof_pos))
        for name, expected in SONIC_0904_READY_LOWER_Q.items():
            self.assertAlmostEqual(actual[name], expected)
        for side, expected in SONIC_0904_READY_ARM_Q.items():
            names = [
                f"{side}_shoulder_pitch_joint",
                f"{side}_shoulder_roll_joint",
                f"{side}_shoulder_yaw_joint",
                f"{side}_elbow_joint",
                f"{side}_wrist_roll_joint",
                f"{side}_wrist_pitch_joint",
                f"{side}_wrist_yaw_joint",
            ]
            np.testing.assert_allclose([actual[name] for name in names], expected)

    def test_root_obs_holds_local_before_r1_then_blends_to_tracker(self):
        from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv

        def make_env():
            env = HAT4OnlineMotionTrackingEnv.__new__(HAT4OnlineMotionTrackingEnv)
            env.reference_forcing = False
            env.localization_paused = False
            env.active_indices = {"pelvis": 0}
            env.local_to_global_transition_steps = 4
            env._root_transition_active = False
            env._root_transition_index = 0
            body_pos = torch.zeros(1, 6, 1, 3)
            body_pos[0, :, 0] = torch.tensor([1.0, 2.0, 0.8])
            env.state_buffer = {
                "body_pos_w_future": body_pos,
                "root_pos_buffer": torch.zeros(1, 4, 3),
            }
            return env

        measured = torch.tensor([[1.4, 2.0, 0.8]])

        # Pre-R1 (playing gate closed): root obs pinned to the reference root,
        # tracker position ignored.
        env = make_env()
        env.reference_playing = False
        env.state_buffer["root_pos_buffer"][:, -1] = measured
        env._update_root_observation(measured)
        torch.testing.assert_close(
            env.state_buffer["root_pos_buffer"][:, -1], torch.tensor([[1.0, 2.0, 0.8]])
        )

        # R1 pressed: walk to the tracker in local_to_global_transition_steps
        # equal increments, then deactivate and leave the buffer untouched.
        env = make_env()
        env.reference_playing = True
        env._root_transition_active = True
        expected_x = [1.1, 1.2, 1.3, 1.4]
        for x in expected_x:
            env.state_buffer["root_pos_buffer"][:, -1] = measured
            env._update_root_observation(measured)
            torch.testing.assert_close(
                env.state_buffer["root_pos_buffer"][:, -1],
                torch.tensor([[x, 2.0, 0.8]]),
            )
        self.assertFalse(env._root_transition_active)
        env.state_buffer["root_pos_buffer"][:, -1] = measured
        env._update_root_observation(measured)
        torch.testing.assert_close(env.state_buffer["root_pos_buffer"][:, -1], measured)

    def test_inspire_feedback_is_reordered_to_hat_training_semantics(self):
        np.testing.assert_allclose(
            inspire_motor_to_hat_hand_q([0, 1, 2, 3, 4, 5]),
            [3, 2, 0, 1, 5, 4],
        )

    def test_chunk_alignment_makes_frame_zero_equal_actual_root(self):
        aligned = align_chunk_to_root(
            chunk(), [3.0, -2.0, 0.9], [1.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            aligned["root_pos"][0], [3.0, -2.0, 0.9], atol=1e-6
        )
        np.testing.assert_allclose(
            aligned["root_quat_wxyz"][0], [1.0, 0.0, 0.0, 0.0], atol=1e-6
        )

    def test_alignment_keeps_heading_but_not_the_robots_own_tilt(self):
        # A pelvis pitched 41 degrees, as measured during a failing MuJoCo run.
        half = np.deg2rad(41.0) / 2.0
        tilted = [np.cos(half), 0.0, np.sin(half), 0.0]
        aligned = align_chunk_to_root(chunk(), [0.0, 0.0, 0.9], tilted)
        # The tilt must not be inherited, or the controller sees no tilt error.
        np.testing.assert_allclose(
            aligned["root_quat_wxyz"][0], [1.0, 0.0, 0.0, 0.0], atol=1e-6
        )
        # Yaw is still followed.
        yaw = np.deg2rad(90.0) / 2.0
        turned = align_chunk_to_root(
            chunk(), [0.0, 0.0, 0.9], [np.cos(yaw), 0.0, 0.0, np.sin(yaw)]
        )
        np.testing.assert_allclose(
            turned["root_quat_wxyz"][0],
            [np.cos(yaw), 0.0, 0.0, np.sin(yaw)],
            atol=1e-6,
        )
        # Frame 0 still lands exactly on the measured root position either way.
        for out in (aligned, turned):
            np.testing.assert_allclose(out["root_pos"][0], [0.0, 0.0, 0.9], atol=1e-6)

    def test_fixed_anchor_does_not_integrate_per_chunk_heading_bias(self):
        # Every HAT chunk carries a small heading-rate bias (~-3 deg over the
        # chunk in the 2026-08-21 runs).  Re-anchoring each chunk to the robot
        # ("follow") integrates that bias without bound; a locked anchor
        # ("fixed") keeps the reference heading within one chunk's bias.
        bias = np.deg2rad(-3.0)
        biased = validate_hat_chunk(chunk())
        n = len(biased["root_quat_wxyz"])
        half = 0.5 * np.linspace(0.0, bias, n)
        biased["root_quat_wxyz"] = np.stack(
            (np.cos(half), np.zeros(n), np.zeros(n), np.sin(half)), axis=-1
        ).astype(np.float32)

        def yaw_of(quat_wxyz):
            w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
            return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        start_pos = np.array([0.0, 0.0, 0.9])
        start_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Follow mode: the robot tracks each chunk perfectly, then the next
        # identical chunk is re-anchored to where the robot ended up.
        robot_pos, robot_quat = start_pos, start_quat
        for _ in range(10):
            aligned = align_chunk_to_root(biased, robot_pos, robot_quat, validated=True)
            robot_pos = aligned["root_pos"][-1]
            robot_quat = aligned["root_quat_wxyz"][-1]
        follow_yaw = yaw_of(robot_quat)
        self.assertLess(follow_yaw, np.deg2rad(-25.0))

        # Fixed mode: the transform is computed once and reused, so the
        # reference never asks for more heading than one chunk's own bias.
        anchor = alignment_transform(biased, start_pos, start_quat)
        for _ in range(10):
            aligned = apply_alignment(biased, *anchor)
        self.assertAlmostEqual(yaw_of(aligned["root_quat_wxyz"][0]), 0.0, places=5)
        self.assertGreater(yaw_of(aligned["root_quat_wxyz"][-1]), np.deg2rad(-3.5))

    def test_sampling_30hz_at_50hz_offsets_and_head_to_torso(self):
        checked = validate_hat_chunk(chunk())
        sampled = sample_chunk(checked, 0.0, np.arange(6) / 50.0)
        np.testing.assert_allclose(
            sampled["root_pos"][:, 0], np.arange(6) / 50.0, atol=1e-6
        )
        # Head is root +0.6m; torso is head -0.4m in its local z.
        np.testing.assert_allclose(sampled["torso_pos"][:, 2], 1.0, atol=1e-6)
        self.assertEqual(sampled["left_fingertip_local"].shape, (6, 5, 3))
        # The head HAT predicted is kept, and torso -> head round-trips exactly.
        np.testing.assert_allclose(sampled["head_pos"][:, 2], 1.4, atol=1e-6)
        from scalebridge.online.reference import quat_apply
        offset = np.broadcast_to([0.0, 0.0, 0.4], sampled["torso_pos"].shape)
        np.testing.assert_allclose(
            sampled["torso_pos"] + quat_apply(sampled["torso_quat_wxyz"], offset),
            sampled["head_pos"],
            atol=1e-6,
        )

    def test_invalid_shape_and_quaternion_are_rejected(self):
        bad = chunk()
        bad["left_fingertip_local"] = [[[0.0, 0.0, 0.0]]]
        with self.assertRaisesRegex(ValueError, "left_fingertip_local"):
            validate_hat_chunk(bad)
        bad = chunk()
        bad["root_quat_wxyz"][0] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            validate_hat_chunk(bad)

    def test_robot_state_rejects_wrong_hand_width(self):
        state = robot_state()
        state["left_hand_q"] = [0.0] * 12
        from scalebridge.online.protocol import validate_robot_state
        with self.assertRaisesRegex(ValueError, "left_hand_q"):
            validate_robot_state(state)


if __name__ == "__main__":
    unittest.main()
