"""Online HAT action chunks as a live HAT-4 ScaleBFM reference."""

from __future__ import annotations

import atexit
from collections import deque
import json
import os
import sys
import threading
import time

import numpy as np
import torch
from loguru import logger

from scalebridge.env.base_env import BaseEnv
from scalebridge.online.current_pose import G1CurrentPoseFK
from scalebridge.online.focus_worker import LatestFocusProcess
from scalebridge.online.protocol import (
    HAT_BODY_JOINT_NAMES,
    inspire_motor_to_hat_hand_q,
    make_robot_state,
    monotonic_ns,
    validate_hat_chunk,
)
from scalebridge.online.reference import (
    alignment_transform,
    apply_alignment,
    quat_slerp,
    sample_chunk,
)
from scalebridge.online.transport import BackgroundLatestSubscriber, LatestPublisher


ACTIVE_LINKS = (
    "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link",
)
# G1's raised-arm ready stance, measured directly from rt/lowstate while the real
# robot held the pose for 16 s (16837 samples on 2026-08-20, drift < 0.04 rad).
# Order: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
RAISED_READY_ARM_Q = {
    "left": np.array(
        [-0.491819, 0.109332, -0.026056, 0.550454, 0.034504, -0.048234, 0.032199],
        dtype=np.float32,
    ),
    "right": np.array(
        [-0.479003, 0.074063, -0.109075, 0.475867, 0.007072, 0.076193, -0.128430],
        dtype=np.float32,
    ),
}
# One colour per reference marker so the spheres can be told apart on screen.
# The head is drawn as a fifth marker: it is the target HAT actually predicts,
# and the torso marker is only the result of inverting head = torso * [0,0,0.4].
LINK_MARKER_RGBA = {
    "pelvis": (1.0, 0.2, 0.2, 1.0),                # red
    "torso_link": (1.0, 1.0, 1.0, 1.0),            # white
    "left_wrist_yaw_link": (0.2, 0.4, 1.0, 1.0),   # blue
    "right_wrist_yaw_link": (0.2, 1.0, 0.3, 1.0),  # green
}
HEAD_MARKER_RGBA = (1.0, 0.85, 0.0, 1.0)           # yellow
class _AsyncBilateralHandIK:
    """Latest-only bilateral IK worker, isolated from the 50 Hz policy loop."""

    def __init__(self, solvers):
        self.solvers = solvers
        self._state = {side: np.zeros(12, dtype=np.float32) for side in solvers}
        self._condition = threading.Condition()
        self._stop = False
        self._targets = None
        self._submitted_sequence = -1
        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="hat-hand-ik", daemon=True
        )
        self._thread.start()

    def submit(self, left, right):
        with self._condition:
            self._submitted_sequence += 1
            self._targets = {
                "lh": np.asarray(left, dtype=np.float32).copy(),
                "rh": np.asarray(right, dtype=np.float32).copy(),
            }
            self._condition.notify()

    def latest(self):
        with self._condition:
            if self._error is not None:
                error = self._error
                self._error = None
                raise RuntimeError(f"Online hand IK worker failed: {error}")
            return self._result

    def _run(self):
        completed = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._submitted_sequence > completed
                )
                if self._stop:
                    return
                sequence = self._submitted_sequence
                targets = self._targets
            try:
                solved = {}
                for side in ("lh", "rh"):
                    q, _ = self.solvers[side].solve_frame(
                        targets[side], self._state[side]
                    )
                    if not np.isfinite(q).all():
                        raise RuntimeError(f"{side} returned a non-finite target")
                    self._state[side] = q
                    solved[side] = q
                with self._condition:
                    self._result = solved
            except Exception as exc:
                with self._condition:
                    self._error = exc
            completed = sequence

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify()
        self._thread.join(timeout=2.0)


class HAT4OnlineMotionTrackingEnv(BaseEnv):
    """Consume latest-only HAT chunks while preserving ScaleBridge's 50 Hz loop."""

    def __init__(self, config, metadata_dict, device):
        self.chunk_subscriber = BackgroundLatestSubscriber(
            config.chunk_endpoint, bind=False, validator=validate_hat_chunk
        )
        self.state_publisher = LatestPublisher(config.state_endpoint, bind=True)
        self.state_sequence_id = 0
        # Exact monotonic timestamp of every RobotState sent to HAT. Chunks
        # carry source_state_sequence_id, so this compensates the complete
        # capture + inference + transport delay without a protocol change.
        self.state_timestamp_history = deque(maxlen=512)
        self.last_published_robot_state = None
        self.last_chunk_sequence_id = -1
        self.minimum_source_state_sequence_id = None
        self.last_chunk_received_ns = None
        self.chunk = None
        self.chunk_started_ns = None
        self.previous_chunk = None
        self.previous_chunk_started_ns = None
        self.blend_started_ns = None
        self.chunk_start_frame = 0
        self.chunk_phase_state = None
        self.held_sample = None
        self.last_reference_sample = None
        self.prestart_reference_sample = None
        self.prestart_last_target = None
        self.takeover_hold_dof_pos = None
        self.takeover_started_ns = None
        self.takeover_alpha = 1.0
        self.requires_restart = False
        self.awaiting_fresh_chunk = False
        self.localization_paused = False
        self.hand_ik_worker = None
        self.open_hand_fingertips = None
        self.hand_motor_to_dof12 = None
        self.focus_worker = None
        self.focus_pad_s = 1.0
        self.hand_focus_history = deque(maxlen=250)
        self._hand_state_valid = False
        self._focus_result_logged = False
        self._closed_online = False
        self.alignment_anchor = None
        self._logged_protocol_origin = False
        self._prestart_link_poses = None
        self._prestart_debug_counter = 0
        self._marker_accepts_rgba = None
        self._root_transition_active = False
        self._root_transition_index = 0
        atexit.register(self.close_online)
        super().__init__(config, metadata_dict, device)

    def close_online(self):
        if self._closed_online:
            return
        self._closed_online = True
        if self.hand_ik_worker is not None:
            self.hand_ik_worker.close()
        if self.focus_worker is not None:
            self.focus_worker.close()
        self.chunk_subscriber.close()
        self.state_publisher.close()

    def _setup_metadata(self):
        self.reference_forcing = bool(self.cfg.get("reference_forcing", False))
        self.metadata_dict["enable_root_localization"] = not self.reference_forcing
        self.chunk_alignment = str(self.cfg.get("chunk_alignment", "fixed"))
        if self.chunk_alignment not in ("fixed", "follow"):
            raise ValueError(
                f"chunk_alignment must be 'fixed' or 'follow', got {self.chunk_alignment!r}"
            )
        logger.info(
            f"[Online HAT] Chunk alignment mode: {self.chunk_alignment} "
            + ("(anchor locks at start; heading errors self-correct)."
               if self.chunk_alignment == "fixed"
               else "(re-anchors to the measured root on every chunk).")
        )
        self.chunk_phase_alignment = bool(
            self.cfg.get("chunk_phase_alignment", True)
        )
        self.phase_rotation_weight = max(
            0.0, float(self.cfg.get("phase_rotation_weight", 0.08))
        )
        self.phase_velocity_weight = max(
            0.0, float(self.cfg.get("phase_velocity_weight", 0.03))
        )
        self.phase_measured_weight = max(
            0.0, float(self.cfg.get("phase_measured_weight", 0.15))
        )
        self.phase_max_root_jump_m = max(
            0.0, float(self.cfg.get("phase_max_root_jump_m", 0.20))
        )
        logger.info(
            "[Online HAT] Source-timestamp chunk alignment "
            + (f"enabled (same-time overlap blend="
               f"{float(self.cfg.get('chunk_blend_s', 0.22)):.2f}s)."
               if self.chunk_phase_alignment
               else "disabled; new chunks start from latency-compensated time only.")
        )
        self.future_frame_offset = torch.as_tensor(
            self.cfg.future_idx, dtype=torch.long, device=self.device
        )
        self.future_offsets_s = np.asarray(self.cfg.future_idx, dtype=np.float64) / 50.0
        self.reference_play_gate = bool(
            self.cfg.simulator.config.get("reference_play_gate", True)
        )
        self.reference_play_control = self.cfg.simulator.config.get(
            "reference_play_control", "R1"
        )
        self.reference_playing = not self.reference_play_gate
        selected = tuple(self.metadata_dict["selected_body_names"])
        missing = [name for name in ACTIVE_LINKS if name not in selected]
        if missing:
            raise KeyError(f"HAT4 metadata is missing active links: {missing}")
        self.active_indices = {name: selected.index(name) for name in ACTIVE_LINKS}
        # raised: hold the fixed real-G1 raised-arm ready pose before R1.
        # stand:  hold whatever pose calibration measured (arms stay down) --
        #         the pre-raised-arm behaviour, for A/B when the raised hold
        #         is unstable on hardware.
        self.prestart_arm_pose = str(self.cfg.get("prestart_arm_pose", "raised"))
        if self.prestart_arm_pose not in ("raised", "stand"):
            raise ValueError(
                f"prestart_arm_pose={self.prestart_arm_pose!r}; "
                "expected 'raised' or 'stand'"
            )
        # Same key and default the proven GMT env reads: blend the root
        # observation from the held local root to the live tracker over this
        # many steps at R1 (and after a Vive recovery) instead of a step jump.
        self.local_to_global_transition_steps = max(
            0,
            int(self.cfg.simulator.config.get(
                "tracker_local_to_global_transition_steps", 25
            )),
        )
        if not self.reference_forcing:
            logger.info(
                "[Online HAT] Root obs local hold before R1; local-to-global "
                f"blend over {self.local_to_global_transition_steps} steps."
            )
        # Closed-loop wrist landing compensation, ported from the offline env
        # (validated 2026-08-22: FOCUS global wrist error 5.2 -> 2.3cm mean).
        # Only the winning "offset" variant is ported; root_override doubled the
        # root drift offline and is deliberately not available online.  Unlike
        # the always-on offline A/B version, this one is gated by the FOCUS
        # window with a fade, as prescribed for deployment.
        self.wrist_compensation = str(self.cfg.get("wrist_compensation", "none"))
        if self.wrist_compensation not in ("none", "offset"):
            raise ValueError(
                f"wrist_compensation={self.wrist_compensation!r}; online supports "
                "'none' or 'offset' (root_override was rejected offline)"
            )
        # gain 0 = feedforward (offset = clamped root error, ~20% effective);
        # gain > 0 integrates the measured world wrist residual instead.
        self.wrist_compensation_gain = float(
            self.cfg.get("wrist_compensation_gain", 0.1)
        )
        self.wrist_compensation_clamp_m = float(
            self.cfg.get("wrist_compensation_clamp_m", 0.15)
        )
        self.wrist_compensation_fade_s = float(
            self.cfg.get("wrist_compensation_fade_s", 0.4)
        )
        self._wrist_offset = torch.zeros(2, 3)
        self._comp_fade_alpha = 0.0
        self._comp_last = None
        self._comp_log_file = None
        if self.wrist_compensation != "none":
            logger.info(
                f"[Online HAT] Wrist compensation: {self.wrist_compensation} "
                f"(gain={self.wrist_compensation_gain:.2f}, "
                f"clamp={self.wrist_compensation_clamp_m:.3f}m, "
                f"fade={self.wrist_compensation_fade_s:.2f}s, FOCUS-gated)"
            )

    @staticmethod
    def _quat_pitch_roll_deg(quat_wxyz):
        w, x, y, z = [float(v) for v in quat_wxyz]
        pitch = np.degrees(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
        roll = np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
        return pitch, roll

    def _log_prestart_hold_error(self, measured_root_pos):
        """Every ~2s of hold, log the obs error the policy is actually fed."""
        self._prestart_debug_counter += 1
        if self._prestart_debug_counter % 100 != 1:
            return
        sample = self.prestart_reference_sample
        if sample is None:
            return
        ref_quat = torch.as_tensor(sample["root_quat_wxyz"][0], dtype=torch.float32)
        meas_quat = self.state_buffer["root_quat_wxyz_buffer"][0, -1].detach().cpu()
        # Rotation error inv(q_ref) * q_meas, expressed as pitch/roll.
        w1, x1, y1, z1 = [float(v) for v in ref_quat]
        w2, x2, y2, z2 = [float(v) for v in meas_quat]
        err = (
            w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
            w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2,
            w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2,
            w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2,
        )
        pitch_deg, roll_deg = self._quat_pitch_roll_deg(err)
        ref_root = torch.as_tensor(sample["root_pos"][0], dtype=torch.float32)
        drift = (measured_root_pos[0].detach().cpu() - ref_root).numpy()
        logger.info(
            "[Online HAT] Pre-start hold obs error: rot pitch "
            f"{pitch_deg:+.2f} deg roll {roll_deg:+.2f} deg, root drift "
            f"[{drift[0]:+.3f} {drift[1]:+.3f} {drift[2]:+.3f}] m "
            "(all should stay near zero while standing)."
        )

    def _make_prestart_reference(self, root_pos, root_quat, live_first):
        del live_first  # Live HAT must not move the robot before SPACE.
        root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
        # The observations rotate every target by the FULL measured root quat
        # (components.py: quat_apply_inverse / quat_mul_inverse_left on
        # root_quat_wxyz_buffer).  The reference therefore must be built with
        # that same quat.  Building it heading-only injects a constant
        # whole-body pitch/roll error equal to the measured pelvis tilt
        # (natural G1 stand pitch + Vive mounting/calibration bias): invisible
        # in MuJoCo (~level truth quat), but on the real robot the policy
        # rotates the body to cancel it and tips over backwards.
        root_quat = np.asarray(root_quat, dtype=np.float32).reshape(4)
        pitch_deg, roll_deg = self._quat_pitch_roll_deg(root_quat)
        logger.info(
            "[Online HAT] Pre-start measured pelvis tilt: "
            f"pitch {pitch_deg:+.2f} deg, roll {roll_deg:+.2f} deg "
            "(reference is built with this same quat, so the hold obs error "
            "starts at zero)."
        )
        if abs(pitch_deg) > 15.0 or abs(roll_deg) > 15.0:
            logger.warning(
                "[Online HAT] Measured pelvis tilt exceeds 15 deg -- check the "
                "Vive waist tracker mounting/calibration before pressing R1."
            )
        ready_dof = (
            self.state_buffer["dof_pos_buffer"][0, -1]
            .detach().cpu().numpy().astype(np.float32).copy()
        )
        if self.prestart_arm_pose == "raised":
            joint_names = list(self.metadata_dict["joint_names"])
            for side, arm_q in RAISED_READY_ARM_Q.items():
                names = [
                    f"{side}_shoulder_pitch_joint",
                    f"{side}_shoulder_roll_joint",
                    f"{side}_shoulder_yaw_joint",
                    f"{side}_elbow_joint",
                    f"{side}_wrist_roll_joint",
                    f"{side}_wrist_pitch_joint",
                    f"{side}_wrist_yaw_joint",
                ]
                ready_dof[[joint_names.index(name) for name in names]] = arm_q
        ready = self.current_pose_fk.compute(
            root_pos, root_quat, ready_dof
        )
        torso_pos, torso_quat = self.current_pose_fk.last_body_poses["torso_link"]
        # Freeze the whole self-consistent ready pose.  Before R1 every
        # non-task link reference is served from this snapshot instead of the
        # live tracker FK, so Vive jitter/drift cannot leak into the standing
        # hold observations -- the same fully-static local hold the GMT env
        # stands on.
        self._prestart_link_poses = {
            name: (np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32))
            for name, (pos, quat) in self.current_pose_fk.last_body_poses.items()
        }
        logger.info(
            "[Online HAT] Pre-start reference uses the fixed "
            + ("real-G1 raised-arm pose" if self.prestart_arm_pose == "raised"
               else "as-measured stand pose (arms stay down)")
            + "; hands remain open until SPACE."
        )
        return {
            "root_pos": root_pos[None],
            "root_quat_wxyz": root_quat[None],
            "head_pos": ready["head_pos_world"][None],
            "head_quat_wxyz": ready["head_quat_world_wxyz"][None],
            "left_wrist_pos": ready["left_wrist_pos_world"][None],
            "left_wrist_quat_wxyz": ready["left_wrist_quat_world_wxyz"][None],
            "right_wrist_pos": ready["right_wrist_pos_world"][None],
            "right_wrist_quat_wxyz": ready["right_wrist_quat_world_wxyz"][None],
            "torso_pos": torso_pos.astype(np.float32)[None],
            "torso_quat_wxyz": torso_quat.astype(np.float32)[None],
            # Ready means open hands.  The first live HAT chunk is deliberately
            # not used here: it is inferred before SPACE and must not be able
            # to close the fingers while the body is holding the proven frame.
            "left_fingertip_local": self.open_hand_fingertips["lh"][None].copy(),
            "right_fingertip_local": self.open_hand_fingertips["rh"][None].copy(),
            "focus_phase": np.array([[0.0, 2.0]], dtype=np.float32),
        }

    def _setup_state_manager(self):
        super()._setup_state_manager()
        token_count = len(self.future_offsets_s)
        link_count = len(self.metadata_dict["selected_body_names"])
        self.state_buffer["body_pos_w_future"] = torch.zeros(
            1, token_count, link_count, 3, device=self.device
        )
        quat = torch.zeros(1, token_count, link_count, 4, device=self.device)
        quat[..., 0] = 1.0
        self.state_buffer["body_quat_w_wxyz_future"] = quat
        self.state_buffer["future_frame_offset"] = self.future_frame_offset[None, :, None]
        focus = torch.empty(1, token_count, 2, device=self.device)
        focus[..., 0] = 0.0
        focus[..., 1] = 2.0
        self.state_buffer["focus_phase"] = focus
        # Neutral until the asynchronous grasp-derived result for a chunk is ready.
        self._setup_hand_ik()
        self._setup_current_pose_fk()
        self._gather_reference_state(monotonic_ns())

    def _setup_current_pose_fk(self):
        xml_path = str(self.cfg.simulator.config.asset.xml_path)
        self.current_pose_fk = G1CurrentPoseFK(
            xml_path,
            self.metadata_dict["joint_names"],
            head_offset=(0.0, 0.0, 0.4),
            link_names=self.metadata_dict["selected_body_names"],
        )
        logger.info(
            "[Online HAT] Current global head/wrist FK enabled from measured G1 state."
        )

    def _setup_hand_ik(self):
        self.hand_ik_solvers = None
        if not hasattr(self.simulator, "set_hand_ik_target"):
            logger.warning("[Online HAT] Simulator has no Inspire hand target interface.")
            return
        config = self.cfg.simulator.config
        module_root = str(config.get(
            "hand_ik_module_root", "/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM"
        ))
        if module_root not in sys.path:
            sys.path.insert(0, module_root)
        os.environ["SHENGYIN_SKIP_ISAACGYM_IMPORT"] = "1"
        from human_policy.twist_hand_gmt_bridge import _PinocchioInspireIK
        def make_solver(side):
            return _PinocchioInspireIK(
                side,
                iters=int(config.get("hand_ik_iters", 20)),
                damping=float(config.get("hand_ik_damping", 1e-3)),
                step=float(config.get("hand_ik_step", 0.7)),
                smooth_w=float(config.get("hand_ik_smooth_weight", 1e-2)),
                reg_w=float(config.get("hand_ik_reg_weight", 1e-4)),
            )

        self.hand_ik_solvers = {side: make_solver(side) for side in ("lh", "rh")}
        self.open_hand_fingertips = {
            side: solver._fk_tips(np.zeros(12, dtype=np.float64)).astype(np.float32)
            for side, solver in self.hand_ik_solvers.items()
        }
        self._reset_hand_ik()
        mapping_path = str(config.get(
            "hand_mapping_path", "/home/nerv/qingyaoxu/TWIST2/deploy_real"
        ))
        if mapping_path not in sys.path:
            sys.path.insert(0, mapping_path)
        from cmd_to_dofpos import cmd6_to_dofpos12
        from focus_definition import FOCUS_PAD_S
        # cmd_to_dofpos uses TWIST2 joint order.  Pinocchio's hand model uses
        # index, middle, pinky, ring, then thumb.
        twist2_from_ik = np.array(
            [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5], dtype=np.int64
        )
        ik_from_twist2 = np.argsort(twist2_from_ik)
        self.hand_motor_to_dof12 = lambda motor: cmd6_to_dofpos12(
            np.clip(np.asarray(motor, dtype=np.float64), 0.0, 1.0) * 1000.0
        )[ik_from_twist2]
        self.focus_pad_s = float(FOCUS_PAD_S)
        solver_kwargs = {
            "iters": int(config.get("hand_ik_iters", 20)),
            "damping": float(config.get("hand_ik_damping", 1e-3)),
            "step": float(config.get("hand_ik_step", 0.7)),
            "smooth_w": float(config.get("hand_ik_smooth_weight", 1e-2)),
            "reg_w": float(config.get("hand_ik_reg_weight", 1e-4)),
        }
        self.focus_worker = LatestFocusProcess(
            module_root=module_root,
            mapping_path=mapping_path,
            solver_kwargs=solver_kwargs,
            focus_pad_s=self.focus_pad_s,
        )
        logger.info("[Online HAT] Warm-started bilateral Pinocchio hand IK enabled.")
        logger.info(
            "[Online HAT] Grasp-derived FOCUS enabled from canonical definition."
        )

    def _reset_hand_ik(self):
        if self.hand_ik_worker is not None:
            self.hand_ik_worker.close()
            self.hand_ik_worker = None
        if self.hand_ik_solvers is not None:
            self.hand_ik_worker = _AsyncBilateralHandIK(self.hand_ik_solvers)

    def _measured_hand_state(self):
        self._hand_state_valid = False
        if hasattr(self.simulator, "get_hand_motor_state"):
            left, right = self.simulator.get_hand_motor_state()
            left = np.asarray(left, dtype=np.float32)
            right = np.asarray(right, dtype=np.float32)
            if left.shape == (6,) and right.shape == (6,):
                self._hand_state_valid = bool(
                    np.isfinite(left).all() and np.isfinite(right).all()
                )
                if not self._hand_state_valid:
                    left = right = np.ones(6, dtype=np.float32)
                keypoints = {}
                for side, motor in (("lh", left), ("rh", right)):
                    q12 = self.hand_motor_to_dof12(motor)
                    tips = self.hand_ik_solvers[side]._fk_tips(q12).astype(np.float32)
                    keypoints[side] = np.concatenate(
                        (np.zeros((1, 3), dtype=np.float32), tips), axis=0
                    )
                return (
                    inspire_motor_to_hat_hand_q(left), keypoints["lh"],
                    inspire_motor_to_hat_hand_q(right), keypoints["rh"],
                )
        # The hand interface should exist for HAT-4.  Use an explicit open-hand
        # state as the safe fallback, which is close to the training mean.
        zero_palm = np.zeros((1, 3), dtype=np.float32)
        return (
            np.ones(6, dtype=np.float32),
            np.concatenate((zero_palm, self.open_hand_fingertips["lh"]), axis=0),
            np.ones(6, dtype=np.float32),
            np.concatenate((zero_palm, self.open_hand_fingertips["rh"]), axis=0),
        )

    def _arm_q(self, side):
        names = [
            f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint", f"{side}_elbow_joint",
            f"{side}_wrist_roll_joint", f"{side}_wrist_pitch_joint",
            f"{side}_wrist_yaw_joint",
        ]
        all_names = self.metadata_dict["joint_names"]
        indices = [all_names.index(name) for name in names]
        return self.state_buffer["dof_pos_buffer"][0, -1, indices].detach().cpu().numpy()

    def _body_q19(self):
        """Legs, waist and left arm in G1 29-DOF standard order.

        Indexed by name because ``dof_pos_buffer`` follows the simulator's own
        joint list, which is not guaranteed to match the standard order.
        """
        all_names = self.metadata_dict["joint_names"]
        indices = [all_names.index(name) for name in HAT_BODY_JOINT_NAMES]
        return self.state_buffer["dof_pos_buffer"][0, -1, indices].detach().cpu().numpy()

    def _focus_history_snapshot(self, now_ns, fps):
        count = max(1, int(round(self.focus_pad_s * float(fps))))
        query_ns = now_ns - (
            np.arange(count, 0, -1, dtype=np.float64) / float(fps) * 1e9
        )
        if not self.hand_focus_history:
            return np.zeros((count, 12), dtype=np.float32)
        timestamps = np.asarray(
            [item[0] for item in self.hand_focus_history], dtype=np.float64
        )
        values = np.stack(
            [item[1] for item in self.hand_focus_history], axis=0
        ).astype(np.float32)
        if len(values) == 1:
            return np.repeat(values, count, axis=0)
        return np.stack(
            [np.interp(query_ns, timestamps, values[:, channel]) for channel in range(12)],
            axis=1,
        ).astype(np.float32)

    def _submit_chunk_focus(self, chunk, now_ns):
        if self.focus_worker is None:
            return
        self.focus_worker.submit(
            chunk["sequence_id"],
            chunk["left_fingertip_local"],
            chunk["right_fingertip_local"],
            self._focus_history_snapshot(now_ns, chunk["fps"]),
            chunk["fps"],
        )

    def _update_chunk_focus(self):
        if self.focus_worker is None:
            return
        result = self.focus_worker.latest()
        if result is None:
            return
        sequence_id, phase = result
        candidates = [self.chunk, self.previous_chunk]
        seen = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if candidate is not None and candidate["sequence_id"] == sequence_id:
                if phase.shape != (candidate["frame_count"], 2):
                    raise RuntimeError(
                        f"FOCUS result has wrong shape {phase.shape} for chunk {sequence_id}"
                    )
                candidate["focus_phase"] = phase
                if not self._focus_result_logged:
                    logger.info(
                        f"[Online HAT] Canonical FOCUS result attached to chunk "
                        f"{sequence_id} ({int(phase[:, 0].sum())}/"
                        f"{len(phase)} focused frames)."
                    )
                    self._focus_result_logged = True
                return

    def _publish_robot_state(self):
        left_hand, left_keypoints, right_hand, right_keypoints = (
            self._measured_hand_state()
        )
        timestamp_ns = monotonic_ns()
        if self._hand_state_valid:
            closure = 1.0 - np.clip(
                np.concatenate((left_hand, right_hand)), 0.0, 1.0
            )
            self.hand_focus_history.append((timestamp_ns, closure.astype(np.float32)))
        root_pos = self.state_buffer["root_pos_buffer"][0, -1].detach().cpu().numpy()
        root_quat = self.state_buffer["root_quat_wxyz_buffer"][0, -1].detach().cpu().numpy()
        dof_pos = self.state_buffer["dof_pos_buffer"][0, -1].detach().cpu().numpy()
        link_poses = self.current_pose_fk.compute(root_pos, root_quat, dof_pos)
        message = make_robot_state(
            sequence_id=self.state_sequence_id,
            timestamp_ns=timestamp_ns,
            root_pos_world=root_pos,
            root_quat_world_wxyz=root_quat,
            **link_poses,
            body_q19=self._body_q19(),
            left_arm_q=self._arm_q("left"),
            left_hand_q=left_hand,
            left_hand_keypoints_local=left_keypoints,
            right_arm_q=self._arm_q("right"),
            right_hand_q=right_hand,
            right_hand_keypoints_local=right_keypoints,
        )
        if hasattr(self.simulator, "record_hat_robot_state"):
            self.simulator.record_hat_robot_state(message)
        self.state_timestamp_history.append((self.state_sequence_id, timestamp_ns))
        self.last_published_robot_state = message
        self.state_publisher.send(message)
        self.state_sequence_id += 1

    def _source_state_timestamp_ns(self, sequence_id):
        """Return the capture timestamp for a chunk's source observation."""
        sequence_id = int(sequence_id)
        for candidate_id, timestamp_ns in reversed(self.state_timestamp_history):
            if candidate_id == sequence_id:
                return timestamp_ns
            if candidate_id < sequence_id:
                break
        return None

    @staticmethod
    def _quat_angle(a, b):
        """Shortest quaternion geodesic angle in radians."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        dots = np.abs(np.sum(a * b, axis=-1))
        return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))

    def _latency_start_frame(self, chunk, now_ns):
        """SONIC-style stale-prefix estimate from source observation time."""
        source_timestamp_ns = self._source_state_timestamp_ns(
            chunk["source_state_sequence_id"]
        )
        if source_timestamp_ns is not None:
            age_s = max(0.0, (now_ns - source_timestamp_ns) * 1e-9)
            source = "timestamp"
        else:
            # Compatibility fallback: state_sequence_id advances with the
            # 50 Hz policy loop. generated_at_ns would omit inference latency.
            latest_sent = max(0, self.state_sequence_id - 1)
            lag = max(0, latest_sent - int(chunk["source_state_sequence_id"]))
            age_s = lag * float(self.dt)
            source = "sequence"
        frame = int(np.rint(age_s * float(chunk["fps"])))
        return int(np.clip(frame, 0, chunk["frame_count"] - 1)), age_s, source

    def _phase_match_start_frame(self, chunk, now_ns):
        """Select the source-timestamp frame and diagnose its outgoing seam.

        A receding-horizon chunk owns its own time axis.  Pose is useful for
        continuity diagnostics, but never overrides that clock; overlapping
        chunks are made continuous later by same-time sampling and blending.
        """
        latency_frame, source_age_s, time_source = self._latency_start_frame(
            chunk, now_ns
        )
        measured = self.last_published_robot_state
        outgoing = None
        outgoing_frame = None
        if (
            getattr(self, "reference_playing", False)
            and not getattr(self, "localization_paused", False)
            and not getattr(self, "awaiting_fresh_chunk", False)
            and getattr(self, "chunk", None) is not None
            and getattr(self, "chunk_started_ns", None) is not None
        ):
            old_elapsed_s = self._elapsed(now_ns, self.chunk_started_ns)
            outgoing_frame = old_elapsed_s * float(self.chunk["fps"])
            # Two samples provide both the exact outgoing pose and its tangent.
            outgoing = sample_chunk(
                self.chunk,
                old_elapsed_s,
                np.array([0.0, 1.0 / float(self.chunk["fps"])], dtype=np.float64),
            )

        if not self.chunk_phase_alignment or (measured is None and outgoing is None):
            return latency_frame, {
                "mode": "latency_only",
                "expected_frame": latency_frame,
                "selected_frame": latency_frame,
                "source_age_s": source_age_s,
                "time_source": time_source,
                "pose_cost": None,
                "transition_valid": True,
            }

        # Every HAT inference is a fresh receding-horizon prediction whose
        # frame zero belongs to its own source-state timestamp.  Frame numbers
        # therefore are *not* episode phases shared by consecutive chunks:
        # old frame 80 may correspond to new frame 1.  Select the new chunk's
        # timestamp-derived stale-prefix frame; the outgoing reference below
        # is only a continuity diagnostic and emergency corruption guard.  In
        # particular, never copy/clamp the old frame number into the new chunk;
        # doing so maps old frame 99 to new frame 99 and holds forever.
        search_center = latency_frame
        mode = "measured_pose" if outgoing_frame is None else "time_aligned_reference"

        # Execute the prediction that belongs to *now*.  Pose matching is kept
        # below as a diagnostic, not as a second clock: allowing it to choose a
        # different index makes repeated poses ambiguous and can repeatedly
        # rewind or fast-forward a receding-horizon stream.  Consecutive chunks
        # are made continuous by sampling both on their source timestamps and
        # blending their overlapping predictions in _sample_reference.
        lower = upper = search_center
        indices = np.asarray([search_center], dtype=np.int64)
        pose_fields = (
            ("root", "root_pos_world", "root_quat_world_wxyz", 2.0),
            ("head", "head_pos_world", "head_quat_world_wxyz", 1.0),
            ("left_wrist", "left_wrist_pos_world", "left_wrist_quat_world_wxyz", 0.5),
            ("right_wrist", "right_wrist_pos_world", "right_wrist_quat_world_wxyz", 0.5),
        )

        def pose_cost(target_getter):
            position_sq = np.zeros(len(indices), dtype=np.float64)
            rotation_sq = np.zeros(len(indices), dtype=np.float64)
            weight_sum = 0.0
            for prefix, state_pos, state_quat, weight in pose_fields:
                target_pos, target_quat = target_getter(
                    prefix, state_pos, state_quat
                )
                pos_error = np.linalg.norm(
                    np.asarray(chunk[f"{prefix}_pos"])[indices] - target_pos,
                    axis=-1,
                )
                rot_error = self._quat_angle(
                    np.asarray(chunk[f"{prefix}_quat_wxyz"])[indices], target_quat
                )
                position_sq += weight * pos_error * pos_error
                rotation_sq += weight * rot_error * rot_error
                weight_sum += weight
            return (
                np.sqrt(position_sq / weight_sum),
                np.sqrt(rotation_sq / weight_sum),
            )

        measured_position = measured_rotation = np.zeros(len(indices), dtype=np.float64)
        if measured is not None:
            measured_position, measured_rotation = pose_cost(
                lambda _prefix, state_pos, state_quat: (
                    np.asarray(measured[state_pos], dtype=np.float64),
                    np.asarray(measured[state_quat], dtype=np.float64),
                )
            )

        reference_position = reference_rotation = np.zeros(len(indices), dtype=np.float64)
        if outgoing is not None:
            reference_position, reference_rotation = pose_cost(
                lambda prefix, _state_pos, _state_quat: (
                    np.asarray(outgoing[f"{prefix}_pos"][0], dtype=np.float64),
                    np.asarray(outgoing[f"{prefix}_quat_wxyz"][0], dtype=np.float64),
                )
            )

        velocity_rms = np.zeros(len(indices), dtype=np.float64)
        velocity_target = outgoing
        if velocity_target is None:
            previous = getattr(self, "last_reference_sample", None)
            if previous is not None and all(
                f"{prefix}_pos" in previous for prefix, *_ in pose_fields
            ):
                velocity_target = previous
        if velocity_target is not None:
            velocity_sq = np.zeros(len(indices), dtype=np.float64)
            velocity_weight_sum = 0.0
            target_dt = (
                1.0 / float(self.chunk["fps"])
                if outgoing is not None
                else float(self.future_offsets_s[1] - self.future_offsets_s[0])
            )
            next_indices = np.minimum(indices + 1, chunk["frame_count"] - 1)
            for prefix, _, _, weight in pose_fields:
                key = f"{prefix}_pos"
                old_velocity = (
                    np.asarray(velocity_target[key][1], dtype=np.float64)
                    - np.asarray(velocity_target[key][0], dtype=np.float64)
                ) / target_dt
                values = np.asarray(chunk[key], dtype=np.float64)
                new_velocity = (
                    values[next_indices] - values[indices]
                ) * float(chunk["fps"])
                velocity_sq += weight * np.sum(
                    (new_velocity - old_velocity) ** 2, axis=-1
                )
                velocity_weight_sum += weight
            velocity_rms = np.sqrt(velocity_sq / velocity_weight_sum)

        if outgoing is None:
            primary_position = measured_position
            primary_rotation = measured_rotation
            consistency = 0.0
        else:
            primary_position = reference_position
            primary_rotation = reference_rotation
            consistency = self.phase_measured_weight * (
                measured_position + self.phase_rotation_weight * measured_rotation
            )
        total = (
            primary_position
            + self.phase_rotation_weight * primary_rotation
            + consistency
            + self.phase_velocity_weight * velocity_rms
        )

        valid = np.ones(len(indices), dtype=bool)
        root_jump = np.zeros(len(indices), dtype=np.float64)
        backtrack = np.zeros(len(indices), dtype=np.float64)
        if outgoing is not None:
            outgoing_root = np.asarray(outgoing["root_pos"][0], dtype=np.float64)
            candidate_roots = np.asarray(chunk["root_pos"])[indices]
            root_delta = candidate_roots - outgoing_root
            root_jump = np.linalg.norm(root_delta, axis=-1)
            tangent = (
                np.asarray(outgoing["root_pos"][1], dtype=np.float64)
                - outgoing_root
            )
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-5:
                backtrack = root_delta @ (tangent / tangent_norm)
            # This is only an emergency corruption guard.  Ordinary HAT
            # re-prediction differences (including legitimate return motion)
            # are handled by time-aligned overlap blending instead of being
            # rejected and leaving the old chunk frozen at its endpoint.
            valid &= root_jump <= self.phase_max_root_jump_m

        if np.any(valid):
            masked_total = np.where(valid, total, np.inf)
            best_offset = int(np.argmin(masked_total))
            transition_valid = True
        else:
            best_offset = int(np.argmin(total))
            transition_valid = outgoing is None
        selected = int(indices[best_offset])
        return selected, {
            "mode": mode,
            "expected_frame": latency_frame,
            "search_center_frame": search_center,
            "outgoing_frame": None if outgoing_frame is None else float(outgoing_frame),
            "selected_frame": selected,
            "search_range": [int(lower), int(upper)],
            "source_age_s": source_age_s,
            "time_source": time_source,
            "pose_cost": float(primary_position[best_offset]),
            "rotation_cost_rad": float(primary_rotation[best_offset]),
            "measured_pose_cost": float(measured_position[best_offset]),
            "velocity_cost_mps": float(velocity_rms[best_offset]),
            "root_jump_m": float(root_jump[best_offset]),
            "backtrack_m": float(backtrack[best_offset]),
            "total_cost": float(total[best_offset]),
            "transition_valid": bool(transition_valid),
        }

    def _accept_latest_chunk(self, now_ns):
        receive_error = self.chunk_subscriber.receive_error()
        if receive_error is not None:
            logger.warning(f"[Online HAT] Rejected invalid chunk: {receive_error}")
        raw = self.chunk_subscriber.receive_latest()
        if raw is None:
            return False
        try:
            candidate = raw
            if candidate["sequence_id"] <= self.last_chunk_sequence_id:
                return False
            if (
                self.minimum_source_state_sequence_id is not None
                and candidate["source_state_sequence_id"]
                < self.minimum_source_state_sequence_id
            ):
                logger.warning(
                    "[Online HAT] Dropped chunk predating calibration or Vive recovery."
                )
                return False
            max_lag = int(self.cfg.get("max_source_state_lag", 30))
            if candidate["source_state_sequence_id"] < self.state_sequence_id - max_lag:
                logger.warning("[Online HAT] Dropped chunk generated from stale robot state.")
                return False
            if self.localization_paused:
                logger.warning("[Online HAT] Dropped chunk received during Vive outage.")
                return False
            root_pos = self.state_buffer["root_pos_buffer"][0, -1].detach().cpu().numpy()
            root_quat = self.state_buffer["root_quat_wxyz_buffer"][0, -1].detach().cpu().numpy()
            if "position_origin_world" in candidate:
                # dex5/HAT position channels already share the robot's world
                # axes; only the fixed opening-head translation was subtracted
                # during preprocessing.  Restore that exact translation and
                # leave world rotations unchanged.  Aligning predicted frame
                # zero to the measured root would erase the model's immediate
                # global displacement command (13 cm in the 16:32 real run).
                align_pos = np.asarray(
                    candidate["position_origin_world"], dtype=np.float32
                )
                align_quat = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float32
                )
                if not self._logged_protocol_origin:
                    logger.info(
                        "[Online HAT] Restoring HAT global positions from "
                        f"protocol head origin {align_pos.tolist()}; frame 0 "
                        "is not re-anchored to the measured robot root."
                    )
                    self._logged_protocol_origin = True
            else:
                # Backward compatibility for recordings/fake producers created
                # before position_origin_world was added to HatChunkV4.
                # "fixed": while holding pre-start the anchor follows the
                # measured root, then freezes once the reference plays.
                if (
                    self.chunk_alignment == "fixed"
                    and self.reference_playing
                    and self.alignment_anchor is not None
                ):
                    align_pos, align_quat = self.alignment_anchor
                else:
                    align_pos, align_quat = alignment_transform(
                        candidate, root_pos, root_quat
                    )
                    if self.chunk_alignment == "fixed":
                        self.alignment_anchor = (align_pos, align_quat)
            aligned = apply_alignment(candidate, align_pos, align_quat)
        except Exception as exc:
            logger.warning(f"[Online HAT] Rejected invalid chunk: {exc}")
            return False

        reset_transition = self.awaiting_fresh_chunk or not self.reference_playing
        try:
            start_frame, phase_state = self._phase_match_start_frame(aligned, now_ns)
        except Exception as exc:
            # A first/recovery chunk may safely fall back to latency. During
            # continuous playback, never let a matching error rewind the active
            # reference: keep executing the old chunk and await another one.
            start_frame, source_age_s, time_source = self._latency_start_frame(
                aligned, now_ns
            )
            continuous = (
                self.reference_playing
                and not reset_transition
                and self.chunk is not None
                and self.chunk_started_ns is not None
            )
            phase_state = {
                "mode": "match_error",
                "expected_frame": start_frame,
                "selected_frame": start_frame,
                "source_age_s": source_age_s,
                "time_source": time_source,
                "pose_cost": None,
                "fallback_error": str(exc),
                "transition_valid": not continuous,
            }
            logger.warning(
                f"[Online HAT] Chunk phase matching failed; "
                + ("keeping outgoing chunk" if continuous
                   else f"using latency frame {start_frame}")
                + f": {exc}"
            )
        aligned["execution_alignment"] = dict(phase_state)
        if not phase_state.get("transition_valid", True):
            self.last_chunk_sequence_id = candidate["sequence_id"]
            if hasattr(self.simulator, "record_hat_chunk"):
                self.simulator.record_hat_chunk(candidate, aligned)
            logger.warning(
                f"[Online HAT] Rejected emergency-discontinuous chunk "
                f"{candidate['sequence_id']}: outgoing frame "
                f"{phase_state.get('outgoing_frame')} -> candidate "
                f"{phase_state.get('selected_frame')}, root_jump="
                f"{phase_state.get('root_jump_m')}m. The accepted-stream "
                f"staleness watchdog remains active."
            )
            return False
        if reset_transition:
            self.previous_chunk = None
            self.previous_chunk_started_ns = None
        else:
            self.previous_chunk = self.chunk
            self.previous_chunk_started_ns = self.chunk_started_ns
        self.chunk = aligned
        self.chunk_start_frame = start_frame
        self.chunk_phase_state = phase_state
        # The source RobotState timestamp is the chunk's clock origin.  Keeping
        # that exact origin means old frame 11 and new frame 1 are sampled for
        # the same wall-clock target time during overlap; a rounded/matched
        # frame index must never become a replacement clock.
        source_timestamp_ns = self._source_state_timestamp_ns(
            candidate["source_state_sequence_id"]
        )
        self.chunk_started_ns = (
            int(source_timestamp_ns)
            if source_timestamp_ns is not None
            else now_ns - int(phase_state["source_age_s"] * 1e9)
        )
        self.blend_started_ns = now_ns if self.previous_chunk is not None else None
        self.last_chunk_sequence_id = candidate["sequence_id"]
        self.last_chunk_received_ns = now_ns
        self.held_sample = None
        self.awaiting_fresh_chunk = False
        self.minimum_source_state_sequence_id = None
        if hasattr(self.simulator, "record_hat_chunk"):
            self.simulator.record_hat_chunk(candidate, aligned)
        self._submit_chunk_focus(self.chunk, now_ns)
        logger.info(
            f"[Online HAT] Accepted chunk {self.last_chunk_sequence_id} "
            f"({candidate['frame_count']} frames @ {candidate['fps']:.1f} Hz): "
            f"{phase_state.get('mode')} frame {start_frame}, "
            f"source_age={phase_state['source_age_s']:.3f}s, "
            f"root_jump={phase_state.get('root_jump_m')}m, "
            f"backtrack={phase_state.get('backtrack_m')}m, "
            f"pose_cost={phase_state['pose_cost']}."
        )
        return True

    def _elapsed(self, now_ns, started_ns):
        if started_ns is None or not self.reference_playing or self.localization_paused:
            return 0.0
        return max(0.0, (now_ns - started_ns) * 1e-9)

    @staticmethod
    def _repeat_current(sample, count):
        return {
            key: np.repeat(np.asarray(value)[0:1], count, axis=0)
            for key, value in sample.items()
        }

    def _sample_reference(self, now_ns):
        if self.chunk is None:
            root_pos = self.state_buffer["root_pos_buffer"][0, -1].detach().cpu().numpy()
            root_quat = self.state_buffer["root_quat_wxyz_buffer"][0, -1].detach().cpu().numpy()
            count = len(self.future_offsets_s)
            zeros = np.zeros((count, 5, 3), dtype=np.float32)
            return {
                "root_pos": np.repeat(root_pos[None], count, axis=0),
                "root_quat_wxyz": np.repeat(root_quat[None], count, axis=0),
                "left_wrist_pos": np.repeat(root_pos[None], count, axis=0),
                "left_wrist_quat_wxyz": np.repeat(root_quat[None], count, axis=0),
                "right_wrist_pos": np.repeat(root_pos[None], count, axis=0),
                "right_wrist_quat_wxyz": np.repeat(root_quat[None], count, axis=0),
                "torso_pos": np.repeat(root_pos[None], count, axis=0),
                "torso_quat_wxyz": np.repeat(root_quat[None], count, axis=0),
                "left_fingertip_local": zeros.copy(),
                "right_fingertip_local": zeros.copy(),
                "focus_phase": np.tile(
                    np.array([[0.0, 2.0]], dtype=np.float32), (count, 1)
                ),
            }
        if (
            self.reference_play_gate
            and not self.reference_playing
            and self.prestart_reference_sample is not None
        ):
            # Match the proven real-world R1 hold: every future token is the
            # same fixed frame-0 reference.  Do not expose future motion or let
            # newly replanned HAT chunks move the ready pose before SPACE.
            return self._repeat_current(
                self.prestart_reference_sample, len(self.future_offsets_s)
            )
        if self.held_sample is not None:
            return self._repeat_current(self.held_sample, len(self.future_offsets_s))
        current = sample_chunk(
            self.chunk, self._elapsed(now_ns, self.chunk_started_ns), self.future_offsets_s
        )
        if (
            self.previous_chunk is not None
            and self.previous_chunk_started_ns is not None
            and self.blend_started_ns is not None
        ):
            previous = sample_chunk(
                self.previous_chunk,
                self._elapsed(now_ns, self.previous_chunk_started_ns),
                self.future_offsets_s,
            )
            # Full-chunk finger IK returns asynchronously.  Until the new
            # canonical phase is ready, the overlapping old chunk is a much
            # better estimate than injecting a synthetic neutral [0, 2].
            if (
                "focus_phase" not in self.chunk
                and "focus_phase" in self.previous_chunk
            ):
                current["focus_phase"] = previous["focus_phase"]
            blend_s = float(self.cfg.get("chunk_blend_s", 0.22))
            alpha = min(max((now_ns - self.blend_started_ns) * 1e-9 / max(blend_s, 1e-6), 0.0), 1.0)
            if alpha < 1.0:
                for key in current:
                    if key == "focus_phase":
                        continue
                    if "quat" in key:
                        current[key] = quat_slerp(previous[key], current[key], np.full(len(self.future_offsets_s), alpha)).astype(np.float32)
                    else:
                        current[key] = (previous[key] * (1.0 - alpha) + current[key] * alpha).astype(np.float32)
            elif (
                "focus_phase" in self.chunk
                or "focus_phase" not in self.previous_chunk
            ):
                self.previous_chunk = None
                self.previous_chunk_started_ns = None
        return current

    def _gather_reference_state(self, now_ns):
        # Grab the measured root before the forcing block at the end of this
        # method can overwrite it: with a Vive (or MuJoCo truth) backend the
        # buffer top always holds this step's measured pose here, because
        # _update_state_manager ran first in _compute_observation.
        measured_root_pos = self.state_buffer["root_pos_buffer"][:, -1].clone()
        sample = self._sample_reference(now_ns)
        self.last_reference_sample = sample
        count = len(self.future_offsets_s)
        link_count = len(self.metadata_dict["selected_body_names"])
        selected_names = list(self.metadata_dict["selected_body_names"])
        if (
            self.reference_play_gate
            and not self.reference_playing
            and self._prestart_link_poses is not None
        ):
            # Holding pre-start (or latched after a stale stream): the entire
            # reference must be the static snapshot, not live tracker FK.
            measured = self._prestart_link_poses
            self._log_prestart_hold_error(measured_root_pos)
        else:
            measured = self.current_pose_fk.last_body_poses
        if all(name in measured for name in selected_names):
            # HAT predicts four task links.  Every other ScaleBFM target must be
            # a real link pose, not a copy of the pelvis.  Using current FK for
            # those unconstrained links gives them zero tracking error while
            # leaving the HAT pelvis/torso/wrist objectives untouched.
            measured_pos = np.stack(
                [measured[name][0] for name in selected_names], axis=0
            ).astype(np.float32)
            measured_quat = np.stack(
                [measured[name][1] for name in selected_names], axis=0
            ).astype(np.float32)
            pos = np.repeat(measured_pos[None], count, axis=0)
            quat = np.repeat(measured_quat[None], count, axis=0)
        else:
            # Only reachable during object construction, before the first
            # measured state has passed through current_pose_fk.compute().
            pos = np.repeat(sample["root_pos"][:, None, :], link_count, axis=1)
            quat = np.repeat(sample["root_quat_wxyz"][:, None, :], link_count, axis=1)
        mapping = {
            "pelvis": ("root_pos", "root_quat_wxyz"),
            "left_wrist_yaw_link": ("left_wrist_pos", "left_wrist_quat_wxyz"),
            "right_wrist_yaw_link": ("right_wrist_pos", "right_wrist_quat_wxyz"),
            "torso_link": ("torso_pos", "torso_quat_wxyz"),
        }
        for link, (pos_key, quat_key) in mapping.items():
            index = self.active_indices[link]
            pos[:, index] = sample[pos_key]
            quat[:, index] = sample[quat_key]
        self.state_buffer["body_pos_w_future"] = torch.from_numpy(pos).float().to(self.device).unsqueeze(0)
        self.state_buffer["body_quat_w_wxyz_future"] = torch.from_numpy(quat).float().to(self.device).unsqueeze(0)
        self.state_buffer["focus_phase"] = (
            torch.from_numpy(sample["focus_phase"]).float().to(self.device).unsqueeze(0)
        )
        self._apply_wrist_compensation(measured_root_pos)
        self._update_root_observation(measured_root_pos)

    def _update_root_observation(self, measured_root_pos):
        """Local root hold before R1 and the blended local-to-global handover.

        Mirrors the GMT env's tracker_hold_root_mode=local_then_global: while
        holding (forcing / pre-R1 / localization outage) the root obs is the
        reference root, and when playback (re)starts it walks to the live
        tracker over local_to_global_transition_steps instead of jumping.
        """
        pelvis = self.active_indices["pelvis"]
        if self.reference_forcing or not self.reference_playing or self.localization_paused:
            self.state_buffer["root_pos_buffer"][:, -1] = self.state_buffer["body_pos_w_future"][:, 0, pelvis]
        elif self._root_transition_active:
            local_root = self.state_buffer["body_pos_w_future"][:, 0, pelvis].to(
                device=measured_root_pos.device, dtype=measured_root_pos.dtype
            )
            alpha = min(
                1.0,
                float(self._root_transition_index + 1)
                / max(self.local_to_global_transition_steps, 1),
            )
            self.state_buffer["root_pos_buffer"][:, -1] = (
                (1.0 - alpha) * local_root + alpha * measured_root_pos
            )
            self._root_transition_index += 1
            if alpha >= 1.0:
                self._root_transition_active = False

    def _measured_wrist_pos_world(self):
        """Measured world wrist positions from the same chain the state uses.

        current_pose_fk.compute() ran this step in _publish_robot_state, fed by
        the measured root (Vive on the real robot, MuJoCo truth in sim) and the
        measured joints -- exactly the T_pelvis * FK(q) virtual wrist tracker.
        """
        poses = getattr(getattr(self, "current_pose_fk", None), "last_body_poses", None)
        if not poses:
            return None
        try:
            left = poses["left_wrist_yaw_link"][0]
            right = poses["right_wrist_yaw_link"][0]
        except KeyError:
            return None
        return torch.tensor(
            np.stack((left, right)), dtype=torch.float32
        )

    def _apply_wrist_compensation(self, measured_root_pos):
        """FOCUS-gated wrist world-target offset (mutates body_pos_w_future).

        The offset integrates the measured-wrist-vs-original-target residual
        while grasping (the policy only responds ~0.2x to an inconsistent
        wrist target, so an integrator is required), and the applied value is
        faded in and out of the FOCUS window so the reference never jumps.
        Root and torso targets are never touched: the root loop stays alive.
        Always records a _comp_last snapshot (also in mode none) so the trace
        doubles as the A/B baseline.
        """
        self._comp_last = None
        if (
            not self.reference_playing
            or self.localization_paused
            or self.requires_restart
            or self.awaiting_fresh_chunk
            or self.held_sample is not None
        ):
            # Frozen or held references must not wind up the integrator.
            self._comp_fade_alpha = 0.0
            self._wrist_offset = torch.zeros(2, 3)
            return

        body_pos = self.state_buffer["body_pos_w_future"]
        wrist_indices = [
            self.active_indices["left_wrist_yaw_link"],
            self.active_indices["right_wrist_yaw_link"],
        ]
        ref_root = body_pos[0, 0, self.active_indices["pelvis"]].clone()
        ref_wrists = body_pos[0, 0, wrist_indices].clone()
        measured = measured_root_pos[0]
        error = ref_root - measured
        focus = float(self.state_buffer["focus_phase"][0, 0, 0].item())

        rate = self.dt / max(self.wrist_compensation_fade_s, self.dt)
        target = 1.0 if focus >= 0.5 else 0.0
        delta = max(-rate, min(rate, target - self._comp_fade_alpha))
        self._comp_fade_alpha = min(1.0, max(0.0, self._comp_fade_alpha + delta))
        alpha = self._comp_fade_alpha

        offsets = torch.zeros(2, 3)
        measured_wrists = None
        if self.wrist_compensation == "offset":
            clamp = self.wrist_compensation_clamp_m
            if self.wrist_compensation_gain > 0.0:
                measured_wrists = self._measured_wrist_pos_world()
                if measured_wrists is not None and focus >= 0.5:
                    # Integrate only inside the window; hold (do not decay) the
                    # accumulated offset across brief window exits so re-entry
                    # does not restart the 0.5-1s convergence from zero.
                    self._wrist_offset = (
                        self._wrist_offset
                        + self.wrist_compensation_gain
                        * (ref_wrists.cpu() - measured_wrists)
                    ).clamp(-clamp, clamp)
                offsets = self._wrist_offset.clone()
            else:
                offsets = error.cpu().clamp(-clamp, clamp).expand(2, 3).clone()
            if alpha > 0.0:
                applied = (alpha * offsets).to(body_pos.device, body_pos.dtype)
                body_pos[:, :, wrist_indices] += applied[None, None, :, :]

        self._comp_last = {
            "focus": focus,
            "mode": self.wrist_compensation,
            "alpha": alpha,
            "meas_root": measured.tolist(),
            "ref_root": ref_root.tolist(),
            "root_error": error.tolist(),
            "applied_offset_lw": (alpha * offsets[0]).tolist(),
            "applied_offset_rw": (alpha * offsets[1]).tolist(),
            "ref_lw": ref_wrists[0].tolist(),
            "ref_rw": ref_wrists[1].tolist(),
            "cmd_lw": body_pos[0, 0, wrist_indices[0]].tolist(),
            "cmd_rw": body_pos[0, 0, wrist_indices[1]].tolist(),
        }
        if measured_wrists is None:
            measured_wrists = self._measured_wrist_pos_world()
        if measured_wrists is not None:
            self._comp_last["meas_lw"] = measured_wrists[0].tolist()
            self._comp_last["meas_rw"] = measured_wrists[1].tolist()

    def _log_wrist_compensation(self):
        record = self._comp_last
        if record is None:
            return
        if self._comp_log_file is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                out_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                out_dir = "."
            path = os.path.join(out_dir, "wrist_compensation_trace.jsonl")
            self._comp_log_file = open(path, "a", buffering=1)
            logger.info(f"[Online HAT] Wrist compensation trace: {path}")
        self._comp_log_file.write(json.dumps(record) + "\n")

    def _update_markers(self, obs):
        """Draw the tracked reference links plus the head target behind them.

        ``torso_link`` is not something HAT predicts; it is ``head_pos`` pushed
        back down the head offset.  Drawing the head as well makes that mapping
        directly visible instead of implied.
        """
        names = list(self.metadata_dict["selected_body_names"])
        marker_indices = [names.index(name) for name in ACTIVE_LINKS]
        pos = obs["body_pos_w_future"][0, 0, marker_indices]
        rgba = [LINK_MARKER_RGBA[name] for name in ACTIVE_LINKS]
        sample = self.last_reference_sample
        if sample is not None and "head_pos" in sample:
            head = torch.from_numpy(np.asarray(sample["head_pos"][0], dtype=np.float32))
            pos = torch.cat((pos, head.to(pos.device).unsqueeze(0)), dim=0)
            rgba.append(HEAD_MARKER_RGBA)
        # Only the MuJoCo backend accepts colours; BaseSimulator (real robot)
        # takes marker_pos alone and crashed on the extra argument.
        if self._marker_accepts_rgba is None:
            import inspect

            self._marker_accepts_rgba = (
                "rgba"
                in inspect.signature(self.simulator.update_marker_pos).parameters
            )
        if self._marker_accepts_rgba:
            self.simulator.update_marker_pos(pos, rgba)
        else:
            self.simulator.update_marker_pos(pos)

    def _send_hand_target(self):
        if self.hand_ik_worker is None or self.last_reference_sample is None:
            return
        solved = self.hand_ik_worker.latest()
        if solved is not None:
            self.simulator.set_hand_ik_target(solved["lh"], solved["rh"])
        self.hand_ik_worker.submit(
            self.last_reference_sample["left_fingertip_local"][0],
            self.last_reference_sample["right_fingertip_local"][0],
        )

    def _update_localization(self):
        robust = (
            not self.reference_forcing
            and hasattr(self.simulator, "robust_tracking_enabled")
            and self.simulator.robust_tracking_enabled()
        )
        if not robust or not self.reference_playing:
            return
        healthy = self.simulator.is_localization_healthy()
        if not healthy and not self.localization_paused:
            held_sample = self._sample_reference(monotonic_ns())
            self.localization_paused = True
            self.awaiting_fresh_chunk = True
            self.held_sample = held_sample
            logger.warning("[Online HAT] Vive unavailable: holding reference and discarding chunks.")
        elif healthy and self.localization_paused:
            self.localization_paused = False
            self.awaiting_fresh_chunk = True
            # _compute_observation publishes this next sequence after recovery;
            # reject anything inferred from outage-era state.
            self.minimum_source_state_sequence_id = self.state_sequence_id
            # The root obs was frozen on the held reference during the outage;
            # blend it back to the live tracker instead of jumping.
            self._root_transition_active = self.local_to_global_transition_steps > 0
            self._root_transition_index = 0
            logger.info("[Online HAT] Vive recovered: waiting for a fresh HAT chunk.")

    def _update_staleness(self, now_ns):
        if self.last_chunk_received_ns is None:
            return
        age_s = (now_ns - self.last_chunk_received_ns) * 1e-9
        soft_s = float(self.cfg.get("chunk_hold_timeout_s", 0.5))
        hard_s = float(self.cfg.get("chunk_latch_timeout_s", 2.0))
        if age_s >= soft_s and self.held_sample is None:
            self.held_sample = self._sample_reference(now_ns)
            logger.warning("[Online HAT] Chunk stream stale: freezing the current reference.")
        if age_s >= hard_s and not self.requires_restart:
            self.requires_restart = True
            self.reference_playing = False
            self.chunk_started_ns = None
            logger.error(
                f"[Online HAT] No chunk for {age_s:.2f}s: latched hold; "
                f"press {self.reference_play_control} after recovery."
            )

    def _wait_for_first_chunk(self):
        deadline = time.monotonic() + float(self.cfg.get("startup_chunk_timeout_s", 15.0))
        logger.info("[Online HAT] Waiting for the first valid action chunk ...")
        while time.monotonic() < deadline:
            self._publish_robot_state()
            now_ns = monotonic_ns()
            if self._accept_latest_chunk(now_ns):
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for the first valid HAT action chunk")

    def _wait_for_initial_focus(self):
        if self.focus_worker is None or self.chunk is None:
            return
        deadline = time.monotonic() + float(
            self.cfg.get("startup_focus_timeout_s", 3.0)
        )
        while time.monotonic() < deadline:
            self._update_chunk_focus()
            if "focus_phase" in self.chunk:
                return
            time.sleep(0.005)
        raise RuntimeError(
            "Timed out deriving canonical FOCUS for the first HAT action chunk"
        )

    def reset(self):
        self.episode_length_buf.zero_()
        self.action.zero_()
        self.state_buffer["action_buffer"].zero_()
        self.prestart_reference_sample = None
        self.prestart_last_target = None
        self.takeover_hold_dof_pos = None
        self.takeover_started_ns = None
        self.takeover_alpha = 0.0 if self.reference_play_gate else 1.0
        self.simulator.calibrate({})
        current = self.simulator.refresh_sim()
        for key, value in current.items():
            self.state_buffer[f"{key}_buffer"].copy_(
                torch.broadcast_to(value, self.state_buffer[f"{key}_buffer"].shape)
            )
        self.reference_playing = not self.reference_play_gate
        self.requires_restart = False
        self.localization_paused = False
        self._wrist_offset = torch.zeros(2, 3)
        self._comp_fade_alpha = 0.0
        self._comp_last = None
        self._root_transition_active = False
        self._root_transition_index = 0
        self._prestart_debug_counter = 0
        self._reset_hand_ik()
        self.hand_focus_history.clear()
        if self.focus_worker is not None:
            self.focus_worker.latest()
        # Discard pre-calibration predictions and demand one based on the newly
        # measured robot state.
        self.chunk = None
        self.previous_chunk = None
        self.chunk_phase_state = None
        self.chunk_start_frame = 0
        self.alignment_anchor = None
        self._logged_protocol_origin = False
        self.last_chunk_received_ns = None
        self.awaiting_fresh_chunk = True
        self.minimum_source_state_sequence_id = self.state_sequence_id
        self.chunk_subscriber.receive_latest()
        self._wait_for_first_chunk()
        self._wait_for_initial_focus()
        if self.reference_play_gate:
            first = sample_chunk(
                self.chunk,
                0.0,
                np.zeros(1, dtype=np.float64),
            )
            self.prestart_reference_sample = self._make_prestart_reference(
                current["root_pos"].detach().cpu().numpy(),
                current["root_quat_wxyz"].detach().cpu().numpy(),
                first,
            )
        self._gather_reference_state(monotonic_ns())
        self._send_hand_target()
        logger.info(
            f"[Online HAT] READY: policy is holding the fixed real-G1 raised-arm pose; "
            f"press {self.reference_play_control} to start."
            if self.reference_play_gate else "[Online HAT] Online reference started."
        )
        return self._update_observation_manager()

    def _select_applied_target(self, requested, now_ns):
        """Keep policy balance active before SPACE, then blend to live HAT."""
        if not self.reference_play_gate:
            return requested, 1.0
        if not self.reference_playing:
            return requested, 0.0
        blend_s = float(self.cfg.get("takeover_blend_s", 0.75))
        if (
            self.takeover_started_ns is None
            or self.takeover_hold_dof_pos is None
            or blend_s <= 0.0
        ):
            return requested, 1.0
        alpha = min(max((now_ns - self.takeover_started_ns) * 1e-9 / blend_s, 0.0), 1.0)
        hold = self.takeover_hold_dof_pos.to(
            device=requested.device, dtype=requested.dtype
        )
        return hold * (1.0 - alpha) + requested * alpha, alpha

    def _trace_policy_step(self, requested, applied, action):
        if not hasattr(self.simulator, "record_hat_online_step"):
            return
        obs = self._update_observation_manager()
        selected_obs = {
            key: obs[key]
            for key in (
                "target_body_pos_future_to_robot_base",
                "target_body_rot_future_to_robot_base",
                "focus_phase",
            )
            if key in obs
        }
        self.simulator.record_hat_online_step(
            requested_target_dof_pos=requested,
            applied_target_dof_pos=applied,
            policy_action=action,
            reference_sample=self.last_reference_sample,
            policy_observation=selected_obs,
            chunk_sequence_id=self.last_chunk_sequence_id,
            reference_playing=self.reference_playing,
            takeover_alpha=self.takeover_alpha,
            localization_paused=self.localization_paused,
        )

    def _compute_observation(self):
        self._update_state_manager()
        self._publish_robot_state()
        now_ns = monotonic_ns()
        self._update_chunk_focus()
        self._accept_latest_chunk(now_ns)
        self._update_chunk_focus()
        self._update_staleness(now_ns)
        self._gather_reference_state(now_ns)
        return self._update_observation_manager()

    def step(self, tgt_dof_pos, action):
        now_ns = monotonic_ns()
        if (
            self.reference_play_gate
            and not self.reference_playing
            and self.chunk is not None
            and self.last_chunk_received_ns is not None
            and (now_ns - self.last_chunk_received_ns) * 1e-9 < float(self.cfg.get("chunk_hold_timeout_s", 0.5))
            and self.simulator.consume_reference_play_request()
        ):
            self.takeover_hold_dof_pos = (
                tgt_dof_pos if self.prestart_last_target is None
                else self.prestart_last_target
            ).detach().clone()
            self.reference_playing = True
            self.requires_restart = False
            self.held_sample = None
            self.takeover_started_ns = now_ns
            # Same handover the GMT env does at R1: the root obs was held on
            # the local pre-start reference, walk it to the live tracker.
            self._root_transition_active = (
                self.local_to_global_transition_steps > 0
                and not self.reference_forcing
            )
            self._root_transition_index = 0
            logger.info(f"[Online HAT] {self.reference_play_control} pressed: starting closed loop.")
        self._update_localization()
        self._send_hand_target()
        applied_target, self.takeover_alpha = self._select_applied_target(
            tgt_dof_pos, now_ns
        )
        if not self.reference_playing:
            self.prestart_last_target = tgt_dof_pos.detach().clone()
        # The policy target is applied both before and after SPACE, matching the
        # real-world R1 hold, so its action history must remain closed-loop too.
        self.action.copy_(action)
        if (
            not hasattr(self.simulator, "record_hat_online_step")
            and hasattr(self.simulator, "record_policy_step")
        ):
            self.simulator.record_policy_step(
                applied_target.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                self.localization_paused,
            )
        self.simulator.apply_action(applied_target.detach().cpu().numpy())
        self._trace_policy_step(tgt_dof_pos, applied_target, action)
        if self.reference_playing and not self.localization_paused and not self.awaiting_fresh_chunk:
            self.episode_length_buf += 1
        obs = self._compute_observation()
        self._log_wrist_compensation()
        self._update_markers(obs)
        return obs
