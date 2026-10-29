import json
import os
import sys
import time

import numpy as np
import torch
from loguru import logger

from scalebridge.env.motion_tracking import MotionTrackingEnv
from scalebridge.utils.hand_ik_assets import DEFAULT_SEARCH_DIR, resolve_hand_ik_path

_DEFAULT_IK_MODULE_ROOT = "/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM"

WRIST_COMPENSATION_MODES = ("none", "offset", "root_override")


class HAT4MotionTrackingEnv(MotionTrackingEnv):
    """Offline motion tracking with the two FOCUS phase features."""

    def _setup_metadata(self):
        super()._setup_metadata()
        # A/B experiment for closing the wrist *global* landing error caused by
        # steady-state root lag (local 1.7cm vs global 4.4cm on the 600-traj
        # eval).  Both variants are position-only and leave rotations alone.
        #   offset:        every step add e = ref_root - measured_root to the
        #                  two wrist world targets (root/torso targets keep
        #                  chasing, the arm is asked to absorb the root error).
        #   root_override: inside the FOCUS window blend the pelvis reference
        #                  to the measured root position (zero root error, the
        #                  wrist world targets stay put), gated + blended so
        #                  the root loop revives outside the window.
        mode = str(self.cfg.get("wrist_compensation", "none"))
        if mode not in WRIST_COMPENSATION_MODES:
            raise ValueError(
                f"wrist_compensation={mode!r}; expected one of {WRIST_COMPENSATION_MODES}"
            )
        self.wrist_compensation = mode
        self.wrist_compensation_clamp_m = float(
            self.cfg.get("wrist_compensation_clamp_m", 0.08)
        )
        # gain 0 = pure feedforward (offset = root error).  gain > 0 integrates
        # the measured wrist-vs-target residual instead: the policy only
        # responds ~0.2x to an inconsistent wrist target, so an integrator is
        # what actually drives the landing error to zero.
        self.wrist_compensation_gain = float(
            self.cfg.get("wrist_compensation_gain", 0.0)
        )
        self.root_override_blend_s = float(self.cfg.get("root_override_blend_s", 0.4))
        self._override_alpha = 0.0
        self._wrist_offset = torch.zeros(2, 3)
        selected = list(self.metadata_dict["selected_body_names"])
        self._pelvis_body_index = selected.index("pelvis")
        self._wrist_body_indices = [
            selected.index("left_wrist_yaw_link"),
            selected.index("right_wrist_yaw_link"),
        ]
        self._comp_last = None
        self._comp_log_file = None
        if mode != "none":
            logger.info(
                f"[Env] Wrist compensation mode: {mode} "
                f"(clamp={self.wrist_compensation_clamp_m:.3f}m, "
                f"override_blend={self.root_override_blend_s:.2f}s)"
            )

    def _setup_motion(self, motion_data):
        super()._setup_motion(motion_data)
        valid = (
            "focus_mask" in motion_data
            and "focus_time_to_next_edge_s" in motion_data
            and "focus_valid" in motion_data
            and np.asarray(motion_data["focus_valid"], dtype=bool).all()
        )
        if valid:
            mask = np.asarray(motion_data["focus_mask"], dtype=np.float32)
            time_to_edge = np.clip(
                np.asarray(motion_data["focus_time_to_next_edge_s"], dtype=np.float32),
                -2.0,
                2.0,
            )
            phase = np.stack((mask, time_to_edge), axis=-1)
            logger.info("[Env] Loaded embedded FOCUS phase annotations.")
        else:
            phase = np.empty((self.motion_len, 2), dtype=np.float32)
            phase[:, 0] = 0.0
            phase[:, 1] = 2.0
            logger.warning(
                "[Env] Motion has no valid FOCUS annotations; using neutral phase [0, 2]."
            )
        self.focus_phase = torch.from_numpy(phase).to(self.device)

        self.left_hand_reference = None
        self.right_hand_reference = None
        self.hand_fps = None
        self.hand_ik_solvers = None
        self.hand_ik_targets = None
        self.hand_ik_state = None
        hand_path = self.cfg.simulator.config.get("hand_ik_path", None)
        if hand_path:
            hand_path = resolve_hand_ik_path(
                hand_path,
                self.cfg.motion_path,
                self.cfg.simulator.config.get("hand_ik_search_dir", DEFAULT_SEARCH_DIR),
            )
        if hand_path:
            hand = np.load(hand_path, allow_pickle=False)
            for key in ("fps", "lh_dof_pos", "rh_dof_pos"):
                if key not in hand:
                    raise KeyError(f"Hand IK file is missing {key}: {hand_path}")
            left = np.asarray(hand["lh_dof_pos"], dtype=np.float64)
            right = np.asarray(hand["rh_dof_pos"], dtype=np.float64)
            if left.ndim != 2 or left.shape[1] != 12 or right.shape != left.shape:
                raise ValueError(
                    f"Expected hand IK shaped (N, 12), got left={left.shape}, "
                    f"right={right.shape}"
                )
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(f"Hand IK contains NaN or infinity: {hand_path}")
            self.left_hand_reference = left
            self.right_hand_reference = right
            self.hand_fps = float(hand["fps"])
            logger.info(f"[Env] Loaded synchronized Inspire hand IK from {hand_path}")
            self._setup_realtime_hand_ik(hand, hand_path)

    def _setup_realtime_hand_ik(self, hand, hand_path):
        """Solve the packaged fingertip targets live instead of replaying dof.

        The reference NPZ carries both the fingertip trajectory and the dof
        angles a previous offline pass solved from it.  Deployment feeds
        fingertip coordinates, so prefer the solver and keep the stored dof
        only as the fallback when this is switched off.
        """
        config = self.cfg.simulator.config
        if not config.get("hand_realtime_ik", False):
            logger.info("[Env] Real-time hand IK disabled; replaying stored dof.")
            return

        required = (
            "lh_tip_world", "lh_root_pos", "lh_root_rot",
            "rh_tip_world", "rh_root_pos", "rh_root_rot",
        )
        missing = [key for key in required if key not in hand]
        if missing:
            raise KeyError(
                f"Real-time hand IK needs fingertip targets {missing} in {hand_path}. "
                "Set simulator.config.hand_realtime_ik=False to replay stored dof."
            )

        module_root = str(config.get("hand_ik_module_root", _DEFAULT_IK_MODULE_ROOT))
        if not os.path.isdir(module_root):
            raise FileNotFoundError(f"hand_ik_module_root does not exist: {module_root}")
        if module_root not in sys.path:
            sys.path.insert(0, module_root)
        # Batch IK never touches Isaac Gym, and importing it here would fight
        # the torch already loaded by ScaleBridge.
        os.environ["SHENGYIN_SKIP_ISAACGYM_IMPORT"] = "1"
        from human_policy.twist_hand_gmt_bridge import (
            _PinocchioInspireIK,
            _tips_world_to_local,
        )

        solvers = {}
        targets = {}
        for side in ("lh", "rh"):
            solvers[side] = _PinocchioInspireIK(
                side,
                iters=int(config.get("hand_ik_iters", 20)),
                damping=float(config.get("hand_ik_damping", 1e-3)),
                step=float(config.get("hand_ik_step", 0.7)),
                smooth_w=float(config.get("hand_ik_smooth_weight", 1e-2)),
                reg_w=float(config.get("hand_ik_reg_weight", 1e-4)),
            )
            local = _tips_world_to_local(
                np.asarray(hand[f"{side}_root_pos"]),
                np.asarray(hand[f"{side}_root_rot"]),
                np.asarray(hand[f"{side}_tip_world"]),
            )
            local = np.asarray(local, dtype=np.float64)
            if local.ndim != 3 or local.shape[1:] != (5, 3):
                raise ValueError(
                    f"Expected {side} fingertip targets shaped (N, 5, 3), got {local.shape}"
                )
            if not np.isfinite(local).all():
                raise ValueError(f"{side} fingertip targets contain NaN or infinity")
            targets[side] = local

        self.hand_ik_solvers = solvers
        self.hand_ik_targets = targets
        self._reset_hand_ik_state()

        # One warm-up solve keeps the first control step off the Pinocchio
        # import path and reports the real per-frame cost up front.
        start = time.perf_counter()
        self._solve_hand_ik(0.0)
        elapsed = (time.perf_counter() - start) * 1000.0
        self._reset_hand_ik_state()
        logger.info(
            "[Env] Real-time Pinocchio fingertip IK active "
            f"({solvers['lh'].iters} iters, {elapsed:.2f} ms for both hands)."
        )

    def _reset_hand_ik_state(self):
        if self.hand_ik_solvers is None:
            return
        self.hand_ik_state = {
            side: np.zeros(12, dtype=np.float32) for side in self.hand_ik_solvers
        }

    def _solve_hand_ik(self, frame):
        """Return the left/right 12-dof solution for one fingertip frame."""
        solved = {}
        for side, solver in self.hand_ik_solvers.items():
            target = self._interpolate_hand(self.hand_ik_targets[side], frame)
            # Warm-starting from the previous command is what makes a 20-iter
            # solve match the offline 120-iter result exactly.
            q, _ = solver.solve_frame(target, self.hand_ik_state[side])
            self.hand_ik_state[side] = q
            solved[side] = np.asarray(q, dtype=np.float64)
        return solved["lh"], solved["rh"]

    @staticmethod
    def _interpolate_hand(values, frame):
        frame = float(np.clip(frame, 0.0, len(values) - 1))
        lower = int(np.floor(frame))
        upper = min(lower + 1, len(values) - 1)
        alpha = frame - lower
        return values[lower] * (1.0 - alpha) + values[upper] * alpha

    def _send_hand_reference(self):
        if self.left_hand_reference is None or not hasattr(
            self.simulator, "set_hand_ik_target"
        ):
            return
        # The offline Inspire MuJoCo simulator replays its own hand IK file and
        # raises if the env pushes targets; only online-target backends accept
        # them (real_world_inspire has no hand_online_targets attribute).
        if not getattr(self.simulator, "hand_online_targets", True):
            return
        # Body trajectories are fixed at 50 Hz, while the hand IK may have a
        # different source FPS (the current pillow IK is 30 Hz).  Convert by
        # elapsed time instead of assuming frame indices match.
        body_frame = int(self.episode_length_buf.item())
        if (
            (self.reference_play_gate and not self.reference_playing)
            or self._completion_return_enabled()
        ):
            body_frame = 0
        hand_frame = body_frame * self.hand_fps / 50.0
        if self.hand_ik_solvers is None:
            left, right = (
                self._interpolate_hand(self.left_hand_reference, hand_frame),
                self._interpolate_hand(self.right_hand_reference, hand_frame),
            )
        else:
            left, right = self._solve_hand_ik(hand_frame)
        self.simulator.set_hand_ik_target(left, right)

    def _calibrate(self):
        super()._calibrate()
        # The first hand command is sent only after the second R2 completes
        # calibration.  Repeated ready-frame commands then approach frame 0
        # through the worker's rate limiter while the body waits for R1.
        self._reset_hand_ik_state()
        self._send_hand_reference()

    def step(self, tgt_dof_pos, action):
        self._send_hand_reference()
        obs_dict = super().step(tgt_dof_pos, action)
        self._log_wrist_compensation()
        return obs_dict

    def _gather_reference_state(self):
        # The base class overwrites the measured root under reference_forcing;
        # grab it first so the compensation always sees the true MuJoCo root.
        measured_root_pos = self.state_buffer["root_pos_buffer"][:, -1].clone()
        super()._gather_reference_state()
        # During a Local-to-Global transition episode_length_buf is frozen on
        # the anchored frame (0 at R1, the paused frame on a resume), so the
        # plain clamp already yields the correct focus phase in both cases.
        if (
            (self.reference_play_gate and not self.reference_playing)
            or self._completion_return_enabled()
        ):
            frame = torch.zeros_like(self.episode_length_buf)
        else:
            frame = torch.clamp(self.episode_length_buf, 0, self.motion_len - 1)
        phase = self.focus_phase.index_select(0, frame)
        self.state_buffer["focus_phase"] = phase[:, None, :].expand(
            -1, self.future_frame_offset.numel(), -1
        )
        self._apply_wrist_compensation(measured_root_pos, int(frame[0].item()))

    def _apply_wrist_compensation(self, measured_root_pos, frame_index):
        """Mutate body_pos_w_future according to the wrist_compensation mode.

        Called after the base reference gather, so under reference_forcing the
        root observation already holds the pelvis entry of body_pos_w_future;
        when root_override moves that entry it must be re-forced afterwards.
        Always records a `_comp_last` snapshot (also in mode none) so the same
        trace file serves as the A/B baseline.
        """
        self._comp_last = None
        if (
            not self.reference_playing
            or self.motion_complete
            or self._completion_return_enabled()
            or self.localization_paused
        ):
            self._override_alpha = 0.0
            self._wrist_offset = torch.zeros(2, 3)
            return

        body_pos = self.state_buffer["body_pos_w_future"]
        ref_root = body_pos[0, 0, self._pelvis_body_index].clone()
        ref_wrists = body_pos[0, 0, self._wrist_body_indices].clone()
        measured = measured_root_pos[0]
        error = ref_root - measured
        focus = float(self.focus_phase[frame_index, 0].item())

        applied_offset = torch.zeros(3, device=body_pos.device)
        offsets = torch.zeros(2, 3, device=body_pos.device)
        alpha = 0.0
        if self.wrist_compensation == "offset":
            clamp = self.wrist_compensation_clamp_m
            measured_wrists = (
                self._measured_wrist_pos_world()
                if self.wrist_compensation_gain > 0.0
                else None
            )
            if measured_wrists is not None:
                # Integrate the world-frame wrist residual so the offset grows
                # until the measured wrist lands on the original target, no
                # matter how weakly the policy responds to each extra cm.
                self._wrist_offset = (
                    self._wrist_offset
                    + self.wrist_compensation_gain * (ref_wrists.cpu() - measured_wrists)
                ).clamp(-clamp, clamp)
                offsets = self._wrist_offset.to(body_pos.device, body_pos.dtype)
            else:
                offsets = error.clamp(-clamp, clamp).expand(2, 3).clone()
            applied_offset = offsets[0]
            body_pos[:, :, self._wrist_body_indices] += offsets[None, None, :, :]
        elif self.wrist_compensation == "root_override":
            # Ramp toward 1 inside the FOCUS window and back to 0 outside so
            # the root loop is cut only while grasping, never with a step jump.
            rate = self.dt / max(self.root_override_blend_s, self.dt)
            target = 1.0 if focus >= 0.5 else 0.0
            delta = max(-rate, min(rate, target - self._override_alpha))
            self._override_alpha = min(1.0, max(0.0, self._override_alpha + delta))
            alpha = self._override_alpha
            if alpha > 0.0:
                override = measured[None, None, :].to(body_pos.dtype)
                original = body_pos[:, :, self._pelvis_body_index]
                body_pos[:, :, self._pelvis_body_index] = (
                    (1.0 - alpha) * original + alpha * override
                )
                if self.reference_forcing:
                    # Keep the forced root observation consistent with the
                    # modified pelvis reference target.
                    self.state_buffer["root_pos_buffer"][:, -1] = body_pos[:, 0, self._pelvis_body_index]

        self._comp_last = {
            "frame": frame_index,
            "focus": focus,
            "mode": self.wrist_compensation,
            "alpha": alpha,
            "meas_root": measured.tolist(),
            "ref_root": ref_root.tolist(),
            "root_error": error.tolist(),
            "applied_offset": applied_offset.tolist(),
            "applied_offset_lw": offsets[0].tolist(),
            "applied_offset_rw": offsets[1].tolist(),
            "ref_lw": ref_wrists[0].tolist(),
            "ref_rw": ref_wrists[1].tolist(),
            "cmd_lw": body_pos[0, 0, self._wrist_body_indices[0]].tolist(),
            "cmd_rw": body_pos[0, 0, self._wrist_body_indices[1]].tolist(),
        }

    def _measured_wrist_pos_world(self):
        data = getattr(getattr(self, "simulator", None), "mujoco_data", None)
        if data is None:
            return None
        return torch.tensor(
            np.stack((
                data.body("left_wrist_yaw_link").xpos,
                data.body("right_wrist_yaw_link").xpos,
            )),
            dtype=torch.float32,
        )

    def _log_wrist_compensation(self):
        record = self._comp_last
        if record is None:
            return
        data = getattr(self.simulator, "mujoco_data", None)
        if data is not None:
            record = dict(record)
            record["meas_lw"] = data.body("left_wrist_yaw_link").xpos.tolist()
            record["meas_rw"] = data.body("right_wrist_yaw_link").xpos.tolist()
            record["meas_root_now"] = data.qpos[:3].tolist()
        if self._comp_log_file is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                out_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                out_dir = "."
            path = os.path.join(out_dir, "wrist_compensation_trace.jsonl")
            self._comp_log_file = open(path, "a", buffering=1)
            logger.info(f"[Env] Wrist compensation trace: {path}")
        self._comp_log_file.write(json.dumps(record) + "\n")
