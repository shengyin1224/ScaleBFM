from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger

from scalebridge.env.motion_tracking_xsens import MotionTrackingXsensEnv
from scalebridge.online.current_pose import G1CurrentPoseFK
from scalebridge.utils.pico_dataloader import PicoOnlineClient
from scalebridge.utils.torch_utils import (
    calc_heading_quat,
    calc_heading_quat_inv,
    quat_apply_inverse,
    quat_mul,
)


class MotionTrackingPicoEnv(MotionTrackingXsensEnv):
    """ScaleBFM teleoperation driven by PICO instead of Xsens.

    Everything downstream of the mocap source is inherited unchanged from
    ``MotionTrackingXsensEnv``: the same receding-horizon reference sampling,
    the same calibration and heading alignment, the same observation assembly.
    Only the source of the 14 link poses differs.

    The reference is deliberately delayed by ``future_idx[-1]`` policy steps
    (200 ms at the shipped settings) so that the policy's "future" frames are
    human motion that has already been observed rather than an extrapolation.
    That latency is the price of the lookahead; it is inherited from the Xsens
    path, not introduced here.

    Pause/resume: L1 on the Unitree remote, or PICO A+B+X+Y, toggles tracking.
    While paused the reference is frozen, so the robot balances in place at the
    last commanded pose. On resume the incoming stream is re-anchored --
    position and heading -- onto the frozen reference root, so the operator can
    walk around, come back in any orientation, and resume without the robot
    jumping. Episode recording uses Unitree remote B/X edges; Pico B/X are
    reserved for the four-button safety combo and never start recording.
    """

    def _setup_metadata(self):
        super()._setup_metadata()
        self.pico_endpoint = str(self.cfg.get("pico_endpoint", "tcp://127.0.0.1:5571"))
        self.pico_timeout_s = float(self.cfg.get("pico_timeout_s", 120.0))
        self.pico_stamp_on_recv = bool(self.cfg.get("pico_stamp_on_recv", False))
        self.hat4_mode = bool(self.cfg.get("hat4_mode", False))
        self._paused = False
        self._pico_prev_controls = None
        # The Xsens base env never implemented the reference_play_gate that
        # motion_tracking.py has, so tracking used to start the instant the
        # second R2 press returned from calibration. Honor the gate here: hold
        # the Skill BFM stand target until R1 (or L1) releases it.
        self._play_gate = bool(self.cfg.simulator.config.get("reference_play_gate", False))
        self._initial_hold_done = False
        # Match Skill BFM's proven pre-start mode: hold the robot's measured
        # stand pose (arms down), rather than the operator's Pico pose.
        self.prestart_arm_pose = str(self.cfg.get("prestart_arm_pose", "stand"))
        if self.prestart_arm_pose != "stand":
            raise ValueError(
                "motion_tracking_pico currently supports prestart_arm_pose='stand' only"
            )
        if self.has_hand:
            raise ValueError(
                "motion_tracking_pico supports has_hand=False only; the PICO sender "
                "does not publish hand qpos. Drive the hands through the existing "
                "glove pipeline instead."
            )

    def _setup_state_manager(self):
        super()._setup_state_manager()
        self.current_pose_fk = G1CurrentPoseFK(
            self.cfg.simulator.config.asset.xml_path,
            self.metadata_dict["joint_names"],
            link_names=self.metadata_dict["selected_body_names"],
        )
        self._prestart_link_poses = None
        if self.hat4_mode:
            # Pico has no FOCUS annotations.  HAT-4's neutral phase means the
            # policy still receives its required two extra task dimensions,
            # without inventing a wrist-contact event from tracker motion.
            self.state_buffer["focus_phase"] = torch.zeros(
                1, int(self.future_frame_offset.numel()), 2, device=self.device
            )
            self.state_buffer["focus_phase"][..., 1] = 2.0

    def _build_prestart_stand(self):
        """Build the as-calibrated, arms-down stand used by Skill BFM."""
        current = self.simulator.refresh_sim()
        # RealWorld omits root_pos when reference_forcing=true.  MuJoCo's
        # free-joint translation is the pelvis height itself, so use the same
        # calibrated asset height instead of accidentally building a stand at
        # z=0.
        root_pos = current.get("root_pos")
        if root_pos is None:
            root_pos = torch.tensor(
                [0.0, 0.0, float(self.cfg.simulator.config.asset.root_height)],
                dtype=torch.float32,
            )
        root_pos = root_pos.detach().cpu().numpy()
        root_quat = current["root_quat_wxyz"].detach().cpu().numpy()
        dof_pos = current["dof_pos"].detach().cpu().numpy()
        self.current_pose_fk.compute(root_pos, root_quat, dof_pos)
        self._prestart_link_poses = {
            name: (
                torch.from_numpy(pos).to(self.device),
                torch.from_numpy(quat).to(self.device),
            )
            for name, (pos, quat) in self.current_pose_fk.last_body_poses.items()
        }
        missing = [
            name for name in self.metadata_dict["selected_body_names"]
            if name not in self._prestart_link_poses
        ]
        if missing:
            raise RuntimeError(f"Stand FK is missing selected links: {missing}")

    def _hold_prestart_stand(self):
        selected = self.metadata_dict["selected_body_names"]
        pos = torch.stack([self._prestart_link_poses[name][0] for name in selected])
        quat = torch.stack([self._prestart_link_poses[name][1] for name in selected])
        self.body_pos_w_future[:] = pos[None, None]
        self.body_quat_w_wxyz_future[:] = quat[None, None]
        self.state_buffer.update({
            "body_pos_w_future": self.body_pos_w_future,
            "body_quat_w_wxyz_future": self.body_quat_w_wxyz_future,
        })
        if self.reference_forcing:
            self.state_buffer["root_pos_buffer"][:, -1] = pos[0]

    def _calibrate(self):
        self._paused = False
        self._initial_hold_done = False
        self._pico_prev_controls = None
        self._pico_prev_combo = False
        super()._calibrate()
        self._build_prestart_stand()

    def reset(self):
        # BaseEnv.reset returns before the Xsens subclass gets a chance to
        # gather a reference. Gather once here so the first policy observation
        # is already the static stand target, not zero-filled placeholders.
        obs = super().reset()
        if self._play_gate:
            self._gather_reference_state()
            obs = self._update_observation_manager()
        self._latest_policy_observation = obs
        return obs

    def _launch_xsens(self) -> None:
        """Override the Xsens TCP handshake with the PICO ZMQ subscription."""
        if self._xsens_client is not None:
            self._xsens_client.stop()

        num_bodies = len(self.metadata_dict["selected_body_names"])
        self._xsens_client = PicoOnlineClient(
            endpoint=self.pico_endpoint,
            expected_num_bodies=num_bodies,
            # [LT, LG, RT, RG, A, B, X, Y] from the sender.  The body policy
            # remains 29-DOF; this side channel is for hands and controls.
            num_hand_qpos=8,
            stamp_on_recv=self.pico_stamp_on_recv,
        )

        logger.info(f"[PICO] Subscribing to {self.pico_endpoint} ({num_bodies} links) ...")
        self._xsens_client.start()

        logger.info(
            "[PICO] Waiting for the first retargeted frame. Start "
            "pico_scalebfm_sender.py now if it is not already running."
        )
        if not self._xsens_client.wait_first_frame(timeout_s=self.pico_timeout_s):
            raise RuntimeError(
                f"PICO: no frames on {self.pico_endpoint} within {self.pico_timeout_s}s. "
                "Check that pico_scalebfm_sender.py is publishing (run it with "
                "--dry-run first to confirm the PICO stream itself is alive)."
            )
        logger.info("[PICO] Streaming ready.")

        if hasattr(self.simulator, "left_upper_switch_pressed"):
            logger.info("[PICO] Pause/resume available: press L1 to toggle tracking.")
        else:
            logger.info(
                "[PICO] Pause/resume needs the remote (real robot only); "
                "in MuJoCo stop with Ctrl-C."
            )

    def step(self, tgt_dof_pos, action):
        """Apply one Pico target and mirror it into the optional recorder.

        The inherited Xsens step predates the Pico dataset recorder and only
        publishes the target.  Keep its tracking behavior identical while
        adding the recorder hook before the command is sent.
        """
        self.action.copy_(action)

        tgt_dof_pos = tgt_dof_pos.detach().cpu().numpy()
        if self.has_hand:
            hand_qpos = self._xsens_client.buffer.latest()[3][None, :]
            tgt_dof_pos = np.concatenate([tgt_dof_pos, hand_qpos], axis=-1)

        if hasattr(self.simulator, "record_reference_target"):
            policy_obs = getattr(self, "_latest_policy_observation", {})
            self.simulator.record_reference_target(
                self.body_pos_w_future[0].detach().cpu().numpy(),
                self.body_quat_w_wxyz_future[0].detach().cpu().numpy(),
                policy_body_pos=policy_obs.get(
                    "target_body_pos_future_to_robot_base"
                ),
                policy_body_quat=policy_obs.get(
                    "target_body_rot_future_to_robot_base"
                ),
                future_time_offsets=policy_obs.get("future_time_offsets"),
                focus_phase=policy_obs.get("focus_phase"),
            )
        if hasattr(self.simulator, "record_policy_step"):
            self.simulator.record_policy_step(
                tgt_dof_pos,
                action.detach().cpu().numpy(),
                getattr(self, "localization_paused", False),
            )
        self.simulator.apply_action(tgt_dof_pos)

        self.episode_length_buf += 1

        obs_dict = self._compute_observation()
        self._latest_policy_observation = obs_dict

        obs_last = obs_dict["body_pos_w_future"][0, -1].cpu()
        self.simulator.update_marker_pos(obs_last)

        return obs_dict

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def _gather_reference_state(self):
        self._poll_pause_toggle()

        if self._paused:
            # Reference tensors keep their frozen values; only re-assert root
            # forcing, because root_pos_buffer is refreshed from the robot
            # state every step and its last entry must stay on the reference.
            if self.reference_forcing:
                self.state_buffer["root_pos_buffer"][:, -1] = self.state_buffer["body_pos_w_future"][:, 0, 0]
            if self.hat4_mode:
                self.state_buffer["focus_phase"][..., 0] = 0.0
                self.state_buffer["focus_phase"][..., 1] = 2.0
            return

        super()._gather_reference_state()

        if self.hat4_mode:
            self.state_buffer["focus_phase"][..., 0] = 0.0
            self.state_buffer["focus_phase"][..., 1] = 2.0

        if self._play_gate and not self._initial_hold_done:
            # Match the existing Skill BFM stand mode after the second R2:
            # freeze the robot's measured stand pose until R1 releases it.
            self._hold_prestart_stand()
            self._initial_hold_done = True
            self._paused = True
            logger.info(
                "[PICO] Holding prestart stand pose. Stand ready, then press R1 "
                "(or L1) to start tracking."
            )

    def _poll_pause_toggle(self) -> None:
        # L1: latched rising-edge flag maintained by RealWorld's LCM handler;
        # the MuJoCo simulator has no remote, so getattr stays 0 and it is
        # inert there. R1 additionally releases a hold (initial or L1-paused),
        # via the same consume API real_world already exposes for play gates.
        l1 = bool(getattr(self.simulator, "left_upper_switch_pressed", 0))
        if l1:
            self.simulator.left_upper_switch_pressed = 0

        # PICO face buttons travel in the optional PSB1 hand section.  A+B+X+Y
        # is intentionally mapped to the same latched pause/resume operation
        # as the Unitree remote L1.  PICO B/X never control recording; episode
        # recording uses the physical Unitree remote B/X below.
        combo = False
        latest = self._xsens_client.buffer.latest()
        if latest is not None and latest[3].shape == (8,):
            controls = np.asarray(latest[3], dtype=np.float32)
            a, b, x, y = controls[4:] > 0.5
            combo = bool(a and b and x and y)
            self._pico_prev_controls = controls.copy()
            if hasattr(self.simulator, "record_pico_controls"):
                self.simulator.record_pico_controls(controls)
        if combo and not bool(getattr(self, "_pico_prev_combo", False)):
            l1 = True
        self._pico_prev_combo = combo

        # Real Unitree B/X are carried by the spare LCM lower-switch fields:
        # left_lower_right_switch=B and right_lower_left_switch=X.  RealWorld
        # latches rising edges in *_pressed so a short button press cannot be
        # missed by the 50 Hz policy loop.
        remote_b = bool(getattr(self.simulator, "left_lower_right_switch_pressed", 0))
        remote_x = bool(getattr(self.simulator, "right_lower_left_switch_pressed", 0))
        if remote_b:
            self.simulator.left_lower_right_switch_pressed = 0
        if remote_x:
            self.simulator.right_lower_left_switch_pressed = 0
        if remote_b and hasattr(self.simulator, "record_toggle"):
            self.simulator.record_toggle()
        if remote_x and hasattr(self.simulator, "record_discard"):
            self.simulator.record_discard()

        r1 = False
        if self._paused and hasattr(self.simulator, "consume_reference_play_request"):
            r1 = bool(self.simulator.consume_reference_play_request())

        if not self._paused:
            if l1:
                self._paused = True
                logger.info("[PICO] L1/A+B+X+Y: PAUSED. Reference frozen; robot holds pose. Press R1/L1/A+B+X+Y to resume.")
        elif l1 or r1:
            self._realign_to_frozen_reference()
            self._paused = False
            logger.info("[PICO] RESUMED. Stream re-anchored to the held pose; tracking.")

    def _realign_to_frozen_reference(self) -> None:
        """Recompute the mocap alignment so the live stream continues from the
        frozen reference instead of jumping to wherever the operator is now.

        Mirrors the math in ``MotionTrackingXsensEnv._calibrate``: the mapping
        is ``ref = R_offset @ (stream - pos_offset)``. We choose the offsets so
        that the operator's *current* root lands exactly on the frozen
        reference root (xy) with the frozen reference heading.
        """
        latest = self._xsens_client.buffer.latest()
        if latest is None:
            logger.warning("[PICO] Resume requested but the buffer is empty; keeping old alignment.")
            return

        stream_root_pos = torch.from_numpy(latest[1][0].copy()).float().to(self.device)
        stream_root_pos[-1] = 0.0
        stream_root_quat = torch.from_numpy(latest[2][0].copy()).float().to(self.device)

        # Frame index 0 is the frame the robot is currently tracking (the most
        # delayed one); its root is where the robot was told to be.
        frozen_root_pos = self.state_buffer["body_pos_w_future"][0, 0, 0].clone()
        frozen_root_quat = self.state_buffer["body_quat_w_wxyz_future"][0, 0, 0].clone()

        target_heading = calc_heading_quat(frozen_root_quat)
        source_heading_inv = calc_heading_quat_inv(stream_root_quat)
        self._quat_offset.copy_(quat_mul(target_heading, source_heading_inv))

        # ref = R @ (stream - off). Solving R @ (stream_root - off) = frozen_xy
        # gives off = stream_root - R^{-1} @ frozen_xy.
        rot = self._quat_offset[0, 0, 0]
        shift = frozen_root_pos.clone()
        shift[-1] = 0.0
        self._pos_offset.copy_(stream_root_pos - quat_apply_inverse(rot, shift))
