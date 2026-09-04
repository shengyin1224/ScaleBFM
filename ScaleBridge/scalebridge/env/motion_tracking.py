import os
import torch
import numpy as np
from loguru import logger
from scalebridge.env.base_env import BaseEnv
from scalebridge.utils.torch_utils import calc_heading_quat, calc_heading_quat_inv, quat_mul, quat_apply, quat_inv

class MotionTrackingEnv(BaseEnv):

    def _setup_metadata(self):
        # weishuai: We explicitly pass motion dict as arguments to support heritage on motion data processing
        assert os.path.exists(self.cfg.motion_path), f"You have to ensure the correct path to motion file. Current motion path is: {self.cfg.motion_path}."
        logger.info(f"[Env] Loading offline motion trajectory from {self.cfg.motion_path}.")
        motion_data = np.load(self.cfg.motion_path)
        assert motion_data['fps'] == 50, f"You should first process the data to ensure the same format and FPS with IsaacLab compatible format."
        self._setup_motion(motion_data)

        self.reference_forcing = self.cfg.get('reference_forcing', True)
        logger.info(f"[Env] Using reference root position to apply forcing: {self.reference_forcing}")
        self.metadata_dict["enable_root_localization"] = not self.reference_forcing

        self.reference_play_gate = self.cfg.simulator.config.get('reference_play_gate', False)
        self.reference_playing = not self.reference_play_gate
        self.reference_play_control = self.cfg.simulator.config.get('reference_play_control', 'R1')
        self.localization_paused = False
        # After the last reference frame the policy has no scripted future and
        # can slowly destabilize (observed as a sideways fall ~15s after
        # completion). Local root hold removes residual global position error;
        # hardened deployment also returns to the proven frame-0 stand target.
        self.motion_complete = False
        self.motion_complete_local_hold = bool(
            self.cfg.get('motion_complete_local_hold', True)
        )
        # 2026-08-19 hardening (single master switch): (a) resume from a
        # localization pause through the same 25-step Local-to-Global blend
        # used at R1 instead of an instant switch, (b) require a sustained
        # healthy streak before resuming so local/global cannot flap every
        # ~0.5s, (c) the Inspire backend logs hand commands and feedback, and
        # (d) return completed motions to the proven frame-0 stand target.
        self.deployment_hardening = bool(
            self.cfg.simulator.config.get('deployment_hardening', False)
        )
        self.tracker_resume_min_healthy_steps = max(
            0, int(self.cfg.simulator.config.get('tracker_resume_min_healthy_steps', 50))
        )
        self.tracker_local_to_global_transition_steps = max(
            0,
            int(self.cfg.simulator.config.get(
                'tracker_local_to_global_transition_steps', 25
            )),
        )
        self.motion_complete_return_to_start_steps = max(
            0,
            int(self.cfg.simulator.config.get(
                'motion_complete_return_to_start_steps', 100
            )),
        )
        self._completion_return_active = False
        self._completion_return_index = 0
        self._completion_return_finished = False
        self._completion_stand_body_pos = None
        self._completion_stand_body_quat = None
        self._resume_healthy_streak = 0
        if self.deployment_hardening:
            logger.info(
                f'[Env] Deployment hardening active: resume needs '
                f'{self.tracker_resume_min_healthy_steps} healthy steps, then blends '
                f'local-to-global over {self.tracker_local_to_global_transition_steps} steps; '
                f'motion completion returns to frame 0 over '
                f'{self.motion_complete_return_to_start_steps} steps.'
            )
        self.tracker_hold_root_mode = str(
            self.cfg.simulator.config.get('tracker_hold_root_mode', 'local_then_global')
        )
        valid_hold_modes = {'local_then_global', 'global_from_calibration'}
        if self.tracker_hold_root_mode not in valid_hold_modes:
            raise ValueError(
                f"Unsupported tracker_hold_root_mode={self.tracker_hold_root_mode!r}; "
                f"expected one of {sorted(valid_hold_modes)}"
            )
        logger.info(
            f"[Env] Tracker hold root mode: {self.tracker_hold_root_mode}"
        )
        self._root_transition_active = False
        self._root_transition_index = 0
        if self.reference_play_gate:
            logger.info(
                f'[Env] Reference play gate enabled: hold the initial pose until '
                f'{self.reference_play_control} is pressed.'
            )

    def _global_time_indices(self, frame_offsets):
        if self.reference_play_gate and not self.reference_playing:
            # Repeat the initial pose across the complete future horizon. Merely
            # freezing episode_length_buf is insufficient because non-zero future
            # offsets would still reveal the upcoming motion to the policy.
            return torch.zeros(
                (self.episode_length_buf.shape[0], frame_offsets.numel()),
                dtype=torch.long,
                device=self.device,
            )
        if self._root_transition_active:
            # Keep the policy on the CURRENT reference frame while the root
            # observation is handed from Local hold to calibrated Global. At
            # R1 the episode buffer is zero, so this matches the original
            # frame-0 hold; on a mid-motion resume it holds the paused frame.
            return self.episode_length_buf.clamp(0, self.motion_len - 1).unsqueeze(-1).expand(
                -1, frame_offsets.numel()
            )
        if self.localization_paused:
            # Hold the current reference frame during a Global localization
            # outage; do not expose future motion while episode time is frozen.
            return self.episode_length_buf.clamp(0, self.motion_len - 1).unsqueeze(-1).expand(
                -1, frame_offsets.numel()
            )
        return torch.clamp(
            self.episode_length_buf.unsqueeze(-1) + frame_offsets,
            min=0, max=self.motion_len - 1
        )
    
    def _setup_motion(self, motion_data):
        selected_links = self.metadata_dict["selected_body_names"]
        complete_links = self.metadata_dict["body_names"]
        body_indexes = np.array([complete_links.index(link) for link in selected_links])

        self.body_pos_w = torch.from_numpy(motion_data["body_pos_w"][:, body_indexes]).to(self.device)
        self.body_quat_w = torch.from_numpy(motion_data["body_quat_w"][:, body_indexes]).to(self.device) # wxyz
        self.motion_len = len(self.body_pos_w)
        self.future_frame_offset = torch.as_tensor(self.cfg.future_idx, dtype=torch.long, device=self.device)
        self.joint_pos = motion_data["joint_pos"]
        self.joint_vel = motion_data["joint_vel"]
        self.body_lin_vel_w = motion_data["body_lin_vel_w"]
        self.body_ang_vel_w = motion_data["body_ang_vel_w"]
        # self.future_frames = self.future_frame_offset.shape[-1]

    def _completion_return_enabled(self):
        return (
            self.deployment_hardening
            and self.motion_complete
            and self.motion_complete_local_hold
            and self.motion_complete_return_to_start_steps > 0
        )

    def _completion_return_alpha(self):
        if not self._completion_return_active:
            return 1.0
        return min(
            1.0,
            float(self._completion_return_index)
            / max(self.motion_complete_return_to_start_steps - 1, 1),
        )

    def _prepare_completion_stand_target(self):
        """Place the frame-0 stand at the robot's completion heading.

        Reusing frame 0 in its original world heading would retain a global yaw
        error after a motion that turns the robot. Preserve frame 0's link pose
        relative to its pelvis, but anchor it at the final reference position
        and the robot's measured completion heading.
        """
        measured_root_quat = self.state_buffer["root_quat_wxyz_buffer"][0, -1]
        stand_root_quat = calc_heading_quat(measured_root_quat)
        source_root_pos = self.body_pos_w[0, 0]
        source_root_quat = self.body_quat_w[0, 0]
        body_count = self.body_pos_w.shape[1]
        source_root_quat_expand = source_root_quat[None].expand(body_count, -1)
        stand_root_quat_expand = stand_root_quat[None].expand(body_count, -1)

        frame0_relative_pos = quat_apply(
            quat_inv(source_root_quat_expand),
            self.body_pos_w[0] - source_root_pos,
        )
        frame0_relative_quat = quat_mul(
            quat_inv(source_root_quat_expand), self.body_quat_w[0]
        )
        stand_root_pos = self.body_pos_w[-1, 0]
        self._completion_stand_body_pos = stand_root_pos + quat_apply(
            stand_root_quat_expand, frame0_relative_pos
        )
        self._completion_stand_body_quat = quat_mul(
            stand_root_quat_expand, frame0_relative_quat
        )

    def _gather_reference_state(self):
        if self._completion_return_enabled():
            # The policy can hold frame 0 reliably before R1, whereas repeating
            # this motion's final frame eventually destabilizes it.  Blend the
            # complete target pose directly from final -> frame 0 instead of
            # replaying the whole motion backwards.
            alpha = self._completion_return_alpha()
            final_pos = self.body_pos_w[-1]
            start_pos = (
                self._completion_stand_body_pos
                if self._completion_stand_body_pos is not None
                else self.body_pos_w[0]
            )
            blended_pos = (1.0 - alpha) * final_pos + alpha * start_pos

            final_quat = self.body_quat_w[-1]
            start_quat = (
                self._completion_stand_body_quat
                if self._completion_stand_body_quat is not None
                else self.body_quat_w[0]
            )
            # q and -q encode the same rotation. Align hemispheres before a
            # normalized linear interpolation so the blend takes the short arc.
            same_hemisphere_start = torch.where(
                (final_quat * start_quat).sum(dim=-1, keepdim=True) < 0,
                -start_quat,
                start_quat,
            )
            blended_quat = (
                (1.0 - alpha) * final_quat + alpha * same_hemisphere_start
            )
            blended_quat = blended_quat / torch.linalg.vector_norm(
                blended_quat, dim=-1, keepdim=True
            ).clamp_min(1e-8)

            future_count = self.future_frame_offset.numel()
            self.state_buffer.update({
                "body_pos_w_future": blended_pos[None, None].expand(
                    1, future_count, -1, -1
                ),
                "body_quat_w_wxyz_future": blended_quat[None, None].expand(
                    1, future_count, -1, -1
                ),
                "future_frame_offset": self.future_frame_offset[None, :, None],
            })

            if self._completion_return_active:
                self._completion_return_index += 1
                if (
                    self._completion_return_index
                    >= self.motion_complete_return_to_start_steps
                ):
                    self._completion_return_active = False
                    self._completion_return_finished = True
                    logger.info(
                        '[Env] Motion-complete return finished: holding the '
                        'frame-0 stand at the completion heading with Local root '
                        'localization.'
                    )
        else:
            temporal_index = self._global_time_indices(self.future_frame_offset).reshape(-1)
            self.state_buffer.update({
                "body_pos_w_future": self.body_pos_w.index_select(0, temporal_index).unsqueeze(0),
                "body_quat_w_wxyz_future": self.body_quat_w.index_select(0, temporal_index).unsqueeze(0),
                "future_frame_offset": self.future_frame_offset[None, :, None]
            })
        
        # Global localization is unnecessary while waiting for R1 and can make
        # the standing hold react to tracker jitter/dropouts. Use the same local
        # root convention as reference_forcing=True until playback begins.
        local_hold_before_r1 = (
            self.reference_play_gate
            and not self.reference_playing
            and self.tracker_hold_root_mode == 'local_then_global'
        )
        if (
            self.reference_forcing
            or local_hold_before_r1
            or self.localization_paused
            or (self.motion_complete and self.motion_complete_local_hold)
        ):
            self.state_buffer["root_pos_buffer"][:, -1] = self.state_buffer["body_pos_w_future"][:, 0, 0]
        elif self._root_transition_active:
            # Blend only the policy observation. The calibrated Tracker
            # coordinates and the R2 origin remain unchanged in the simulator
            # and in the raw recording.
            global_root = self.state_buffer["root_pos_buffer"][:, -1].clone()
            # body_pos_w_future already holds the frame this transition is
            # anchored to (frame 0 at R1, the paused frame on a resume).
            local_root = self.state_buffer["body_pos_w_future"][:, 0, 0].to(
                device=self.state_buffer["root_pos_buffer"].device,
                dtype=self.state_buffer["root_pos_buffer"].dtype,
            )
            alpha = min(
                1.0,
                float(self._root_transition_index + 1)
                / max(self.tracker_local_to_global_transition_steps, 1),
            )
            self.state_buffer["root_pos_buffer"][:, -1] = (
                (1.0 - alpha) * local_root + alpha * global_root
            )
            self._root_transition_index += 1
            if (
                self._root_transition_index
                >= self.tracker_local_to_global_transition_steps
            ):
                self._root_transition_active = False

    def _setup_state_manager(self):
        super()._setup_state_manager()
        self._gather_reference_state()
        
    def _update_state_manager(self):
        super()._update_state_manager()
        self._gather_reference_state()

    def reset(self):
        self.motion_complete = False
        self._completion_return_active = False
        self._completion_return_index = 0
        self._completion_return_finished = False
        self._completion_stand_body_pos = None
        self._completion_stand_body_quat = None
        obs_dict = super().reset()
        # BaseEnv.reset refreshes the measured root after _calibrate(). Restore
        # local hold before the warm-up and first policy action.
        if self.reference_play_gate and not self.reference_playing:
            self._gather_reference_state()
            obs_dict = self._update_observation_manager()
        self._root_transition_active = False
        self._root_transition_index = 0
        return obs_dict
        
    def _calibrate(self):
        
        use_rsi = self.cfg.get('rsi', False)
        
        if use_rsi:
            logger.warning(f"Reference State Initialization activated! This should only be used in simulator!")
            init_state_dict = {
                "root_pos": self.body_pos_w[0,0].cpu().numpy(),
                "root_quat": self.body_quat_w[0,0].cpu().numpy(),
                "root_lin_vel": self.body_lin_vel_w[0,0],
                "root_ang_vel": self.body_ang_vel_w[0,0],
                "dof_pos": self.joint_pos[0],
                "dof_vel": self.joint_vel[0],
            }
        else:
            init_state_dict = {}

        root_pos, root_quat = self.simulator.calibrate(init_state_dict) # weishuai: We default to not applying RSI to pure motion tracking
        
        if not use_rsi:
            # weishuai: We do not use RSI but adjust motion based on the current state;
            logger.info(f"[Env] Adjusting the xy-offset of offline trajectories ...")
            # Indexing returns a view. Clone before zeroing z so frame 0 pelvis
            # height in the source trajectory is not modified in place.
            pos_offset = self.body_pos_w[0,0].clone()
            pos_offset[..., -1] = 0

            logger.info(f"[Env] Adjusting the heading direction of offline trajectories ...")
            root_quat_wxyz = torch.from_numpy(root_quat).float().to(self.device)
            target_heading = calc_heading_quat(root_quat_wxyz)
            source_q0 = self.body_quat_w[0,0]
            source_heading_inv = calc_heading_quat_inv(source_q0)
            
            q_align = quat_mul(target_heading, source_heading_inv) # (4)

            # Diagnostic only: a large q_align means the motion is being spun to
            # meet the IMU yaw, which is zeroed at robot power-on and is not the
            # direction the robot physically faces.
            def _yaw_deg(q):
                q = q.detach().cpu().double()
                return float(torch.rad2deg(torch.atan2(
                    2.0 * (q[0] * q[3] + q[1] * q[2]),
                    1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
                )))
            logger.info(
                f"[Env] Heading alignment: robot IMU yaw={_yaw_deg(root_quat_wxyz):+.1f}deg, "
                f"motion frame-0 yaw={_yaw_deg(source_q0):+.1f}deg, "
                f"applied rotation q_align yaw={_yaw_deg(q_align):+.1f}deg"
            )

            q_align_expand = q_align[None,None,:].expand(self.body_quat_w.shape[0], self.body_quat_w.shape[1], -1)
            
            self.body_quat_w = quat_mul(q_align_expand, self.body_quat_w)            
            self.body_pos_w = quat_apply(q_align_expand, self.body_pos_w - pos_offset)

            # Populate observations only after alignment. Updating before the
            # transform leaves reset() with one stale, unaligned reference frame.
            self._update_state_manager()

        self.reference_playing = not self.reference_play_gate
        if self.reference_play_gate:
            hold_description = (
                'calibrated Global root tracking'
                if self.tracker_hold_root_mode == 'global_from_calibration'
                else 'Local hold before the R1 Local-to-Global transition'
            )
            logger.info(
                f'[Env] READY: policy is holding the initial reference pose with {hold_description}. '
                f'Press {self.reference_play_control} to play.'
            )

    def step(self, tgt_dof_pos, action):
        started_this_step = False
        if (
            self.reference_play_gate
            and not self.reference_playing
            and self.simulator.consume_reference_play_request()
        ):
            self.reference_playing = True
            started_this_step = True
            if (
                self.tracker_hold_root_mode == 'local_then_global'
                and self.tracker_local_to_global_transition_steps > 0
            ):
                self._root_transition_active = True
                self._root_transition_index = 0
            if hasattr(self.simulator, 'record_localization_event'):
                self.simulator.record_localization_event('r1_global_playback_start')
            logger.info(
                f'[Env] {self.reference_play_control} pressed: starting reference motion playback'
                + (' and switching to global root localization.' if not self.reference_forcing else '.')
            )

        robust_enabled = (
            not self.reference_forcing
            and hasattr(self.simulator, 'robust_tracking_enabled')
            and self.simulator.robust_tracking_enabled()
        )
        # Once motion-complete local hold is active the observation no longer
        # consumes the global tracker, so pause/resume toggling (and its voice
        # alerts) would only add noise.
        if (
            robust_enabled
            and self.reference_playing
            and not (self.motion_complete and self.motion_complete_local_hold)
        ):
            healthy = self.simulator.is_localization_healthy()
            self._resume_healthy_streak = self._resume_healthy_streak + 1 if healthy else 0
            if not healthy and not self.localization_paused:
                self.localization_paused = True
                if hasattr(self.simulator, 'record_localization_event'):
                    self.simulator.record_localization_event('policy_paused_local_hold')
                logger.warning(
                    '[Env] Global localization unavailable: freezing reference time and '
                    'switching observations to local hold.'
                )
            elif healthy and self.localization_paused:
                # Hardening: demand a sustained healthy streak so a briefly
                # recovering tracker cannot flap local/global every ~0.5s, and
                # hand the observation back to Global through the same blend
                # used at R1 instead of an instant switch.
                required_steps = (
                    self.tracker_resume_min_healthy_steps if self.deployment_hardening else 0
                )
                if self._resume_healthy_streak >= required_steps:
                    self.localization_paused = False
                    if (
                        self.deployment_hardening
                        and self.tracker_hold_root_mode == 'local_then_global'
                        and self.tracker_local_to_global_transition_steps > 0
                    ):
                        self._root_transition_active = True
                        self._root_transition_index = 0
                    if hasattr(self.simulator, 'record_localization_event'):
                        self.simulator.record_localization_event('policy_resumed_global')
                    logger.info('[Env] Global localization recovered: resuming reference playback.')

        self.action.copy_(action)
        if hasattr(self.simulator, 'record_policy_step'):
            self.simulator.record_policy_step(
                tgt_dof_pos.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                self.localization_paused,
            )
        self.simulator.apply_action(tgt_dof_pos.detach().cpu().numpy())
        # On the R1 edge, first expose the normal frame-0 future horizon. Advance
        # to frame 1 only after the policy has acted on that observation once.
        if (
            self.reference_playing
            and not self.localization_paused
            and not started_this_step
            and not self._root_transition_active
            and not self.motion_complete
        ):
            self.episode_length_buf += 1
        if (
            self.reference_playing
            and not self.motion_complete
            and int(self.episode_length_buf.max().item()) >= self.motion_len - 1
        ):
            self.motion_complete = True
            if (
                self.deployment_hardening
                and self.motion_complete_local_hold
                and self.motion_complete_return_to_start_steps > 0
            ):
                self._completion_return_active = True
                self._completion_return_index = 0
                self._completion_return_finished = False
                self._prepare_completion_stand_target()
            if hasattr(self.simulator, 'record_localization_event'):
                self.simulator.record_localization_event('motion_playback_complete')
            if hasattr(self.simulator, '_speak_tracker_event'):
                self.simulator._speak_tracker_event('motion_playback_complete')
            if self._completion_return_enabled():
                logger.info(
                    '[Env] Reference motion complete: returning from the final '
                    f'frame to frame 0 over '
                    f'{self.motion_complete_return_to_start_steps} steps, then '
                    'holding the frame-0 stand at the completion heading with '
                    'Local root localization.'
                )
            elif self.motion_complete_local_hold:
                logger.info(
                    '[Env] Reference motion complete: freezing on the final frame '
                    'and switching the root observation to local stand hold.'
                )
            else:
                logger.info('[Env] Reference motion complete: holding the final frame.')
        obs_dict = self._compute_observation()
        self.simulator.update_marker_pos(obs_dict["body_pos_w_future"][0,0])
        return obs_dict
