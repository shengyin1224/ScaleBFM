"""Variable-speed playback for HAT4 focus fine-tuning; factory commands stay unchanged."""
import math
import torch
from isaaclab.utils.math import quat_error_magnitude
from .commands import MotionCommand
from .reference_playback import (
    eased_progress, truncated_rates, interpolate_rates, interpolate_samples, precision_scales,
)


class TimeWarpMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        for prefix in ("playback", "precision"):
            eased_progress(0, getattr(cfg, prefix + "_start_step"), getattr(cfg, prefix + "_end_step"))
        if cfg.playback_node_interval <= 0 or not 0 < cfg.playback_rate_low < 1 < cfg.playback_rate_high:
            raise ValueError("Invalid playback interval or bounds")
        super().__init__(cfg, env)
        self.sim_time = torch.zeros(self.num_envs)
        self.ref_time = self.time_steps.float() / 50.0
        self.training_steps = int(cfg.schedule_step_offset)
        max_frames = max(int(self.time_totals_train.max()),
                         int(self.time_totals_test.max()) if self.has_test_set else 0)
        horizon = max_frames / 50.0 / cfg.playback_rate_low
        # Include lookahead beyond the final control step.
        horizon += cfg.ref_frame_buffer_size * env.step_dt + cfg.playback_node_interval
        nodes = math.ceil(horizon / cfg.playback_node_interval) + 2
        self.playback_curve = torch.ones(self.num_envs, nodes)
        self._sample_cache = {}

    @property
    def playback_std(self):
        p = eased_progress(self.training_steps, self.cfg.playback_start_step, self.cfg.playback_end_step)
        return self.cfg.playback_std_start + (self.cfg.playback_std_end - self.cfg.playback_std_start) * p

    def _reset_playback(self, env_ids):
        self.sim_time[env_ids] = 0
        self.ref_time[env_ids] = self.time_steps[env_ids].float() / 50.0
        self.playback_curve[env_ids] = (
            1.0 if self.is_evaluating else truncated_rates(
                (len(env_ids), self.playback_curve.shape[1]), self.playback_std,
                self.cfg.playback_rate_low, self.cfg.playback_rate_high)
        )
        self._sample_cache.clear()
        self._future_manual_cache.clear()

    def _rates(self, sim_times):
        if not self.cfg.playback_enabled or self.is_evaluating:
            return torch.ones_like(sim_times)
        return interpolate_rates(self.playback_curve, sim_times, self.cfg.playback_node_interval)

    @property
    def playback_rate(self):
        return self._rates(self.sim_time)

    def _future_times(self, frame_offsets):
        """Project the same per-control-step integration used by _update_command."""
        offsets = frame_offsets.long()
        if torch.any(offsets < 0):
            raise ValueError("Future offsets must be nonnegative")
        if not self.cfg.playback_enabled:
            return (self.time_steps[:, None] + offsets).float() / 50.0, torch.ones_like(offsets, dtype=torch.float)
        max_offset = int(offsets.max())
        delta = torch.arange(max_offset + 1) * self._env.step_dt
        rates = self._rates(self.sim_time[:, None] + delta[None])
        advance = torch.cat((torch.zeros(self.num_envs, 1),
                             (rates[:, :-1] * self._env.step_dt).cumsum(1)), dim=1)
        return self.ref_time[:, None] + advance.gather(1, offsets), rates.gather(1, offsets)

    def _sample(self, field, frame_offsets=None):
        key = field
        # Current samples are reused across rewards/metrics; future samples are
        # cached by the public manual-future API below, without large CPU keys.
        if frame_offsets is None and key in self._sample_cache:
            return self._sample_cache[key]
        if frame_offsets is None:
            times = self.ref_time if self.cfg.playback_enabled else self.time_steps.float() / 50.0
            rates = self.playback_rate
            mids = self.motion_ids
        else:
            times, rates = self._future_times(frame_offsets)
            mids = self.motion_ids[:, None]
        frames = (times * 50.0).clamp_min(0)
        frames = torch.minimum(frames, (self.time_totals[mids] - 1).float())
        lo = frames.floor().long()
        hi = torch.minimum(lo + 1, self.time_totals[mids] - 1)
        base = self.time_offsets[mids]
        result = interpolate_samples(getattr(self, "cat_" + field), base + lo, base + hi,
                                     frames - lo, quaternion=field == "body_quat_w")
        if field in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            result = result * rates.reshape(*rates.shape, *([1] * (result.ndim - rates.ndim)))
        result = result.to(self.device)
        if field == "body_pos_w":
            origins = self._env.scene.env_origins
            result = result + (origins[:, None] if frame_offsets is None else origins[:, None, None])
        if frame_offsets is None:
            self._sample_cache[key] = result
        return result

    def _manual_sample(self, field, future_idx):
        cache = self._future_manual_cache.setdefault(tuple(future_idx), {})
        if field not in cache:
            cache[field] = self._sample(field, self._future_frame_offsets_manual(future_idx))
        return cache[field]

    def _resample_command(self, env_ids):
        # Also invalidate samples if playback is explicitly disabled.
        self._sample_cache.clear()
        self._future_manual_cache.clear()
        super()._resample_command(env_ids)

    def _update_command(self):
        if not self.is_evaluating:
            self.training_steps += 1
        if self.cfg.playback_enabled:
            self.ref_time += self._env.step_dt * self.playback_rate
            self.sim_time += self._env.step_dt
            # Compatibility with existing focus labels, failure resampling and timeout.
            self.time_steps[:] = (self.ref_time * 50.0 + 1e-5).floor().long()
        else:
            self.time_steps += 1
        self._sample_cache.clear()
        self._future_manual_cache.clear()

    def switch_motion_set(self, train_to_test=True):
        super().switch_motion_set(train_to_test)
        self._sample_cache.clear()
        self._future_manual_cache.clear()

    @property
    def joint_pos(self):
        return self._sample("joint_pos")

    @property
    def joint_pos_future(self):
        return self._sample("joint_pos", self._future_frame_offsets())

    @property
    def joint_vel(self):
        return self._sample("joint_vel")

    @property
    def joint_vel_future(self):
        return self._sample("joint_vel", self._future_frame_offsets())

    @property
    def body_pos_w(self):
        return self._sample("body_pos_w")

    @property
    def body_pos_w_future(self):
        return self._sample("body_pos_w", self._future_frame_offsets())

    def body_pos_w_future_manual(self, future_idx):
        return self._manual_sample("body_pos_w", future_idx)

    @property
    def body_quat_w(self):
        return self._sample("body_quat_w")

    @property
    def body_quat_w_future(self):
        return self._sample("body_quat_w", self._future_frame_offsets())

    def body_quat_w_future_manual(self, future_idx):
        return self._manual_sample("body_quat_w", future_idx)

    @property
    def body_lin_vel_w(self):
        return self._sample("body_lin_vel_w")

    @property
    def body_lin_vel_w_future(self):
        return self._sample("body_lin_vel_w", self._future_frame_offsets())

    def body_lin_vel_w_future_manual(self, future_idx):
        return self._manual_sample("body_lin_vel_w", future_idx)

    @property
    def body_ang_vel_w(self):
        return self._sample("body_ang_vel_w")

    @property
    def body_ang_vel_w_future(self):
        return self._sample("body_ang_vel_w", self._future_frame_offsets())

    def body_ang_vel_w_future_manual(self, future_idx):
        return self._manual_sample("body_ang_vel_w", future_idx)

    @property
    def anchor_pos_w_future(self):
        return self.body_pos_w_future[:, :, self.motion_anchor_body_index]

    @property
    def anchor_quat_w_future(self):
        return self.body_quat_w_future[:, :, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w_future(self):
        return self.body_lin_vel_w_future[:, :, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w_future(self):
        return self.body_ang_vel_w_future[:, :, self.motion_anchor_body_index]


def adaptive_wrist_precision(
    env, command_name="motion", wrist_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    pelvis_name="pelvis", focus_weight=2.0, w_pos=1.0, w_rot=1.0,
    sigma_pos_start=0.10, sigma_pos_end=0.01, sigma_pos_max=0.10,
    speed_range=(0.05, 0.10), sigma_rot_range=(math.radians(5), math.radians(20)),
    pelvis_speed_threshold=0.02,
):
    """One extra reward: mean squared wrist errors, one gate, one focus multiplier."""
    command = env.command_manager.get_term(command_name)
    if focus_weight <= 0:
        raise ValueError("focus_weight must be positive")
    indexes = [command.cfg.body_names.index(name) for name in wrist_names]
    pelvis = command.cfg.body_names.index(pelvis_name)
    progress = eased_progress(command.training_steps, command.cfg.precision_start_step,
                              command.cfg.precision_end_step)
    sigma_min = sigma_pos_start + (sigma_pos_end - sigma_pos_start) * progress
    ref_vel = command.body_lin_vel_w
    sigma_pos, sigma_rot = precision_scales(ref_vel[:, indexes], sigma_min, speed_range,
                                           sigma_pos_max, sigma_rot_range)
    pos_error_sq = (command.robot_body_pos_w[:, indexes] - command.body_pos_w[:, indexes]).square().sum(-1).mean(-1)
    rot_error_sq = quat_error_magnitude(command.robot_body_quat_w[:, indexes],
                                       command.body_quat_w[:, indexes]).square().mean(-1)
    focus = (command.focus_flag & command.focus_valid).to(pos_error_sq.dtype)
    multiplier = 1 + (focus_weight - 1) * focus
    gate = (torch.linalg.vector_norm(ref_vel[:, pelvis], dim=-1) <= pelvis_speed_threshold).to(pos_error_sq.dtype)
    return gate * multiplier * (
        w_pos * torch.exp(-pos_error_sq / sigma_pos.square())
        + w_rot * torch.exp(-rot_error_sq / sigma_rot.square())
    )
