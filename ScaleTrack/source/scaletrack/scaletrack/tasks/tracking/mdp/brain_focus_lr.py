"""Independent left/right phase labels over unchanged August body trajectories."""
import os
from pathlib import Path
import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude
from .commands import MotionCommand

class BrainFocusLRMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._lr_sets = {}
        root = Path(os.environ['FOCUS_LR_ROOT'])
        if not (root / 'READY').is_file():
            raise RuntimeError(f'Independent focus corpus is not validated: {root}')
        for split in ('train', 'test'):
            if split == 'test' and not self.has_test_set:
                continue
            arrays = {key: [] for key in ('focus_mask_lr','focus_time_to_next_edge_s_lr','focus_valid_lr')}
            names = getattr(self, f'motion_names_{split}')
            totals = getattr(self, f'time_totals_{split}')
            for name, total in zip(names, totals):
                with np.load(root / 'labels' / f'{name}.npz') as data:
                    for key in arrays:
                        value = data[key]
                        if value.shape != (int(total),2):
                            raise ValueError(f'{name}: {key} shape {value.shape} != {(int(total),2)}')
                        arrays[key].append(value)
            self._lr_sets[split] = {k: torch.from_numpy(np.concatenate(v)) for k,v in arrays.items()}
        self._lr_active = self._lr_sets['train']
        counts = self._lr_active['focus_mask_lr'].sum(0).tolist()
        print(f'[BrainFocusLR] sidecar_root={root}; hand_order=left,right; focus_frames={counts}; phase_dims=4', flush=True)

    def switch_motion_set(self, train_to_test=True):
        super().switch_motion_set(train_to_test)
        if hasattr(self, '_lr_sets'):
            self._lr_active = self._lr_sets['test' if train_to_test else 'train']

    @property
    def focus_flag_lr(self):
        return self._lr_active['focus_mask_lr'][self._global_time_index()].to(self.device)

    @property
    def focus_valid_lr(self):
        return self._lr_active['focus_valid_lr'][self._global_time_index()].to(self.device)

    @property
    def focus_time_lr(self):
        return self._lr_active['focus_time_to_next_edge_s_lr'][self._global_time_index()].to(self.device)

    @property
    def focus_flag(self):
        if not hasattr(self, '_lr_active'):
            return super().focus_flag
        return (self.focus_flag_lr & self.focus_valid_lr).any(-1)

    @property
    def focus_valid(self):
        if not hasattr(self, '_lr_active'):
            return super().focus_valid
        return self.focus_valid_lr.any(-1)

def phase_observation_lr(env, command_name, token_count):
    command = env.command_manager.get_term(command_name)
    valid = command.focus_valid_lr
    flag = command.focus_flag_lr & valid
    time = torch.where(valid, command.focus_time_lr.clamp(-2,2), 2.)
    # [left flag, left edge time, right flag, right edge time]
    phase = torch.stack((flag.float(), time), -1).flatten(-2)
    return phase[:,None].expand(-1,token_count,-1)

def weighted_errors_lr(error, names, flags, strength):
    weights = torch.ones_like(error)
    for column, name in enumerate(names):
        if name == 'left_wrist_yaw_link':
            active = flags[:,0]
        elif name == 'right_wrist_yaw_link':
            active = flags[:,1]
        elif name == 'torso_link':
            active = flags.any(-1)
        else:
            continue
        weights[:,column] = 1 + (strength - 1) * active.to(error.dtype)
    return (error * weights).mean(-1)

def body_position_lr(env, command_name, std, focus_weight=2.):
    c = env.command_manager.get_term(command_name)
    error = (c.body_pos_w - c.robot_body_pos_w).square().sum(-1)
    weighted = weighted_errors_lr(error, c.cfg.body_names, c.focus_flag_lr & c.focus_valid_lr, focus_weight)
    return torch.exp(-weighted / std**2)

def body_rotation_lr(env, command_name, std, focus_weight=2.):
    c = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(c.body_quat_w, c.robot_body_quat_w).square()
    weighted = weighted_errors_lr(error, c.cfg.body_names, c.focus_flag_lr & c.focus_valid_lr, focus_weight)
    return torch.exp(-weighted / std**2)
