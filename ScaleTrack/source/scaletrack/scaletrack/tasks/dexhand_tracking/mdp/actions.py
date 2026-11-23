"""Action terms, including the same-PD IK replay executor."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .commands import DexHandMotionCommand


class ReferenceJointPositionAction(JointPositionAction):
    """Ignore policy input and replay q_ref[t+1] through the same implicit PD."""

    cfg: "ReferenceJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        command: DexHandMotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
        self._processed_actions = command._get(f"q_{self.cfg.hand}", frame_offset=1)


@configclass
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ReferenceJointPositionAction
    command_name: str = "motion"
    hand: str = "rh"

