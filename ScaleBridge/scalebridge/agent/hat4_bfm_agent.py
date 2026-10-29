import json
import os

import torch
import torch_tensorrt  # Registers Torch-TensorRT engine classes for torch.jit.load.
from loguru import logger

from scalebridge.agent.base_agent import BaseAgent


class HAT4BFMAgent(BaseAgent):
    """Opt-in ScaleBridge agent for the 269-D HAT-4/FOCUS policy."""

    def __init__(self, config, device):
        super().__init__(config, device)
        if self.device != "cuda":
            raise ValueError("The HAT-4 deployment policy requires CUDA")
        self.control_mode = torch.tensor(
            [self.config.control_mode], dtype=torch.long, device=self.device
        )
        logger.info(
            f"[Agent] Activating control mode {self.config.control_mode}: HAT-4"
        )

    def _load_policy(self):
        logger.info(f"[Agent] Loading HAT-4 checkpoint from {self.checkpoint}")
        self.policy = torch.jit.load(self.checkpoint).to(self.device)
        self.policy.eval()
        path_base, _ = os.path.splitext(self.checkpoint)
        metadata_path = path_base + "_metadata.json"
        logger.info(f"[Agent] Loading HAT-4 metadata from {metadata_path}")
        with open(metadata_path, "r") as stream:
            self.meta_data_dict = json.load(stream)
        self.prestart_policy = None
        prestart_checkpoint = self.config.get("prestart_checkpoint")
        if prestart_checkpoint:
            prestart_metadata_path = os.path.splitext(str(prestart_checkpoint))[0] + "_metadata.json"
            with open(prestart_metadata_path, "r") as stream:
                prestart_metadata = json.load(stream)
            for key in (
                "joint_names", "action_names", "selected_body_names",
                "history_buffer_size", "future_idx", "default_dof_pos",
                "action_scale", "stiffness", "damping", "torque_limit",
                "policy_architecture", "control_modes", "focus_phase_observation",
            ):
                if prestart_metadata.get(key) != self.meta_data_dict.get(key):
                    raise ValueError(f"Standing checkpoint is incompatible: metadata {key!r} differs")
            logger.info(f"[Agent] Loading pre-R1 standing checkpoint from {prestart_checkpoint}")
            self.prestart_policy = torch.jit.load(str(prestart_checkpoint)).to(self.device).eval()
        self._using_prestart_policy = None

    def get_meta_data(self):
        return self.meta_data_dict

    def get_action(self, obs_dict):
        use_prestart = self.prestart_policy is not None and bool(obs_dict.get("prestart_hold", False))
        if use_prestart != self._using_prestart_policy:
            logger.info("[Agent] Active controller: {}", "pre-R1 standing checkpoint" if use_prestart else "tracking checkpoint")
            self._using_prestart_policy = use_prestart
        policy = self.prestart_policy if use_prestart else self.policy
        control_mode = obs_dict.get("policy_control_mode", self.control_mode)
        return policy(
            obs_dict["root_quat_buffer"],
            obs_dict["base_ang_vel_buffer"],
            obs_dict["dof_pos_buffer"],
            obs_dict["dof_vel_buffer"],
            obs_dict["actions_buffer"],
            obs_dict["target_body_pos_future_to_robot_base"],
            obs_dict["target_body_rot_future_to_robot_base"],
            control_mode,
            obs_dict["future_time_offsets"],
            obs_dict["focus_phase"],
        )
