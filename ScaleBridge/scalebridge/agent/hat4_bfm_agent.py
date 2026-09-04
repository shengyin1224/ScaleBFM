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

    def get_meta_data(self):
        return self.meta_data_dict

    def get_action(self, obs_dict):
        return self.policy(
            obs_dict["root_quat_buffer"],
            obs_dict["base_ang_vel_buffer"],
            obs_dict["dof_pos_buffer"],
            obs_dict["dof_vel_buffer"],
            obs_dict["actions_buffer"],
            obs_dict["target_body_pos_future_to_robot_base"],
            obs_dict["target_body_rot_future_to_robot_base"],
            self.control_mode,
            obs_dict["future_time_offsets"],
            obs_dict["focus_phase"],
        )
