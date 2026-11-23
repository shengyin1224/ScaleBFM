#!/usr/bin/env python3
"""Export the HAT-4 training checkpoint for the existing ScaleBridge runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SELECTED_BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]

MODE_BODY_NAMES = [
    ["pelvis"],
    ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    ["left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
    ["pelvis", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
    SELECTED_BODY_NAMES,
    ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--network-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_network_types(path: Path):
    spec = importlib.util.spec_from_file_location("_hat4_network", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HumanoidTransformer, module.TaskEmbedder


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v - q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ), dim=-1)


def quat_mul_inverse_right(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q2_inv = torch.cat((q2[..., :1], -q2[..., 1:]), dim=-1)
    return quat_mul(q1, q2_inv)


def parse_robot_xml(path: Path):
    root = ET.parse(path).getroot()
    world_root = root.find("worldbody").find("body")
    body_names, joint_names = [], []
    parents, translations, rotations, axes = [], [], [], []

    def visit(node, parent_index: int) -> None:
        body_index = len(body_names)
        body_names.append(node.attrib["name"])
        parents.append(parent_index)
        if body_index:
            joint = node.findall("joint")
            if len(joint) != 1:
                raise ValueError(f"Expected one joint on {node.attrib['name']}, got {len(joint)}")
            joint_names.append(joint[0].attrib["name"])
            translations.append(np.fromstring(node.attrib.get("pos", "0 0 0"), sep=" "))
            rotations.append(np.fromstring(node.attrib.get("quat", "1 0 0 0"), sep=" "))
            axes.append(np.fromstring(joint[0].attrib["axis"], sep=" "))
        for child in node.findall("body"):
            visit(child, body_index)

    visit(world_root, -1)
    return (
        body_names,
        joint_names,
        torch.tensor(parents, dtype=torch.long),
        torch.tensor(np.asarray(translations), dtype=torch.float32),
        torch.tensor(np.asarray(rotations), dtype=torch.float32),
        torch.tensor(np.asarray(axes), dtype=torch.float32),
    )


class ScaleBridgeHAT4Policy(nn.Module):
    def __init__(
        self, actor: nn.Module, task_embedder: nn.Module, mode_vectors: torch.Tensor,
        mode_mappings: torch.Tensor, default_dof_pos: torch.Tensor,
        action_scale: torch.Tensor, parent_indices: torch.Tensor,
        local_translation: torch.Tensor, local_rotation: torch.Tensor,
        joint_axis: torch.Tensor, selected_indices: torch.Tensor,
        actor_to_xml: torch.Tensor,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.task_embedder = task_embedder
        self.register_buffer("mode_vectors", mode_vectors)
        self.register_buffer("mode_mappings", mode_mappings)
        self.register_buffer("default_dof_pos", default_dof_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("parent_indices", parent_indices)
        # Keep topology as Python constants as well.  Tensor-valued parent
        # indices become data-dependent symbols under torch.export/Dynamo and
        # prevent full TensorRT compilation, even though the kinematic tree is
        # static for this robot.
        self.parent_indices_python = tuple(int(v) for v in parent_indices.tolist())
        self.num_bodies = len(self.parent_indices_python)
        self.register_buffer("local_translation", local_translation)
        self.register_buffer("local_rotation", local_rotation)
        self.register_buffer("joint_axis", joint_axis)
        self.register_buffer("selected_indices", selected_indices)
        self.register_buffer("actor_to_xml", actor_to_xml)
        tangent = torch.zeros(1, 6, 14, 3)
        normal = torch.zeros(1, 6, 14, 3)
        tangent[..., 0] = 1.0
        normal[..., 2] = 1.0
        self.register_buffer("tangent", tangent)
        self.register_buffer("normal", normal)

    def forward(
        self, root_quat_buffer: torch.Tensor, base_ang_vel_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor, dof_vel_buffer: torch.Tensor,
        action_buffer: torch.Tensor, target_pos: torch.Tensor,
        target_quat: torch.Tensor, mode_index: torch.Tensor,
        time_offsets: torch.Tensor, focus_phase: torch.Tensor,
    ):
        gravity = torch.zeros_like(base_ang_vel_buffer)
        gravity[..., 2] = -1.0
        prop_obs = torch.cat((
            quat_apply_inverse(root_quat_buffer, gravity), base_ang_vel_buffer,
            dof_pos_buffer - self.default_dof_pos, dof_vel_buffer * 0.05,
        ), dim=-1)

        dof_pos_xml = dof_pos_buffer[:, -1, self.actor_to_xml]
        half_angle = dof_pos_xml.unsqueeze(-1) * 0.5
        joint_quat = torch.cat((
            torch.cos(half_angle), self.joint_axis.unsqueeze(0) * torch.sin(half_angle)
        ), dim=-1)
        body_pos = torch.zeros(1, self.num_bodies, 3, device=dof_pos_buffer.device)
        body_quat = torch.zeros(1, self.num_bodies, 4, device=dof_pos_buffer.device)
        body_quat[..., 0] = 1.0
        for index in range(1, self.num_bodies):
            parent = self.parent_indices_python[index]
            parent_pos = body_pos[:, parent]
            parent_quat = body_quat[:, parent]
            body_pos[:, index] = parent_pos + quat_apply(parent_quat, self.local_translation[index-1:index])
            local_joint_quat = quat_mul(self.local_rotation[index-1:index], joint_quat[:, index-1])
            body_quat[:, index] = quat_mul(parent_quat, local_joint_quat)

        current_pos = body_pos[:, self.selected_indices]
        current_quat = body_quat[:, self.selected_indices]
        pos_error = target_pos - current_pos[:, None]
        target_rot_6d = torch.cat((
            quat_apply(target_quat, self.tangent), quat_apply(target_quat, self.normal)
        ), dim=-1)
        rot_error = quat_mul_inverse_right(
            target_quat, current_quat[:, None].expand_as(target_quat)
        )
        rot_error_6d = torch.cat((
            quat_apply(rot_error, self.tangent), quat_apply(rot_error, self.normal)
        ), dim=-1)
        task_obs = torch.cat((
            target_pos.flatten(2), pos_error.flatten(2), target_rot_6d.flatten(2),
            rot_error_6d.flatten(2), time_offsets.float(), focus_phase,
        ), dim=-1)
        mapping = self.mode_mappings[mode_index].unsqueeze(1)
        mode = self.mode_vectors[mode_index].unsqueeze(1).expand(-1, task_obs.shape[1], -1)
        task_input = torch.cat((task_obs * mapping, mode), dim=-1)
        action = self.actor(prop_obs, action_buffer, self.task_embedder(task_input))
        return action * self.action_scale + self.default_dof_pos, action


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    metadata = json.loads(args.base_metadata.read_text())
    HumanoidTransformer, TaskEmbedder = load_network_types(args.network_file)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]

    actor = HumanoidTransformer(64, 29, 29, 256, 4, 256, 4).to(device)
    embedder = TaskEmbedder(269, 256).to(device)
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
    embedder.load_state_dict({k.removeprefix("actor_task_embedder."): v for k, v in state.items() if k.startswith("actor_task_embedder.")})
    actor.eval()
    embedder.eval()

    body_names, xml_joints, parents, translations, rotations, axes = parse_robot_xml(args.robot_xml)
    actor_joints = metadata["joint_names"]
    actor_to_xml = torch.tensor([actor_joints.index(name) for name in xml_joints], dtype=torch.long)
    selected_indices = torch.tensor([body_names.index(name) for name in SELECTED_BODY_NAMES], dtype=torch.long)

    mode_vectors = torch.zeros(len(MODE_BODY_NAMES), len(SELECTED_BODY_NAMES))
    for row, names in enumerate(MODE_BODY_NAMES):
        for name in names:
            mode_vectors[row, SELECTED_BODY_NAMES.index(name)] = 1.0
    mappings = [mode_vectors.repeat_interleave(dim, dim=1) for dim in (3, 3, 6, 6)]
    mode_mappings = torch.cat((*mappings, torch.ones(9, 3)), dim=1)

    wrapper = ScaleBridgeHAT4Policy(
        actor, embedder, mode_vectors, mode_mappings,
        torch.tensor(metadata["default_dof_pos"], dtype=torch.float32),
        torch.tensor(metadata["action_scale"], dtype=torch.float32),
        parents, translations, rotations / rotations.norm(dim=-1, keepdim=True), axes,
        selected_indices, actor_to_xml,
    ).to(device).eval()

    inputs = (
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(1, 3, 4).to(device),
        torch.zeros(1, 3, 3, device=device),
        torch.tensor(metadata["default_dof_pos"], device=device).view(1, 1, 29).expand(1, 3, 29),
        torch.zeros(1, 3, 29, device=device), torch.zeros(1, 3, 29, device=device),
        torch.zeros(1, 6, 14, 3, device=device),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 1, 1, 4).expand(1, 6, 14, 4),
        torch.tensor([8], dtype=torch.long, device=device),
        torch.tensor([0, 1, 2, 3, 4, 5], device=device).view(1, 6, 1),
        torch.tensor([0.0, 2.0], device=device).view(1, 1, 2).expand(1, 6, 2),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, inputs, strict=False, check_trace=True)
        traced.save(str(args.output))
        loaded = torch.jit.load(str(args.output), map_location=device)
        expected = wrapper(*inputs)
        actual = loaded(*inputs)
        max_error = max((a-b).abs().max().item() for a, b in zip(expected, actual))
    if max_error > 1e-5:
        raise RuntimeError(f"TorchScript validation failed: max error {max_error}")

    metadata["policy_architecture"]["task_obs_dim"] = 269
    metadata["policy_architecture"]["focus_phase_dim"] = 2
    metadata["control_modes"] = [
        "Pelvis-1", "UMI-2", "VR-3", "UMI-4", "VR-5", "UpperBody-6",
        "UpperBody-Mobile-7", "WholeBody-14", "HAT-4",
    ]
    metadata["focus_phase_observation"] = True
    metadata["source_checkpoint"] = str(args.checkpoint.resolve())
    metadata_path = args.output.with_name(args.output.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output.resolve()}")
    print(f"Saved {metadata_path.resolve()}")
    print(f"Validation max error: {max_error:.3e}")


if __name__ == "__main__":
    main()
