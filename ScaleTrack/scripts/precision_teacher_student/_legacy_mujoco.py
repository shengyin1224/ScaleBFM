#!/usr/bin/env python3
"""Run ScaleBFM's pelvis-and-hands mode in MuJoCo and record metrics/video."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from my_rsl_rl.networks import HumanoidTransformer, TaskEmbedder


SELECTED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
ACTIVE_BODY_NAMES = ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]
HAT4_BODY_NAMES = [
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--isaac-result-npz", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--video-out", type=Path, required=True)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MuJoCo rendering while retaining the same rollout and metrics.",
    )
    parser.add_argument(
        "--hand-reference-ik",
        type=Path,
        help="TWIST hand-reference NPZ containing rh/lh fingertip targets and Pinocchio IK DOFs.",
    )
    parser.add_argument("--focus-cache", type=Path)
    parser.add_argument(
        "--focus-motion-name",
        help="Unique motion basename substring used to select one entry from the TWIST FOCUS cache.",
    )
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tracking-mode",
        choices=("3", "4", "14"),
        default="3",
        help="ScaleBFM target-link mode: VR-3, HAT-4, or WholeBody-14.",
    )
    parser.add_argument("--future-offset", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument(
        "--root-drift-cm",
        type=float,
        default=0.0,
        help="Translate the initial MuJoCo root along world +x by this many centimeters.",
    )
    parser.add_argument(
        "--reanchor-mode",
        choices=("on", "off"),
        default="on",
        help=(
            "With injected root drift, 'on' retains the fixed world reference; "
            "'off' translates the policy reference with the stale root-local target."
        ),
    )
    return parser.parse_args()


def quat_conjugate(q):
    result = q.clone()
    result[..., 1:] *= -1
    return result


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(q, v):
    q_vec = q[..., 1:]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_error_deg(actual, reference):
    dot = np.abs(np.sum(actual * reference, axis=-1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def np_quat_conjugate(q):
    result = np.asarray(q, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def np_quat_mul(q1, q2):
    w1, x1, y1, z1 = np.moveaxis(np.asarray(q1), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(q2), -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def np_quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_vec = q[..., 1:]
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def heading_quat(q):
    """Extract the world-z heading component from a wxyz quaternion."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    heading = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray(
        [np.cos(0.5 * heading), 0.0, 0.0, np.sin(0.5 * heading)],
        dtype=np.float64,
    )


def np_quat_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_error_deg(actual, reference):
    relative = np_quat_to_matrix(actual) @ np_quat_to_matrix(reference).T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "rmse": None, "p95": None, "max": None}
    return {
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def interpolate_sequence(values, frame_f):
    values = np.asarray(values)
    frame_f = float(np.clip(frame_f, 0.0, max(len(values) - 1, 0)))
    lower = int(np.floor(frame_f))
    upper = min(lower + 1, len(values) - 1)
    alpha = frame_f - lower
    return values[lower] * (1.0 - alpha) + values[upper] * alpha


def pose_from_parent(parent_pos, parent_quat, local_pos, local_quat):
    return (
        parent_pos + np_quat_apply(parent_quat, local_pos),
        np_quat_mul(parent_quat, local_quat),
    )


def local_point(root_pos, root_quat, point):
    return np_quat_apply(np_quat_conjugate(root_quat), point - root_pos)


def local_quat(root_quat, body_quat):
    return np_quat_mul(np_quat_conjugate(root_quat), body_quat)


def actuator_parameters(joint_names):
    natural_frequency = 10.0 * 2.0 * math.pi
    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425

    stiffness_5020 = armature_5020 * natural_frequency**2
    stiffness_7520_14 = armature_7520_14 * natural_frequency**2
    stiffness_7520_22 = armature_7520_22 * natural_frequency**2
    stiffness_4010 = armature_4010 * natural_frequency**2
    damping_5020 = 4.0 * armature_5020 * natural_frequency
    damping_7520_14 = 4.0 * armature_7520_14 * natural_frequency
    damping_7520_22 = 4.0 * armature_7520_22 * natural_frequency
    damping_4010 = 4.0 * armature_4010 * natural_frequency

    stiffness = []
    damping = []
    torque_limit = []
    for name in joint_names:
        if re.search(r"hip_(pitch|yaw)", name):
            stiffness.append(stiffness_7520_14)
            damping.append(damping_7520_14)
            torque_limit.append(88.0)
        elif "hip_roll" in name or "knee" in name:
            stiffness.append(stiffness_7520_22)
            damping.append(damping_7520_22)
            torque_limit.append(139.0)
        elif "ankle" in name:
            stiffness.append(2.0 * stiffness_5020)
            damping.append(2.0 * damping_5020)
            torque_limit.append(50.0)
        elif name in ("waist_roll_joint", "waist_pitch_joint"):
            stiffness.append(300.0)
            damping.append(5.0)
            torque_limit.append(50.0)
        elif name == "waist_yaw_joint":
            stiffness.append(100.0)
            damping.append(2.0)
            torque_limit.append(88.0)
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            stiffness.append(stiffness_4010)
            damping.append(damping_4010)
            torque_limit.append(5.0)
        else:
            stiffness.append(stiffness_5020)
            damping.append(damping_5020)
            torque_limit.append(25.0)
    return tuple(np.asarray(x, dtype=np.float32) for x in (stiffness, damping, torque_limit))


def add_reference_markers(renderer, positions):
    colors = (
        np.array([1.0, 0.25, 0.15, 0.85], dtype=np.float32),
        np.array([0.15, 0.8, 1.0, 0.85], dtype=np.float32),
        np.array([0.2, 1.0, 0.25, 0.85], dtype=np.float32),
    )
    for position, color in zip(positions, colors):
        if renderer.scene.ngeom >= renderer.scene.maxgeom:
            break
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.035, 0.035, 0.035]),
            position,
            np.eye(3).reshape(-1),
            color,
        )
        renderer.scene.ngeom += 1


def main():
    global ACTIVE_BODY_NAMES
    args = parse_args()
    if args.tracking_mode == "14":
        ACTIVE_BODY_NAMES = list(SELECTED_BODY_NAMES)
    elif args.tracking_mode == "4":
        ACTIVE_BODY_NAMES = list(HAT4_BODY_NAMES)
    device = torch.device(args.device)
    reference = np.load(args.reference)
    focus_state_mask = None
    focus_time_to_next_edge_s = None
    focus_metadata = None
    if {
        "focus_mask",
        "focus_time_to_next_edge_s",
        "focus_valid",
    }.issubset(reference.files):
        embedded_valid = np.asarray(reference["focus_valid"], dtype=bool)
        if embedded_valid.shape != (len(reference["joint_pos"]),):
            raise ValueError("Reference focus_valid has the wrong shape")
        if np.all(embedded_valid):
            focus_state_mask = np.asarray(reference["focus_mask"], dtype=bool)
            focus_time_to_next_edge_s = np.asarray(
                reference["focus_time_to_next_edge_s"], dtype=np.float32
            )
            focus_metadata = {
                "source": "embedded_reference_npz",
                "reference": str(args.reference.resolve()),
            }
    if args.focus_cache is not None:
        if not args.focus_motion_name:
            raise ValueError("--focus-motion-name is required with --focus-cache.")
        focus_cache = torch.load(
            args.focus_cache, map_location="cpu", weights_only=False
        )
        matches = [
            index
            for index, path in enumerate(focus_cache["motion_files"])
            if args.focus_motion_name in Path(str(path)).name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one FOCUS-cache match for {args.focus_motion_name!r}, "
                f"found {len(matches)}: {[focus_cache['motion_files'][i] for i in matches]}"
            )
        focus_motion_index = matches[0]
        focus_start = int(focus_cache["motion_start_idx"][focus_motion_index])
        focus_frames = int(focus_cache["motion_num_frames"][focus_motion_index])
        source_focus = (
            focus_cache["focus"][focus_start : focus_start + focus_frames]
            .cpu()
            .numpy()
            .astype(bool)
        )
        source_valid = (
            focus_cache["valid"][focus_start : focus_start + focus_frames]
            .cpu()
            .numpy()
            .astype(bool)
        )
        if not np.all(source_valid):
            raise RuntimeError("Selected FOCUS-cache motion contains invalid frames.")
        mapping = np.rint(
            np.arange(len(reference["joint_pos"]), dtype=np.float64)
            * (len(source_focus) - 1)
            / max(len(reference["joint_pos"]) - 1, 1)
        ).astype(np.int64)
        focus_state_mask = source_focus[mapping]
        if "time_to_next_edge_s" in focus_cache:
            source_next_edge = (
                focus_cache["time_to_next_edge_s"][
                    focus_start : focus_start + focus_frames
                ]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            focus_time_to_next_edge_s = source_next_edge[mapping]
        focus_metadata = {
            "cache": str(args.focus_cache.resolve()),
            "motion_file": str(focus_cache["motion_files"][focus_motion_index]),
            "source_frames": focus_frames,
            "source_focus_frames": int(source_focus.sum()),
            "source_focus_fraction": float(source_focus.mean()),
            "focus_pad_s": float(focus_cache["focus_pad_s"]),
            "focus_mode": str(focus_cache["focus_mode"]),
        }
    isaac_arrays = np.load(args.isaac_result_npz)
    actor_joint_names = [str(x) for x in isaac_arrays["joint_names"]]
    default_dof_pos = torch.as_tensor(
        isaac_arrays["action_offset"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    action_scale = torch.as_tensor(
        isaac_arrays["action_scale"], dtype=torch.float32, device=device
    ).unsqueeze(0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    task_input_dim = int(
        state["actor_task_embedder.task_projection.weight"].shape[1]
    )
    if task_input_dim not in (267, 269):
        raise ValueError(
            f"Unsupported actor task input width {task_input_dim}; expected 267 or 269"
        )
    has_focus_phase_observation = task_input_dim == 269

    actor = HumanoidTransformer(
        prop_obs_dim=64,
        action_dim=29,
        output_dim=29,
        embed_dim=256,
        num_heads=4,
        ff_dim=256,
        num_layers=4,
    ).to(device)
    task_embedder = TaskEmbedder(task_input_dim, 256).to(device)
    actor.load_state_dict(
        {key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")}
    )
    task_embedder.load_state_dict(
        {
            key.removeprefix("actor_task_embedder."): value
            for key, value in state.items()
            if key.startswith("actor_task_embedder.")
        }
    )
    actor.eval()
    task_embedder.eval()

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.opt.timestep = 0.001
    if not args.no_video:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.video_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.video_height)
    data = mujoco.MjData(model)
    xml_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    actor_to_xml = np.asarray([xml_joint_names.index(name) for name in actor_joint_names])
    xml_to_actor = np.asarray([actor_joint_names.index(name) for name in xml_joint_names])
    extra_body_names = ["head_mocap"]
    if args.hand_reference_ik is not None:
        extra_body_names.extend(["L_hand_base_link", "R_hand_base_link"])
    body_ids = {
        name: model.body(name).id
        for name in dict.fromkeys(SELECTED_BODY_NAMES + extra_body_names)
    }

    # The packaged NPZ body order is IsaacLab articulation order.
    isaac_body_names = ["pelvis"] + [name.replace("_joint", "_link") for name in actor_joint_names]
    isaac_body_names[isaac_body_names.index("waist_pitch_link")] = "torso_link"
    reference_body_indices = [isaac_body_names.index(name) for name in SELECTED_BODY_NAMES]
    active_reference_indices = [isaac_body_names.index(name) for name in ACTIVE_BODY_NAMES]
    torso_reference_index = isaac_body_names.index("torso_link")

    stiffness, damping, torque_limit = actuator_parameters(xml_joint_names)
    data.qpos[:3] = reference["body_pos_w"][0, 0]
    injected_root_drift = np.asarray(
        [args.root_drift_cm / 100.0, 0.0, 0.0], dtype=np.float64
    )
    data.qpos[:3] += injected_root_drift
    data.qpos[3:7] = reference["body_quat_w"][0, 0]
    data.qpos[7:] = reference["joint_pos"][0, xml_to_actor]
    data.qvel[:3] = reference["body_lin_vel_w"][0, 0]
    data.qvel[3:6] = reference["body_ang_vel_w"][0, 0]
    data.qvel[6:] = reference["joint_vel"][0, xml_to_actor]
    mujoco.mj_forward(model, data)

    control_decimation = 20
    head_body_id = body_ids["head_mocap"]
    head_local_pos = model.body_pos[head_body_id].copy()
    head_local_quat = model.body_quat[head_body_id].copy()
    head_parent_id = int(model.body_parentid[head_body_id])
    head_parent_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, head_parent_id
    )
    if head_parent_name != "torso_link":
        raise RuntimeError(
            f"head_mocap must be a rigid child of torso_link, got {head_parent_name}"
        )

    hand_context = None
    if args.hand_reference_ik is not None:
        shengyin_root = Path(__file__).resolve().parents[4]
        if str(shengyin_root) not in sys.path:
            sys.path.insert(0, str(shengyin_root))
        os.environ["SHENGYIN_SKIP_ISAACGYM_IMPORT"] = "1"
        from human_policy.twist_hand_gmt_bridge import (
            _PinocchioInspireIK,
            _tips_world_to_local,
        )

        hand_reference = np.load(args.hand_reference_ik, allow_pickle=False)
        required_hand_keys = {
            "fps",
            "lh_root_pos",
            "lh_root_rot",
            "lh_tip_world",
            "lh_dof_pos",
            "rh_root_pos",
            "rh_root_rot",
            "rh_tip_world",
            "rh_dof_pos",
        }
        missing_hand_keys = sorted(required_hand_keys.difference(hand_reference.files))
        if missing_hand_keys:
            raise KeyError(
                f"Hand IK reference is missing required arrays: {missing_hand_keys}"
            )
        hand_context = {
            "fps": float(hand_reference["fps"]),
            "reference": hand_reference,
            "sides": {},
        }
        for side, prefix, hand_base_name, wrist_name in (
            ("left", "lh", "L_hand_base_link", "left_wrist_yaw_link"),
            ("right", "rh", "R_hand_base_link", "right_wrist_yaw_link"),
        ):
            solver = _PinocchioInspireIK(
                prefix,
                iters=1,
                damping=1e-3,
                step=0.7,
                smooth_w=1e-2,
                reg_w=1e-4,
            )
            hand_base_id = body_ids[hand_base_name]
            hand_context["sides"][side] = {
                "prefix": prefix,
                "solver": solver,
                "dof_pos": hand_reference[f"{prefix}_dof_pos"],
                "target_local": _tips_world_to_local(
                    hand_reference[f"{prefix}_root_pos"],
                    hand_reference[f"{prefix}_root_rot"],
                    hand_reference[f"{prefix}_tip_world"],
                ),
                "hand_base_id": hand_base_id,
                "hand_base_local_pos": model.body_pos[hand_base_id].copy(),
                "hand_base_local_quat": model.body_quat[hand_base_id].copy(),
                "wrist_reference_index": isaac_body_names.index(wrist_name),
            }

    context_len = 3
    root_quat = torch.as_tensor(data.qpos[3:7], dtype=torch.float32, device=device)
    dof_pos = torch.as_tensor(data.qpos[7:][actor_to_xml], dtype=torch.float32, device=device)
    dof_vel = torch.as_tensor(data.qvel[6:][actor_to_xml], dtype=torch.float32, device=device)
    gyro_name = (
        "base_gyro"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "base_gyro") >= 0
        else "imu-pelvis-angular-velocity"
    )
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, gyro_name)
    if gyro_id < 0:
        raise RuntimeError("MuJoCo asset has no pelvis gyro sensor.")
    gyro_adr = model.sensor_adr[gyro_id]
    base_ang_vel = torch.as_tensor(
        data.sensordata[gyro_adr : gyro_adr + 3], dtype=torch.float32, device=device
    )
    root_quat_history = root_quat.view(1, 1, 4).repeat(1, context_len, 1)
    base_ang_vel_history = base_ang_vel.view(1, 1, 3).repeat(1, context_len, 1)
    dof_pos_history = dof_pos.view(1, 1, 29).repeat(1, context_len, 1)
    dof_vel_history = dof_vel.view(1, 1, 29).repeat(1, context_len, 1)
    action_history = torch.zeros(1, context_len, 29, dtype=torch.float32, device=device)
    previous_action = torch.zeros(1, 29, dtype=torch.float32, device=device)

    mode = torch.zeros(14, dtype=torch.float32, device=device)
    for name in ACTIVE_BODY_NAMES:
        mode[SELECTED_BODY_NAMES.index(name)] = 1.0
    mode_mapping = torch.cat(
        [mode.repeat_interleave(size) for size in (3, 3, 6, 6)]
        + [torch.ones(1, device=device)]
    )
    if has_focus_phase_observation:
        mode_mapping = torch.cat(
            (mode_mapping, torch.ones(2, device=device)), dim=0
        )
    future_offsets = np.asarray([0, 1, 2, 3, 4, args.future_offset], dtype=np.int64)
    time_offsets = torch.as_tensor(
        future_offsets, dtype=torch.float32, device=device
    ).view(1, 6, 1)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    writer = None
    if not args.no_video:
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        writer = imageio.get_writer(
            args.video_out,
            fps=args.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.5
    camera.azimuth = 135
    camera.elevation = -15

    metrics = {
        name: {"position_m": [], "rotation_deg": []} for name in ACTIVE_BODY_NAMES
    }
    local_metrics = {
        name: {"position_m": [], "rotation_deg": []} for name in ACTIVE_BODY_NAMES
    }
    pelvis_frame_metrics = {
        name: {"position_m": [], "rotation_deg": []} for name in ACTIVE_BODY_NAMES
    }
    root_relative_position_metrics = {name: [] for name in ACTIVE_BODY_NAMES}
    rotation_audit = {
        name: {
            "reference_angular_speed_deg_s": [],
            "global_rotation_deg": [],
            "local_rotation_deg": [],
        }
        for name in ACTIVE_BODY_NAMES
    }
    head_metrics = {
        "global": {"position_m": [], "rotation_deg": []},
        "pelvis_frame_local": {"position_m": [], "rotation_deg": []},
    }
    head_fixed_transform_position_error_m = []
    head_fixed_transform_rotation_error_deg = []
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    fingertip_metrics = (
        {
            side: {
                finger: {
                    "global_position_m": [],
                    "pelvis_frame_local_position_m": [],
                    "wrist_frame_local_position_m": [],
                }
                for finger in finger_names
            }
            for side in ("left", "right")
        }
        if hand_context is not None
        else None
    )
    max_quaternion_vs_matrix_rotation_difference_deg = 0.0
    focus_trace = []
    joint_rmse = []
    max_active_position = []
    target_xml = data.qpos[7:].copy()
    render_every = max(1, round(1.0 / (args.video_fps * model.opt.timestep)))
    total_control_steps = len(reference["joint_pos"]) - 1

    with torch.inference_mode():
        for sim_step in range(total_control_steps * control_decimation):
            if sim_step % control_decimation == 0:
                frame = sim_step // control_decimation
                root_quat = torch.as_tensor(
                    data.qpos[3:7].copy(), dtype=torch.float32, device=device
                )
                base_ang_vel = torch.as_tensor(
                    data.sensordata[gyro_adr : gyro_adr + 3].copy(),
                    dtype=torch.float32,
                    device=device,
                )
                dof_pos = torch.as_tensor(
                    data.qpos[7:][actor_to_xml].copy(), dtype=torch.float32, device=device
                )
                dof_vel = torch.as_tensor(
                    data.qvel[6:][actor_to_xml].copy(), dtype=torch.float32, device=device
                )
                for history, value in (
                    (root_quat_history, root_quat),
                    (base_ang_vel_history, base_ang_vel),
                    (dof_pos_history, dof_pos),
                    (dof_vel_history, dof_vel),
                    (action_history, previous_action[0]),
                ):
                    history[:] = torch.roll(history, shifts=-1, dims=1)
                    history[:, -1] = value

                gravity = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 1, 3)
                projected_gravity = quat_apply(
                    quat_conjugate(root_quat_history), gravity.expand(1, context_len, 3)
                )
                prop_obs = torch.cat(
                    (
                        projected_gravity,
                        base_ang_vel_history,
                        dof_pos_history - default_dof_pos.unsqueeze(1),
                        dof_vel_history * 0.05,
                    ),
                    dim=-1,
                )

                future_frames = np.minimum(frame + future_offsets, len(reference["joint_pos"]) - 1)
                ref_pos = torch.as_tensor(
                    reference["body_pos_w"][future_frames][:, reference_body_indices],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                if args.reanchor_mode == "off":
                    ref_pos = ref_pos + torch.as_tensor(
                        injected_root_drift, dtype=torch.float32, device=device
                    ).view(1, 1, 1, 3)
                ref_quat = torch.as_tensor(
                    reference["body_quat_w"][future_frames][:, reference_body_indices],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                current_root_pos = torch.as_tensor(
                    data.xpos[body_ids["pelvis"]].copy(), dtype=torch.float32, device=device
                )
                current_body_pos = torch.stack(
                    [
                        torch.as_tensor(data.xpos[body_ids[name]].copy(), device=device)
                        for name in SELECTED_BODY_NAMES
                    ]
                ).float()
                current_body_quat = torch.stack(
                    [
                        torch.as_tensor(data.xquat[body_ids[name]].copy(), device=device)
                        for name in SELECTED_BODY_NAMES
                    ]
                ).float()
                root_inverse = quat_conjugate(root_quat).view(1, 1, 1, 4)
                root_expand = root_quat.view(1, 1, 1, 4)
                pos_to_root = quat_apply(
                    root_inverse.expand(1, 6, 14, 4),
                    ref_pos - current_root_pos.view(1, 1, 1, 3),
                )
                pos_error = quat_apply(
                    root_inverse.expand(1, 6, 14, 4),
                    ref_pos - current_body_pos.view(1, 1, 14, 3),
                )
                rot_to_root = quat_mul(root_inverse.expand(1, 6, 14, 4), ref_quat)
                rot_error = quat_mul(
                    quat_mul(
                        root_inverse.expand(1, 6, 14, 4),
                        quat_mul(
                            ref_quat,
                            quat_conjugate(current_body_quat).view(1, 1, 14, 4),
                        ),
                    ),
                    root_expand.expand(1, 6, 14, 4),
                )
                tangent = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 1, 3)
                normal = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3)
                rot_to_root_6d = torch.cat(
                    (
                        quat_apply(rot_to_root, tangent.expand(1, 6, 14, 3)),
                        quat_apply(rot_to_root, normal.expand(1, 6, 14, 3)),
                    ),
                    dim=-1,
                )
                rot_error_6d = torch.cat(
                    (
                        quat_apply(rot_error, tangent.expand(1, 6, 14, 3)),
                        quat_apply(rot_error, normal.expand(1, 6, 14, 3)),
                    ),
                    dim=-1,
                )
                task_obs = torch.cat(
                    (
                        pos_to_root.flatten(2),
                        pos_error.flatten(2),
                        rot_to_root_6d.flatten(2),
                        rot_error_6d.flatten(2),
                        time_offsets,
                    ),
                    dim=-1,
                )
                if has_focus_phase_observation:
                    phase_frame = min(frame, len(reference["joint_pos"]) - 1)
                    phase_valid = (
                        focus_state_mask is not None
                        and focus_time_to_next_edge_s is not None
                    )
                    phase_values = torch.tensor(
                        [
                            float(
                                bool(focus_state_mask[phase_frame])
                                if phase_valid
                                else False
                            ),
                            float(
                                np.clip(
                                    focus_time_to_next_edge_s[phase_frame],
                                    -2.0,
                                    2.0,
                                )
                                if phase_valid
                                else 2.0
                            ),
                        ],
                        dtype=torch.float32,
                        device=device,
                    ).view(1, 1, 2).expand(1, 6, 2)
                    task_obs = torch.cat((task_obs, phase_values), dim=-1)
                task_input = torch.cat(
                    (
                        task_obs * mode_mapping.view(1, 1, -1),
                        mode.view(1, 1, -1).expand(1, 6, 14),
                    ),
                    dim=-1,
                )
                action = actor(prop_obs, action_history, task_embedder(task_input))
                previous_action = action
                target_actor = (default_dof_pos + action * action_scale)[0].cpu().numpy()
                target_xml = target_actor[xml_to_actor]

            torque = stiffness * (target_xml - data.qpos[7:]) - damping * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, -torque_limit, torque_limit)
            mujoco.mj_step(model, data)

            if (sim_step + 1) % control_decimation == 0:
                reference_frame = min((sim_step + 1) // control_decimation, len(reference["joint_pos"]) - 1)
                active_errors = []
                actual_root_pos = data.xpos[body_ids["pelvis"]].copy()
                actual_root_quat = data.xquat[body_ids["pelvis"]].copy()
                reference_root_pos = reference["body_pos_w"][
                    reference_frame, active_reference_indices[0]
                ]
                reference_root_quat = reference["body_quat_w"][
                    reference_frame, active_reference_indices[0]
                ]
                heading_delta = heading_quat(
                    np_quat_mul(actual_root_quat, np_quat_conjugate(reference_root_quat))
                )
                aligned_reference_origin = actual_root_pos.copy()
                aligned_reference_origin[2] = reference_root_pos[2]
                for name, ref_index in zip(ACTIVE_BODY_NAMES, active_reference_indices):
                    body_id = body_ids[name]
                    actual_position = data.xpos[body_id].copy()
                    actual_quat = data.xquat[body_id].copy()
                    reference_position = reference["body_pos_w"][reference_frame, ref_index]
                    reference_quat = reference["body_quat_w"][reference_frame, ref_index]
                    position_error = float(
                        np.linalg.norm(actual_position - reference_position)
                    )
                    rotation_error = float(
                        quat_error_deg(
                            actual_quat[None],
                            reference_quat[None],
                        )[0]
                    )
                    matrix_rotation_error = rotation_matrix_error_deg(
                        actual_quat, reference_quat
                    )
                    max_quaternion_vs_matrix_rotation_difference_deg = max(
                        max_quaternion_vs_matrix_rotation_difference_deg,
                        abs(rotation_error - matrix_rotation_error),
                    )
                    metrics[name]["position_m"].append(position_error)
                    metrics[name]["rotation_deg"].append(rotation_error)

                    aligned_reference_position = aligned_reference_origin + np_quat_apply(
                        heading_delta, reference_position - reference_root_pos
                    )
                    aligned_reference_quat = np_quat_mul(heading_delta, reference_quat)
                    local_metrics[name]["position_m"].append(
                        float(np.linalg.norm(actual_position - aligned_reference_position))
                    )
                    local_rotation_error = float(
                        quat_error_deg(
                            actual_quat[None], aligned_reference_quat[None]
                        )[0]
                    )
                    local_metrics[name]["rotation_deg"].append(local_rotation_error)
                    previous_reference_frame = max(0, reference_frame - 1)
                    previous_reference_quat = reference["body_quat_w"][
                        previous_reference_frame, ref_index
                    ]
                    reference_angular_speed = float(
                        quat_error_deg(
                            reference_quat[None], previous_reference_quat[None]
                        )[0]
                        / (control_decimation * model.opt.timestep)
                    )
                    rotation_audit[name]["reference_angular_speed_deg_s"].append(
                        reference_angular_speed
                    )
                    rotation_audit[name]["global_rotation_deg"].append(rotation_error)
                    rotation_audit[name]["local_rotation_deg"].append(
                        local_rotation_error
                    )
                    root_relative_position_metrics[name].append(
                        float(
                            np.linalg.norm(
                                (actual_position - actual_root_pos)
                                - (reference_position - reference_root_pos)
                            )
                        )
                    )
                    actual_position_pelvis = local_point(
                        actual_root_pos, actual_root_quat, actual_position
                    )
                    reference_position_pelvis = local_point(
                        reference_root_pos, reference_root_quat, reference_position
                    )
                    actual_quat_pelvis = local_quat(
                        actual_root_quat, actual_quat
                    )
                    reference_quat_pelvis = local_quat(
                        reference_root_quat, reference_quat
                    )
                    pelvis_frame_metrics[name]["position_m"].append(
                        float(
                            np.linalg.norm(
                                actual_position_pelvis
                                - reference_position_pelvis
                            )
                        )
                    )
                    pelvis_frame_metrics[name]["rotation_deg"].append(
                        float(
                            quat_error_deg(
                                actual_quat_pelvis[None],
                                reference_quat_pelvis[None],
                            )[0]
                        )
                    )
                    active_errors.append(position_error)

                reference_torso_pos = reference["body_pos_w"][
                    reference_frame, torso_reference_index
                ]
                reference_torso_quat = reference["body_quat_w"][
                    reference_frame, torso_reference_index
                ]
                reference_head_pos, reference_head_quat = pose_from_parent(
                    reference_torso_pos,
                    reference_torso_quat,
                    head_local_pos,
                    head_local_quat,
                )
                actual_head_pos = data.xpos[head_body_id].copy()
                actual_head_quat = data.xquat[head_body_id].copy()
                actual_torso_pos = data.xpos[body_ids["torso_link"]].copy()
                actual_torso_quat = data.xquat[body_ids["torso_link"]].copy()
                observed_head_local_pos = local_point(
                    actual_torso_pos, actual_torso_quat, actual_head_pos
                )
                observed_head_local_quat = local_quat(
                    actual_torso_quat, actual_head_quat
                )
                head_fixed_transform_position_error_m.append(
                    float(np.linalg.norm(observed_head_local_pos - head_local_pos))
                )
                head_fixed_transform_rotation_error_deg.append(
                    float(
                        quat_error_deg(
                            observed_head_local_quat[None],
                            head_local_quat[None],
                        )[0]
                    )
                )
                head_metrics["global"]["position_m"].append(
                    float(np.linalg.norm(actual_head_pos - reference_head_pos))
                )
                head_metrics["global"]["rotation_deg"].append(
                    float(
                        quat_error_deg(
                            actual_head_quat[None], reference_head_quat[None]
                        )[0]
                    )
                )
                actual_head_local_pos = local_point(
                    actual_root_pos, actual_root_quat, actual_head_pos
                )
                reference_head_local_pos = local_point(
                    reference_root_pos, reference_root_quat, reference_head_pos
                )
                actual_head_local_quat = local_quat(
                    actual_root_quat, actual_head_quat
                )
                reference_head_local_quat = local_quat(
                    reference_root_quat, reference_head_quat
                )
                head_metrics["pelvis_frame_local"]["position_m"].append(
                    float(
                        np.linalg.norm(
                            actual_head_local_pos - reference_head_local_pos
                        )
                    )
                )
                head_metrics["pelvis_frame_local"]["rotation_deg"].append(
                    float(
                        quat_error_deg(
                            actual_head_local_quat[None],
                            reference_head_local_quat[None],
                        )[0]
                    )
                )

                if hand_context is not None:
                    hand_frame_f = (
                        reference_frame
                        * control_decimation
                        * model.opt.timestep
                        * hand_context["fps"]
                    )
                    for side, values in hand_context["sides"].items():
                        hand_q = interpolate_sequence(values["dof_pos"], hand_frame_f)
                        target_tip_local = interpolate_sequence(
                            values["target_local"], hand_frame_f
                        )
                        actual_tip_local = values["solver"]._fk_tips(
                            np.asarray(hand_q, dtype=np.float64)
                        )

                        actual_hand_base_pos = data.xpos[
                            values["hand_base_id"]
                        ].copy()
                        actual_hand_base_quat = data.xquat[
                            values["hand_base_id"]
                        ].copy()
                        wrist_reference_index = values["wrist_reference_index"]
                        reference_wrist_pos = reference["body_pos_w"][
                            reference_frame, wrist_reference_index
                        ]
                        reference_wrist_quat = reference["body_quat_w"][
                            reference_frame, wrist_reference_index
                        ]
                        (
                            reference_hand_base_pos,
                            reference_hand_base_quat,
                        ) = pose_from_parent(
                            reference_wrist_pos,
                            reference_wrist_quat,
                            values["hand_base_local_pos"],
                            values["hand_base_local_quat"],
                        )
                        actual_tip_world = actual_hand_base_pos + np_quat_apply(
                            actual_hand_base_quat, actual_tip_local
                        )
                        reference_tip_world = (
                            reference_hand_base_pos
                            + np_quat_apply(
                                reference_hand_base_quat, target_tip_local
                            )
                        )
                        for finger_index, finger_name in enumerate(finger_names):
                            finger_values = fingertip_metrics[side][finger_name]
                            finger_values["global_position_m"].append(
                                float(
                                    np.linalg.norm(
                                        actual_tip_world[finger_index]
                                        - reference_tip_world[finger_index]
                                    )
                                )
                            )
                            actual_tip_pelvis = local_point(
                                actual_root_pos,
                                actual_root_quat,
                                actual_tip_world[finger_index],
                            )
                            reference_tip_pelvis = local_point(
                                reference_root_pos,
                                reference_root_quat,
                                reference_tip_world[finger_index],
                            )
                            finger_values[
                                "pelvis_frame_local_position_m"
                            ].append(
                                float(
                                    np.linalg.norm(
                                        actual_tip_pelvis - reference_tip_pelvis
                                    )
                                )
                            )
                            finger_values[
                                "wrist_frame_local_position_m"
                            ].append(
                                float(
                                    np.linalg.norm(
                                        actual_tip_local[finger_index]
                                        - target_tip_local[finger_index]
                                    )
                                )
                            )
                focus_trace.append(
                    bool(focus_state_mask[reference_frame])
                    if focus_state_mask is not None
                    else False
                )
                max_active_position.append(max(active_errors))
                actual_actor = data.qpos[7:][actor_to_xml]
                joint_rmse.append(
                    float(
                        np.sqrt(
                            np.square(actual_actor - reference["joint_pos"][reference_frame]).mean()
                        )
                    )
                )

            if writer is not None and renderer is not None and sim_step % render_every == 0:
                frame = min(sim_step // control_decimation, len(reference["joint_pos"]) - 1)
                camera.lookat[:] = data.xpos[body_ids["pelvis"]]
                renderer.update_scene(data, camera=camera)
                add_reference_markers(
                    renderer, reference["body_pos_w"][frame, active_reference_indices]
                )
                writer.append_data(renderer.render())

    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()
    max_active_position = np.asarray(max_active_position)
    fingertip_report = None
    if fingertip_metrics is not None:
        metric_names = (
            "global_position_m",
            "pelvis_frame_local_position_m",
            "wrist_frame_local_position_m",
        )
        fingertip_report = {
            "definition": {
                "global_position_m": "virtual IK/FK fingertip versus target fingertip in world coordinates",
                "pelvis_frame_local_position_m": "each fingertip expressed in its own actual/reference full pelvis frame",
                "wrist_frame_local_position_m": "finger-only Pinocchio IK/FK residual in each hand-base frame",
            },
            "per_finger": {
                side: {
                    finger: {
                        metric_name: summary(values[metric_name])
                        for metric_name in metric_names
                    }
                    for finger, values in side_values.items()
                }
                for side, side_values in fingertip_metrics.items()
            },
            "per_hand_all_five_fingers": {},
            "both_hands_all_ten_fingers": {},
        }
        for side, side_values in fingertip_metrics.items():
            fingertip_report["per_hand_all_five_fingers"][side] = {
                metric_name: summary(
                    np.concatenate(
                        [
                            np.asarray(values[metric_name], dtype=np.float64)
                            for values in side_values.values()
                        ]
                    )
                )
                for metric_name in metric_names
            }
        fingertip_report["both_hands_all_ten_fingers"] = {
            metric_name: summary(
                np.concatenate(
                    [
                        np.asarray(values[metric_name], dtype=np.float64)
                        for side_values in fingertip_metrics.values()
                        for values in side_values.values()
                    ]
                )
            )
            for metric_name in metric_names
        }

    phase_report = None
    focus_array = np.asarray(focus_trace, dtype=bool)
    if focus_state_mask is not None:
        phase_report = {"metadata": focus_metadata, "buckets": {}}
        for phase_name, phase_mask in (
            ("focus", focus_array),
            ("non_focus", ~focus_array),
        ):
            bucket = {
                "frames": int(phase_mask.sum()),
                "duration_s": float(
                    phase_mask.sum() * control_decimation * model.opt.timestep
                ),
                "active_links_global": {
                    name: {
                        "position_m": summary(
                            np.asarray(values["position_m"])[phase_mask]
                        ),
                        "rotation_deg": summary(
                            np.asarray(values["rotation_deg"])[phase_mask]
                        ),
                    }
                    for name, values in metrics.items()
                },
                "active_links_pelvis_frame_local_twist": {
                    name: {
                        "position_m": summary(
                            np.asarray(values["position_m"])[phase_mask]
                        ),
                        "rotation_deg": summary(
                            np.asarray(values["rotation_deg"])[phase_mask]
                        ),
                    }
                    for name, values in pelvis_frame_metrics.items()
                },
                "active_links_heading_aligned_local_paper": {
                    name: {
                        "position_m": summary(
                            np.asarray(values["position_m"])[phase_mask]
                        ),
                        "rotation_deg": summary(
                            np.asarray(values["rotation_deg"])[phase_mask]
                        ),
                    }
                    for name, values in local_metrics.items()
                },
                "head_link_proxy": {
                    scope: {
                        metric_name: summary(np.asarray(values)[phase_mask])
                        for metric_name, values in scope_values.items()
                    }
                    for scope, scope_values in head_metrics.items()
                },
            }
            if fingertip_metrics is not None:
                bucket["inspire_hand_ik_fingertips"] = {
                    "per_finger": {
                        side: {
                            finger: {
                                metric_name: summary(
                                    np.asarray(values)[phase_mask]
                                )
                                for metric_name, values in finger_values.items()
                            }
                            for finger, finger_values in side_values.items()
                        }
                        for side, side_values in fingertip_metrics.items()
                    },
                    "both_hands_all_ten_fingers": {
                        metric_name: summary(
                            np.concatenate(
                                [
                                    np.asarray(values[metric_name])[phase_mask]
                                    for side_values in fingertip_metrics.values()
                                    for values in side_values.values()
                                ]
                            )
                        )
                        for metric_name in metric_names
                    },
                }
            phase_report["buckets"][phase_name] = bucket

    if args.trace_out is not None:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "focus_mask": focus_array,
            "active_link_names": np.asarray(ACTIVE_BODY_NAMES),
            "active_global_position_m": np.stack(
                [np.asarray(metrics[name]["position_m"]) for name in ACTIVE_BODY_NAMES],
                axis=1,
            ),
            "active_global_rotation_deg": np.stack(
                [np.asarray(metrics[name]["rotation_deg"]) for name in ACTIVE_BODY_NAMES],
                axis=1,
            ),
            "active_pelvis_local_position_m": np.stack(
                [
                    np.asarray(pelvis_frame_metrics[name]["position_m"])
                    for name in ACTIVE_BODY_NAMES
                ],
                axis=1,
            ),
            "active_pelvis_local_rotation_deg": np.stack(
                [
                    np.asarray(pelvis_frame_metrics[name]["rotation_deg"])
                    for name in ACTIVE_BODY_NAMES
                ],
                axis=1,
            ),
            "active_reference_angular_speed_deg_s": np.stack(
                [
                    np.asarray(rotation_audit[name]["reference_angular_speed_deg_s"])
                    for name in ACTIVE_BODY_NAMES
                ],
                axis=1,
            ),
            "active_heading_local_position_m": np.stack(
                [
                    np.asarray(local_metrics[name]["position_m"])
                    for name in ACTIVE_BODY_NAMES
                ],
                axis=1,
            ),
            "active_heading_local_rotation_deg": np.stack(
                [
                    np.asarray(local_metrics[name]["rotation_deg"])
                    for name in ACTIVE_BODY_NAMES
                ],
                axis=1,
            ),
            "head_global_position_m": np.asarray(
                head_metrics["global"]["position_m"]
            ),
            "head_global_rotation_deg": np.asarray(
                head_metrics["global"]["rotation_deg"]
            ),
            "head_pelvis_local_position_m": np.asarray(
                head_metrics["pelvis_frame_local"]["position_m"]
            ),
            "head_pelvis_local_rotation_deg": np.asarray(
                head_metrics["pelvis_frame_local"]["rotation_deg"]
            ),
        }
        if fingertip_metrics is not None:
            for metric_name in metric_names:
                trace_payload[f"fingertip_{metric_name}"] = np.stack(
                    [
                        np.stack(
                            [
                                np.asarray(
                                    fingertip_metrics[side][finger][metric_name]
                                )
                                for finger in finger_names
                            ],
                            axis=1,
                        )
                        for side in ("left", "right")
                    ],
                    axis=1,
                )
        np.savez_compressed(args.trace_out, **trace_payload)

    report = {
        "simulator": "MuJoCo",
        "control_mode": {
            "index": {"3": 2, "4": 8, "14": 7}[args.tracking_mode],
            "name": {
                "3": "VR-3",
                "4": "HAT-4",
                "14": "WholeBody-14",
            }[args.tracking_mode],
            "active_links": ACTIVE_BODY_NAMES,
        },
        "actor_task_input_dim": task_input_dim,
        "focus_phase_observation_active": has_focus_phase_observation,
        "checkpoint": str(args.checkpoint.resolve()),
        "reference": str(args.reference.resolve()),
        "hand_reference_ik": (
            str(args.hand_reference_ik.resolve())
            if args.hand_reference_ik is not None
            else None
        ),
        "xml": str(args.xml.resolve()),
        "pelvis_gyro_sensor": gyro_name,
        "control_dt": control_decimation * model.opt.timestep,
        "simulation_dt": model.opt.timestep,
        "reanchor_ablation": {
            "mode": args.reanchor_mode,
            "injected_initial_root_drift_xyz_m": injected_root_drift.tolist(),
            "evaluation_reference": "original fixed world reference",
            "policy_reference": (
                "original fixed world reference"
                if args.reanchor_mode == "on"
                else "reference translated by injected drift (stale root-local target)"
            ),
        },
        "evaluated_transitions": total_control_steps,
        "duration_sec": total_control_steps * control_decimation * model.opt.timestep,
        "future_offsets": future_offsets.tolist(),
        "active_link_pose_vs_reference": {
            name: {
                "position_m": summary(values["position_m"]),
                "rotation_deg": summary(values["rotation_deg"]),
            }
            for name, values in metrics.items()
        },
        "active_link_pose_local_heading_aligned": {
            name: {
                "position_m": summary(values["position_m"]),
                "rotation_deg": summary(values["rotation_deg"]),
            }
            for name, values in local_metrics.items()
        },
        "active_link_pose_pelvis_frame_local_twist": {
            name: {
                "position_m": summary(values["position_m"]),
                "rotation_deg": summary(values["rotation_deg"]),
            }
            for name, values in pelvis_frame_metrics.items()
        },
        "active_link_position_root_relative_no_heading_alignment": {
            name: summary(values)
            for name, values in root_relative_position_metrics.items()
        },
        "head_link_proxy": {
            "mujoco_body": "head_mocap",
            "note": "The TWIST XML merges the visual head_link mesh into torso_link; head_mocap is its fixed tracking body.",
            "global": {
                "position_m": summary(head_metrics["global"]["position_m"]),
                "rotation_deg": summary(head_metrics["global"]["rotation_deg"]),
            },
            "pelvis_frame_local": {
                "position_m": summary(
                    head_metrics["pelvis_frame_local"]["position_m"]
                ),
                "rotation_deg": summary(
                    head_metrics["pelvis_frame_local"]["rotation_deg"]
                ),
            },
            "fixed_transform_audit": {
                "parent_body": head_parent_name,
                "configured_parent_local_position_m": head_local_pos.tolist(),
                "configured_parent_local_quaternion_wxyz": head_local_quat.tolist(),
                "runtime_position_error_m": summary(
                    head_fixed_transform_position_error_m
                ),
                "runtime_rotation_error_deg": summary(
                    head_fixed_transform_rotation_error_deg
                ),
            },
        },
        "inspire_hand_ik_fingertips": fingertip_report,
        "focus_evaluation": phase_report,
        "trace_out": str(args.trace_out.resolve()) if args.trace_out else None,
        "rotation_metric_audit": {
            "max_quaternion_vs_rotation_matrix_difference_deg": (
                max_quaternion_vs_matrix_rotation_difference_deg
            ),
            "reference_motion_frames_over_30_deg_s": {
                name: {
                    "frames": int(
                        np.sum(
                            np.asarray(values["reference_angular_speed_deg_s"]) >= 30.0
                        )
                    ),
                    "global_rotation_deg": summary(
                        np.asarray(values["global_rotation_deg"])[
                            np.asarray(values["reference_angular_speed_deg_s"]) >= 30.0
                        ]
                    ),
                    "local_rotation_deg": summary(
                        np.asarray(values["local_rotation_deg"])[
                            np.asarray(values["reference_angular_speed_deg_s"]) >= 30.0
                        ]
                    ),
                }
                for name, values in rotation_audit.items()
            },
        },
        "joint_rmse_rad": summary(joint_rmse),
        "max_active_link_position_m": summary(max_active_position),
        "success_threshold_m": 0.5,
        "success_rate_frames": float(np.mean(max_active_position <= 0.5)),
        "first_failure_frame": (
            int(np.flatnonzero(max_active_position > 0.5)[0])
            if np.any(max_active_position > 0.5)
            else None
        ),
        "video": None if args.no_video else str(args.video_out.resolve()),
    }
    args.metrics_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
