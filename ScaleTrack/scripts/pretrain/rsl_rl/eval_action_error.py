"""Run one ScaleTrack reference and report policy-action tracking errors."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="G1-BFM-Transformer-Tracking")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument(
    "--mode_index",
    type=int,
    default=2,
    help="Fixed ScaleBFM control mode. Index 2 is pelvis + left/right hand.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

import scaletrack.tasks  # noqa: F401


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def error_summary(error, joint_names, unit):
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    result = {
        f"all_joint_values_{unit}": scalar_summary(absolute),
        f"per_frame_joint_l2_{unit}": scalar_summary(np.linalg.norm(error, axis=1)),
        "per_joint": {},
    }
    for index, name in enumerate(joint_names):
        result["per_joint"][name] = scalar_summary(absolute[:, index])
    wrist_indices = [index for index, name in enumerate(joint_names) if "wrist" in name]
    arm_indices = [
        index
        for index, name in enumerate(joint_names)
        if any(token in name for token in ("shoulder", "elbow", "wrist"))
    ]
    result[f"wrist_joint_values_{unit}"] = scalar_summary(absolute[:, wrist_indices])
    result[f"arm_joint_values_{unit}"] = scalar_summary(absolute[:, arm_indices])
    return result


def quaternion_error_deg(reference, actual):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    reference /= np.linalg.norm(reference, axis=-1, keepdims=True)
    actual /= np.linalg.norm(actual, axis=-1, keepdims=True)
    dot = np.abs(np.sum(reference * actual, axis=-1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.enable_reset_disturbance = False
    env_cfg.commands.motion.debug_vis = False
    mode_candidates = list(env_cfg.commands.motion.mode_candidates.items())
    if not 0 <= args_cli.mode_index < len(mode_candidates):
        raise ValueError(
            f"mode_index must be in [0, {len(mode_candidates) - 1}], got {args_cli.mode_index}"
        )
    mode_name, active_link_names = mode_candidates[args_cli.mode_index]
    env_cfg.commands.motion.mode_candidates = {mode_name: active_link_names}
    env_cfg.commands.motion.mode_sampling_weights = {mode_name: 1.0}

    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    checkpoint = get_checkpoint_path(
        str(log_root.resolve()), agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] motion={args_cli.motion_file}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    runner._set_env_is_evaluating()
    # Normal inference deliberately uses the complete 14-link target and replaces
    # the mode with all ones. Reuse the training observation path so this
    # evaluation really applies the selected sparse-link mask.
    actor_critic = runner.alg.policy
    get_actor_obs = actor_critic.get_actor_obs

    def get_fixed_mode_actor_obs(obs, inference=False):
        return get_actor_obs(obs, inference=False)

    actor_critic.get_actor_obs = get_fixed_mode_actor_obs
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command = env.unwrapped.command_manager.get_term("motion")
    command.motion_ids[:] = 0
    motion_length = int(command.time_totals[0].item())
    if args_cli.num_envs != 1:
        raise ValueError("This evaluator intentionally supports exactly one environment.")

    obs, _ = env.reset()
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    robot = env.unwrapped.scene["robot"]
    joint_names = list(action_term._joint_names)
    if joint_names != list(robot.joint_names):
        raise RuntimeError("Action joint order does not match robot joint-state order.")

    raw_action_log = []
    desired_joint_log = []
    reference_before_log = []
    reference_after_log = []
    actual_joint_log = []
    time_before_log = []
    time_after_log = []
    policy_obs_log = []
    policy_task_log = []
    action_obs_log = []
    mode_obs_log = []
    mode_mapping_log = []
    robot_root_pos_log = []
    robot_root_quat_log = []
    robot_base_ang_vel_log = []
    robot_body_pos_log = []
    robot_body_quat_log = []
    wrist_pose = {
        "left": {"pos_m": [], "rot_deg": []},
        "right": {"pos_m": [], "rot_deg": []},
    }
    wrist_pose_pelvis_local = {
        "left": {"pos_m": [], "rot_deg": []},
        "right": {"pos_m": [], "rot_deg": []},
    }
    wrist_pose_focus_pelvis_local = {
        "left": {"pos_m": [], "rot_deg": []},
        "right": {"pos_m": [], "rot_deg": []},
    }
    focus_trace = []
    body_pos_g_trace = []
    diagnostic_pose = {
        "root": {
            "global": {"pos_m": [], "rot_deg": []},
        },
        "torso": {
            "global": {"pos_m": [], "rot_deg": []},
            "pelvis_local": {"pos_m": [], "rot_deg": []},
        },
        "head_proxy": {
            "global": {"pos_m": [], "rot_deg": []},
            "pelvis_local": {"pos_m": [], "rot_deg": []},
        },
    }
    active_link_pose = {
        body_name: {"pos_m": [], "rot_deg": []} for body_name in active_link_names
    }
    command_body_names = list(command.cfg.body_names)
    torso_command_index = command_body_names.index("torso_link")
    torso_robot_index = robot.body_names.index("torso_link")
    body_pairs = {
        "left": (
            command_body_names.index("left_wrist_yaw_link"),
            robot.body_names.index("left_wrist_yaw_link"),
        ),
        "right": (
            command_body_names.index("right_wrist_yaw_link"),
            robot.body_names.index("right_wrist_yaw_link"),
        ),
    }

    # There are N reference states and N-1 transitions.  Each action is
    # conditioned on frame t; after the simulator step the reference is t+1.
    for _ in range(motion_length - 1):
        time_before_log.append(int(command.time_steps[0].item()))
        reference_before_log.append(command.joint_pos[0].detach().cpu().numpy().copy())
        policy_obs_log.append(obs["policy"][0].detach().cpu().numpy().copy())
        policy_task_log.append(obs["policy_task"][0].detach().cpu().numpy().copy())
        action_obs_log.append(obs["action"][0].detach().cpu().numpy().copy())
        mode_obs_log.append(obs["mode"][0].detach().cpu().numpy().copy())
        mode_mapping_log.append(obs["mode_mapping"][0].detach().cpu().numpy().copy())
        robot_root_pos_log.append(command.robot_anchor_pos_w[0].detach().cpu().numpy().copy())
        robot_root_quat_log.append(command.robot_anchor_quat_w[0].detach().cpu().numpy().copy())
        robot_base_ang_vel_log.append(robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy())
        robot_body_pos_log.append(command.robot_body_pos_w[0].detach().cpu().numpy().copy())
        robot_body_quat_log.append(command.robot_body_quat_w[0].detach().cpu().numpy().copy())
        with torch.inference_mode():
            raw_action = policy(obs)
            obs, _, _, _ = env.step(raw_action)

        reference_anchor_pos = command.anchor_pos_w[0]
        reference_anchor_quat = command.anchor_quat_w[0]
        actual_anchor_pos = command.robot_anchor_pos_w[0]
        actual_anchor_quat = command.robot_anchor_quat_w[0]
        in_focus = bool((command.focus_flag[0] & command.focus_valid[0]).item())
        focus_trace.append(in_focus)
        body_pos_g_trace.append(
            float(
                torch.linalg.vector_norm(
                    command.body_pos_w[0] - command.robot_body_pos_w[0], dim=-1
                ).mean()
            )
        )
        diagnostic_pose["root"]["global"]["pos_m"].append(
            float(torch.linalg.vector_norm(reference_anchor_pos - actual_anchor_pos))
        )
        diagnostic_pose["root"]["global"]["rot_deg"].append(
            float(
                quaternion_error_deg(
                    reference_anchor_quat.detach().cpu().numpy()[None],
                    actual_anchor_quat.detach().cpu().numpy()[None],
                )[0]
            )
        )

        reference_torso_pos = command.body_pos_w[0, torso_command_index]
        reference_torso_quat = command.body_quat_w[0, torso_command_index]
        actual_torso_pos = robot.data.body_pos_w[0, torso_robot_index]
        actual_torso_quat = robot.data.body_quat_w[0, torso_robot_index]
        diagnostic_pose["torso"]["global"]["pos_m"].append(
            float(torch.linalg.vector_norm(reference_torso_pos - actual_torso_pos))
        )
        diagnostic_pose["torso"]["global"]["rot_deg"].append(
            float(
                quaternion_error_deg(
                    reference_torso_quat.detach().cpu().numpy()[None],
                    actual_torso_quat.detach().cpu().numpy()[None],
                )[0]
            )
        )
        reference_torso_local_pos = quat_apply(
            quat_inv(reference_anchor_quat),
            reference_torso_pos - reference_anchor_pos,
        )
        actual_torso_local_pos = quat_apply(
            quat_inv(actual_anchor_quat),
            actual_torso_pos - actual_anchor_pos,
        )
        reference_torso_local_quat = quat_mul(
            quat_inv(reference_anchor_quat), reference_torso_quat
        )
        actual_torso_local_quat = quat_mul(
            quat_inv(actual_anchor_quat), actual_torso_quat
        )
        diagnostic_pose["torso"]["pelvis_local"]["pos_m"].append(
            float(
                torch.linalg.vector_norm(
                    reference_torso_local_pos - actual_torso_local_pos
                )
            )
        )
        diagnostic_pose["torso"]["pelvis_local"]["rot_deg"].append(
            float(
                quaternion_error_deg(
                    reference_torso_local_quat.detach().cpu().numpy()[None],
                    actual_torso_local_quat.detach().cpu().numpy()[None],
                )[0]
            )
        )
        head_offset = torch.tensor(
            [0.0, 0.0, 0.4],
            dtype=reference_torso_pos.dtype,
            device=reference_torso_pos.device,
        )
        reference_head_pos = reference_torso_pos + quat_apply(
            reference_torso_quat, head_offset
        )
        actual_head_pos = actual_torso_pos + quat_apply(
            actual_torso_quat, head_offset
        )
        diagnostic_pose["head_proxy"]["global"]["pos_m"].append(
            float(torch.linalg.vector_norm(reference_head_pos - actual_head_pos))
        )
        diagnostic_pose["head_proxy"]["global"]["rot_deg"].append(
            diagnostic_pose["torso"]["global"]["rot_deg"][-1]
        )
        reference_head_local_pos = quat_apply(
            quat_inv(reference_anchor_quat),
            reference_head_pos - reference_anchor_pos,
        )
        actual_head_local_pos = quat_apply(
            quat_inv(actual_anchor_quat),
            actual_head_pos - actual_anchor_pos,
        )
        diagnostic_pose["head_proxy"]["pelvis_local"]["pos_m"].append(
            float(
                torch.linalg.vector_norm(
                    reference_head_local_pos - actual_head_local_pos
                )
            )
        )
        diagnostic_pose["head_proxy"]["pelvis_local"]["rot_deg"].append(
            diagnostic_pose["torso"]["pelvis_local"]["rot_deg"][-1]
        )
        raw_action_log.append(raw_action[0].detach().cpu().numpy().copy())
        desired_joint_log.append(
            action_term.processed_actions[0].detach().cpu().numpy().copy()
        )
        reference_after_log.append(command.joint_pos[0].detach().cpu().numpy().copy())
        actual_joint_log.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        time_after_log.append(int(command.time_steps[0].item()))

        for side, (command_index, robot_index) in body_pairs.items():
            ref_pos = command.body_pos_w[0, command_index].detach().cpu().numpy()
            act_pos = robot.data.body_pos_w[0, robot_index].detach().cpu().numpy()
            wrist_pose[side]["pos_m"].append(float(np.linalg.norm(ref_pos - act_pos)))
            ref_quat = command.body_quat_w[0, command_index].detach().cpu().numpy()
            act_quat = robot.data.body_quat_w[0, robot_index].detach().cpu().numpy()
            wrist_pose[side]["rot_deg"].append(
                float(quaternion_error_deg(ref_quat[None], act_quat[None])[0])
            )
            reference_local_pos = quat_apply(
                quat_inv(reference_anchor_quat),
                command.body_pos_w[0, command_index] - reference_anchor_pos,
            )
            actual_local_pos = quat_apply(
                quat_inv(actual_anchor_quat),
                robot.data.body_pos_w[0, robot_index] - actual_anchor_pos,
            )
            reference_local_quat = quat_mul(
                quat_inv(reference_anchor_quat),
                command.body_quat_w[0, command_index],
            )
            actual_local_quat = quat_mul(
                quat_inv(actual_anchor_quat),
                robot.data.body_quat_w[0, robot_index],
            )
            local_pos_error = float(
                torch.linalg.vector_norm(
                    reference_local_pos - actual_local_pos
                ).item()
            )
            local_rot_error = float(
                quaternion_error_deg(
                    reference_local_quat.detach().cpu().numpy()[None],
                    actual_local_quat.detach().cpu().numpy()[None],
                )[0]
            )
            wrist_pose_pelvis_local[side]["pos_m"].append(local_pos_error)
            wrist_pose_pelvis_local[side]["rot_deg"].append(local_rot_error)
            if in_focus:
                wrist_pose_focus_pelvis_local[side]["pos_m"].append(
                    local_pos_error
                )
                wrist_pose_focus_pelvis_local[side]["rot_deg"].append(
                    local_rot_error
                )
        for body_name, values in active_link_pose.items():
            command_index = command_body_names.index(body_name)
            robot_index = robot.body_names.index(body_name)
            ref_pos = command.body_pos_w[0, command_index].detach().cpu().numpy()
            act_pos = robot.data.body_pos_w[0, robot_index].detach().cpu().numpy()
            values["pos_m"].append(float(np.linalg.norm(ref_pos - act_pos)))
            ref_quat = command.body_quat_w[0, command_index].detach().cpu().numpy()
            act_quat = robot.data.body_quat_w[0, robot_index].detach().cpu().numpy()
            values["rot_deg"].append(
                float(quaternion_error_deg(ref_quat[None], act_quat[None])[0])
            )

    raw_action = np.asarray(raw_action_log)
    desired_joint = np.asarray(desired_joint_log)
    reference_before = np.asarray(reference_before_log)
    reference_after = np.asarray(reference_after_log)
    actual_joint = np.asarray(actual_joint_log)
    action_scale = action_term._scale
    action_offset = action_term._offset
    if isinstance(action_scale, torch.Tensor):
        action_scale = action_scale[0].detach().cpu().numpy()
    else:
        action_scale = np.full(len(joint_names), action_scale)
    if isinstance(action_offset, torch.Tensor):
        action_offset = action_offset[0].detach().cpu().numpy()
    else:
        action_offset = np.full(len(joint_names), action_offset)
    reference_action_next = (reference_after - action_offset) / action_scale

    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "motion_file": str(Path(args_cli.motion_file).resolve()),
        "motion_name": command.motion_names[0],
        "control_mode": {
            "index": args_cli.mode_index,
            "name": mode_name,
            "active_links": active_link_names,
        },
        "sampled_long_future_offset": int(command._rand_timestep[0, 0].item()),
        "reference_fps": 50,
        "reference_frames": motion_length,
        "evaluated_transitions": motion_length - 1,
        "duration_sec": (motion_length - 1) / 50.0,
        "tracking_success": bool(max(body_pos_g_trace) <= 0.5),
        "tracking_success_definition": (
            "max over frames of mean 14-link global position error <= 0.5 m"
        ),
        "body_pos_g_m": scalar_summary(body_pos_g_trace),
        "focus_frames": int(sum(focus_trace)),
        "joint_names": joint_names,
        "alignment": {
            "policy_observation_reference": "frame_t",
            "primary_error_reference": "frame_t_plus_1_after_sim_step",
            "time_before_first_last": [time_before_log[0], time_before_log[-1]],
            "time_after_first_last": [time_after_log[0], time_after_log[-1]],
        },
        "raw_policy_action": scalar_summary(np.abs(raw_action)),
        "normalized_action_vs_reference_next": error_summary(
            raw_action - reference_action_next, joint_names, "normalized"
        ),
        "desired_joint_vs_reference_next": error_summary(
            desired_joint - reference_after, joint_names, "rad"
        ),
        "desired_joint_vs_reference_current": error_summary(
            desired_joint - reference_before, joint_names, "rad"
        ),
        "sim_joint_vs_reference_next": error_summary(
            actual_joint - reference_after, joint_names, "rad"
        ),
        "wrist_pose_vs_reference_next": {
            side: {
                "position_m": scalar_summary(values["pos_m"]),
                "rotation_deg": scalar_summary(values["rot_deg"]),
            }
            for side, values in wrist_pose.items()
        },
        "wrist_pose_pelvis_local_vs_reference_next": {
            side: {
                "position_m": scalar_summary(values["pos_m"]),
                "rotation_deg": scalar_summary(values["rot_deg"]),
            }
            for side, values in wrist_pose_pelvis_local.items()
        },
        "wrist_pose_focus_pelvis_local_vs_reference_next": {
            side: {
                "position_m": scalar_summary(values["pos_m"]),
                "rotation_deg": scalar_summary(values["rot_deg"]),
            }
            for side, values in wrist_pose_focus_pelvis_local.items()
        },
        "diagnostic_pose_vs_reference_next": {
            body_name: {
                frame_name: {
                    "position_m": scalar_summary(frame_values["pos_m"]),
                    "rotation_deg": scalar_summary(frame_values["rot_deg"]),
                }
                for frame_name, frame_values in body_values.items()
            }
            for body_name, body_values in diagnostic_pose.items()
        },
        "active_link_pose_vs_reference_next": {
            body_name: {
                "position_m": scalar_summary(values["pos_m"]),
                "rotation_deg": scalar_summary(values["rot_deg"]),
            }
            for body_name, values in active_link_pose.items()
        },
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"),
        raw_action=raw_action,
        desired_joint=desired_joint,
        reference_before=reference_before,
        reference_after=reference_after,
        actual_joint=actual_joint,
        time_before=np.asarray(time_before_log),
        time_after=np.asarray(time_after_log),
        joint_names=np.asarray(joint_names),
        action_scale=action_scale,
        action_offset=action_offset,
        reference_action_next=reference_action_next,
        policy_obs=np.asarray(policy_obs_log),
        policy_task=np.asarray(policy_task_log),
        action_obs=np.asarray(action_obs_log),
        mode_obs=np.asarray(mode_obs_log),
        mode_mapping=np.asarray(mode_mapping_log),
        robot_root_pos=np.asarray(robot_root_pos_log),
        robot_root_quat=np.asarray(robot_root_quat_log),
        robot_base_ang_vel=np.asarray(robot_base_ang_vel_log),
        robot_body_pos=np.asarray(robot_body_pos_log),
        robot_body_quat=np.asarray(robot_body_quat_log),
        focus=np.asarray(focus_trace, dtype=np.bool_),
        body_pos_g_m=np.asarray(body_pos_g_trace),
        left_wrist_pelvis_local_position_m=np.asarray(
            wrist_pose_pelvis_local["left"]["pos_m"]
        ),
        left_wrist_pelvis_local_rotation_deg=np.asarray(
            wrist_pose_pelvis_local["left"]["rot_deg"]
        ),
        right_wrist_pelvis_local_position_m=np.asarray(
            wrist_pose_pelvis_local["right"]["pos_m"]
        ),
        right_wrist_pelvis_local_rotation_deg=np.asarray(
            wrist_pose_pelvis_local["right"]["rot_deg"]
        ),
        diagnostic_root_global_position_m=np.asarray(
            diagnostic_pose["root"]["global"]["pos_m"]
        ),
        diagnostic_root_global_rotation_deg=np.asarray(
            diagnostic_pose["root"]["global"]["rot_deg"]
        ),
        diagnostic_torso_global_position_m=np.asarray(
            diagnostic_pose["torso"]["global"]["pos_m"]
        ),
        diagnostic_torso_global_rotation_deg=np.asarray(
            diagnostic_pose["torso"]["global"]["rot_deg"]
        ),
        diagnostic_torso_pelvis_local_position_m=np.asarray(
            diagnostic_pose["torso"]["pelvis_local"]["pos_m"]
        ),
        diagnostic_torso_pelvis_local_rotation_deg=np.asarray(
            diagnostic_pose["torso"]["pelvis_local"]["rot_deg"]
        ),
        diagnostic_head_proxy_global_position_m=np.asarray(
            diagnostic_pose["head_proxy"]["global"]["pos_m"]
        ),
        diagnostic_head_proxy_global_rotation_deg=np.asarray(
            diagnostic_pose["head_proxy"]["global"]["rot_deg"]
        ),
        diagnostic_head_proxy_pelvis_local_position_m=np.asarray(
            diagnostic_pose["head_proxy"]["pelvis_local"]["pos_m"]
        ),
        diagnostic_head_proxy_pelvis_local_rotation_deg=np.asarray(
            diagnostic_pose["head_proxy"]["pelvis_local"]["rot_deg"]
        ),
    )
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
