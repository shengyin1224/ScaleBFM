import argparse
import numpy as np
import os
import glob
from pathlib import Path
import torch
import joblib
import math
import yaml
import hashlib
from isaaclab.app import AppLauncher

# Joblib uses threads below; allowing every interpolation task to spawn a full
# OpenMP pool causes severe 32 x N oversubscription on the 3000-motion corpus.
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description="Process retargeted motion files")
parser.add_argument("--data_dir", type=str, required=True, help="The path to the input motion directory.")
parser.add_argument("--data_format", type=str, default="pkl", help="Format of the retargeted motions.")
parser.add_argument("--output_dir", type=str, default="source/scaletrack/data/motions", help="Path to save the obtained files")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument(
    "--loader_workers",
    type=int,
    default=32,
    help="In-process interpolation threads (threads are required after Isaac launch).",
)
parser.add_argument(
    "--skip_existing",
    action="store_true",
    help="Resume packaging by excluding motions whose output NPZ already exists.",
)
parser.add_argument("--robot_type", type=str, default="g1_29dof", help="Type of humanoid robot")
# parser.add_argument("--num_splits", type=int, default=1, help="Number of splits to divide motions into")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_slerp

##
# Pre-defined configs
##
if args_cli.robot_type == 'g1_29dof':
    from scaletrack.robots.g1_29dof import (
        G1_29DOF_CYLINDER_CFG as ROBOT_CFG,
        G1_29DOF_JOINT_NAMES as ROBOT_JOINT_NAMES,
    )
else:
    raise NotImplementedError

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_slerp_batch(
    q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
    """Vectorized shortest-path SLERP for wxyz quaternion batches."""
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1_short = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    sin_angle = torch.sin(angle)
    alpha = blend.unsqueeze(-1)
    safe = sin_angle.abs() > 1.0e-7
    weight0 = torch.where(
        safe,
        torch.sin((1.0 - alpha) * angle) / sin_angle.clamp_min(1.0e-7),
        1.0 - alpha,
    )
    weight1 = torch.where(
        safe,
        torch.sin(alpha * angle) / sin_angle.clamp_min(1.0e-7),
        alpha,
    )
    output = weight0 * q0 + weight1 * q1_short
    return torch.nn.functional.normalize(output, dim=-1)


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def load_single_motion(input_dir, motion_file, output_dt):
    try:
        """Loads a single motion from file."""

        motion_dict = joblib.load(motion_file)
        fps = motion_dict['fps']
        base_pos = torch.from_numpy(motion_dict['root_pos']).float()
        base_rot = torch.from_numpy(motion_dict['root_rot'])[:, [3,0,1,2]].float()
        dof_pos = torch.from_numpy(motion_dict['dof_pos']).float()
        focus_mask = np.asarray(
            motion_dict.get("focus_mask", np.zeros(len(base_pos), dtype=bool)),
            dtype=bool,
        )
        focus_time_to_next_edge_s = np.asarray(
            motion_dict.get(
                "focus_time_to_next_edge_s",
                np.full(len(base_pos), 2.0, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        focus_valid = np.asarray(
            motion_dict.get(
                "focus_valid",
                np.zeros(len(base_pos), dtype=bool),
            ),
            dtype=bool,
        )
        
        input_frames = base_pos.shape[0]
        input_dt = 1.0 / fps  # Assuming 30 fps input, adjust if needed
        duration = (input_frames - 1) * input_dt
        
        # Interpolate to output fps
        times = torch.arange(0, duration, output_dt, dtype=torch.float32)
        output_frames = times.shape[0]
        for field_name, values in (
            ("focus_mask", focus_mask),
            ("focus_time_to_next_edge_s", focus_time_to_next_edge_s),
            ("focus_valid", focus_valid),
        ):
            if values.shape != (input_frames,):
                raise ValueError(
                    f"{field_name} must be shaped ({input_frames},), got {values.shape}"
                )

        phase = times / duration
        index_0 = (phase * (input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(input_frames - 1))
        blend = phase * (input_frames - 1) - index_0
        
        motion_base_pos = base_pos[index_0] * (1 - blend.unsqueeze(1)) + base_pos[index_1] * blend.unsqueeze(1)
        motion_base_rot = quat_slerp_batch(
            base_rot[index_0], base_rot[index_1], blend
        )
        motion_dof_pos = dof_pos[index_0] * (1 - blend.unsqueeze(1)) + dof_pos[index_1] * blend.unsqueeze(1)
        nearest_index = torch.clamp(
            torch.round(phase * (input_frames - 1)).long(),
            0,
            input_frames - 1,
        ).numpy()
        motion_focus_mask = focus_mask[nearest_index]
        motion_focus_time_to_next_edge_s = focus_time_to_next_edge_s[nearest_index]
        motion_focus_valid = focus_valid[nearest_index]
        
        # Compute velocities
        base_lin_vel = torch.gradient(motion_base_pos, spacing=output_dt, dim=0)[0]
        dof_vel = torch.gradient(motion_dof_pos, spacing=output_dt, dim=0)[0]
        q_prev, q_next = motion_base_rot[:-2], motion_base_rot[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * output_dt)
        base_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

        print(f"Loaded motion: {Path(motion_file).stem}, duration: {output_frames * output_dt:.2f}s")

        return {
            'base_pos': motion_base_pos,
            'base_rot': motion_base_rot,
            'base_lin_vel': base_lin_vel,
            'base_ang_vel': base_ang_vel,
            'dof_pos': motion_dof_pos,
            'dof_vel': dof_vel,
            'focus_mask': motion_focus_mask,
            'focus_time_to_next_edge_s': motion_focus_time_to_next_edge_s,
            'focus_valid': motion_focus_valid,
            'output_frames': output_frames,
            'file_name': "_".join(os.path.relpath(os.path.splitext(motion_file)[0], input_dir).split("/"))
        }

    except Exception as e:
        print(f"Failed to load {motion_file}. There is an error: {e}")
        return None


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class BatchMotionLoader:
    def __init__(
        self,
        input_dir: str,
        motion_files: list,
        output_fps: int,
        device: torch.device,
        num_envs: int,
    ):
        self.input_dir = input_dir
        self.motion_files = motion_files
        self.output_fps = output_fps
        self.output_dt = 1.0 / self.output_fps
        self.device = device
        self.num_envs = num_envs
        self.current_batch_idx = 0
        self.batch_motions = []
        
        # Load all motions
        self._load_all_motions()
        self.num_motions = len(self.batch_motions)
        self.num_batches = (self.num_motions + num_envs - 1) // num_envs
        
        print(f"Loaded {self.num_motions} motions, will process in {self.num_batches} batches")

    def _load_all_motions(self):
        """Load all motions into memory."""

        # A loky child process cannot import Isaac Sim's omni/carb runtime after
        # this script has launched the app.  Keep loading in-process: one motion
        # runs directly, while batches use threads for the CPU interpolation.
        if len(self.motion_files) == 1:
            self.batch_motions = [
                load_single_motion(self.input_dir, self.motion_files[0], self.output_dt)
            ]
        else:
            self.batch_motions = joblib.Parallel(
                n_jobs=min(args_cli.loader_workers, len(self.motion_files)),
                prefer="threads",
                verbose=10,
            )(
                joblib.delayed(load_single_motion)(self.input_dir, motion_file, self.output_dt)
                for motion_file in self.motion_files
            )
        self.batch_motions = [motion for motion in self.batch_motions if motion is not None and motion['output_frames'] > 2]
        self.batch_motions = sorted(self.batch_motions, key=lambda x: x['output_frames'])


    def get_current_batch(self):
        """Get the current batch of motions."""
        start_idx = self.current_batch_idx * self.num_envs
        end_idx = min((self.current_batch_idx + 1) * self.num_envs, self.num_motions)
        
        if start_idx >= self.num_motions:
            return None, None
        
        batch_motions = self.batch_motions[start_idx:end_idx]
        batch_size = len(batch_motions)
        
        # Pad batch if necessary
        if batch_size < self.num_envs:
            # Repeat the last motion to fill the batch
            padding_needed = self.num_envs - batch_size
            for _ in range(padding_needed):
                batch_motions.append(batch_motions[-1])
        
        return batch_motions, batch_size

    def next_batch(self):
        """Move to next batch."""
        self.current_batch_idx += 1
        return self.current_batch_idx < self.num_batches

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop for batch motion processing."""
    
    # Find all motion files
    motion_files = glob.glob(f"{args_cli.data_dir}/**/*.{args_cli.data_format}", recursive=True)
    if args_cli.skip_existing:
        before = len(motion_files)
        motion_files = [
            path
            for path in motion_files
            if not os.path.exists(
                os.path.join(
                    args_cli.output_dir,
                    "_".join(
                        os.path.relpath(
                            os.path.splitext(path)[0], args_cli.data_dir
                        ).split("/")
                    )
                    + ".npz",
                )
            )
        ]
        print(f"Resume packaging: skipped {before - len(motion_files)} existing outputs")
    if not motion_files:
        print(f"No motion files found in {args_cli.data_dir} with format {args_cli.data_format}")
        return
    
    print(f"Found {len(motion_files)} motion files")
    
    # Create batch motion loader
    motion_loader = BatchMotionLoader(
        input_dir=args_cli.data_dir,
        motion_files=motion_files,
        output_fps=args_cli.output_fps,
        device=torch.device('cpu'),
        num_envs=args_cli.num_envs,
    )
    
    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    
    # Process each batch
    for batch_idx in range(motion_loader.num_batches):
        batch_motions, actual_batch_size = motion_loader.get_current_batch()
        if batch_motions is None:
            break
            
        print(f"Processing batch {batch_idx + 1}/{motion_loader.num_batches} with {actual_batch_size} motions")
        
        # Initialize data loggers for this batch
        batch_logs = {}
        for env_idx in range(actual_batch_size):
            motion_name = batch_motions[env_idx]['file_name']
            batch_logs[env_idx] = {
                "fps": args_cli.output_fps,
                "joint_pos": [],
                "joint_vel": [],
                "body_pos_w": [],
                "body_quat_w": [],
                "body_lin_vel_w": [],
                "body_ang_vel_w": [],
                "file_name": motion_name
            }
        
        # Get maximum frame count in this batch
        max_frames = max(motion['output_frames'] for motion in batch_motions[:actual_batch_size])
        frame_counters = [0] * args_cli.num_envs
        completed_envs = set()
        
        # Simulation loop for current batch
        while simulation_app.is_running() and len(completed_envs) < actual_batch_size:
            # Prepare batch data for current frame
            batch_base_pos = []
            batch_base_rot = []
            batch_base_lin_vel = []
            batch_base_ang_vel = []
            batch_dof_pos = []
            batch_dof_vel = []
            
            for env_idx in range(args_cli.num_envs):
                if env_idx < actual_batch_size and frame_counters[env_idx] < batch_motions[env_idx]['output_frames']:
                    motion = batch_motions[env_idx]
                    frame_idx = frame_counters[env_idx]
                    
                    batch_base_pos.append(motion['base_pos'][frame_idx:frame_idx+1])
                    batch_base_rot.append(motion['base_rot'][frame_idx:frame_idx+1])
                    batch_base_lin_vel.append(motion['base_lin_vel'][frame_idx:frame_idx+1])
                    batch_base_ang_vel.append(motion['base_ang_vel'][frame_idx:frame_idx+1])
                    batch_dof_pos.append(motion['dof_pos'][frame_idx:frame_idx+1])
                    batch_dof_vel.append(motion['dof_vel'][frame_idx:frame_idx+1])
                    
                    frame_counters[env_idx] += 1
                else:
                    # Use last frame for completed or padded environments
                    if env_idx < actual_batch_size:
                        motion = batch_motions[env_idx]
                        last_idx = motion['output_frames'] - 1
                        batch_base_pos.append(motion['base_pos'][last_idx:last_idx+1])
                        batch_base_rot.append(motion['base_rot'][last_idx:last_idx+1])
                        batch_base_lin_vel.append(motion['base_lin_vel'][last_idx:last_idx+1])
                        batch_base_ang_vel.append(motion['base_ang_vel'][last_idx:last_idx+1])
                        batch_dof_pos.append(motion['dof_pos'][last_idx:last_idx+1])
                        batch_dof_vel.append(motion['dof_vel'][last_idx:last_idx+1])
                    else:
                        # For padded environments, use data from last actual motion
                        motion = batch_motions[actual_batch_size - 1]
                        last_idx = motion['output_frames'] - 1
                        batch_base_pos.append(motion['base_pos'][last_idx:last_idx+1])
                        batch_base_rot.append(motion['base_rot'][last_idx:last_idx+1])
                        batch_base_lin_vel.append(motion['base_lin_vel'][last_idx:last_idx+1])
                        batch_base_ang_vel.append(motion['base_ang_vel'][last_idx:last_idx+1])
                        batch_dof_pos.append(motion['dof_pos'][last_idx:last_idx+1])
                        batch_dof_vel.append(motion['dof_vel'][last_idx:last_idx+1])
            
            # Stack batch data
            base_pos = torch.cat(batch_base_pos, dim=0).to(sim.device)
            base_rot = torch.cat(batch_base_rot, dim=0).to(sim.device)
            base_lin_vel = torch.cat(batch_base_lin_vel, dim=0).to(sim.device)
            base_ang_vel = torch.cat(batch_base_ang_vel, dim=0).to(sim.device)
            dof_pos = torch.cat(batch_dof_pos, dim=0).to(sim.device)
            dof_vel = torch.cat(batch_dof_vel, dim=0).to(sim.device)
            
            # Set root state
            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = base_pos
            root_states[:, :2] += scene.env_origins[:, :2]
            root_states[:, 3:7] = base_rot
            root_states[:, 7:10] = base_lin_vel
            root_states[:, 10:] = base_ang_vel
            robot.write_root_state_to_sim(root_states)
            
            # Set joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos[:, robot_joint_indexes] = dof_pos
            joint_vel[:, robot_joint_indexes] = dof_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            sim.render()
            scene.update(sim.get_physics_dt())
            
            # One batched device-to-host transfer per field.  The previous
            # per-env .cpu() calls introduced 6 * num_envs synchronizations at
            # every frame and dominated full-corpus packaging time.
            host_joint_pos = robot.data.joint_pos[
                :actual_batch_size
            ].cpu().numpy()
            host_joint_vel = robot.data.joint_vel[
                :actual_batch_size
            ].cpu().numpy()
            host_body_pos = (
                robot.data.body_pos_w[:actual_batch_size]
                - scene.env_origins[:actual_batch_size, None, :]
            ).cpu().numpy()
            host_body_quat = robot.data.body_quat_w[
                :actual_batch_size
            ].cpu().numpy()
            host_body_lin_vel = robot.data.body_lin_vel_w[
                :actual_batch_size
            ].cpu().numpy()
            host_body_ang_vel = robot.data.body_ang_vel_w[
                :actual_batch_size
            ].cpu().numpy()

            # Log data for active environments
            for env_idx in range(actual_batch_size):
                if frame_counters[env_idx] <= batch_motions[env_idx]['output_frames']:
                    batch_logs[env_idx]["joint_pos"].append(host_joint_pos[env_idx].copy())
                    batch_logs[env_idx]["joint_vel"].append(host_joint_vel[env_idx].copy())
                    batch_logs[env_idx]["body_pos_w"].append(host_body_pos[env_idx].copy())
                    batch_logs[env_idx]["body_quat_w"].append(host_body_quat[env_idx].copy())
                    batch_logs[env_idx]["body_lin_vel_w"].append(host_body_lin_vel[env_idx].copy())
                    batch_logs[env_idx]["body_ang_vel_w"].append(host_body_ang_vel[env_idx].copy())
            
            # Check for completed environments
            for env_idx in range(actual_batch_size):
                if env_idx not in completed_envs and frame_counters[env_idx] >= batch_motions[env_idx]['output_frames']:
                    completed_envs.add(env_idx)
                    # Save completed motion
                    motion_name = batch_logs[env_idx]["file_name"]
                    output_path = os.path.join(args_cli.output_dir, f"{motion_name}.npz")
                    
                    # Convert lists to numpy arrays
                    log_data = {}
                    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                        log_data[k] = np.stack(batch_logs[env_idx][k], axis=0)
                    log_data["fps"] = batch_logs[env_idx]["fps"]
                    source_motion = batch_motions[env_idx]
                    log_data["focus_mask"] = source_motion["focus_mask"].astype(bool)
                    log_data["focus_time_to_next_edge_s"] = source_motion[
                        "focus_time_to_next_edge_s"
                    ].astype(np.float32)
                    log_data["focus_valid"] = source_motion["focus_valid"].astype(bool)
                    
                    np.savez(output_path, **log_data)
                    print(f"{len(completed_envs)} / {actual_batch_size}; Saved motion: {output_path}.")
        
        if not motion_loader.next_batch():
            break

def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    
    # Design scene with multiple environments
    scene_cfg = ReplayMotionsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=ROBOT_JOINT_NAMES,
    )


if __name__ == "__main__":
    main()
    exit()
    # simulation_app.close()
