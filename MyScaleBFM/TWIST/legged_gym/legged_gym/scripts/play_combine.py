# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# # 
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# import os
# import json
# import sys

# from isaacgym import gymutil
# from legged_gym.envs import *
# from legged_gym.gym_utils import task_registry
# from legged_gym.gym_utils.helpers import update_class_from_dict
# import torch
# import faulthandler
# from tqdm import tqdm
# from termcolor import cprint

# def get_combine_args():
#     custom_parameters = [
#         {"name": "--task", "type": str, "default": "h1_mimic", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
#         {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
#         {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
#         {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
#         {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
#         {"name": "--checkpoint", "type": int, "default": -1, "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
#         {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
#         {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
#         {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
#         {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
#         {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
#         {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
#         {"name": "--device", "type": str, "default": "cuda:0", "help": 'Device for sim, rl, and graphics'},

#         {"name": "--rows", "type": int, "help": "num_rows."},
#         {"name": "--cols", "type": int, "help": "num_cols"},
#         {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
#         {"name": "--proj_name", "type": str,  "default": "h1", "help": "run folder name."},
        
#         {"name": "--exptid", "type": str, "help": "exptid"},
#         {"name": "--entity", "type": str, "default": "", "help": "wandb entity"},
#         {"name": "--resumeid", "type": str, "help": "exptid"},

#         {"name": "--use_jit", "action": "store_true", "default": False, "help": "Load jit script when playing"},
#         {"name": "--draw", "action": "store_true", "default": False, "help": "draw debug plot when playing"},
#         {"name": "--save", "action": "store_true", "default": False, "help": "save data for evaluation"},

#         {"name": "--action_delay", "action": "store_true", "default": False, "help": "Add action delay"},

#         {"name": "--web", "action": "store_true", "default": False, "help": "if use web viewer"},
#         {"name": "--no_wandb", "action": "store_true", "default": False, "help": "no wandb"},

#         {"name": "--record_video", "action": "store_true", "default": False, "help": "record video"},
        
#         {"name": "--fix_action_std", "action": "store_true", "default": False, "help": "fix std"},
#         {"name": "--no_rand", "action": "store_true", "default": False, "help": "no domain randomization"},
        
#         {"name": "--teleop_mode", "action": "store_true", "default": False, "help": "teleop mode"},
#         {"name": "--record_log", "action": "store_true", "default": False, "help": "record log"},
        
#         {"name": "--use_transformer", "action": "store_true", "default": False, "help": "use transformer"},

#         {"name": "--teacher_exptid", "type": str, "help": "teacher exptid", "default": "mimic"},
#         {"name": "--teacher_checkpoint", "type": int, "help": "teacher checkpoint", "default": -1},
#         {"name": "--eval_student", "action": "store_true", "default": False, "help": "eval student"},
        
#         # motion file
#         {"name": "--motion_file", "type": str, "help": "motion file", "default": "g1_zhen_phc"},

#         # --- Combined Control Specific Arguments ---
#         {"name": "--body_proj_name", "type": str, "help": "body project name"},
#         {"name": "--body_exptid", "type": str, "help": "body experiment id"},
#         {"name": "--hand_proj_name", "type": str, "help": "hand project name"},
#         {"name": "--hand_exptid", "type": str, "help": "hand experiment id"},
#     ]
#     # parse arguments
#     args = gymutil.parse_arguments(
#         description="Combined RL Policy Play",
#         custom_parameters=custom_parameters)

#     # name alignment
#     args.sim_device_id = args.compute_device_id
#     args.sim_device = args.sim_device_type
#     if args.sim_device=='cuda':
#         args.sim_device += f":{args.sim_device_id}"
#     return args

# def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
#     if checkpoint==-1:
#         models = [file for file in os.listdir(root) if model_name_include in file]
#         models.sort(key=lambda m: '{0:0>15}'.format(m))
#         model = models[-1]
#         checkpoint = model.split("_")[-1].split(".")[0]
#     return model, checkpoint

# def set_play_cfg(env_cfg, args):
#     env_cfg.env.num_envs = args.num_envs if args.num_envs else 2
#     env_cfg.env.episode_length_s = 60
#     env_cfg.terrain.num_rows = 5
#     env_cfg.terrain.num_cols = 5
#     env_cfg.terrain.curriculum = False
#     env_cfg.terrain.max_difficulty = True
    
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = True
#     env_cfg.domain_rand.push_robots = False
#     env_cfg.domain_rand.push_interval_s = 5
#     env_cfg.domain_rand.max_push_vel_xy = 2.5
#     env_cfg.domain_rand.randomize_base_mass = False
#     env_cfg.domain_rand.randomize_base_com = False
#     env_cfg.domain_rand.action_delay = False
    
#     if hasattr(env_cfg, "motion"):
#         env_cfg.motion.motion_curriculum = False


# def play(args):
#     faulthandler.enable()
    
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

#     # Resolve names from the parser (before loading saved configs, so we know which log dir to check)
#     body_exptid = args.body_exptid if args.body_exptid is not None else (args.exptid if args.exptid is not None else env_cfg.env.body_exptid)
#     hand_exptid = args.hand_exptid if args.hand_exptid is not None else (args.teacher_exptid if (args.teacher_exptid is not None and args.teacher_exptid != "mimic") else env_cfg.env.hand_exptid)
    
#     body_proj = args.body_proj_name if args.body_proj_name is not None else (args.proj_name if args.proj_name != "h1" else env_cfg.env.body_proj_name)
#     hand_proj = args.hand_proj_name if args.hand_proj_name is not None else env_cfg.env.hand_proj_name

#     # Try to load saved configs from checkpoint directory (similar to play.py)
#     # Use body log directory for config loading
#     try:
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
#         body_log_dir = os.path.join(project_root, "logs", body_proj, body_exptid)
        
#         env_cfg_path = os.path.join(body_log_dir, 'env_cfg.json')
#         train_cfg_path = os.path.join(body_log_dir, 'train_cfg.json')

#         if os.path.exists(env_cfg_path):
#             cprint(f"Loading saved env config from: {env_cfg_path}", "green")
#             with open(env_cfg_path, 'r') as f:
#                 saved_env_cfg_dict = json.load(f)
#             update_class_from_dict(env_cfg, saved_env_cfg_dict)
#             cprint("✓ Successfully loaded env config", "green")

#         if os.path.exists(train_cfg_path):
#             cprint(f"Loading saved train config from: {train_cfg_path}", "green")
#             with open(train_cfg_path, 'r') as f:
#                 saved_train_cfg_dict = json.load(f)
#             update_class_from_dict(train_cfg, saved_train_cfg_dict)
#             cprint("✓ Successfully loaded train config", "green")
#     except Exception as e:
#         cprint(f"Warning: Could not load saved configs: {e}", "yellow")
#         cprint("Using default configs", "yellow")

#     # Update config with resolved values (after loading saved configs)
#     env_cfg.env.body_exptid = body_exptid
#     env_cfg.env.hand_exptid = hand_exptid
#     env_cfg.env.body_proj_name = body_proj
#     env_cfg.env.hand_proj_name = hand_proj

#     set_play_cfg(env_cfg, args)

#     env_cfg.env.record_video = args.record_video
#     env_cfg.env.rand_reset = False
    
#     if_normalize = env_cfg.env.normalize_obs
#     cprint(f"if_normalize: {if_normalize}", "green")
    
#     if env_cfg.env.record_video:
#         env_cfg.env.episode_length_s = 10

#     # The G1CombinedControl env will load its own policies based on updated env_cfg
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
 
    
   
    
#     # Now get the properly initialized observations
#     body_obs, hand_obs = env.get_observations()

#     actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

#     if args.record_video:
#         mp4_writers = []
#         import imageio
#         env.enable_viewer_sync = True
#         for i in range(env.num_envs):
#             video_name = body_proj + "-" + body_exptid + "_hand_" + hand_exptid + ".mp4"
#             run_name = body_exptid
#             path = f"../../logs/videos_retarget/{run_name}"
#             if not os.path.exists(path):
#                 os.makedirs(path)
#             video_name = os.path.join(path, video_name)
#             mp4_writer = imageio.get_writer(video_name, fps=int(1/env.dt))
#             cprint(f"Recording video to {video_name}", "green")
#             mp4_writers.append(mp4_writer)

#     if args.record_log:
#         run_name = body_exptid
#         logs_dict = []
#         dict_name = body_proj + "-" + body_exptid + "_hand_" + hand_exptid + ".json"
#         path = f"../../logs/env_logs/{run_name}"
#         if not os.path.exists(path):
#             os.makedirs(path)
#         dict_name = os.path.join(path, dict_name)
        
    
#     if not (args.record_video or args.record_log):
#         traj_length = 100*int(env.max_episode_length)
#     else:
#         traj_length = 2 * int(env.max_episode_length)
        
#     env_id = env.lookat_id

#     for i in tqdm(range(traj_length)):
#         # Get current observations (computed in previous step or initial)
#         body_obs, hand_obs = env.get_observations()
        
#         # Compute actions from policies
#         with torch.no_grad():
#             body_actions = body_policy(body_obs.detach())
#             hand_actions = hand_policy(hand_obs.detach())
#             # Zero out hand actions if needed
#             hand_actions = torch.zeros_like(hand_actions)
        
#         # Combine actions
#         actions = torch.cat([body_actions, hand_actions], dim=-1)
        
#         # Step environment with computed actions (this will compute new observations internally)
#         body_obs, hand_obs, _, rews, dones, infos = env.step(actions.detach())
        
#         if args.record_video:
#             imgs = env.render_record(mode='rgb_array')
#             if imgs is not None:
#                 for i in range(env.num_envs):
#                     mp4_writers[i].append_data(imgs[i])
                    
#         if args.record_log:
#             log_dict = env.get_episode_log()
#             logs_dict.append(log_dict)
        
#         if env.button_pressed:
#             print(f"env_id: {env.lookat_id:<{5}}")
    
#     if args.record_video:
#         for mp4_writer in mp4_writers:
#             mp4_writer.close()
            
#     if args.record_log:
#         with open(dict_name, 'w') as f:
#             json.dump(logs_dict, f)
    

# if __name__ == '__main__':
#     # 1. Use the new extended argument parser
#     args = get_combine_args()
    
#     # 2. Start play
#     play(args)

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import json
import atexit
import tempfile
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# Isaac Gym builds gymtorch on first import.  When this script is launched via
# an absolute conda Python path (as in the evaluation command), conda's bin
# directory is not necessarily on PATH, so torch cannot find its bundled
# ninja executable.
_CONDA_BIN = str(Path(os.sys.prefix) / "bin")
if os.path.isdir(_CONDA_BIN) and _CONDA_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _CONDA_BIN + os.pathsep + os.environ.get("PATH", "")
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry
from legged_gym.gym_utils.helpers import update_class_from_dict
import torch
import faulthandler
from tqdm import tqdm
from termcolor import cprint
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse
from legged_gym.envs.g1.g1_wrist_common import (
    quat_log_error_xyzw,
    quat_nlerp_shortest,
    quat_to_rot6d_xyzw,
    wrist_pose_in_root,
)
from rsl_rl.modules import ActorCritic, ActorCriticMimic
from twist2_transformer_fk import KinematicsModel


# This is deliberately kept here (rather than changing the input NPZ in
# place): play_combine references are normally shared by several experiments.
_DEFAULT_PH2D_CANONICAL_REF = Path(
    "./DATASETS/PH2D/402-pick_on_color_pad_right-2025_01_09-16_36_15/processed_episode_0.hdf5"
)
_TEMP_HAND_REFERENCE_FILES = []
_SHENGYIN_ROOT = Path(__file__).resolve().parents[4]
_HUMAN_POLICY_ROOT = _SHENGYIN_ROOT / "human_policy"


class OnlineWristIK:
    """Solve only the six wrist joints against pelvis-frame wrist quaternions.

    Shoulder, elbow and torso angles come from the robot's measured state, so
    the wrist solution compensates their tracking error instead of blindly
    replaying the offline-retarget wrist angles.
    """

    def __init__(self, env, iters=3, damping=1e-3, fd_eps=1e-3):
        self.env = env
        self.iters = int(iters)
        self.damping = float(damping)
        self.fd_eps = float(fd_eps)
        self.model = KinematicsModel(
            xml_path=str(_SHENGYIN_ROOT / "TWIST/assets/g1inspire/g1_29dof_inspire_hand.xml"),
            key_body_names=["L_hand_base_link", "R_hand_base_link"],
            device=str(env.device),
        )
        train_global = [int(x) for x in env.train_dof_indices.detach().cpu().tolist()]
        wrist_global = [int(x) for x in env.wrist_dof_indices.detach().cpu().tolist()]
        self.wrist_global = wrist_global
        self.wrist_slots = [train_global.index(x) for x in wrist_global]
        model_wrist_slots = sorted(
            joint.dof_idx for joint in self.model._joints
            if joint.dof_idx >= 0 and "wrist" in joint.name.lower()
        )
        if self.wrist_slots != model_wrist_slots:
            raise RuntimeError(
                f"online wrist IK DOF mismatch: env slots={self.wrist_slots}, model slots={model_wrist_slots}"
            )
        self.left_body_idx = self.model.body_names.index("L_hand_base_link")
        self.right_body_idx = self.model.body_names.index("R_hand_base_link")
        limits = env.dof_pos_limits[:, 0:2]
        self.lower = limits[wrist_global, 0]
        self.upper = limits[wrist_global, 1]

    def solve(self, target_quat_b):
        base_q = self.env.dof_pos[:, self.env.train_dof_indices].detach()
        wrist_q = base_q[:, self.wrist_slots].clone()
        batch = base_q.shape[0]
        target6 = quat_to_rot6d_xyzw(target_quat_b.reshape(-1, 4)).reshape(batch, 12)
        eye6 = torch.eye(6, device=base_q.device, dtype=base_q.dtype).unsqueeze(0).expand(batch, -1, -1)
        with torch.no_grad():
            for _ in range(self.iters):
                # One batched FK evaluates the current pose plus six finite-
                # difference perturbations.  Damped least squares then solves
                # the 12D rot6d error using only the six wrist joints.
                q_variants = base_q[:, None, :].expand(-1, 7, -1).clone()
                q_variants[:, :, self.wrist_slots] = wrist_q[:, None, :]
                for j, slot in enumerate(self.wrist_slots):
                    q_variants[:, j + 1, slot] += self.fd_eps
                flat_q = q_variants.reshape(batch * 7, -1)
                root_pos = torch.zeros(batch * 7, 3, device=base_q.device, dtype=base_q.dtype)
                root_quat = torch.zeros(batch * 7, 4, device=base_q.device, dtype=base_q.dtype)
                root_quat[:, 3] = 1.0
                _, body_quat = self.model.forward_kinematics(root_pos, root_quat, flat_q)
                actual = body_quat[:, [self.left_body_idx, self.right_body_idx], :]
                actual6 = quat_to_rot6d_xyzw(actual.reshape(-1, 4)).reshape(batch, 7, 12)
                jac = ((actual6[:, 1:, :] - actual6[:, :1, :]) / self.fd_eps).transpose(1, 2)
                error = (target6 - actual6[:, 0, :]).unsqueeze(-1)
                jt = jac.transpose(1, 2)
                delta = torch.linalg.solve(
                    jt @ jac + self.damping * eye6,
                    jt @ error,
                ).squeeze(-1).clamp(-0.35, 0.35)
                wrist_q = (wrist_q + delta).clamp(self.lower, self.upper)
        return wrist_q


def _cleanup_temp_hand_references():
    for path in _TEMP_HAND_REFERENCE_FILES:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


atexit.register(_cleanup_temp_hand_references)


def _unit_np(values):
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values, axis=-1, keepdims=True) + 1e-9)


def _hand_local_basis(kpts6):
    """Return the fixed hand-frame basis used by twist_hand_gmt_bridge.py."""
    kpts6 = np.asarray(kpts6, dtype=np.float64)
    if kpts6.ndim != 3 or kpts6.shape[1:] != (6, 3):
        raise ValueError(f"Expected hand keypoints shaped (T,6,3), got {kpts6.shape}")
    rel = kpts6[:, 1:] - kpts6[:, 0:1]
    finger = _unit_np(rel.mean(axis=(0, 1), keepdims=True))[0, 0]
    normal = _unit_np(np.cross(rel[:, 1], rel[:, 4]).mean(axis=0, keepdims=True))[0]
    normal = _unit_np((normal - finger * np.dot(normal, finger))[None])[0]
    third = _unit_np(np.cross(normal, finger)[None])[0]
    finger = _unit_np(np.cross(third, normal)[None])[0]
    return np.stack([normal, finger, third], axis=1)


def _ph2d_hand_basis(reference_path, side):
    import h5py
    if str(_HUMAN_POLICY_ROOT) not in sys.path:
        sys.path.insert(0, str(_HUMAN_POLICY_ROOT))
    import hdt.constants as C

    indices = C.OUTPUT_LEFT_KEYPOINTS if side == "lh" else C.OUTPUT_RIGHT_KEYPOINTS
    with h5py.File(reference_path, "r") as handle:
        actions = handle["action"][()].astype(np.float32)
    return _hand_local_basis(actions[:, indices].reshape(actions.shape[0], 6, 3))


def _canonicalize_hand_reference_npz(path, reference_path):
    """Canonicalize local hand observations while keeping NPZ world tips fixed.

    The combined environment stores world fingertip targets.  Rotating those
    targets would change the task.  Instead, derive the local points, compute
    the bridge correction, and apply its inverse to the wrist orientation.
    Consequently R_new * local_new == R_old * local_old and the simulator
    still tracks the exact same world fingertip trajectory.
    """
    from scipy.spatial.transform import Rotation

    source = np.load(path, allow_pickle=False)
    required = [
        "rh_root_pos", "rh_root_rot", "rh_tip_world",
        "lh_root_pos", "lh_root_rot", "lh_tip_world",
    ]
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"Cannot canonicalize hand reference; NPZ is missing {missing}")

    data = {key: source[key] for key in source.files}
    for side in ("lh", "rh"):
        root_pos = np.asarray(data[f"{side}_root_pos"], dtype=np.float32)
        root_rot = np.asarray(data[f"{side}_root_rot"], dtype=np.float32)
        tips_world = np.asarray(data[f"{side}_tip_world"], dtype=np.float32)
        world_to_local = Rotation.from_quat(root_rot).as_matrix().transpose(0, 2, 1)
        local_tips = np.einsum("tij,tkj->tki", world_to_local, tips_world - root_pos[:, None])
        current_basis = _hand_local_basis(np.concatenate([np.zeros_like(local_tips[:, :1]), local_tips], axis=1))
        target_basis = _ph2d_hand_basis(reference_path, side)
        correction = (target_basis @ current_basis.T).astype(np.float32)
        root_matrix = Rotation.from_quat(root_rot).as_matrix()
        data[f"{side}_root_rot"] = Rotation.from_matrix(root_matrix @ correction.T).as_quat().astype(np.float32)
        # tip_world intentionally remains unchanged; this is the invariant the
        # GMT canonicalization relies on.
        cprint(
            f"[hand refs] canonicalized {side} local frame to PH2D; "
            f"world tips unchanged; corr={np.array2string(correction, precision=3, suppress_small=True)}",
            "green",
        )

    fd, output_path = tempfile.mkstemp(prefix="play_combine_ph2d_", suffix=".npz")
    os.close(fd)
    np.savez(output_path, **data)
    _TEMP_HAND_REFERENCE_FILES.append(output_path)
    return output_path


def _ensure_ik_hand_reference(path, args):
    """Populate Inspire joint references with the same IK implementation as the bridge."""
    source = np.load(path, allow_pickle=False)
    if "rh_dof_pos" in source.files and "lh_dof_pos" in source.files:
        return path

    # Reuse the bridge solver instead of carrying a subtly different hand IK
    # implementation in the combined-control evaluator.
    if str(_SHENGYIN_ROOT) not in sys.path:
        sys.path.insert(0, str(_SHENGYIN_ROOT))
    from human_policy.twist_hand_gmt_bridge import _add_ik_refs, _load_refs_npz, _save_refs_npz

    refs = _load_refs_npz(path)
    ik_args = SimpleNamespace(
        hand_control_mode="ik",
        exact_ik_reset=True,
        full_ik_in_gmt=False,
        force_recompute_ik=bool(args.force_recompute_ik),
        ik_backend=args.ik_backend,
        ik_device=args.ik_device or args.device,
        ik_lr=args.ik_lr,
        ik_iters=args.ik_iters,
        ik_damping=args.ik_damping,
        ik_step=args.ik_step,
        ik_smooth_weight=args.ik_smooth_weight,
        ik_reg_weight=args.ik_reg_weight,
        ik_cache=False,
        ik_cache_root=args.ik_cache_root,
        canonicalize_hand_frame=args.canonicalize_hand_frame,
        interp_ref_fps=None,
        _ik_cache_source=None,
    )
    refs = _add_ik_refs(refs, ik_args)
    fd, output_path = tempfile.mkstemp(prefix="play_combine_ik_", suffix=".npz")
    os.close(fd)
    _save_refs_npz(output_path, refs)
    _TEMP_HAND_REFERENCE_FILES.append(output_path)
    cprint(f"[hand ik] generated full-sequence Inspire IK reference: {output_path}", "green")
    return output_path


def _prepare_hand_motion_file(args):
    path = args.hand_motion_file
    if not path:
        return path
    path = os.path.abspath(os.path.expanduser(path))
    if args.canonicalize_hand_frame == "ph2d":
        reference_path = Path(args.canonical_hand_reference).expanduser()
        if not reference_path.exists():
            raise FileNotFoundError(
                f"--canonical-hand-reference does not exist: {reference_path}. "
                "Pass an actual PH2D HDF5 file; the historic /home/ubuntu default is not present here."
            )
        if not path.endswith(".npz"):
            raise ValueError("--canonicalize-hand-frame ph2d currently requires the NPZ hand reference used by play_combine")
        path = _canonicalize_hand_reference_npz(path, reference_path)
    if args.hand_control_mode == "ik" and not args.online_hand_ik:
        path = _ensure_ik_hand_reference(path, args)
    return path


def _configure_hand_gmt_checkpoint(env_cfg, checkpoint_path):
    """Infer observation layout from a GMT checkpoint before constructing env/runner.

    The legacy direct checkpoint has a 39-dim per-step motion input.  The
    2026-05-24 GMT checkpoint has 42 and was trained with global hand features
    plus wrist-position proprioception.  The state dict is the authoritative
    source here because the supplied checkpoint directory does not contain its
    original JSON config.
    """
    if not checkpoint_path:
        return None
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Hand GMT checkpoint not found: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = state["model_state_dict"]
    single_motion_obs = int(model["actor.motion_encoder.encoder.0.weight"].shape[1])
    normalizer = state.get("normalizer")
    critic_normalizer = state.get("critic_normalizer")
    observation_dim = int(normalizer._mean_sq.numel()) if normalizer is not None else None
    privileged_dim = int(critic_normalizer._mean_sq.numel()) if critic_normalizer is not None else None
    if single_motion_obs != 42:
        cprint(f"[hand gmt] checkpoint motion observation={single_motion_obs} (legacy layout)", "yellow")
        return {"single_motion_obs": single_motion_obs, "observation_dim": observation_dim}

    # MotionEncoder's 6/4 kernels identify the standard 20 future-reference
    # frames.  The combined hand env keeps an 11-frame proprio history; the
    # resulting 1808-vector is exactly the normalizer shape stored with this
    # checkpoint.
    env_cfg.env.hand_obs_global = False
    env_cfg.env.hand_gmt_future_layout = True
    env_cfg.env.include_wrist_pos_in_proprio = True
    env_cfg.env.include_wrist_error_obs = False
    # Saved training cfg: 840 motion values = 20 future steps *
    # (21 RH + 21 LH), followed by 11 * 88 proprio/history values.
    env_cfg.env.hand_n_mimic_obs = 0
    env_cfg.env.hand_n_proprio = 44
    env_cfg.env.hand_n_priv_mimic_obs = 42 * len(env_cfg.env.tar_obs_steps)
    if observation_dim is not None:
        env_cfg.env.hand_num_observations = observation_dim
    if privileged_dim is not None:
        env_cfg.env.hand_num_privileged_obs = privileged_dim
    cprint(
        "[hand gmt] checkpoint motion observation=42: enabled 20-step future-motion layout "
        f"and wrist-position proprio (obs={observation_dim}, priv={privileged_dim})",
        "green",
    )
    return {"single_motion_obs": single_motion_obs, "observation_dim": observation_dim}

def get_combine_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "g1inspire_combined_control", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--body_task", "type": str, "default": "g1inspire_fuse_combine_direct", "help": "body task name."},
        {"name": "--hand_task", "type": str, "default": "dexhand_mimic_direct", "help": "hand task name."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--device", "type": str, "default": "cuda:0", "help": 'Device for sim, rl, and graphics'},

        {"name": "--rows", "type": int, "help": "num_rows."},
        {"name": "--cols", "type": int, "help": "num_cols"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
        {"name": "--proj_name", "type": str,  "default": "h1", "help": "run folder name."},

        {"name": "--exptid", "type": str, "help": "exptid"},
        {"name": "--entity", "type": str, "default": "", "help": "wandb entity"},
        {"name": "--resumeid", "type": str, "help": "exptid"},

        {"name": "--use_jit", "action": "store_true", "default": False, "help": "Load jit script when playing"},
        {"name": "--draw", "action": "store_true", "default": False, "help": "draw debug plot when playing"},
        {"name": "--save", "action": "store_true", "default": False, "help": "save data for evaluation"},

        {"name": "--action_delay", "action": "store_true", "default": False, "help": "Add action delay"},

        {"name": "--web", "action": "store_true", "default": False, "help": "if use web viewer"},
        {"name": "--no_wandb", "action": "store_true", "default": False, "help": "no wandb"},

        {"name": "--record_video", "action": "store_true", "default": False, "help": "record video"},

        {"name": "--fix_action_std", "action": "store_true", "default": False, "help": "fix std"},
        {"name": "--no_rand", "action": "store_true", "default": False, "help": "no domain randomization"},

        {"name": "--teleop_mode", "action": "store_true", "default": False, "help": "teleop mode"},
        {"name": "--record_log", "action": "store_true", "default": False, "help": "record log"},

        {"name": "--use_transformer", "action": "store_true", "default": False, "help": "use transformer"},

        {"name": "--teacher_exptid", "type": str, "help": "teacher exptid", "default": "mimic"},
        {"name": "--teacher_checkpoint", "type": int, "help": "teacher checkpoint", "default": -1},
        {"name": "--eval_student", "action": "store_true", "default": False, "help": "eval student"},

        # motion file
        {"name": "--motion_file", "type": str, "help": "motion file", "default": "g1_zhen_phc"},

        # --- Combined Control Specific Arguments ---
        {"name": "--body_proj_name", "type": str, "help": "body project name"},
        {"name": "--body_exptid", "type": str, "help": "body experiment id"},
        {"name": "--hand_proj_name", "type": str, "help": "hand project name"},
        {"name": "--hand_exptid", "type": str, "help": "hand experiment id"},
        {"name": "--body_motion_file", "type": str, "help": "body motion yaml/pkl/npz for fixed play"},
        {"name": "--hand_motion_file", "type": str, "help": "hand motion yaml/hdf5 for fixed play"},
        {"name": "--body_checkpoint", "type": int, "default": -1, "help": "body checkpoint; -1 loads latest"},
        {"name": "--hand_checkpoint", "type": int, "default": -1, "help": "hand checkpoint; -1 loads latest"},
        {"name": "--body_checkpoint_path", "type": str, "help": "explicit body checkpoint path; overrides --body_checkpoint"},
        {"name": "--hand_checkpoint_path", "type": str, "help": "explicit hand checkpoint path; overrides --hand_checkpoint"},
        {"name": "--play_steps", "type": int, "help": "number of control steps to run"},
        {"name": "--metrics_out", "type": str, "help": "write tracking metrics as JSON"},
        {"name": "--root_trace_out", "type": str, "help": "write per-step root height diagnostics as JSON"},
        {"name": "--wrist_trace_out", "type": str, "help": "write per-step wrist ref/actual pelvis-frame traces as JSON (env 0) for error decomposition"},
        {"name": "--hand_finger_trace_out", "type": str, "help": "write per-step per-finger actual/ref world + wrist world (env 0) as JSON for per-finger & wrist-relative error"},
        {"name": "--stop_on_done", "action": "store_true", "default": False, "help": "stop after the first episode"},
        {"name": "--sparse_tracking_eval", "action": "store_true", "default": False, "help": "ignore unobserved-body termination for sparse references"},
        {"name": "--zero_hand_policy", "action": "store_true", "default": False, "help": "use zero hand actions instead of loading/running a hand policy"},
        {"name": "--hand-control-mode", "type": str, "choices": ["gmt", "ik", "zero"], "default": "gmt", "help": "gmt: hand policy; ik: direct Inspire IK targets; zero: legacy frozen/zero hand actions"},
        {"name": "--online-hand-ik", "action": "store_true", "default": False, "help": "with hand-control-mode=ik, solve current fingertip targets causally at every simulation control step instead of reading precomputed DOFs"},
        {"name": "--hand-gmt-ckpt", "type": str, "help": "explicit GMT hand checkpoint; alias for --hand_checkpoint_path"},
        {"name": "--canonicalize-hand-frame", "type": str, "choices": ["none", "ph2d"], "default": "none", "help": "rotate local hand observations into the PH2D/Inspire frame without changing world fingertips"},
        {"name": "--canonical-hand-reference", "type": str, "default": str(_DEFAULT_PH2D_CANONICAL_REF), "help": "existing PH2D HDF5 used to define --canonicalize-hand-frame ph2d"},
        {"name": "--ik-backend", "type": str, "choices": ["auto", "pytorch", "pino"], "default": "auto", "help": "IK backend for --hand-control-mode ik"},
        {"name": "--ik-device", "type": str, "help": "torch device for pytorch IK; defaults to --device"},
        {"name": "--ik-iters", "type": int, "default": 120, "help": "IK solver iterations per frame"},
        {"name": "--ik-lr", "type": float, "default": 0.03, "help": "pytorch IK learning rate"},
        {"name": "--ik-damping", "type": float, "default": 1e-3, "help": "pinocchio IK damping"},
        {"name": "--ik-step", "type": float, "default": 0.7, "help": "pinocchio IK step size"},
        {"name": "--ik-smooth-weight", "type": float, "default": 1e-2, "help": "temporal smoothness weight for IK"},
        {"name": "--ik-reg-weight", "type": float, "default": 1e-4, "help": "zero-pose regularization weight for IK"},
        {"name": "--ik-cache-root", "type": str, "default": "/root/shengyin/DATASETS/IK", "help": "reserved IK cache directory (play_combine writes only a temporary ref)"},
        {"name": "--force-recompute-ik", "action": "store_true", "default": False, "help": "ignore existing NPZ rh/lh_dof_pos and solve IK again"},
        {"name": "--use_residual_body_policy", "action": "store_true", "default": False, "help": "compose base body policy with a wrist residual policy"},
        {"name": "--residual_task", "type": str, "default": "g1_wrist_residual_direct", "help": "wrist residual task that defines policy dimensions"},
        {"name": "--residual_proj_name", "type": str, "help": "residual project name"},
        {"name": "--residual_exptid", "type": str, "help": "residual experiment id"},
        {"name": "--residual_checkpoint", "type": int, "default": -1, "help": "residual checkpoint; -1 loads latest"},
        {"name": "--residual_checkpoint_path", "type": str, "help": "explicit residual checkpoint path"},
        {"name": "--base_policy_path", "type": str, "help": "explicit base body policy checkpoint for residual composition"},
        {"name": "--wrist_cache_file", "type": str, "help": "wrist FK cache for image residual retrieval"},
        {"name": "--image_cache_file", "type": str, "help": "image embedding cache for image residual retrieval"},
        {"name": "--focus_cache_file", "type": str, "help": "grasp focus cache for v2 (480-dim) residual phase features"},
        {"name": "--focus_motion_name", "type": str, "help": "motion name substring to index the focus cache (e.g. Pickup_Pillow_ep_0000)"},
        {"name": "--body_wrist_phase_obs", "action": "store_true", "default": False, "help": "append 2-dim grasp phase to body obs for v2 finetune body policies (obs +2)"},
        {"name": "--body_wrist_ori_error_tail_obs", "action": "store_true", "default": False, "help": "append pelvis-frame wrist rot6d reference-minus-current error at the flat body-observation tail (obs +12)"},
        {"name": "--body_wrist_relori_tail_obs", "action": "store_true", "default": False, "help": "append SO(3)-log wrist error + actual pelvis-frame wrist angular velocity (obs +12)"},
        {"name": "--body_wrist_relori_clamp_mode", "type": str, "choices": ["hard", "soft"], "default": "hard", "help": "relori tail error clamp: hard=clamp(+/-0.4) [ctrl/p1], soft=0.4*tanh(err/0.4) [p2]"},
        {"name": "--body_wrist_future_ori_tail_obs", "action": "store_true", "default": False, "help": "P0 feed-forward tail: log-error(6)+omega(6)+K future ref rot6d (obs +12+K*12)"},
        {"name": "--body_wrist_future_horizon", "type": str, "default": "5,10,20", "help": "comma-separated future horizon steps for --body_wrist_future_ori_tail_obs (control steps)"},
        {"name": "--body_wrist_relrot6d_tail_obs", "action": "store_true", "default": False, "help": "append per-wrist rot6d(actual^-1 * reference) from online reference FK (obs +12)"},
        {"name": "--body_direct_no_global", "action": "store_true", "default": False, "help": "use the direct-v2 20x30 no-global motion layout (600 motion + 102 proprio)"},
        {"name": "--body_wrist_rot_obs", "action": "store_true", "default": False, "help": "feed ref wrist orientation (rpy, +6/mimic frame) to body obs for wristrot finetune policies"},
        {"name": "--body_torso_rot_obs", "action": "store_true", "default": False, "help": "feed ref waist/torso orientation (+3/mimic frame) to body obs for wristrot+torso finetune policies"},
        {"name": "--wrist-ff-alpha", "type": float, "default": 0.0, "help": "blend six body wrist actions with retarget reference joint targets (0=off, 0.7=mix, 1=override)"},
        {"name": "--wrist-online-ik", "action": "store_true", "default": False, "help": "solve wrist targets online from measured upstream joints and desired pelvis-frame wrist orientation"},
        {"name": "--wrist-online-ik-iters", "type": int, "default": 3, "help": "DLS iterations per control step for online wrist IK"},
        {"name": "--wrist-online-ik-damping", "type": float, "default": 1e-3, "help": "online wrist IK damped-least-squares regularization"},
        {"name": "--wrist-online-ik-fd-eps", "type": float, "default": 1e-3, "help": "finite-difference angle used for online wrist Jacobian"},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="Combined RL Policy Play",
        custom_parameters=custom_parameters)

    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


class WristPhaseProvider:
    """Reproduces g1_wrist_finetune._wrist_phase_obs / g1_wrist_residual._wrist_phase_features:
    a 2-dim [focus, phase] feature indexed into the per-motion grasp focus cache (60 Hz).
    Used to append the phase obs to a v2 finetune body policy (obs +2) at eval time."""

    def __init__(self, focus_cache_file, focus_motion_name, device):
        self.device = device
        self.ok = False
        if not focus_cache_file or not os.path.exists(focus_cache_file):
            cprint(f"[phase] focus cache not found ({focus_cache_file}); body phase -> neutral", "yellow")
            return
        cache = torch.load(focus_cache_file, map_location=device, weights_only=False)
        names = [str(n) for n in cache["motion_files"]]
        idx = None
        if focus_motion_name:
            key = focus_motion_name.lower()
            idx = next((i for i, n in enumerate(names) if key in n.lower()), None)
        if idx is None:
            cprint(f"[phase] motion '{focus_motion_name}' not in focus cache; body phase -> neutral", "yellow")
            return
        s = int(cache["motion_start_idx"][idx]); n = int(cache["motion_num_frames"][idx])
        self.focus = cache["focus"][s:s + n].float()
        self.valid = cache["valid"][s:s + n].float()
        self.phase = cache["phase"][s:s + n].float()
        self.n = n
        self.fps = 60.0
        self.ok = True
        cprint(f"[phase] body phase obs matched: '{names[idx].split('/')[-1]}' ({n} frames @60Hz)", "green")

    def features(self, env):
        if not self.ok:
            return torch.zeros(env.num_envs, 2, device=self.device)
        t = env._get_motion_times()
        fi = torch.round(t * self.fps).long().clamp(0, self.n - 1)
        valid = self.valid[fi]
        focus = self.focus[fi] * valid
        phase = self.phase[fi].clamp(-1.0, 1.0) * valid
        return torch.stack([focus, phase], dim=-1)


class WristOriTailProvider:
    """Reference wrist quaternions for an ori-tail finetune policy.

    The training environment indexes ``wrist_quat_b`` with frame-0 from the
    60 Hz wrist FK cache.  Keep that behavior here, including nearest/floor
    sampling (orientation is deliberately not interpolated during training).
    """

    def __init__(self, wrist_cache_file, focus_motion_name, device):
        if not wrist_cache_file or not os.path.exists(wrist_cache_file):
            raise RuntimeError(
                "--wrist_cache_file is required with --body_wrist_ori_error_tail_obs"
            )
        cache = torch.load(wrist_cache_file, map_location=device, weights_only=False)
        if "wrist_quat_b" not in cache:
            raise RuntimeError(f"Wrist cache missing wrist_quat_b: {wrist_cache_file}")
        names = [str(name) for name in cache["motion_files"]]
        key = (focus_motion_name or "").lower()
        idx = next((i for i, name in enumerate(names) if key and key in name.lower()), None)
        if idx is None:
            raise RuntimeError(
                f"[ori-tail] motion '{focus_motion_name}' not found in wrist cache"
            )
        start = int(cache["motion_start_idx"][idx])
        count = int(cache["motion_num_frames"][idx])
        self.quat_b = cache["wrist_quat_b"][start:start + count].float()
        self.n = count
        self.fps = 60.0
        cprint(
            f"[ori-tail] wrist quaternion cache matched: '{names[idx].split('/')[-1]}' "
            f"({count} frames @60Hz)",
            "green",
        )

    def quaternions(self, env):
        times = env._get_motion_times()
        frame = torch.floor(times * self.fps).long().clamp(0, self.n - 1)
        return self.quat_b[frame]


class WristRelativeOriTailProvider(WristOriTailProvider):
    """Exact direct-relori tail: Log(R_ref R_actual^T) + actual wrist omega.

    clamp_mode "hard" (ctrl/p1): error.clamp(+/-0.4).
    clamp_mode "soft" (p2): 0.4*tanh(error/0.4) -- matches training
    g1_wrist_finetune.py use_wrist_relative_ori_tail_obs branch exactly.
    """

    def __init__(self, wrist_cache_file, focus_motion_name, device, clamp_mode="hard", error_clamp=0.4):
        super().__init__(wrist_cache_file, focus_motion_name, device)
        self.clamp_mode = clamp_mode
        self.error_clamp = float(error_clamp)

    def quaternions(self, env):
        frame = env._get_motion_times() * self.fps
        f0 = torch.floor(frame).long().clamp(0, self.n - 1)
        f1 = (f0 + 1).clamp(0, self.n - 1)
        blend = (frame - f0.float()).clamp(0.0, 1.0)
        return quat_nlerp_shortest(self.quat_b[f0], self.quat_b[f1], blend)

    def _clamp_error(self, error):
        c = self.error_clamp
        if c > 0 and self.clamp_mode == "hard":
            return error.clamp(-c, c)
        if c > 0 and self.clamp_mode == "soft":
            return c * torch.tanh(error / c)
        return error

    def _actual_quat_omega_b(self, env):
        wrist_state = env.rigid_body_states[:, env._wrist_body_ids]
        root_quat = env.root_states[:, None, 3:7].expand(-1, 2, -1)
        actual_quat_b = quat_mul(
            quat_conjugate(root_quat.reshape(-1, 4)),
            wrist_state[:, :, 3:7].reshape(-1, 4),
        ).reshape(env.num_envs, 2, 4)
        rel_omega_w = wrist_state[:, :, 10:13] - env.root_states[:, None, 10:13]
        actual_omega_b = quat_rotate_inverse(
            root_quat.reshape(-1, 4), rel_omega_w.reshape(-1, 3)
        ).reshape(env.num_envs, 2, 3)
        return actual_quat_b, actual_omega_b

    def features(self, env):
        ref_quat_b = self.quaternions(env)
        actual_quat_b, actual_omega_b = self._actual_quat_omega_b(env)
        error = self._clamp_error(quat_log_error_xyzw(ref_quat_b, actual_quat_b))
        return torch.cat((error.reshape(env.num_envs, -1), actual_omega_b.reshape(env.num_envs, -1)), dim=-1)


class WristFutureOriTailProvider(WristRelativeOriTailProvider):
    """P0 feed-forward tail: current log-error(6, hard +/-0.4) + actual wrist
    omega(6) + K future reference orientations as rot6d (K*12).

    Method A: future reference is read from the SAME 60 Hz wrist_quat_b cache at
    times t + h*dt (h in horizon_steps), nlerp-interpolated -- numerically ~=
    the training online_fk of _future_ref_wrist_quat_b. Layout matches
    g1_wrist_finetune.py use_wrist_future_ori_tail_obs branch exactly.
    """

    def __init__(self, wrist_cache_file, focus_motion_name, device, horizon_steps, error_clamp=0.4):
        # P0 always hard-clamps the current-error channel (training clamp>0 branch).
        super().__init__(wrist_cache_file, focus_motion_name, device, clamp_mode="hard", error_clamp=error_clamp)
        self.horizon_steps = [int(h) for h in horizon_steps]

    def _future_quat(self, env):
        times = env._get_motion_times()  # (num_envs,)
        offsets = torch.tensor(
            [float(h) * env.dt for h in self.horizon_steps], device=env.device, dtype=times.dtype
        )
        future_times = times.unsqueeze(-1) + offsets.unsqueeze(0)  # (num_envs, K)
        frame = future_times * self.fps
        f0 = torch.floor(frame).long().clamp(0, self.n - 1)
        f1 = (f0 + 1).clamp(0, self.n - 1)
        blend = (frame - f0.float()).clamp(0.0, 1.0)
        # self.quat_b: (n, 2, 4) -> index (num_envs, K) -> (num_envs, K, 2, 4).
        # quat_nlerp_shortest auto-unsqueezes blend (num_envs, K) to (num_envs, K, 1, 1).
        q0 = self.quat_b[f0]
        q1 = self.quat_b[f1]
        return quat_nlerp_shortest(q0, q1, blend)

    def features(self, env):
        ref_quat_b = self.quaternions(env)
        actual_quat_b, actual_omega_b = self._actual_quat_omega_b(env)
        error = self._clamp_error(quat_log_error_xyzw(ref_quat_b, actual_quat_b))
        future_quat = self._future_quat(env)  # (num_envs, K, 2, 4)
        future_ori6d = quat_to_rot6d_xyzw(future_quat.reshape(-1, 4)).reshape(env.num_envs, -1)
        return torch.cat(
            (
                error.reshape(env.num_envs, -1),
                actual_omega_b.reshape(env.num_envs, -1),
                future_ori6d,
            ),
            dim=-1,
        )


class WristRelativeRot6DTailProvider:
    """Exact base-relrot6d tail using the same online 29-DoF reference FK."""

    def __init__(self, xml_path, device):
        self.model = KinematicsModel(
            xml_path=xml_path,
            key_body_names=["L_hand_base_link", "R_hand_base_link"],
            device=str(device),
        ).eval()
        self.ids = torch.tensor(
            [
                self.model.body_names.index("L_hand_base_link"),
                self.model.body_names.index("R_hand_base_link"),
            ],
            device=device,
            dtype=torch.long,
        )
        self.cached_counter = None
        self.cached_ref_quat_b = None

    def _reference_quaternions(self, env):
        counter = int(env.global_counter)
        if counter != self.cached_counter:
            dof = env._ref_dof_pos[:, env.body_dof_indices]
            if dof.shape[-1] != 29:
                raise RuntimeError(f"online wrist FK expects 29 reference DOFs, got {dof.shape[-1]}")
            root_pos = torch.zeros(env.num_envs, 3, device=env.device, dtype=dof.dtype)
            root_quat = torch.zeros(env.num_envs, 4, device=env.device, dtype=dof.dtype)
            root_quat[:, 3] = 1.0
            with torch.no_grad():
                _, body_quat = self.model.forward_kinematics(root_pos, root_quat, dof)
            self.cached_ref_quat_b = body_quat[:, self.ids].contiguous()
            self.cached_counter = counter
        return self.cached_ref_quat_b

    def features(self, env):
        ref_quat_b = self._reference_quaternions(env)
        wrist_quat_w = env.rigid_body_states[:, env._wrist_body_ids, 3:7]
        root_quat = env.root_states[:, None, 3:7].expand(-1, 2, -1)
        actual_quat_b = quat_mul(
            quat_conjugate(root_quat.reshape(-1, 4)), wrist_quat_w.reshape(-1, 4)
        ).reshape(env.num_envs, 2, 4)
        error_quat = quat_mul(
            quat_conjugate(actual_quat_b.reshape(-1, 4)), ref_quat_b.reshape(-1, 4)
        )
        return quat_to_rot6d_xyzw(error_quat).reshape(env.num_envs, -1)


def _default_residual_paths(residual_task):
    defaults = {
        "g1_wrist_residual_direct": {
            "body_task": "g1inspire_fuse_combine_direct",
            "base_policy": f"{LEGGED_GYM_ROOT_DIR}/logs/g1inspire_fuse_combine_direct/twist2/model_197500.pt",
            "residual_proj": "g1_wrist_residual_direct",
            "residual_exptid": "residual_direct_twist2_v2",
        },
        "g1_wrist_residual_current": {
            "body_task": "g1inspire_stu_rl_fuse_combine_current",
            "base_policy": f"{LEGGED_GYM_ROOT_DIR}/logs/g1inspire_fuse_combine_stu_current/stu_current_filter/model_172500.pt",
            "residual_proj": "g1_wrist_residual_current",
            "residual_exptid": "residual_stu_current_filter_v2",
        },
        "g1_wrist_residual_no_global": {
            "body_task": "g1inspire_stu_rl_fuse_combine_no_global",
            "base_policy": f"{LEGGED_GYM_ROOT_DIR}/logs/g1inspire_fuse_combine_stu_no_global/stu_no_global_filter/model_172500.pt",
            "residual_proj": "g1_wrist_residual_no_global",
            "residual_exptid": "residual_stu_no_global_filter_v2",
        },
        "g1_wrist_residual_image_direct": {
            "body_task": "g1inspire_fuse_combine_direct",
            "base_policy": f"{LEGGED_GYM_ROOT_DIR}/logs/g1inspire_fuse_combine_direct/twist2/model_197500.pt",
            "residual_proj": "g1_wrist_residual_image_direct",
            "residual_exptid": "residual_image_direct_twist2",
        },
        "g1_wrist_residual_direct_v2": {
            "body_task": "g1inspire_fuse_combine_direct",
            "base_policy": f"{LEGGED_GYM_ROOT_DIR}/logs/g1inspire_fuse_combine_direct/twist2/model_197500.pt",
            "residual_proj": "g1_wrist_residual_direct_v2",
            "residual_exptid": "residual_direct_v2_alpha010_formal",
        },
        "g1_wrist_residual_direct_v2_no_bc": {
            "body_task": "g1inspire_fuse_combine_direct",
            "base_policy": f"{LEGGED_GYM_ROOT_DIR}/logs/g1inspire_fuse_combine_direct/twist2/model_197500.pt",
            "residual_proj": "g1_wrist_residual_direct_v2_no_bc",
            "residual_exptid": "residual_direct_v2_no_bc_alpha010_formal",
        },
    }
    if residual_task not in defaults:
        raise ValueError(f"Unsupported residual_task for play_combine: {residual_task}")
    return defaults[residual_task]


class ResidualBodyPolicyAdapter:
    def __init__(
        self,
        env,
        residual_task,
        base_policy_path,
        residual_checkpoint_path,
        device,
        wrist_cache_file=None,
        image_cache_file=None,
        focus_cache_file=None,
        focus_motion_name=None,
    ):
        from legged_gym.envs.g1 import g1_wrist_tasks  # noqa: F401

        self.env = env
        self.device = device
        self.residual_env_cfg = None
        residual_env_cfg, residual_train_cfg = task_registry.get_cfgs(name=residual_task)
        self.residual_env_cfg = residual_env_cfg
        self.residual_future_steps = list(getattr(residual_env_cfg.env, "residual_future_steps", [2, 4, 6, 8, 10]))
        self.zero_residual_frames = int(getattr(residual_env_cfg.env, "zero_residual_frames", 30))
        self.residual_alpha = self._build_residual_alpha(residual_env_cfg)
        self.last_base_action = torch.zeros(env.num_envs, 29, device=device)
        self.last_residual_delta = torch.zeros(env.num_envs, 29, device=device)
        # v2: 3-step base-action history (obs) + phase-conditioned arm alpha
        self.base_action_hist = torch.zeros(env.num_envs, 3, 29, device=device)
        self.image_residual = False
        self.residual_v2 = False
        self.retr_chunk = int(getattr(residual_env_cfg.env, "retrieval_chunk", 131072))

        self.base_actor_critic, self.base_normalizer = self._load_base_policy(
            residual_env_cfg, base_policy_path
        )
        self.residual_actor_critic, self.residual_normalizer = self._load_residual_policy(
            residual_train_cfg, residual_checkpoint_path
        )
        if self.image_residual:
            wrist_cache_file = wrist_cache_file or getattr(residual_env_cfg.motion, "wrist_cache_file", "")
            image_cache_file = image_cache_file or getattr(residual_env_cfg.motion, "image_cache_file", "")
            self._init_image_retrieval(wrist_cache_file, image_cache_file)
        if self.residual_v2:
            focus_cache_file = focus_cache_file or getattr(residual_env_cfg.motion, "focus_cache_file", "")
            self._init_focus_cache(focus_cache_file, focus_motion_name)

    def _build_residual_alpha(self, residual_env_cfg):
        alpha = torch.full(
            (29,),
            float(getattr(residual_env_cfg.env, "residual_alpha", 0.15)),
            device=self.device,
        )
        alpha[:12] = float(getattr(residual_env_cfg.env, "residual_alpha_legs", 0.05))
        alpha[12:15] = float(getattr(residual_env_cfg.env, "residual_alpha_waist", 0.10))
        alpha[15:] = float(getattr(residual_env_cfg.env, "residual_alpha_arms", 0.15))
        return alpha

    def _init_focus_cache(self, focus_cache_file, focus_motion_name):
        # v2 residual obs carries grasp-phase features + phase-conditioned arm alpha, both
        # indexed into a per-motion focus cache (60 Hz). Resolve the current eval motion into
        # that cache by name so phase is reproduced faithfully; fall back to neutral if absent.
        self.focus = None
        self._focus_valid_motion = False
        if not focus_cache_file or not os.path.exists(focus_cache_file):
            cprint(f"[v2] focus cache not found ({focus_cache_file}); phase features -> neutral", "yellow")
            return
        cache = torch.load(focus_cache_file, map_location=self.device, weights_only=False)
        names = [str(n) for n in cache["motion_files"]]
        idx = None
        if focus_motion_name:
            key = focus_motion_name.lower()
            for i, n in enumerate(names):
                if key in n.lower():
                    idx = i
                    break
        if idx is None:
            cprint(f"[v2] motion '{focus_motion_name}' not in focus cache; phase features -> neutral", "yellow")
            self.focus = {k: cache[k] for k in ("focus", "valid", "phase")}
            self.focus["focus_pad_s"] = float(cache.get("focus_pad_s", 1.0))
            return
        s = int(cache["motion_start_idx"][idx])
        n = int(cache["motion_num_frames"][idx])
        self.focus = {
            "focus": cache["focus"][s:s + n].float(),
            "valid": cache["valid"][s:s + n].float(),
            "phase": cache["phase"][s:s + n].float(),
            "focus_pad_s": float(cache.get("focus_pad_s", 1.0)),
            "num_frames": n,
        }
        self._focus_valid_motion = True
        self._focus_fps = 60.0
        cprint(f"[v2] focus cache motion matched: '{names[idx].split('/')[-1]}' ({n} frames @60Hz)", "green")

    def _phase_features(self):
        # returns (num_envs, 2) = [focus, phase], matching g1_wrist_residual._wrist_phase_features
        if not getattr(self, "_focus_valid_motion", False):
            return torch.zeros(self.env.num_envs, 2, device=self.device)
        t = self.env._get_motion_times()  # seconds
        fi = torch.round(t * self._focus_fps).long().clamp(0, self.focus["num_frames"] - 1)
        valid = self.focus["valid"][fi]
        focus = self.focus["focus"][fi] * valid
        phase = self.focus["phase"][fi].clamp(-1.0, 1.0) * valid
        return torch.stack([focus, phase], dim=-1)

    def _phase_ramp(self, phase_feat):
        # matches g1_wrist_residual._wrist_phase_ramp
        focus = phase_feat[:, 0]
        abs_phase = phase_feat[:, 1].abs()
        pad_s = max(self.focus["focus_pad_s"] if self.focus else 1.0, 1e-6)
        ramp_s = float(getattr(self.residual_env_cfg.env, "residual_alpha_ramp_s", 0.3))
        ramp_frac = max(min(ramp_s / pad_s, 1.0), 1e-6)
        inner = max(1.0 - ramp_frac, 0.0)
        import math
        edge_x = ((abs_phase - inner) / ramp_frac).clamp(0.0, 1.0)
        edge = 0.5 * (1.0 + torch.cos(math.pi * edge_x))
        ramp = torch.where(abs_phase <= inner, torch.ones_like(edge), edge)
        return ramp * focus

    def _current_alpha(self, phase_feat):
        # phase-conditioned arm alpha for v2 (residual_phase_alpha=True); else fixed vector
        alpha = self.residual_alpha.view(1, -1).expand(self.env.num_envs, -1)
        if not (self.residual_v2 and getattr(self.residual_env_cfg.env, "residual_phase_alpha", False)):
            return alpha
        alpha = alpha.clone()
        lo = float(getattr(self.residual_env_cfg.env, "residual_alpha_arms_out_focus", 0.15))
        hi = float(getattr(self.residual_env_cfg.env, "residual_alpha_arms_in_focus",
                           getattr(self.residual_env_cfg.env, "residual_alpha_arms", 0.30)))
        arm_alpha = lo + (hi - lo) * self._phase_ramp(phase_feat)  # (num_envs,)
        alpha[:, 15:] = arm_alpha[:, None]  # arm joints = action indices 15:29
        return alpha

    def _load_base_policy(self, residual_env_cfg, base_policy_path):
        policy_class_name = getattr(residual_env_cfg.env, "base_policy_class_name", "ActorCriticMimic")
        policy_class = ActorCriticMimic if policy_class_name == "ActorCriticMimic" else ActorCritic
        policy_kwargs = dict(getattr(residual_env_cfg.env, "base_policy_kwargs", {}))
        model = policy_class(**policy_kwargs).to(self.device)
        checkpoint = torch.load(base_policy_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        normalizer = checkpoint.get("normalizer", None)
        if normalizer is not None and hasattr(normalizer, "to"):
            normalizer.to(self.device)
        cprint(f"Loaded residual base policy: {base_policy_path}", "green")
        return model, normalizer

    def _load_residual_policy(self, residual_train_cfg, residual_checkpoint_path):
        checkpoint = torch.load(residual_checkpoint_path, map_location=self.device)
        first_weight = checkpoint["model_state_dict"]["actor.actor_backbone.0.weight"]
        residual_obs_dim = int(first_weight.shape[1])
        self.residual_obs_dim = residual_obs_dim
        # 275: base residual (wrist-only future deltas).
        # 420: base residual + per-future-step whole-body dof_delta (29 * len(future_steps)),
        #      matching g1_wrist_residual.py::_future_ref_wrist_deltas anchor+dof variant.
        # 659: 275 base residual + 384 image embedding.
        self.future_include_dof = False
        if residual_obs_dim == 275:
            self.image_residual = False
        elif residual_obs_dim == 420:
            self.image_residual = False
            self.future_include_dof = True
        elif residual_obs_dim == 480:
            # v2: 420 base (future incl dof_delta) + phase(2) + base_action_hist(3*29=87, replaces last_base_action 29)
            self.image_residual = False
            self.future_include_dof = True
            self.residual_v2 = True
        elif residual_obs_dim == 659:
            self.image_residual = True
        else:
            raise RuntimeError(
                f"Residual checkpoint expects obs dim {residual_obs_dim}; "
                "play_combine supports 275/420/480-dim residual and 659-dim image residual policies."
            )
        policy_cfg = residual_train_cfg.policy
        model = ActorCritic(
            num_prop=residual_obs_dim,
            num_critic_obs=residual_obs_dim,
            num_priv_latent=0,
            num_hist=0,
            num_actions=29,
            actor_hidden_dims=list(policy_cfg.actor_hidden_dims),
            critic_hidden_dims=list(policy_cfg.critic_hidden_dims),
            activation=policy_cfg.activation,
            init_noise_std=policy_cfg.init_noise_std,
            priv_encoder_dims=list(policy_cfg.priv_encoder_dims),
            tanh_encoder_output=policy_cfg.tanh_encoder_output,
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        normalizer = checkpoint.get("normalizer", None)
        if normalizer is not None and hasattr(normalizer, "to"):
            normalizer.to(self.device)
        cprint(f"Loaded wrist residual policy: {residual_checkpoint_path}", "green")
        return model, normalizer

    def _init_image_retrieval(self, wrist_cache_file, image_cache_file):
        if not wrist_cache_file:
            raise RuntimeError("--wrist_cache_file or cfg.motion.wrist_cache_file is required for image residual")
        if not image_cache_file:
            raise RuntimeError("--image_cache_file or cfg.motion.image_cache_file is required for image residual")
        wrist_cache = torch.load(wrist_cache_file, map_location="cpu")
        image_cache = torch.load(image_cache_file, map_location="cpu")
        for key in ("wrist_pos_b", "wrist_ori6d_b"):
            if key not in wrist_cache:
                raise RuntimeError(f"wrist cache missing key: {key}")
        for key in ("img_embed", "valid"):
            if key not in image_cache:
                raise RuntimeError(f"image cache missing key: {key}")
        if wrist_cache["wrist_pos_b"].shape[0] != image_cache["img_embed"].shape[0]:
            raise RuntimeError("wrist cache and image cache frame counts do not match")

        valid = image_cache["valid"].to(self.device)
        img_cpu = image_cache["img_embed"].float()
        img_cpu = img_cpu / img_cpu.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.img_embed = img_cpu.half().to(self.device)
        self.img_valid = valid
        feat = torch.cat(
            [
                wrist_cache["wrist_pos_b"].float().reshape(wrist_cache["wrist_pos_b"].shape[0], -1),
                wrist_cache["wrist_ori6d_b"].float().reshape(wrist_cache["wrist_ori6d_b"].shape[0], -1),
            ],
            dim=-1,
        ).to(self.device)
        self.retr_mean = feat[valid].mean(dim=0, keepdim=True)
        self.retr_std = feat[valid].std(dim=0, keepdim=True).clamp_min(1e-4)
        feat_n = ((feat - self.retr_mean) / self.retr_std).half()
        self.valid_global_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        self.retr_feat = feat_n[self.valid_global_idx].contiguous()
        cprint(f"Loaded image residual retrieval caches: {wrist_cache_file}, {image_cache_file}", "green")

    def _normalize(self, normalizer, obs):
        if normalizer is None:
            return obs
        return normalizer.normalize(obs)

    def _ref_wrist_pose_local(self, times):
        ref_data, hand_data = self.env._motion_lib.calc_motion_frame(
            self.env._motion_ids, times, hand_mode="matched"
        )
        root_pos = ref_data["root_pos"].clone()
        root_quat = ref_data["root_quat"]
        root_pos[:, 2] += self.env.cfg.motion.height_offset
        root_pos[:, :2] += self.env.episode_init_origin[:, :2]

        lh_pos = hand_data["lh_wrist_pos"]
        rh_pos = hand_data["rh_wrist_pos"]
        if lh_pos.dim() == 3:
            lh_pos = lh_pos.squeeze(1)
        if rh_pos.dim() == 3:
            rh_pos = rh_pos.squeeze(1)
        wrist_pos_w = torch.stack((lh_pos, rh_pos), dim=1)
        wrist_pos_w[:, :, 2] += self.env.cfg.motion.height_offset
        wrist_pos_w[:, :, :2] += self.env.episode_init_origin[:, None, :2]

        lh_quat = hand_data.get("lh_wrist_rot", root_quat)
        rh_quat = hand_data.get("rh_wrist_rot", root_quat)
        if lh_quat.dim() == 3:
            lh_quat = lh_quat.squeeze(1)
        if rh_quat.dim() == 3:
            rh_quat = rh_quat.squeeze(1)
        wrist_quat_w = torch.stack((lh_quat, rh_quat), dim=1)
        wrist_pos_b, wrist_quat_b = wrist_pose_in_root(root_pos, root_quat, wrist_pos_w, wrist_quat_w)
        ref_body_dof = ref_data["body_dof_pos"]
        return wrist_pos_b, wrist_quat_b, root_pos, root_quat, ref_body_dof

    def _actual_wrist_pose_local(self):
        wrist_state = self.env.rigid_body_states[:, self.env._wrist_body_ids, :]
        return wrist_pose_in_root(
            self.env.root_states[:, 0:3],
            self.env.root_states[:, 3:7],
            wrist_state[:, :, 0:3],
            wrist_state[:, :, 3:7],
        )

    def _future_ref_wrist_deltas(self, curr_pos, curr_quat, curr_times):
        curr_ori6d = quat_to_rot6d_xyzw(curr_quat.reshape(-1, 4)).reshape(self.env.num_envs, 2, 6)
        deltas = []
        for step in self.residual_future_steps:
            times = curr_times + float(step) * self.env.dt
            pos, quat, root_pos, root_quat, ref_body_dof = self._ref_wrist_pose_local(times)
            ori6d = quat_to_rot6d_xyzw(quat.reshape(-1, 4)).reshape(self.env.num_envs, 2, 6)
            rel_pos = quat_rotate_inverse(self.env.root_states[:, 3:7], root_pos - self.env.root_states[:, 0:3])
            rel_quat = quat_mul(root_quat, quat_conjugate(self.env.root_states[:, 3:7]))
            step_terms = [
                (pos - curr_pos).reshape(self.env.num_envs, -1),
                (ori6d - curr_ori6d).reshape(self.env.num_envs, -1),
                rel_pos,
                quat_to_rot6d_xyzw(rel_quat).reshape(self.env.num_envs, -1),
            ]
            if self.future_include_dof:
                # matches g1_wrist_residual.py::_future_anchor_delta dof_delta term
                dof_delta = ref_body_dof - self.env.dof_pos[:, self.env.body_dof_indices]
                step_terms.append(dof_delta)
            deltas.extend(step_terms)
        return torch.cat(deltas, dim=-1)

    def _build_residual_obs(self):
        curr_times = self.env._get_motion_times()
        ref_pos, ref_quat, _, _, _ = self._ref_wrist_pose_local(curr_times)
        actual_pos, actual_quat = self._actual_wrist_pose_local()
        ref_ori6d = quat_to_rot6d_xyzw(ref_quat.reshape(-1, 4)).reshape(self.env.num_envs, -1)
        actual_ori6d = quat_to_rot6d_xyzw(actual_quat.reshape(-1, 4)).reshape(self.env.num_envs, -1)
        body_dof_indices = self.env.body_dof_indices
        proprio = torch.cat(
            [
                self.env.reindex(self.env.dof_pos[:, body_dof_indices]),
                self.env.reindex(self.env.dof_vel[:, body_dof_indices]),
                self.env.projected_gravity,
                self.env.base_ang_vel * self.env.obs_scales.ang_vel,
            ],
            dim=-1,
        )
        wrist_delta = torch.cat(
            [(ref_pos - actual_pos).reshape(self.env.num_envs, -1), ref_ori6d - actual_ori6d],
            dim=-1,
        )
        obs_parts = [
            proprio,
            wrist_delta,
            self._future_ref_wrist_deltas(ref_pos, ref_quat, curr_times),
        ]
        if self.residual_v2:
            # phase(2) + base_action_hist(3*29=87), matching g1_wrist_residual.compute_observations v2 path
            self._last_phase_feat = self._phase_features()
            obs_parts.append(self._last_phase_feat)
            obs_parts.append(self.base_action_hist.reshape(self.env.num_envs, -1))
        else:
            obs_parts.append(self.last_base_action)
        obs_parts.append(self.last_residual_delta)
        residual_obs = torch.cat(obs_parts, dim=-1)
        if self.residual_v2:
            expected_base_dim = 480
        elif self.future_include_dof:
            expected_base_dim = 420
        else:
            expected_base_dim = 275
        if residual_obs.shape[-1] != expected_base_dim:
            raise RuntimeError(
                f"residual observation mismatch: got {residual_obs.shape[-1]}, expected {expected_base_dim}"
            )
        if self.image_residual:
            residual_obs = torch.cat([residual_obs, self._retrieve_current_image_embed(actual_pos, actual_quat)], dim=-1)
        if residual_obs.shape[-1] != self.residual_obs_dim:
            raise RuntimeError(
                f"residual observation mismatch: got {residual_obs.shape[-1]}, expected {self.residual_obs_dim}"
            )
        return residual_obs

    @torch.no_grad()
    def _retrieve_current_image_embed(self, actual_pos, actual_quat):
        ori6d = quat_to_rot6d_xyzw(actual_quat.reshape(-1, 4)).reshape(self.env.num_envs, 2, 6)
        feat = torch.cat([actual_pos.reshape(self.env.num_envs, -1), ori6d.reshape(self.env.num_envs, -1)], dim=-1)
        query = ((feat - self.retr_mean) / self.retr_std).half()
        valid_count = self.retr_feat.shape[0]
        best_val = torch.full((self.env.num_envs,), float("inf"), device=self.device)
        best_row = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        for start in range(0, valid_count, self.retr_chunk):
            block = self.retr_feat[start:start + self.retr_chunk]
            score = block.pow(2).sum(-1)[None, :] - 2.0 * (query @ block.t())
            block_min, block_argmin = score.min(dim=1)
            update = block_min < best_val
            best_val = torch.where(update, block_min, best_val)
            best_row = torch.where(update, start + block_argmin, best_row)
        global_ids = self.valid_global_idx[best_row]
        return self.img_embed[global_ids].float()

    def __call__(self, body_obs):
        base_obs = self._normalize(self.base_normalizer, body_obs.detach())
        # obs uses base_action_hist / last_base_action from the PREVIOUS step (built before we
        # compute this step's base_action), matching the training env timing.
        residual_obs = self._normalize(self.residual_normalizer, self._build_residual_obs())
        with torch.no_grad():
            base_action = self.base_actor_critic.act_inference(base_obs)
            residual_delta = torch.tanh(self.residual_actor_critic.act_inference(residual_obs))
        if self.residual_v2:
            alpha = self._current_alpha(self._last_phase_feat)
        else:
            alpha = self.residual_alpha
        release_mask = (self.env.episode_length_buf >= self.zero_residual_frames).float().unsqueeze(-1)
        final_action = base_action + alpha * residual_delta * release_mask
        self.last_base_action = base_action.detach()
        self.last_residual_delta = residual_delta.detach()
        # push this step's base action into the 3-deep history (hist[0]=most recent)
        self.base_action_hist = torch.cat(
            [base_action.detach().unsqueeze(1), self.base_action_hist[:, :2]], dim=1
        )
        return final_action

def set_play_cfg(env_cfg):
    env_cfg.env.num_envs = 1#2 if not args.num_envs else args.num_envs
    env_cfg.env.episode_length_s = 60
    # env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False


def play(args):
    faulthandler.enable()
    if args.online_hand_ik and args.hand_control_mode != "ik":
        raise ValueError("--online-hand-ik requires --hand-control-mode ik")
    if args.online_hand_ik and args.ik_backend not in ("auto", "pino"):
        raise ValueError("--online-hand-ik currently supports Pinocchio IK only")
    wrist_tail_flags = [
        args.body_wrist_ori_error_tail_obs,
        args.body_wrist_relori_tail_obs,
        args.body_wrist_relrot6d_tail_obs,
    ]
    if sum(bool(flag) for flag in wrist_tail_flags) > 1:
        raise ValueError("select only one wrist tail observation mode")
    if args.hand_gmt_ckpt:
        if args.hand_checkpoint_path and os.path.abspath(os.path.expanduser(args.hand_checkpoint_path)) != os.path.abspath(os.path.expanduser(args.hand_gmt_ckpt)):
            raise ValueError("--hand-gmt-ckpt and --hand_checkpoint_path refer to different files")
        args.hand_checkpoint_path = args.hand_gmt_ckpt
    if args.zero_hand_policy:
        args.hand_control_mode = "zero"
    hand_policy_enabled = args.hand_control_mode == "gmt"

    # 1. Resolve model paths based on config
    # args.body_proj_name =  "g1inspire_fuse_combine_direct"
    # args.body_expid =  "direct"
    # args.hand_proj_name =  "dexhand_mimic_direct"
    # args.hand_expid = "newkpkd"
    
    combined_tasks = {
        "g1inspire_fuse_combine_direct": "g1inspire_combined_control",
        "g1inspire_priv_mimic_fuse_combine": "g1inspire_combined_control_teacher",
        "g1inspire_stu_rl_fuse_combine_current": "g1inspire_combined_control_stu_current",
        "g1inspire_stu_rl_fuse_combine_no_global": "g1inspire_combined_control_stu_no_global",
        "g1inspire_fuse_combine_dofpred": "g1inspire_combined_control_dofpred",
        "g1inspire_fuse_combine_dofpred_kpt": "g1inspire_combined_control_dofpred_kpt",
    }
    residual_defaults = None
    if args.use_residual_body_policy:
        residual_defaults = _default_residual_paths(args.residual_task)
        args.body_task = residual_defaults["body_task"]
    if args.task == "g1inspire_combined_control":
        args.task = combined_tasks.get(args.body_task, args.task)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    args.body_proj_name = args.body_proj_name or env_cfg.env.body_proj_name
    args.body_exptid = args.body_exptid or env_cfg.env.body_exptid
    args.hand_proj_name = args.hand_proj_name or env_cfg.env.hand_proj_name
    args.hand_exptid = args.hand_exptid or env_cfg.env.hand_exptid

    body_log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.body_proj_name, args.body_exptid)
    hand_log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.hand_proj_name, args.hand_exptid)

    def load_saved_env_section(log_dir):
        env_cfg_path = os.path.join(log_dir, "env_cfg.json")
        if not os.path.exists(env_cfg_path):
            return None
        with open(env_cfg_path, "r") as cfg_file:
            return json.load(cfg_file).get("env", {})

    hand_env = load_saved_env_section(hand_log_pth)
    if hand_env and hand_policy_enabled:
        hand_key_map = {
            "n_mimic_obs": "hand_n_mimic_obs",
            "n_proprio": "hand_n_proprio",
            "history_len": "hand_history_len",
            "num_observations": "hand_num_observations",
            "num_privileged_obs": "hand_num_privileged_obs",
            "n_priv_mimic_obs": "hand_n_priv_mimic_obs",
            "num_actions": "hand_num_actions",
        }
        for source_key, target_key in hand_key_map.items():
            if source_key in hand_env:
                setattr(env_cfg.env, target_key, hand_env[source_key])
        cprint(f"Loaded hand observation dims from: {os.path.join(hand_log_pth, 'env_cfg.json')}", "green")
    elif not hand_policy_enabled:
        # Matches G1CombinedControl._compute_hand_obs_internal:
        # per side: 21 mimic values (wrist xyz/rpy + 5 local tips) + 41 proprio.
        env_cfg.env.hand_n_mimic_obs = 21
        env_cfg.env.hand_n_priv_mimic_obs = 21 * len(env_cfg.env.tar_obs_steps)
        env_cfg.env.hand_n_proprio = 41
        env_cfg.env.hand_history_len = 10
        env_cfg.env.hand_num_observations = (21 + 41) * 2 * (env_cfg.env.hand_history_len + 1)
        env_cfg.env.hand_num_privileged_obs = env_cfg.env.hand_num_observations
        cprint(f"Using combine-native hand observation dims for --hand-control-mode {args.hand_control_mode}", "yellow")

    hand_checkpoint_spec = None
    if hand_policy_enabled and args.hand_checkpoint_path:
        hand_checkpoint_spec = _configure_hand_gmt_checkpoint(env_cfg, args.hand_checkpoint_path)


    # # Try to load saved configs from checkpoint directory
    # try:
    #     import json
    #     body_env_cfg_path = os.path.join(body_log_pth, 'env_cfg.json')
    #     body_train_cfg_path = os.path.join(body_log_pth, 'train_cfg.json')
    #     hand_env_cfg_path = os.path.join(hand_log_pth, 'env_cfg.json')
    #     hand_train_cfg_path = os.path.join(hand_log_pth, 'train_cfg.json')


    #     if os.path.exists(body_env_cfg_path):
    #         cprint(f"Loading saved env config from: {body_env_cfg_path}", "green")
    #         with open(body_env_cfg_path, 'r') as f:
    #             saved_env_cfg_dict = json.load(f)
    #         update_class_from_dict(body_env_cfg, saved_env_cfg_dict)
    #         cprint("✓ Successfully loaded env config", "green")

    #     if os.path.exists(body_train_cfg_path):
    #         cprint(f"Loading saved train config from: {body_train_cfg_path}", "green")
    #         with open(body_train_cfg_path, 'r') as f:
    #             saved_train_cfg_dict = json.load(f)
    #         update_class_from_dict(body_train_cfg, saved_train_cfg_dict)
    #         cprint("✓ Successfully loaded train config", "green")
        
    #     if os.path.exists(hand_env_cfg_path):
    #         cprint(f"Loading saved env config from: {hand_env_cfg_path}", "green")
    #         with open(hand_env_cfg_path, 'r') as f:
    #             saved_env_cfg_dict = json.load(f)
    #         update_class_from_dict(hand_env_cfg, saved_env_cfg_dict)
    #         cprint("✓ Successfully loaded env config", "green")

    #     if os.path.exists(hand_train_cfg_path):
    #         cprint(f"Loading saved train config from: {hand_train_cfg_path}", "green")
    #         with open(hand_train_cfg_path, 'r') as f:
    #             saved_train_cfg_dict = json.load(f)
    #         update_class_from_dict(hand_train_cfg, saved_train_cfg_dict)
    #         cprint("✓ Successfully loaded train config", "green")
    # except Exception as e:
    #     cprint(f"Warning: Could not load saved configs: {e}", "yellow")
    #     cprint("Using default configs", "yellow")

    set_play_cfg(env_cfg)

    if args.body_direct_no_global:
        # Direct-v2 no-global is still a direct/future-motion policy; it is not
        # the 11-frame student-history architecture used by the older
        # g1inspire_stu_rl_fuse_combine_no_global task.
        env_cfg.env.mimic_obs_mode = "no_global"
        env_cfg.env.body_n_mimic_obs = 20 * 30
        env_cfg.env.body_n_priv_mimic_obs = 20 * 30
        env_cfg.env.body_n_obs_single = 20 * 30 + 102
        env_cfg.env.body_num_observations = env_cfg.env.body_n_obs_single
        if getattr(env_cfg.env, "body_num_privileged_obs", None) is not None:
            env_cfg.env.body_num_privileged_obs = env_cfg.env.body_n_obs_single
        env_cfg.env.body_num_motion_steps = 20
        env_cfg.env.body_use_history = False

    env_cfg.env.rand_reset = False
    env_cfg.env.record_video = args.record_video
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_yaw = False
    env_cfg.env.sparse_tracking_eval = args.sparse_tracking_eval

    if args.body_motion_file:
        env_cfg.motion.motion_file = args.body_motion_file
    elif args.motion_file != "g1_zhen_phc":
        env_cfg.motion.motion_file = args.motion_file

    prepared_hand_motion_file = _prepare_hand_motion_file(args)
    if prepared_hand_motion_file:
        env_cfg.motion.hand_motion_file = prepared_hand_motion_file
        env_cfg.motion.head_data_dir = prepared_hand_motion_file
    elif not hasattr(env_cfg.motion, "hand_motion_file") or env_cfg.motion.hand_motion_file is None:
        env_cfg.motion.hand_motion_file = getattr(env_cfg.motion, "head_data_dir", None)
    
    if_normalize = env_cfg.env.normalize_obs
    cprint(f"if_normalize: {if_normalize}", "green")
    
    if env_cfg.env.record_video:
        env_cfg.env.episode_length_s = 10

    if args.body_wrist_phase_obs:
        # v2 finetune body policy carries base body obs + 2-dim grasp phase; size the body
        # obs (and thus the body policy input) accordingly before the env / runner are built.
        env_cfg.env.body_wrist_phase_obs = True
        env_cfg.env.body_num_observations = int(env_cfg.env.body_num_observations) + 2
        # critic/privileged obs also carried +2 phase in v2 training; size it so the (unused-at-
        # inference) critic weights still load under strict state_dict loading.
        if hasattr(env_cfg.env, "body_num_privileged_obs") and env_cfg.env.body_num_privileged_obs is not None:
            env_cfg.env.body_num_privileged_obs = int(env_cfg.env.body_num_privileged_obs) + 2

    if args.body_wrist_ori_error_tail_obs:
        env_cfg.env.body_wrist_ori_error_tail_obs = True
        env_cfg.env.body_num_observations = int(env_cfg.env.body_num_observations) + 12
        if hasattr(env_cfg.env, "body_num_privileged_obs") and env_cfg.env.body_num_privileged_obs is not None:
            env_cfg.env.body_num_privileged_obs = int(env_cfg.env.body_num_privileged_obs) + 12

    if args.body_wrist_relori_tail_obs or args.body_wrist_relrot6d_tail_obs:
        env_cfg.env.body_wrist_new_tail_obs = True
        env_cfg.env.body_num_observations = int(env_cfg.env.body_num_observations) + 12
        if hasattr(env_cfg.env, "body_num_privileged_obs") and env_cfg.env.body_num_privileged_obs is not None:
            env_cfg.env.body_num_privileged_obs = int(env_cfg.env.body_num_privileged_obs) + 12

    if args.body_wrist_future_ori_tail_obs:
        future_horizon = [int(h) for h in str(args.body_wrist_future_horizon).split(",") if h.strip()]
        future_tail_dim = 12 + len(future_horizon) * 12
        env_cfg.env.body_wrist_future_tail_obs = True
        env_cfg.env.body_wrist_future_tail_dim = future_tail_dim
        env_cfg.env.body_num_observations = int(env_cfg.env.body_num_observations) + future_tail_dim
        if hasattr(env_cfg.env, "body_num_privileged_obs") and env_cfg.env.body_num_privileged_obs is not None:
            env_cfg.env.body_num_privileged_obs = int(env_cfg.env.body_num_privileged_obs) + future_tail_dim

    if args.body_wrist_rot_obs:
        # wristrot finetune policies feed ref wrist orientation (rpy) into every mimic frame.
        # Direct policies carry 20 future frames, whereas student policies carry
        # one enlarged frame plus history.  Keep these layouts separate.
        env_cfg.env.use_wrist_rot_obs = True
        env_cfg.env.wrist_rot_obs_type = "rpy"
        if getattr(env_cfg.env, "body_use_history", False):
            env_cfg.env.body_n_mimic_obs = int(env_cfg.env.body_n_mimic_obs) + 6
            env_cfg.env.body_n_priv_mimic_obs = int(env_cfg.env.body_n_priv_mimic_obs) + 6
            env_cfg.env.body_n_obs_single = int(env_cfg.env.body_n_obs_single) + 6
            env_cfg.env.body_num_observations = int(env_cfg.env.body_num_observations) + 66
            if getattr(env_cfg.env, "body_num_privileged_obs", None) is not None:
                env_cfg.env.body_num_privileged_obs = int(env_cfg.env.body_num_privileged_obs) + 240
        else:
            extra = int(env_cfg.env.body_num_motion_steps) * 6
            env_cfg.env.body_n_mimic_obs = int(env_cfg.env.body_n_mimic_obs) + extra
            env_cfg.env.body_n_priv_mimic_obs = int(env_cfg.env.body_n_priv_mimic_obs) + extra
            env_cfg.env.body_n_obs_single = int(env_cfg.env.body_n_obs_single) + extra
            env_cfg.env.body_num_observations = int(env_cfg.env.body_num_observations) + extra
            if getattr(env_cfg.env, "body_num_privileged_obs", None) is not None:
                env_cfg.env.body_num_privileged_obs = int(env_cfg.env.body_num_privileged_obs) + extra

    if args.body_torso_rot_obs:
        # torso variant appends ref waist joint targets (+3) to the student mimic frame only
        # (student _get_mimic_obs); teacher/priv obs is untouched.
        env_cfg.env.use_torso_rot_obs = True
        env_cfg.env.body_n_mimic_obs = int(env_cfg.env.body_n_mimic_obs) + 3
        env_cfg.env.body_n_priv_mimic_obs = int(env_cfg.env.body_n_priv_mimic_obs) + 3
        env_cfg.env.body_n_obs_single = int(env_cfg.env.body_n_obs_single) + 3
        env_cfg.env.body_num_observations = int(env_cfg.env.body_num_observations) + 33

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    wrist_online_ik = OnlineWristIK(
        env, args.wrist_online_ik_iters, args.wrist_online_ik_damping, args.wrist_online_ik_fd_eps
    ) if args.wrist_online_ik else None
    if args.body_wrist_phase_obs:
        env._body_phase_provider = WristPhaseProvider(args.focus_cache_file, args.focus_motion_name, env.device)
    if args.body_wrist_ori_error_tail_obs:
        env._body_wrist_ori_provider = WristOriTailProvider(
            args.wrist_cache_file, args.focus_motion_name, env.device
        )
        # make_env computes one observation before providers can be attached;
        # refresh it so the first policy action also receives the real tail.
        env.compute_observations()
    if args.body_wrist_relori_tail_obs:
        env._body_wrist_new_tail_provider = WristRelativeOriTailProvider(
            args.wrist_cache_file, args.focus_motion_name, env.device,
            clamp_mode=args.body_wrist_relori_clamp_mode,
        )
        env.compute_observations()
    if args.body_wrist_future_ori_tail_obs:
        env._body_wrist_future_tail_provider = WristFutureOriTailProvider(
            args.wrist_cache_file, args.focus_motion_name, env.device,
            [int(h) for h in str(args.body_wrist_future_horizon).split(",") if h.strip()],
        )
        env.compute_observations()
    if args.body_wrist_relrot6d_tail_obs:
        env._body_wrist_new_tail_provider = WristRelativeRot6DTailProvider(
            os.path.join(LEGGED_GYM_ROOT_DIR, "../assets/g1inspire/g1_29dof_inspire_hand.xml"),
            env.device,
        )
        env.compute_observations()
    body_obs, hand_obs = env.get_observations()

    # load policy using dual model runner (no resume, load manually)
    train_cfg.runner.resume = False
    dual_ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root=None,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        init_wandb=False,
    )

    def resolve_checkpoint_path(explicit_path, log_root, checkpoint):
        if explicit_path:
            return os.path.abspath(os.path.expanduser(explicit_path))
        return get_load_path(log_root, load_run=-1, checkpoint=checkpoint)

    # Manually load body and hand checkpoints
    from legged_gym.gym_utils.helpers import get_load_path
    body_resume_path = resolve_checkpoint_path(args.body_checkpoint_path, body_log_pth, args.body_checkpoint)
    hand_resume_path = None if not hand_policy_enabled else resolve_checkpoint_path(args.hand_checkpoint_path, hand_log_pth, args.hand_checkpoint)
    residual_resume_path = None
    base_policy_path = None
    residual_body_policy = None

    if args.use_residual_body_policy:
        base_policy_path = os.path.abspath(os.path.expanduser(args.base_policy_path or residual_defaults["base_policy"]))
        residual_proj_name = args.residual_proj_name or residual_defaults["residual_proj"]
        residual_exptid = args.residual_exptid or residual_defaults["residual_exptid"]
        residual_log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", residual_proj_name, residual_exptid)
        residual_resume_path = resolve_checkpoint_path(
            args.residual_checkpoint_path, residual_log_pth, args.residual_checkpoint
        )
        if not os.path.exists(base_policy_path):
            raise FileNotFoundError(f"Base policy checkpoint not found: {base_policy_path}")
        if not os.path.exists(residual_resume_path):
            raise FileNotFoundError(f"Residual checkpoint not found: {residual_resume_path}")
        residual_body_policy = ResidualBodyPolicyAdapter(
            env=env,
            residual_task=args.residual_task,
            base_policy_path=base_policy_path,
            residual_checkpoint_path=residual_resume_path,
            device=env.device,
            wrist_cache_file=args.wrist_cache_file,
            image_cache_file=args.image_cache_file,
            focus_cache_file=args.focus_cache_file,
            focus_motion_name=args.focus_motion_name,
        )

    # Load body model
    if args.use_residual_body_policy:
        print("Skipping dual-runner body checkpoint load because residual body policy is enabled")
    elif os.path.exists(body_resume_path):
        dual_ppo_runner.load_body(body_resume_path, load_optimizer=False)
    else:
        print(f"Body checkpoint not found at {body_resume_path}, using default")

    # Load hand model
    if not hand_policy_enabled:
        print(f"Skipping hand checkpoint load because --hand-control-mode {args.hand_control_mode} does not use GMT")
    elif os.path.exists(hand_resume_path):
        dual_ppo_runner.load_hand(hand_resume_path, load_optimizer=False)
    else:
        print(f"Hand checkpoint not found at {hand_resume_path}, using default")

    body_policy, hand_policy = dual_ppo_runner.get_inference_policy(device=env.device)
    if if_normalize:
        try:
            body_normalizer, body_critic_normalizer, hand_normalizer, hand_critic_normalizer = dual_ppo_runner.get_normalizer(device=env.device)
        except:
            print("No normalizer found")
            body_normalizer = body_critic_normalizer = hand_normalizer = hand_critic_normalizer = None

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    
    if args.play_steps is not None:
        traj_length = args.play_steps
    elif not (args.record_video or args.record_log):
        traj_length = 100*int(env.max_episode_length)
    else:
        traj_length = 2 * int(env.max_episode_length)
        
    env_id = env.lookat_id
    tracking = {
        "root_pos": [],
        "root_rot": [],
        "head_pos": [],
        "head_pos_pelvis": [],
        "wrist_pos": [],
        "wrist_pos_pelvis": [],
        "left_wrist_pos": [],
        "right_wrist_pos": [],
        "left_wrist_pos_pelvis": [],
        "right_wrist_pos_pelvis": [],
        # wrist ORIENTATION error (root-relative geodesic angle, rad)
        "wrist_rot": [],
        "left_wrist_rot": [],
        "right_wrist_rot": [],
        "hand_tip_pos": [],
        # Finger-only tracking: compare target and simulated fingertips in
        # their respective hand-base frames.  Unlike hand_tip_pos, this does
        # not charge the hand IK/GMT for the body policy's wrist pose error.
        "hand_tip_pos_wrist_frame": [],
        "right_hand_tip_pos_wrist_frame": [],
        "left_hand_tip_pos_wrist_frame": [],
        "body_dof": [],
        "arm_dof": [],
        "left_arm_dof": [],
        "right_arm_dof": [],
        "torso_dof": [],
        "non_arm_body_dof": [],
    }
    done_step = None
    termination_reason = []
    root_trace = []
    wrist_trace = []
    finger_trace = []
    train_dof_indices_cpu = [int(i) for i in env.train_dof_indices.detach().cpu().tolist()]
    dof_group_indices = {
        "left_arm_dof": [
            idx for idx in train_dof_indices_cpu
            if env.dof_names[idx].startswith("left_")
            and any(token in env.dof_names[idx] for token in ("shoulder", "elbow", "wrist"))
        ],
        "right_arm_dof": [
            idx for idx in train_dof_indices_cpu
            if env.dof_names[idx].startswith("right_")
            and any(token in env.dof_names[idx] for token in ("shoulder", "elbow", "wrist"))
        ],
        "torso_dof": [
            idx for idx in train_dof_indices_cpu
            if any(token in env.dof_names[idx] for token in ("waist", "torso"))
        ],
    }
    dof_group_indices["arm_dof"] = dof_group_indices["left_arm_dof"] + dof_group_indices["right_arm_dof"]
    dof_group_indices["non_arm_body_dof"] = [
        idx for idx in train_dof_indices_cpu if idx not in set(dof_group_indices["arm_dof"])
    ]
    dof_group_tensors = {
        name: torch.tensor(indices, device=env.device, dtype=torch.long)
        for name, indices in dof_group_indices.items()
    }

    def record_tracking():
        root_dot = torch.sum(env.root_states[:, 3:7] * env._ref_root_rot, dim=-1).abs().clamp(max=1.0)
        tracking["root_pos"].extend(torch.linalg.norm(env.root_states[:, :3] - env._ref_root_pos, dim=-1).cpu().tolist())
        tracking["root_rot"].extend((2.0 * torch.acos(root_dot)).cpu().tolist())
        tracking["head_pos"].extend(torch.linalg.norm(
            env.rigid_body_states[:, env._head_body_ids, :3] - env._ref_body_pos[:, env._head_body_ids, :3], dim=-1
        ).reshape(-1).cpu().tolist())
        head_actual_local = quat_rotate_inverse(
            env.root_states[:, None, 3:7].expand(-1, len(env._head_body_ids), -1).reshape(-1, 4),
            (env.rigid_body_states[:, env._head_body_ids, :3] - env.root_states[:, None, :3]).reshape(-1, 3),
        ).reshape(env.num_envs, len(env._head_body_ids), 3)
        head_ref_local = quat_rotate_inverse(
            env._ref_root_rot[:, None, :].expand(-1, len(env._head_body_ids), -1).reshape(-1, 4),
            (env._ref_body_pos[:, env._head_body_ids, :3] - env._ref_root_pos[:, None, :]).reshape(-1, 3),
        ).reshape(env.num_envs, len(env._head_body_ids), 3)
        tracking["head_pos_pelvis"].extend(
            torch.linalg.norm(head_actual_local - head_ref_local, dim=-1).reshape(-1).cpu().tolist()
        )
        tracking["wrist_pos"].extend(torch.linalg.norm(
            env.rigid_body_states[:, env._wrist_body_ids, :3] - env._ref_body_pos[:, env._wrist_body_ids, :3], dim=-1
        ).reshape(-1).cpu().tolist())
        wrist_world_err = torch.linalg.norm(
            env.rigid_body_states[:, env._wrist_body_ids, :3] - env._ref_body_pos[:, env._wrist_body_ids, :3], dim=-1
        )
        if wrist_world_err.shape[-1] >= 2:
            tracking["left_wrist_pos"].extend(wrist_world_err[:, 0].reshape(-1).cpu().tolist())
            tracking["right_wrist_pos"].extend(wrist_world_err[:, 1].reshape(-1).cpu().tolist())
        wrist_actual_local = quat_rotate_inverse(
            env.root_states[:, None, 3:7].expand(-1, len(env._wrist_body_ids), -1).reshape(-1, 4),
            (env.rigid_body_states[:, env._wrist_body_ids, :3] - env.root_states[:, None, :3]).reshape(-1, 3),
        ).reshape(env.num_envs, len(env._wrist_body_ids), 3)
        wrist_ref_local = quat_rotate_inverse(
            env._ref_root_rot[:, None, :].expand(-1, len(env._wrist_body_ids), -1).reshape(-1, 4),
            (env._ref_body_pos[:, env._wrist_body_ids, :3] - env._ref_root_pos[:, None, :]).reshape(-1, 3),
        ).reshape(env.num_envs, len(env._wrist_body_ids), 3)
        wrist_local_err = torch.linalg.norm(wrist_actual_local - wrist_ref_local, dim=-1)
        tracking["wrist_pos_pelvis"].extend(wrist_local_err.reshape(-1).cpu().tolist())
        if args.wrist_trace_out:
            # env-0 raw vectors for offline error decomposition (static/dynamic, lag, workspace)
            ref_w = env._ref_body_pos[0, env._wrist_body_ids, :3].detach().cpu().tolist()
            act_w = env.rigid_body_states[0, env._wrist_body_ids, :3].detach().cpu().tolist()
            wrist_trace.append({
                "ref_local": wrist_ref_local[0].detach().cpu().tolist(),      # pelvis-frame reference wrist (2x3)
                "act_local": wrist_actual_local[0].detach().cpu().tolist(),   # pelvis-frame actual wrist (2x3)
                "ref_world": ref_w,                                           # world reference wrist (2x3), for speed
                "act_world": act_w,
            })
        if wrist_local_err.shape[-1] >= 2:
            tracking["left_wrist_pos_pelvis"].extend(wrist_local_err[:, 0].reshape(-1).cpu().tolist())
            tracking["right_wrist_pos_pelvis"].extend(wrist_local_err[:, 1].reshape(-1).cpu().tolist())
        # wrist ORIENTATION error: geodesic between actual & reference wrist rotation, each expressed
        # relative to its own root (factors out root orientation). Ref wrist rot from motion lib.
        try:
            _, _hand_ref = env._motion_lib.calc_motion_frame(env._motion_ids, env._get_motion_times(), hand_mode="matched")
            _ref_lh = _hand_ref["lh_wrist_rot"]; _ref_rh = _hand_ref["rh_wrist_rot"]
            if _ref_lh.dim() == 3: _ref_lh = _ref_lh.squeeze(1)
            if _ref_rh.dim() == 3: _ref_rh = _ref_rh.squeeze(1)
            ref_wq = torch.stack((_ref_lh, _ref_rh), dim=1).reshape(-1, 4)          # [L,R] matches _wrist_body_ids
            act_wq = env.rigid_body_states[:, env._wrist_body_ids, 3:7].reshape(-1, 4)
            root_q = env.root_states[:, None, 3:7].expand(-1, len(env._wrist_body_ids), -1).reshape(-1, 4)
            refroot_q = env._ref_root_rot[:, None, :].expand(-1, len(env._wrist_body_ids), -1).reshape(-1, 4)
            act_rel = quat_mul(quat_conjugate(root_q), act_wq)
            ref_rel = quat_mul(quat_conjugate(refroot_q), ref_wq)
            dq = quat_mul(quat_conjugate(act_rel), ref_rel)
            ang = (2.0 * torch.acos(dq[:, 3].abs().clamp(max=1.0))).reshape(env.num_envs, -1)
            tracking["wrist_rot"].extend(ang.reshape(-1).cpu().tolist())
            if ang.shape[-1] >= 2:
                tracking["left_wrist_rot"].extend(ang[:, 0].reshape(-1).cpu().tolist())
                tracking["right_wrist_rot"].extend(ang[:, 1].reshape(-1).cpu().tolist())
        except Exception:
            pass
        hand_ids = torch.cat((env._rh_tips_id, env._lh_tips_id))
        tracking["hand_tip_pos"].extend(torch.linalg.norm(
            env.rigid_body_states[:, hand_ids, :3] - env._ref_body_pos[:, hand_ids, :3], dim=-1
        ).reshape(-1).cpu().tolist())

        # Evaluate the actual keypoints -> DOF -> FK mapping independently of
        # whole-body wrist tracking.  The old world-space metric could report
        # 10+ cm even for ground-truth hand DOFs when the body policy missed
        # wrist orientation by ~30 degrees.
        _, current_hand_data = env._motion_lib.calc_motion_frame(
            env._motion_ids, env._get_motion_times(), hand_mode="matched"
        )
        per_hand_local_errors = []
        for side, tip_ids, wrist_slot in (
            ("rh", env._rh_tips_id, 1),
            ("lh", env._lh_tips_id, 0),
        ):
            wrist_state = env.rigid_body_states[:, env._wrist_body_ids[wrist_slot]]
            actual_local = quat_rotate_inverse(
                wrist_state[:, None, 3:7].expand(-1, len(tip_ids), -1).reshape(-1, 4),
                (env.rigid_body_states[:, tip_ids, :3] - wrist_state[:, None, :3]).reshape(-1, 3),
            ).reshape(env.num_envs, len(tip_ids), 3)

            ref_body = current_hand_data[f"{side}_body_pos"][:, [17, 3, 6, 12, 9]]
            ref_wrist_pos = current_hand_data[f"{side}_wrist_pos"].reshape(env.num_envs, 1, 3)
            ref_wrist_rot = current_hand_data[f"{side}_wrist_rot"]
            ref_local = quat_rotate_inverse(
                ref_wrist_rot[:, None, :].expand(-1, len(tip_ids), -1).reshape(-1, 4),
                (ref_body - ref_wrist_pos).reshape(-1, 3),
            ).reshape(env.num_envs, len(tip_ids), 3)
            local_error = torch.linalg.norm(actual_local - ref_local, dim=-1)
            tracking[f"{'right' if side == 'rh' else 'left'}_hand_tip_pos_wrist_frame"].extend(
                local_error.reshape(-1).cpu().tolist()
            )
            per_hand_local_errors.append(local_error)
        tracking["hand_tip_pos_wrist_frame"].extend(
            torch.cat(per_hand_local_errors, dim=-1).reshape(-1).cpu().tolist()
        )
        if args.hand_finger_trace_out:
            # env-0 per-finger world positions (10: R thumb/index/middle/ring/pinky, then L)
            # + wrist world (env._wrist_body_ids = [L, R]) for per-finger & wrist-relative error
            finger_trace.append({
                "act": env.rigid_body_states[0, hand_ids, :3].detach().cpu().tolist(),
                "ref": env._ref_body_pos[0, hand_ids, :3].detach().cpu().tolist(),
                "wrist_act": env.rigid_body_states[0, env._wrist_body_ids, :3].detach().cpu().tolist(),
                "wrist_ref": env._ref_body_pos[0, env._wrist_body_ids, :3].detach().cpu().tolist(),
                "wrist_act_quat": env.rigid_body_states[0, env._wrist_body_ids, 3:7].detach().cpu().tolist(),
            })
        body_dof_err = torch.abs(env.dof_pos[:, env.train_dof_indices] - env._ref_dof_pos[:, env.train_dof_indices])
        tracking["body_dof"].extend(body_dof_err.reshape(-1).cpu().tolist())
        for group_name, group_indices in dof_group_tensors.items():
            if len(group_indices) > 0:
                tracking[group_name].extend(torch.abs(
                    env.dof_pos[:, group_indices] - env._ref_dof_pos[:, group_indices]
                ).reshape(-1).cpu().tolist())

    def record_root_trace(step_index, done=False):
        debug = getattr(env, "termination_debug", {}) if done else {}
        if debug:
            root_z = float(debug["actual_root_z"][0].detach().cpu())
            ref_z = float(debug["ref_root_z"][0].detach().cpu())
            root_z_diff = float(debug["root_z_diff"][0].detach().cpu())
            abs_root_z_diff = float(debug["abs_root_z_diff"][0].detach().cpu())
            roll = float(debug["roll"][0].detach().cpu())
            pitch = float(debug["pitch"][0].detach().cpu())
        else:
            root_z = float(env.root_states[0, 2].detach().cpu())
            ref_z = float(env._ref_root_pos[0, 2].detach().cpu())
            root_z_diff = root_z - ref_z
            abs_root_z_diff = abs(root_z_diff)
            roll = float(env.roll[0].detach().cpu())
            pitch = float(env.pitch[0].detach().cpu())
        reasons = [
            name for name, mask in getattr(env, "termination_reasons", {}).items() if bool(mask[0])
        ]
        root_trace.append({
            "step": int(step_index),
            "time_s": float(step_index * env.dt),
            "actual_root_z": root_z,
            "ref_root_z": ref_z,
            "root_z_diff": root_z_diff,
            "abs_root_z_diff": abs_root_z_diff,
            "roll": roll,
            "pitch": pitch,
            "done": bool(done),
            "termination_reason": reasons,
        })

    online_hand_ik = None
    online_hand_ik_solve_ms = []
    online_hand_ik_residual_m = {"rh": [], "lh": []}
    if args.online_hand_ik:
        if str(_SHENGYIN_ROOT) not in sys.path:
            sys.path.insert(0, str(_SHENGYIN_ROOT))
        from human_policy.twist_hand_gmt_bridge import _PinocchioInspireIK

        online_hand_ik = {}
        for side in ("rh", "lh"):
            online_hand_ik[side] = {
                "solver": _PinocchioInspireIK(
                    side,
                    iters=args.ik_iters,
                    damping=args.ik_damping,
                    step=args.ik_step,
                    smooth_w=args.ik_smooth_weight,
                    reg_w=args.ik_reg_weight,
                ),
                "q_prev": np.zeros(12, dtype=np.float32),
            }
        cprint(
            "[hand ik] online causal Pinocchio enabled: current five-tip target "
            f"-> {args.ik_iters} IK iterations -> env.step",
            "green",
        )

    for i in tqdm(range(traj_length)):
        record_tracking()

        if if_normalize and body_normalizer is not None and hand_normalizer is not None:
            body_normalized_obs = body_normalizer.normalize(body_obs.detach())
            hand_normalized_obs = hand_normalizer.normalize(hand_obs.detach())
        else:
            body_normalized_obs = body_obs.detach()
            hand_normalized_obs = hand_obs.detach()
        with torch.no_grad():
            if residual_body_policy is not None:
                body_actions = residual_body_policy(body_obs.detach())
            else:
                body_actions = body_policy(body_normalized_obs, hist_encoding=True)
            if args.hand_control_mode == "zero":
                hand_actions = torch.zeros(env.num_envs, 24, device=env.device)
            elif args.hand_control_mode == "gmt":
                hand_actions = hand_policy(hand_normalized_obs, hist_encoding=True)
            elif args.hand_control_mode == "ik":
                _, ik_hand_data = env._motion_lib.calc_motion_frame(
                    env._motion_ids, env._get_motion_times(), hand_mode="matched"
                )
                if args.online_hand_ik:
                    # Solve from the current interpolated Cartesian target.  The
                    # solver state is carried across control steps, but no future
                    # target or precomputed rh/lh_dof_pos is read.
                    tip_indices = [17, 3, 6, 12, 9]
                    solved_q = {}
                    solve_started = time.perf_counter()
                    for side in ("rh", "lh"):
                        body_pos = ik_hand_data[f"{side}_body_pos"]
                        wrist_pos = ik_hand_data[f"{side}_wrist_pos"].reshape(
                            env.num_envs, 1, 3
                        )
                        wrist_rot = ik_hand_data[f"{side}_wrist_rot"]
                        target_local = quat_rotate_inverse(
                            wrist_rot[:, None, :]
                            .expand(-1, len(tip_indices), -1)
                            .reshape(-1, 4),
                            (body_pos[:, tip_indices] - wrist_pos).reshape(-1, 3),
                        ).reshape(env.num_envs, len(tip_indices), 3)
                        if env.num_envs != 1:
                            raise RuntimeError(
                                "--online-hand-ik evaluation currently requires one environment"
                            )
                        context = online_hand_ik[side]
                        q, residual = context["solver"].solve_frame(
                            target_local[0].detach().cpu().numpy(),
                            context["q_prev"],
                        )
                        context["q_prev"] = q
                        solved_q[side] = torch.as_tensor(
                            q[None], device=env.device, dtype=torch.float32
                        )
                        online_hand_ik_residual_m[side].append(residual)
                    online_hand_ik_solve_ms.append(
                        1000.0 * (time.perf_counter() - solve_started)
                    )
                    rh_q = solved_q["rh"]
                    lh_q = solved_q["lh"]
                else:
                    # Offline baseline: read the precomputed IK reference at the
                    # same interpolated motion time.
                    rh_q = ik_hand_data["rh_dof_pos"]
                    lh_q = ik_hand_data["lh_dof_pos"]
                    if rh_q.dim() == 3:
                        rh_q = rh_q.squeeze(1)
                    if lh_q.dim() == 3:
                        lh_q = lh_q.squeeze(1)
                rh_default = env.default_dof_pos_all[:, env.rh_dof_indices]
                lh_default = env.default_dof_pos_all[:, env.lh_dof_indices]
                hand_actions = torch.cat([rh_q - rh_default, lh_q - lh_default], dim=-1)
                hand_actions = hand_actions / float(env.cfg.control.action_scale)
            else:
                raise RuntimeError(f"Unsupported hand control mode: {args.hand_control_mode}")
         
         # Combine actions
        actions = torch.zeros(env.num_envs, 53, device=env.device, requires_grad=False)
        body_dof_indices = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 34, 35, 36, 37, 38, 39, 40]
        rh_dof_indices = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        lh_dof_indices = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        assert body_actions.shape[-1] == len(body_dof_indices), f"body action dim mismatch: {body_actions.shape[-1]}"
        assert hand_actions.shape[-1] == len(rh_dof_indices) + len(lh_dof_indices), f"hand action dim mismatch: {hand_actions.shape[-1]}"
        actions[:, body_dof_indices] = body_actions
        if args.wrist_ff_alpha > 0.0:
            alpha = float(args.wrist_ff_alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"--wrist-ff-alpha must be in [0, 1], got {alpha}")
            wrist_global = [int(idx) for idx in env.wrist_dof_indices.detach().cpu().tolist()]
            if not all(idx in body_dof_indices for idx in wrist_global):
                raise RuntimeError(f"wrist DOFs {wrist_global} are not all controlled by body policy")
            if wrist_online_ik is not None:
                ref_data_now, _ = env._motion_lib.calc_motion_frame(
                    env._motion_ids, env._get_motion_times(), hand_mode="matched"
                )
                ref_root_inv = quat_conjugate(ref_data_now["root_quat"])
                target_quat_b = torch.stack(
                    [
                        quat_mul(ref_root_inv, ref_data_now["lh_wrist_rot"]),
                        quat_mul(ref_root_inv, ref_data_now["rh_wrist_rot"]),
                    ],
                    dim=1,
                )
                wrist_target_q = wrist_online_ik.solve(target_quat_b)
            else:
                wrist_target_q = env._ref_dof_pos[:, wrist_global]
            ref_wrist_action = (
                wrist_target_q - env.default_dof_pos_all[:, wrist_global]
            ) / float(env.cfg.control.action_scale)
            policy_wrist_action = actions[:, wrist_global]
            actions[:, wrist_global] = (1.0 - alpha) * policy_wrist_action + alpha * ref_wrist_action
        actions[:, rh_dof_indices] = hand_actions[:, :12]
        actions[:, lh_dof_indices] = hand_actions[:, 12:]
        body_obs, hand_obs, _, rews, dones, infos = env.step(actions.detach())
        record_root_trace(i + 1, done=torch.any(dones).item())
        if torch.any(dones):
            done_step = i + 1
            termination_reason = [
                name for name, mask in getattr(env, "termination_reasons", {}).items() if bool(mask[0])
            ]
            if args.stop_on_done:
                break

    def metric_summary(values):
        values = torch.tensor(values, dtype=torch.float32)
        return {
            "mean": float(values.mean()),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
        }

    executed_steps = done_step if done_step is not None else min(traj_length, len(tracking["root_pos"]))
    motion_length = float(env._motion_lib.get_motion_length(torch.zeros(1, device=env.device, dtype=torch.long))[0])
    metrics = {
        "body_task": args.body_task,
        "body_checkpoint": body_resume_path,
        "residual_body_policy": bool(args.use_residual_body_policy),
        "base_policy_checkpoint": base_policy_path,
        "residual_task": args.residual_task if args.use_residual_body_policy else None,
        "residual_checkpoint": residual_resume_path,
        "hand_checkpoint": hand_resume_path if hand_resume_path is not None else "zero_hand_policy",
        "hand_control_mode": args.hand_control_mode,
        "online_hand_ik": bool(args.online_hand_ik),
        "online_hand_ik_solver": (
            {
                "backend": "pino",
                "iters_per_control_step": int(args.ik_iters),
                "damping": float(args.ik_damping),
                "step": float(args.ik_step),
                "smooth_weight": float(args.ik_smooth_weight),
                "regularization_weight": float(args.ik_reg_weight),
                "solve_time_ms": metric_summary(online_hand_ik_solve_ms),
                "target_fit_residual_m": {
                    side: metric_summary(values)
                    for side, values in online_hand_ik_residual_m.items()
                },
            }
            if args.online_hand_ik
            else None
        ),
        "hand_reference": prepared_hand_motion_file,
        "hand_checkpoint_spec": hand_checkpoint_spec,
        "canonicalize_hand_frame": args.canonicalize_hand_frame,
        "canonical_hand_reference": str(Path(args.canonical_hand_reference).expanduser()) if args.canonicalize_hand_frame != "none" else None,
        "steps": int(executed_steps),
        "tracked_seconds": float(executed_steps * env.dt),
        "motion_length_seconds": motion_length,
        "completion": float(min(executed_steps * env.dt / max(motion_length, 1e-6), 1.0)),
        "terminated": done_step is not None and executed_steps * env.dt < motion_length - env.dt,
        "termination_reason": termination_reason,
        "dof_groups": {
            name: [env.dof_names[idx] for idx in indices]
            for name, indices in dof_group_indices.items()
        },
        "errors_m": {
            name: metric_summary(values) for name, values in tracking.items() if name != "root_rot" and name != "body_dof"
            and not name.endswith("_dof")
        },
        "root_rotation_error_rad": metric_summary(tracking["root_rot"]),
        "body_dof_error_rad": metric_summary(tracking["body_dof"]),
        "dof_error_rad": {
            name: metric_summary(values) for name, values in tracking.items() if name.endswith("_dof") and values
        },
    }
    print(json.dumps(metrics, indent=2))
    if args.metrics_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, "w") as output:
            json.dump(metrics, output, indent=2)
    if args.root_trace_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.root_trace_out)), exist_ok=True)
        with open(args.root_trace_out, "w") as output:
            json.dump(root_trace, output, indent=2)
    if args.wrist_trace_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.wrist_trace_out)), exist_ok=True)
        with open(args.wrist_trace_out, "w") as output:
            json.dump({"dt": float(env.dt), "trace": wrist_trace}, output)
    if args.hand_finger_trace_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.hand_finger_trace_out)), exist_ok=True)
        with open(args.hand_finger_trace_out, "w") as output:
            json.dump({"finger_order": ["R_thumb","R_index","R_middle","R_ring","R_pinky",
                                        "L_thumb","L_index","L_middle","L_ring","L_pinky"],
                       "wrist_order": ["L","R"], "trace": finger_trace}, output)
    

if __name__ == '__main__':
    # args = get_args()
    args = get_combine_args()
    play(args)
