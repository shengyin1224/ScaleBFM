"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Optional explicit video output directory (useful for batch rendering).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--mode_index", type=int, default=None, help="Fixed motion mode index to use during playback.")
parser.add_argument(
    "--local_tracking",
    action="store_true",
    default=False,
    help="Use the reference root position for observations; requires a root-containing --mode_index.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import scaletrack.tasks  # noqa: F401


class _HiddenMarker:

    def __init__(self, marker):
        self._marker = marker
        self._marker.set_visibility(False)

    def set_visibility(self, _visible):
        self._marker.set_visibility(False)

    def visualize(self, *_args, **_kwargs):
        pass


class _LocalReferenceMarker:

    def __init__(self, marker, motion_command, robot_root_position):
        self._marker = marker
        self._motion_command = motion_command
        self._robot_root_position = robot_root_position

    def set_visibility(self, visible):
        self._marker.set_visibility(visible)

    def visualize(self, positions, orientations, *args, **kwargs):
        root_offset = self._robot_root_position(self._motion_command) - self._motion_command.anchor_pos_w
        self._marker.visualize(positions + root_offset, orientations, *args, **kwargs)


def _enable_local_tracking(motion_command):

    command_type = type(motion_command)
    robot_root_position = command_type.robot_anchor_pos_w.fget

    class _LocalTrackingMotionCommand(command_type):
        @property
        def robot_anchor_pos_w(self):
            return self.anchor_pos_w

    motion_command.__class__ = _LocalTrackingMotionCommand
    return robot_root_position


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.enable_reset_disturbance = False
    if args_cli.mode_index is not None:
        mode_candidates = list(env_cfg.commands.motion.mode_candidates.items())
        if not 0 <= args_cli.mode_index < len(mode_candidates):
            raise ValueError(
                f"mode_index must be between 0 and {len(mode_candidates) - 1}; received {args_cli.mode_index}."
            )
        mode_name, link_names = mode_candidates[args_cli.mode_index]
        env_cfg.commands.motion.mode_candidates = {mode_name: link_names}
        env_cfg.commands.motion.mode_sampling_weights = {mode_name: 1.0}
        print(f"[INFO]: Using fixed mode index {args_cli.mode_index}: {mode_name} ({link_names})")
    if args_cli.local_tracking:
        if args_cli.mode_index is None:
            raise ValueError("--local_tracking requires --mode_index with a mode that includes the root link.")
        root_link_name = env_cfg.commands.motion.anchor_body_name
        if root_link_name not in link_names:
            raise ValueError(
                f"Local tracking requires the selected mode to include the root link '{root_link_name}'; "
                f"mode '{mode_name}' contains {link_names}."
            )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.mode_index is not None or args_cli.local_tracking:
        motion_command = env.unwrapped.command_manager.get_term("motion")

    if args_cli.local_tracking:
        robot_root_position = _enable_local_tracking(motion_command)
        if hasattr(motion_command, "goal_body_visualizers"):
            for index, marker in enumerate(motion_command.goal_body_visualizers):
                motion_command.goal_body_visualizers[index] = _LocalReferenceMarker(
                    marker, motion_command, robot_root_position
                )
        print("[INFO]: Local tracking enabled: observations use the reference root position.")

    if args_cli.mode_index is not None:
        active_link_names = set(link_names)
        if hasattr(motion_command, "current_body_visualizers"):
            for index, body_name in enumerate(motion_command.cfg.body_names):
                if body_name not in active_link_names:
                    motion_command.current_body_visualizers[index] = _HiddenMarker(
                        motion_command.current_body_visualizers[index]
                    )
                    motion_command.goal_body_visualizers[index] = _HiddenMarker(
                        motion_command.goal_body_visualizers[index]
                    )

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": (
                os.path.abspath(args_cli.video_folder)
                if args_cli.video_folder is not None
                else os.path.join(log_dir, "videos", "play")
            ),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    
    ppo_runner._set_env_is_evaluating()

    if args_cli.mode_index is not None:
        # During normal inference the policy intentionally uses the complete task observation and an all-ones mode.
        # For fixed-mode playback, reuse the training observation path so the selected mode masks the task observation
        # and is passed to the actor explicitly. This instance-only override does not affect training.
        actor_critic = ppo_runner.alg.policy
        get_actor_obs = actor_critic.get_actor_obs

        def get_fixed_mode_actor_obs(obs, inference=False):
            return get_actor_obs(obs, inference=False)

        actor_critic.get_actor_obs = get_fixed_mode_actor_obs

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
