"""Train a complete 14-link precision teacher without touching old runs."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--motion-file", required=True)
p.add_argument("--heldout-file", required=True)
p.add_argument("--initial-checkpoint", required=True)
p.add_argument("--run-name", required=True)
p.add_argument("--cap", type=float, choices=(1.0, 2.0), required=True)
p.add_argument("--num-envs", type=int, default=4096)
p.add_argument("--iterations", type=int, default=20000)
p.add_argument("--resume-checkpoint")
p.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import scaletrack.tasks
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_ppo_cfg import G1HAT4FinetunePPORunnerCfg
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from env_cfg import PrecisionEnvCfg
from policy import disable_phase_weights


def main():
    cfg = PrecisionEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.commands.motion.motion_file = args.motion_file
    cfg.commands.motion.test_motion_file = args.heldout_file
    cfg.rewards.wrist_precision.params["cap"] = args.cap
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        cfg.commands.motion.schedule_step_offset = saved["iter"] * 64
    agent = G1HAT4FinetunePPORunnerCfg()
    agent.device = args.device
    agent.save_interval = 1000
    agent.eval_interval = 2000
    agent.eval_during_training = not args.smoke
    agent.eval_focus_metric_keys = []
    agent.eval_metric_keys += ["error_left_wrist_global_pos", "error_right_wrist_global_pos"]
    agent.algorithm.schedule = "fixed"
    agent.algorithm.actor_learning_rate = 2e-5
    agent.algorithm.entropy_final_coef = 0.001
    agent.algorithm.entropy_decay_updates = 8000
    log_dir = Path("logs/rsl_rl/precision_teacher_student") / args.run_name
    if log_dir.exists() and not args.resume_checkpoint:
        raise FileExistsError(log_dir)
    env = gym.make("G1-HAT4-F2O-Finetune", cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log_dir.resolve()), device=args.device)
    runner.load(args.resume_checkpoint or args.initial_checkpoint,
                load_optimizer=bool(args.resume_checkpoint), map_location=args.device)
    if not args.resume_checkpoint:
        runner.current_learning_iteration = 0
    disable_phase_weights(runner.alg.policy)
    # The default inference path overrides sparse modes; explicitly keep sampled mode.
    original = runner.alg.policy.get_actor_obs
    runner.alg.policy.get_actor_obs = lambda obs, inference=False: original(obs, inference=False)
    print("PRECISION_CONTRACT", {"mode": "WholeBody-14", "cap_combined": args.cap,
          "focus": False, "actor_input": 269, "history": 3, "future": [0,1,2,3,4,5],
          "curriculum_updates": [2000,8000], "initial": args.initial_checkpoint}, flush=True)
    runner.learn(args.iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
