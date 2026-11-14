"""Direct HAT-4 PPO ablations, isolated from the running teacher experiments."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion-file", required=True)
parser.add_argument("--heldout-file", required=True)
parser.add_argument("--initial-checkpoint", required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--cap", type=float, choices=(0., 1., 2.), required=True)
parser.add_argument("--playback", action="store_true")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=20000)
parser.add_argument("--resume-checkpoint")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--smoke-full-curriculum", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import scaletrack.tasks
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_ppo_cfg import G1HAT4FinetunePPORunnerCfg
from scaletrack.tasks.tracking.tracking_env_cfg import BFMMaskObservationCfg
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from env_cfg import PrecisionEnvCfg
from policy import disable_phase_weights


def direct_configuration():
    cfg = PrecisionEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.commands.motion.motion_file = args.motion_file
    cfg.commands.motion.test_motion_file = args.heldout_file
    cfg.commands.motion.mode_sampling_weights = {
        name: float(name == "HAT-4") for name in cfg.commands.motion.mode_candidates}
    cfg.commands.motion.playback_enabled = args.playback
    cfg.observations.policy = BFMMaskObservationCfg.PolicyCfg()
    # Direct PPO needs no duplicate teacher histories or commands.
    cfg.observations.teacher_policy = None
    cfg.observations.teacher_policy_task = None
    if args.cap == 0:
        cfg.rewards.wrist_precision = None
    else:
        cfg.rewards.wrist_precision.params["cap"] = args.cap
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        cfg.commands.motion.schedule_step_offset = saved["iter"] * 64
    if args.smoke_full_curriculum:
        if not args.smoke:
            raise ValueError("--smoke-full-curriculum is only allowed with --smoke")
        cfg.commands.motion.schedule_step_offset = 8000 * 64
    return cfg


def check_configuration(cfg):
    assert cfg.commands.motion.mode_sampling_weights["HAT-4"] == 1
    assert sum(cfg.commands.motion.mode_sampling_weights.values()) == 1
    assert cfg.observations.policy.enable_corruption
    assert not cfg.observations.policy_task.enable_corruption
    for name in ("target_body_pos", "target_body_pos_rel", "target_body_rot", "target_body_rot_rel"):
        term = getattr(cfg.observations.policy_task, name)
        assert term.noise is None
        assert term.params["future_idx"] == [0, 1, 2, 3, 4, 5]
    for name in ("motion_body_pos", "motion_body_rot"):
        term = getattr(cfg.rewards, name)
        assert term.params.get("focus_weight", 1.) == 1.
    assert cfg.commands.motion.playback_enabled == args.playback
    assert (cfg.rewards.wrist_precision is None) == (args.cap == 0)


def main():
    if args.smoke:
        original_cap, original_playback = args.cap, args.playback
        for cap, playback in ((0.,False),(0.,True),(1.,False),(2.,False),(1.,True),(2.,True)):
            args.cap, args.playback = cap, playback
            check_configuration(direct_configuration())
        args.cap, args.playback = original_cap, original_playback
        print("ALL_SIX_CONFIGURATIONS_VALIDATED", flush=True)
    cfg = direct_configuration()
    check_configuration(cfg)
    agent = G1HAT4FinetunePPORunnerCfg()
    agent.device = args.device
    agent.save_interval = 1000
    agent.eval_interval = 2000
    agent.eval_during_training = not args.smoke
    agent.eval_focus_metric_keys = []
    agent.eval_metric_keys += ["error_left_wrist_global_pos", "error_right_wrist_global_pos"]
    agent.algorithm.schedule = "fixed"
    agent.algorithm.actor_learning_rate = 2e-5
    agent.algorithm.entropy_final_coef = .001
    agent.algorithm.entropy_decay_updates = 8000
    log_dir = Path("logs/rsl_rl/hat4_direct_ablation") / args.run_name
    if log_dir.exists() and not args.resume_checkpoint:
        raise FileExistsError(log_dir)
    env = RslRlVecEnvWrapper(gym.make("G1-HAT4-F2O-Finetune", cfg=cfg))
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log_dir.resolve()), device=args.device)
    runner.load(args.resume_checkpoint or args.initial_checkpoint,
                load_optimizer=bool(args.resume_checkpoint), map_location=args.device)
    if not args.resume_checkpoint:
        runner.current_learning_iteration = 0
    disable_phase_weights(runner.alg.policy)
    original = runner.alg.policy.get_actor_obs
    runner.alg.policy.get_actor_obs = lambda obs, inference=False: original(obs, inference=False)
    contract = {"mode": "HAT-4", "cap_combined": args.cap, "playback": args.playback,
                "playback_std_end": .3 if args.playback else 0, "focus": False,
                "actor_input": 269, "history": 3, "future": [0, 1, 2, 3, 4, 5],
                "proprioception_noise": True, "reference_noise": False,
                "curriculum_updates": [2000, 8000], "initial": args.initial_checkpoint,
                "train_manifest": args.motion_file, "heldout_manifest": args.heldout_file}
    print("DIRECT_CONTRACT " + json.dumps(contract), flush=True)
    runner.learn(args.iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
