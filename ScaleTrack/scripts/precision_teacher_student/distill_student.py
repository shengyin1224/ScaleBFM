"""On-policy DAgger from a trained 14-link teacher to deployment-compatible HAT-4."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--motion-file", required=True)
p.add_argument("--heldout-file", required=True)
p.add_argument("--teacher-checkpoint", required=True)
p.add_argument("--initial-checkpoint", required=True)
p.add_argument("--run-name", required=True)
p.add_argument("--iterations", type=int, default=5000)
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=16)
p.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import copy
import gymnasium as gym
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import scaletrack.tasks
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_ppo_cfg import G1HAT4FinetunePPORunnerCfg
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from env_cfg import PrecisionEnvCfg, student_configuration
from policy import mean_action, disable_phase_weights


def main():
    torch.manual_seed(42)
    cfg = PrecisionEnvCfg()
    student_configuration(cfg)
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.commands.motion.motion_file = args.motion_file
    cfg.commands.motion.test_motion_file = args.heldout_file
    cfg.commands.motion.playback_start_step = 0
    cfg.commands.motion.playback_end_step = 2000 * args.steps
    agent = G1HAT4FinetunePPORunnerCfg()
    agent.device = args.device
    log_dir = Path("logs/rsl_rl/precision_teacher_student") / args.run_name
    if log_dir.exists():
        raise FileExistsError(log_dir)
    log_dir.mkdir(parents=True)
    env = RslRlVecEnvWrapper(gym.make("G1-HAT4-F2O-Finetune", cfg=cfg))
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log_dir.resolve()), device=args.device)
    runner.load(args.initial_checkpoint, load_optimizer=False, map_location=args.device)
    student = runner.alg.policy
    disable_phase_weights(student)
    teacher = copy.deepcopy(student)
    teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=args.device,
                                      weights_only=False)["model_state_dict"])
    teacher.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(student.actor_parameters, lr=2e-5)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(log_dir))
    obs = env.get_observations()
    noise = torch.zeros(args.num_envs, 3, device=args.device)
    for iteration in range(args.iterations):
        student.eval()
        batches = []
        with torch.no_grad():
            for _ in range(args.steps):
                noise = 0.9 * noise + (1 - 0.9**2)**0.5 * 0.003 * torch.randn_like(noise)
                target = mean_action(teacher, obs, teacher=True)
                action = mean_action(student, obs, position_noise=noise)
                batch = TensorDict({k: obs[k].detach().clone() for k in
                                   ("policy", "policy_task", "action", "mode", "mode_mapping")},
                                   batch_size=[args.num_envs])
                batch["noise"] = noise.clone()
                batch["target"] = target.clone()
                batches.append(batch)
                obs, _, dones, _ = env.step(action)
                noise[dones.bool().reshape(-1)] = 0
        data = torch.cat(batches, dim=0)
        student.train()
        total = 0.0
        count = 0
        for _ in range(2):
            for indices in torch.randperm(len(data), device=args.device).split(2048):
                batch = data[indices]
                predicted = mean_action(student, batch, position_noise=batch["noise"])
                loss = (predicted - batch["target"]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite distillation loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.actor_parameters, 0.5)
                optimizer.step()
                total += float(loss.detach())
                count += 1
        disable_phase_weights(student)
        writer.add_scalar("DAgger/action_mse", total/count, iteration+1)
        if iteration % 25 == 0:
            print("DAGGER", iteration+1, "action_mse", total/count, flush=True)
        if (iteration+1) % 500 == 0 or iteration+1 == args.iterations:
            torch.save({"model_state_dict": student.state_dict(), "iter": iteration+1,
                        "actor_optimizer_state_dict": optimizer.state_dict(),
                        "critic_optimizer_state_dict": runner.alg.critic_optimizer.state_dict(),
                        "infos": {"teacher": args.teacher_checkpoint, "phase_disabled": True,
                                  "mode": "HAT-4", "task_width": 269},
                        "distillation_optimizer_state_dict": optimizer.state_dict()},
                       log_dir / f"model_{iteration+1}.pt")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
