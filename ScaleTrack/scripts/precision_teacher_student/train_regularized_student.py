"""A/B/C on-policy distillation; only training losses differ between groups."""
import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
for name in ("motion-file", "heldout-file", "teacher-checkpoint", "initial-checkpoint", "run-name"):
    p.add_argument("--" + name, required=True)
p.add_argument("--variant", choices=("A", "B", "C"), required=True)
p.add_argument("--iterations", type=int, default=5000)
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=16)
p.add_argument("--minibatch", type=int, default=1024)
p.add_argument("--calibration-update", type=int, default=100)
p.add_argument("--ramp-updates", type=int, default=500)
p.add_argument("--save-interval", type=int, default=500)
p.add_argument("--eval-interval", type=int, default=1000)
p.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import copy
import gymnasium as gym
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import scaletrack.tasks
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_ppo_cfg import G1HAT4FinetunePPORunnerCfg
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from env_cfg import PrecisionEnvCfg, student_configuration
from policy import mean_action, disable_phase_weights
from student_losses import (sequence_indices, consistency_loss, temporal_loss,
                            gradient_norm, calibrated_weight)
from student_validation import evaluate_heldout
from student_lifecycle import run_with_cleanup


def main(resources):
    if args.steps < 3 or args.iterations <= args.calibration_update or args.ramp_updates <= 0:
        raise ValueError("Insufficient rollout/training length for regularization")
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
    # DAgger does not optimize rewards. Avoid computing an unused precision term.
    cfg.rewards.wrist_precision = None
    agent = G1HAT4FinetunePPORunnerCfg()
    agent.device = args.device
    agent.eval_focus_metric_keys = []
    agent.eval_metric_keys += ["error_left_wrist_global_pos", "error_right_wrist_global_pos"]
    log_dir = Path("logs/rsl_rl/student_regularized") / args.run_name
    log_dir.mkdir(parents=True, exist_ok=False)
    env = RslRlVecEnvWrapper(gym.make("G1-HAT4-F2O-Finetune", cfg=cfg))
    resources.callback(env.close)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(log_dir.resolve()), device=args.device)
    runner.load(args.initial_checkpoint, load_optimizer=False, map_location=args.device)
    student = runner.alg.policy
    # Bypass the legacy inference=True path, which overrides HAT-4 with 14 links.
    runner.get_inference_policy = lambda device=None: lambda observation: mean_action(student, observation)
    disable_phase_weights(student)
    teacher = copy.deepcopy(student)
    teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=args.device,
                                      weights_only=False)["model_state_dict"])
    teacher.eval().requires_grad_(False)
    parameters = list(student.actor_parameters)
    optimizer = torch.optim.Adam(parameters, lr=2e-5)
    contract = json.loads(Path(__file__).with_name("deployment_contract.json").read_text())
    robot = env.unwrapped.scene["robot"]
    assert list(robot.joint_names) == contract["joint_names"]
    scale = torch.as_tensor(env.unwrapped.action_manager.get_term("joint_pos")._scale,
                            device=args.device).reshape(-1, 29)[0].clone()
    torch.testing.assert_close(scale, torch.tensor(contract["action_scale"], device=args.device))
    arm_ids = [i for i,n in enumerate(robot.joint_names)
               if any(s in n for s in ("shoulder", "elbow", "wrist"))]
    command = env.unwrapped.command_manager.get_term("motion")
    wrist_ids = [command.cfg.body_names.index(s) for s in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    assert cfg.commands.motion.mode_sampling_weights["HAT-4"] == 1
    assert not cfg.observations.policy_task.enable_corruption
    assert cfg.observations.policy.enable_corruption
    writer = SummaryWriter(str(log_dir))
    resources.callback(writer.close)
    reg_generator = torch.Generator(device=args.device).manual_seed(73042)
    obs = env.get_observations()
    noise = torch.zeros(args.num_envs, 3, device=args.device)
    continuation = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
    weights = {"consistency": 0.0, "temporal": 0.0}
    fields = ("policy", "policy_task", "action", "mode", "mode_mapping")
    configuration = {**vars(args), "seed": 42, "lr": 2e-5, "epochs": 2,
                     "rollout_noise_mm": 3, "consistency_noise_mm": 1,
                     "calibration_gradient_fraction": .1, "history": 3,
                     "task_width": 269, "focus": False, "arm_joints": arm_ids,
                     "action_scale": scale.tolist(), "deployment_changed": False}
    print("STUDENT_CONTRACT " + json.dumps(configuration, default=str), flush=True)
    started = time.monotonic()
    for iteration in range(args.iterations):
        student.eval()
        batches = []
        with torch.no_grad():
            for _ in range(args.steps):
                noise = .9 * noise + (1 - .9**2)**.5 * .003 * torch.randn_like(noise)
                batch = TensorDict({k: obs[k].detach().clone() for k in fields}, batch_size=[args.num_envs])
                batch["noise"] = noise.clone()
                batch["target"] = mean_action(teacher, obs, teacher=True).detach().clone()
                batch["continuation"] = continuation.clone()
                batch["wrist_speed"] = torch.linalg.vector_norm(
                    command.body_lin_vel_w[:, wrist_ids].to(args.device), dim=-1).max(-1).values
                action = mean_action(student, obs, position_noise=noise)
                obs, _, dones, _ = env.step(action)
                continuation = ~dones.bool().reshape(-1)
                noise[~continuation] = 0
                batches.append(batch)
        data = torch.cat(batches, dim=0)
        student.train()
        totals = {"bc": 0., "consistency": 0., "temporal": 0., "total": 0.}
        count = 0
        ramp = min(1., max(0., (iteration - args.calibration_update) / args.ramp_updates))
        for epoch in range(2):
            for indices in torch.randperm(len(data), device=args.device).split(args.minibatch):
                batch = data[indices]
                previous = previous2 = valid = None
                if args.variant == "C" and iteration >= args.calibration_update:
                    i2, i1, valid = sequence_indices(indices, args.num_envs, data["continuation"])
                    sequence = data[torch.cat((i2, i1, indices))]
                    all_predicted = mean_action(student, sequence, position_noise=sequence["noise"])
                    previous2, previous, predicted = all_predicted.chunk(3)
                else:
                    predicted = mean_action(student, batch, position_noise=batch["noise"])
                bc = (predicted - batch["target"]).square().mean()
                spatial = predicted.sum() * 0
                temporal = predicted.sum() * 0
                if args.variant != "A" and iteration >= args.calibration_update:
                    perturbation = torch.randn((len(batch),3), device=args.device,
                                                generator=reg_generator) * .001
                    perturbed = mean_action(student, batch, position_noise=batch["noise"] + perturbation)
                    spatial = consistency_loss(predicted, perturbed, scale)
                if previous is not None:
                    temporal = temporal_loss(predicted, previous, previous2, valid,
                                             batch["wrist_speed"], scale, arm_ids)
                if iteration == args.calibration_update and count == 0 and args.variant != "A":
                    norms = {"bc": gradient_norm(bc, parameters), "consistency": gradient_norm(spatial, parameters)}
                    weights["consistency"] = calibrated_weight(norms["bc"], norms["consistency"])
                    if args.variant == "C":
                        norms["temporal"] = gradient_norm(temporal, parameters)
                        weights["temporal"] = calibrated_weight(norms["bc"], norms["temporal"])
                    print("REGULARIZER_CALIBRATION " + json.dumps({"update": iteration+1, "norms": norms, "weights": weights}), flush=True)
                loss = bc + ramp * (weights["consistency"] * spatial + weights["temporal"] * temporal)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite student loss")
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
                optimizer.step()
                for name,value in (("bc",bc),("consistency",spatial),("temporal",temporal),("total",loss)):
                    totals[name] += float(value.detach())
                count += 1
        disable_phase_weights(student)
        for name,value in totals.items():
            writer.add_scalar("Student/"+name, value/count, iteration+1)
        writer.add_scalar("Student/regularizer_ramp", ramp, iteration+1)
        writer.add_scalar("Student/consistency_weight", weights["consistency"], iteration+1)
        writer.add_scalar("Student/temporal_weight", weights["temporal"], iteration+1)
        writer.add_scalar("Student/rollout_reset_fraction", float((~data["continuation"]).float().mean()), iteration+1)
        if iteration < 3 or (iteration+1) % 25 == 0 or args.smoke:
            elapsed = time.monotonic() - started
            print("STUDENT_UPDATE " + json.dumps({"update":iteration+1, "variant":args.variant,
                "losses":{k:v/count for k,v in totals.items()}, "ramp":ramp,
                "elapsed_s":elapsed, "eta_s":elapsed/(iteration+1)*(args.iterations-iteration-1),
                "grad_norm":float(norm), "peak_gpu_gb":torch.cuda.max_memory_allocated()/1e9}), flush=True)
        if (iteration+1) % args.save_interval == 0 or iteration+1 == args.iterations:
            destination = log_dir / f"model_{iteration+1}.pt"
            temporary = destination.with_suffix(".pt.tmp")
            torch.save({"model_state_dict":student.state_dict(), "iter":iteration+1,
                        "actor_optimizer_state_dict":optimizer.state_dict(),
                        "critic_optimizer_state_dict":runner.alg.critic_optimizer.state_dict(),
                        "infos":configuration, "regularizer_weights":weights}, temporary)
            temporary.replace(destination)
        if (iteration+1) % args.eval_interval == 0 or (args.smoke and iteration+1 == args.iterations):
            metrics = evaluate_heldout(runner)
            for name,value in metrics.items():
                writer.add_scalar("Eval_Test/"+name,value,iteration+1)
            print("STUDENT_HELDOUT "+json.dumps({"update":iteration+1,"metrics":metrics}),flush=True)
            command.resample_motions()
            command.randomize_next_resampling = True
            obs, _ = env.reset()
            noise.zero_()
            continuation.zero_()
    print("STUDENT_TRAINING_COMPLETE " + args.run_name, flush=True)


if __name__ == "__main__":
    from isaaclab.sim import SimulationContext
    resources = ExitStack()
    run_with_cleanup(lambda: main(resources), app, resources, SimulationContext.instance)
