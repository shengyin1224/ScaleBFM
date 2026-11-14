"""Paired old/new checkpoint evaluation; only policy phase inputs are ablated."""
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--motion-file', required=True)
parser.add_argument('--july-checkpoint', required=True)
parser.add_argument('--august-checkpoint', required=True)
parser.add_argument('--num-envs', type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args)

import hashlib
import json
import random
import numpy as np
import torch
import yaml
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import scaletrack.tasks
from scaletrack.tasks.tracking.config.g1_29dof.hat4_finetune_env_cfg import G1HAT4PhaseObsOnlyFinetuneEnvCfg
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_ppo_cfg import G1HAT4F3HeadFinetunePPORunnerCfg
from scaletrack.tasks.tracking.mdp.observations import focus_phase
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
BASE_METRICS = ['error_anchor_height', 'error_anchor_rot', 'error_anchor_pos', 'error_anchor_lin_vel', 'error_anchor_ang_vel', 'error_body_pos_g', 'error_body_pos', 'error_body_pos_relative', 'error_body_rot', 'error_body_rot_relative', 'error_joint_pos', 'error_joint_vel']
FOCUS_METRICS = ['error_anchor_pos', 'cosine_pelvis_left_wrist_global_pos_error', 'cosine_pelvis_right_wrist_global_pos_error', 'error_left_wrist_global_pos', 'error_left_wrist_global_rot', 'error_right_wrist_global_pos', 'error_right_wrist_global_rot', 'error_head_proxy_global_pos', 'error_head_proxy_global_rot', 'error_left_wrist_pelvis_local_pos', 'error_left_wrist_pelvis_local_rot', 'error_right_wrist_pelvis_local_pos', 'error_right_wrist_pelvis_local_rot', 'error_head_proxy_pelvis_local_pos', 'error_head_proxy_pelvis_local_rot']

phase_mode = 'correct'
def policy_phase(env, command_name, token_count):
    if phase_mode == 'correct':
        return focus_phase(env, command_name, token_count)
    phase = torch.zeros(env.num_envs, token_count, 2, device=env.device)
    phase[..., 1] = 2.
    return phase

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
manifest = yaml.safe_load(Path(args.motion_file).read_text())
label_hash = hashlib.sha256()
for name, path in manifest.items():
    with np.load(path) as data:
        label_hash.update(name.encode())
        for key in ('focus_mask', 'focus_time_to_next_edge_s', 'focus_valid'):
            label_hash.update(np.asarray(data[key]).tobytes())
label_digest = label_hash.hexdigest()

# F2O has the historical command/physics and the exact 2-D phase interface,
# without the later time-warp / wrist-precision changes now present in F3h.
cfg = G1HAT4PhaseObsOnlyFinetuneEnvCfg()
cfg.seed = 42
cfg.sim.device = args.device
cfg.scene.num_envs = args.num_envs
cfg.commands.motion.motion_file = str(Path(args.motion_file).resolve())
cfg.commands.motion.test_motion_file = ''
cfg.commands.motion.enable_reset_disturbance = False
cfg.commands.motion.debug_vis = False
cfg.commands.motion.mode_sampling_weights = {name: float(name == 'HAT-4') for name in cfg.commands.motion.mode_candidates}
cfg.observations.policy_task.focus_phase.func = policy_phase
# Critic inputs do not drive actions. Keep their genuine labels unchanged.
agent = G1HAT4F3HeadFinetunePPORunnerCfg()
agent.device = args.device
agent.eval_metric_keys = BASE_METRICS + FOCUS_METRICS
agent.eval_focus_metric_keys = FOCUS_METRICS
agent.eval_max_steps = 1000
agent.success_metric_dict = {'error_body_pos_g': .5}
env = RslRlVecEnvWrapper(gym.make('G1-HAT4-F2O-Finetune', cfg=cfg))
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(out / 'runtime'), device=args.device)
Path(runner.log_dir).mkdir(exist_ok=True)
actor = runner.alg.policy
original_get_actor_obs = actor.get_actor_obs
actor.get_actor_obs = lambda obs, inference=False: original_get_actor_obs(obs, inference=False)
reports = {}
for version, checkpoint in [('july', args.july_checkpoint), ('august', args.august_checkpoint)]:
    runner.load(checkpoint, load_optimizer=False, map_location=args.device)
    for mode in ('correct', 'constant_0_2'):
        key = f'{version}_{mode}'
        phase_mode = mode
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        env.unwrapped.seed(42)
        with torch.inference_mode():
            metrics = runner.evaluate_policy(test_set=False)
        per_motion = runner.last_eval_per_motion_metrics
        torch.save(per_motion, out / f'{key}.per_motion.pt')
        report = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                      phase_input=mode, constant=[0.,2.] if mode != 'correct' else None,
                      motion_file=str(Path(args.motion_file).resolve()), label_sha256=label_digest,
                      motions=len(manifest), seed=42, eval_max_steps=1000, mode='HAT-4',
                      statistics_focus='unchanged correct v2 labels', metrics=metrics)
        (out / f'{key}.json').write_text(json.dumps(report, indent=2))
        reports[key] = report
        print('ABLATION_RESULT', key, json.dumps(metrics), flush=True)
(out / 'summary.json').write_text(json.dumps(reports, indent=2))
from summarize_focus_input_ablation import summarize
summarize(out)
env.close()
app.app.close()
