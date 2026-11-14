"""Evaluate native ScaleBFM VR-3, reporting genuine focus and non-focus stages."""
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True)
p.add_argument('--motion-file',required=True)
p.add_argument('--output',required=True)
p.add_argument('--condition',required=True)
p.add_argument('--label-definition',required=True)
p.add_argument('--num-envs',type=int,default=600)
p.add_argument('--max-steps',type=int,default=1000)
AppLauncher.add_app_launcher_args(p)
args,remaining=p.parse_known_args()
sys.argv=[sys.argv[0]]+remaining
app=AppLauncher(args)

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
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
from focus_bucket_statistics import METRICS,partition,summarize

out=Path(args.output)
out.parent.mkdir(parents=True,exist_ok=True)
manifest=yaml.safe_load(Path(args.motion_file).read_text())
digest=hashlib.sha256()
for name,path in manifest.items():
    with np.load(path) as data:
        if not np.asarray(data['focus_valid']).all():
            raise ValueError(f'Unknown focus validity in {name}')
        digest.update(name.encode())
        for field in ('focus_mask','focus_time_to_next_edge_s','focus_valid'):
            digest.update(data[field].tobytes())
cfg=G1HAT4PhaseObsOnlyFinetuneEnvCfg()
cfg.seed=42
cfg.sim.device=args.device
cfg.scene.num_envs=args.num_envs
cfg.commands.motion.motion_file=str(Path(args.motion_file).resolve())
cfg.commands.motion.test_motion_file=''
cfg.commands.motion.enable_reset_disturbance=False
cfg.commands.motion.debug_vis=False
links=cfg.commands.motion.mode_candidates['VR-3']
assert links==['pelvis','left_wrist_yaw_link','right_wrist_yaw_link']
cfg.commands.motion.mode_candidates={'VR-3':links}
cfg.commands.motion.mode_sampling_weights={'VR-3':1.}
agent=G1HAT4F3HeadFinetunePPORunnerCfg()
agent.device=args.device
# Deduplicate metric keys, particularly anchor_pos, to avoid double averaging.
agent.eval_metric_keys=['error_body_pos_g']+METRICS
agent.eval_focus_metric_keys=METRICS
agent.eval_max_steps=args.max_steps or None
agent.success_metric_dict={'error_body_pos_g':.5}
env=RslRlVecEnvWrapper(gym.make('G1-HAT4-F2O-Finetune',cfg=cfg))
runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out.with_suffix(''))+'_runtime',device=args.device)
Path(runner.log_dir).mkdir(exist_ok=True)
runner.load(args.checkpoint,load_optimizer=False,map_location=args.device)
actor=runner.alg.policy
get_obs=actor.get_actor_obs
actor.get_actor_obs=lambda obs,inference=False:get_obs(obs,inference=False)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
env.unwrapped.seed(42)
with torch.inference_mode():
    metrics=runner.evaluate_policy(test_set=False)
raw=runner.last_eval_per_motion_metrics
command=env.unwrapped.command_manager.get_term('motion')
lengths=command.time_totals[raw['motion_ids']].clone()
if args.max_steps:
    lengths=lengths.clamp_max(args.max_steps)
buckets=partition(raw,lengths)
success=raw['error_body_pos_g_max']<=.5
names=[command.motion_names[int(i)] for i in raw['motion_ids']]
payload=dict(raw=raw,partitioned=buckets,success=success,motion_names=names,lengths=lengths)
torch.save(payload,out.with_suffix('.per_motion.pt'))
report=dict(condition=args.condition,checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            mode='VR-3',active_links=links,motions=len(manifest),seed=42,
            max_steps=args.max_steps,label_definition=args.label_definition,label_sha256=digest.hexdigest(),
            success_count=int(success.sum()),success_rate=float(success.float().mean()),
            phase_input='native motion-file labels; official factory projection gains only zero phase columns',
            success_buckets=summarize(buckets,success),
            all_motion_buckets=summarize(buckets,torch.ones_like(success)),
            original_evaluator_metrics=metrics)
out.write_text(json.dumps(report,indent=2))
print('VR3_RESULT',json.dumps(report),flush=True)
env.close()
app.app.close()
