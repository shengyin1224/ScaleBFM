import gymnasium as gym

from . import agents, flat_env_cfg, hat4_finetune_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="G1-BFM-Transformer-Tracking",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1BFMTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BFMTransformerPPORunnerCfg",
    },
)


_HAT4_VARIANTS = {
    "G1-HAT4-F1-Finetune": (
        hat4_finetune_env_cfg.G1HAT4FinetuneEnvCfg,
        "G1HAT4FinetunePPORunnerCfg",
    ),
    "G1-HAT4-F2-Finetune": (
        hat4_finetune_env_cfg.G1HAT4FocusRewardFinetuneEnvCfg,
        "G1HAT4F2FinetunePPORunnerCfg",
    ),
    "G1-HAT4-F3-Finetune": (
        hat4_finetune_env_cfg.G1HAT4FocusFinetuneEnvCfg,
        "G1HAT4F3FinetunePPORunnerCfg",
    ),
    "G1-HAT4-F3W4-Finetune": (
        hat4_finetune_env_cfg.G1HAT4FocusW4FinetuneEnvCfg,
        "G1HAT4F3W4FinetunePPORunnerCfg",
    ),
    "G1-HAT4-F3H-Finetune": (
        hat4_finetune_env_cfg.G1HAT4FocusHeadFinetuneEnvCfg,
        "G1HAT4F3HeadFinetunePPORunnerCfg",
    ),
    "G1-HAT4-F2O-Finetune": (
        hat4_finetune_env_cfg.G1HAT4PhaseObsOnlyFinetuneEnvCfg,
        "G1HAT4F2ObsFinetunePPORunnerCfg",
    ),
    "G1-HAT4-F1M-Finetune": (
        hat4_finetune_env_cfg.G1HAT4OnlyFinetuneEnvCfg,
        "G1HAT4OnlyFinetunePPORunnerCfg",
    ),
    "G1-HAT4-F3S-Finetune": (
        hat4_finetune_env_cfg.G1HAT4FocusFinetuneEnvCfg,
        "G1HAT4F3SeedFinetunePPORunnerCfg",
    ),
}

for task_id, (env_cfg, runner_cfg_name) in _HAT4_VARIANTS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg_name}"
            ),
        },
    )

gym.register(
    id="G1-BFM-Mlp-Tracking",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1BFMTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BFMMlpPPORunnerCfg",
    },
)
