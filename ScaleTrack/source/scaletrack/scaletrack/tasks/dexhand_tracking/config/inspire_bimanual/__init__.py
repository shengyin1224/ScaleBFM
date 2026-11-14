"""Gym registrations for Inspire bimanual tracking."""

import gymnasium as gym

from . import agents
from .ik_replay_env_cfg import (
    DexHandGMTHeldoutEnvCfg,
    DexHandIKReplayEnvCfg,
    DexHandNoObjectEnvCfg,
)
from ...dexhand_env_cfg import DexHandTrackingEnvCfg


_AGENT_MODULE = f"{agents.__name__}.rsl_rl_ppo_cfg"

gym.register(
    id="ScaleTrack-DexHand-GMT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DexHandTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:DexHandPPORunnerCfg",
    },
)
gym.register(
    id="ScaleTrack-DexHand-RLScratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DexHandTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:DexHandScratchPPORunnerCfg",
    },
)
gym.register(
    id="ScaleTrack-DexHand-NoObject-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DexHandNoObjectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:DexHandNoObjectPPORunnerCfg",
    },
)
gym.register(
    id="ScaleTrack-DexHand-GMTHeldout-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DexHandGMTHeldoutEnvCfg,
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:DexHandPPORunnerCfg",
    },
)
gym.register(
    id="ScaleTrack-DexHand-IKReplay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DexHandIKReplayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:DexHandPPORunnerCfg",
    },
)

