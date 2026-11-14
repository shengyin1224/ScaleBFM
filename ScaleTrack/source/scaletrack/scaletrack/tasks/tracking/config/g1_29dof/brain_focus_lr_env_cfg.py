"""Historical F3h recipe with exact brain transition windows and independent hands."""
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from scaletrack.tasks.tracking.mdp.brain_focus_lr import (
    BrainFocusLRMotionCommand, phase_observation_lr, body_position_lr, body_rotation_lr,
)
from .hat4_finetune_env_cfg import G1HAT4FinetuneEnvCfg, HAT4FocusObservationCfg
from .agents.rsl_rl_ppo_cfg import G1HAT4F3HeadFinetunePPORunnerCfg

@configclass
class BrainFocusLRObservations(HAT4FocusObservationCfg):
    @configclass
    class PolicyTaskCfg(HAT4FocusObservationCfg.PolicyTaskCfg):
        focus_phase = ObsTerm(func=phase_observation_lr, params={'command_name':'motion','token_count':6})
    @configclass
    class PrivilegedTaskCfg(HAT4FocusObservationCfg.PrivilegedTaskCfg):
        focus_phase = ObsTerm(func=phase_observation_lr, params={'command_name':'motion','token_count':7})
    @configclass
    class ModeMappingCfg(HAT4FocusObservationCfg.ModeMappingCfg):
        def __post_init__(self):
            self.mode_mapping.params['extra_unmasked_dims'] = 4
    policy_task: PolicyTaskCfg = PolicyTaskCfg()
    critic_task: PrivilegedTaskCfg = PrivilegedTaskCfg()
    mode_mapping: ModeMappingCfg = ModeMappingCfg()

@configclass
class BrainFocusLREnvCfg(G1HAT4FinetuneEnvCfg):
    observations: BrainFocusLRObservations = BrainFocusLRObservations()
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = BrainFocusLRMotionCommand
        self.commands.motion.playback_enabled = False
        self.observations.mode_mapping.mode_mapping.params['extra_unmasked_dims'] = 4
        self.rewards.motion_body_pos.func = body_position_lr
        self.rewards.motion_body_rot.func = body_rotation_lr
        for reward in (self.rewards.motion_body_pos, self.rewards.motion_body_rot):
            reward.params = {k:v for k,v in reward.params.items() if k in ('command_name','std')}
            reward.params['focus_weight'] = 2.

@configclass
class BrainFocusLRRunnerCfg(G1HAT4F3HeadFinetunePPORunnerCfg):
    experiment_name = 'g1_hat4_brain_focus_lr'
    run_name = 'brain_focus_lr_20261114'
