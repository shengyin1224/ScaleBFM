"""Opt-in precision experiments; no existing task or deployment config changes."""
import torch
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from scaletrack.tasks.tracking.config.g1_29dof.hat4_finetune_env_cfg import (
    G1HAT4PhaseObsOnlyFinetuneEnvCfg, HAT4FocusObservationCfg,
)
from scaletrack.tasks.tracking.tracking_env_cfg import BFMMaskObservationCfg, RewardsCfg
from scaletrack.tasks.tracking.mdp.time_warp import TimeWarpMotionCommand, adaptive_wrist_precision
from scaletrack.tasks.tracking.mdp.reference_playback import eased_progress


def neutral_phase(env, command_name="motion", token_count=6):
    return torch.zeros(env.num_envs, token_count, 2, device=env.device)


def wrist_precision(env, command_name="motion", cap=1.0):
    command = env.command_manager.get_term(command_name)
    progress = eased_progress(command.training_steps, command.cfg.precision_start_step,
                              command.cfg.precision_end_step)
    # cap is the combined maximum of position + orientation, not per-term.
    return cap * progress * adaptive_wrist_precision(
        env, command_name=command_name, focus_weight=1.0, w_pos=0.5, w_rot=0.5)


@configclass
class PrecisionObservations(HAT4FocusObservationCfg):
    teacher_policy = BFMMaskObservationCfg.PolicyCfg()
    teacher_policy_task = HAT4FocusObservationCfg.PolicyTaskCfg()


@configclass
class PrecisionRewards(RewardsCfg):
    wrist_precision = RewTerm(func=wrist_precision, weight=1.0,
                             params={"command_name": "motion", "cap": 1.0})


@configclass
class PrecisionEnvCfg(G1HAT4PhaseObsOnlyFinetuneEnvCfg):
    observations: PrecisionObservations = PrecisionObservations()
    rewards: PrecisionRewards = PrecisionRewards()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = TimeWarpMotionCommand
        self.commands.motion.playback_enabled = True
        self.commands.motion.playback_start_step = 2000 * 64
        self.commands.motion.playback_end_step = 8000 * 64
        self.commands.motion.precision_start_step = 2000 * 64
        self.commands.motion.precision_end_step = 8000 * 64
        self.commands.motion.playback_std_end = 0.3
        self.commands.motion.debug_vis = False
        self.commands.motion.mode_sampling_weights = {
            name: float(name == "WholeBody-14") for name in self.commands.motion.mode_candidates}
        for name in ("policy", "policy_task", "teacher_policy", "teacher_policy_task"):
            group = getattr(self.observations, name)
            group.enable_corruption = False
            for term in vars(group).values():
                if isinstance(term, ObsTerm):
                    term.noise = None
        for name in ("policy_task", "teacher_policy_task"):
            group = getattr(self.observations, name)
            for term in vars(group).values():
                if isinstance(term, ObsTerm) and "future_idx" in term.params:
                    term.params["future_idx"] = [0, 1, 2, 3, 4, 5]
            group.focus_phase.func = neutral_phase
        self.observations.critic_task.focus_phase.func = neutral_phase


def student_configuration(cfg):
    cfg.commands.motion.mode_sampling_weights = {
        name: float(name == "HAT-4") for name in cfg.commands.motion.mode_candidates}
    # Keep the deployed 3-step history, 6 future tokens, 269-D task and 29 actions.
    cfg.observations.policy = BFMMaskObservationCfg.PolicyCfg()
    cfg.commands.motion.playback_std_end = 0.15
