"""HAT-4 and FOCUS finetuning variants for the G1 ScaleBFM tracker."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import scaletrack.tasks.tracking.mdp as mdp
from scaletrack.tasks.tracking.tracking_env_cfg import BFMMaskObservationCfg

from .flat_env_cfg import G1BFMTrackingEnvCfg


HAT4_NAME = "HAT-4"
WRIST_LINKS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


@configclass
class HAT4FocusObservationCfg(BFMMaskObservationCfg):
    """Add two phase features without changing proprioception or action history."""

    @configclass
    class PolicyTaskCfg(BFMMaskObservationCfg.PolicyTaskCfg):
        focus_phase = ObsTerm(
            func=mdp.focus_phase,
            params={"command_name": "motion", "token_count": 6},
        )

    @configclass
    class PrivilegedTaskCfg(BFMMaskObservationCfg.PrivilegedTaskCfg):
        focus_phase = ObsTerm(
            func=mdp.focus_phase,
            params={"command_name": "motion", "token_count": 7},
        )

    @configclass
    class ModeMappingCfg(BFMMaskObservationCfg.ModeMappingCfg):
        mode_mapping = ObsTerm(
            func=mdp.mode_mapping,
            params={
                "command_name": "motion",
                "feature_dims_per_link": [3, 3, 6, 6],
                "with_time": True,
                "extra_unmasked_dims": 2,
            },
        )

    policy_task: PolicyTaskCfg = PolicyTaskCfg()
    critic_task: PrivilegedTaskCfg = PrivilegedTaskCfg()
    mode_mapping: ModeMappingCfg = ModeMappingCfg()


def mixed_hat4_weights(mode_names) -> dict[str, float]:
    old_modes = [name for name in mode_names if name != HAT4_NAME]
    if len(old_modes) != 8:
        raise ValueError(f"Expected eight factory modes, got {old_modes}")
    return {
        **{name: 0.5 / len(old_modes) for name in old_modes},
        HAT4_NAME: 0.5,
    }


@configclass
class G1HAT4FinetuneEnvCfg(G1BFMTrackingEnvCfg):
    """F1: HAT-4 interface with mixed old-mode replay, no phase mechanism."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.mode_sampling_weights = mixed_hat4_weights(
            self.commands.motion.mode_candidates
        )


@configclass
class G1HAT4FocusRewardFinetuneEnvCfg(G1HAT4FinetuneEnvCfg):
    """F2: F1 plus FOCUS-weighted wrist position and rotation rewards."""

    focus_weight: float = 2.0
    phase_weighted_body_names: list[str] = WRIST_LINKS

    def __post_init__(self):
        super().__post_init__()
        for reward in (
            self.rewards.motion_body_pos,
            self.rewards.motion_body_rot,
        ):
            reward.params["focus_weight"] = float(self.focus_weight)
            reward.params["phase_weighted_body_names"] = list(
                self.phase_weighted_body_names
            )


@configclass
class G1HAT4FocusFinetuneEnvCfg(G1HAT4FocusRewardFinetuneEnvCfg):
    """F3: complete HAT-4 + phase reward + phase observation method."""

    observations: HAT4FocusObservationCfg = HAT4FocusObservationCfg()


@configclass
class G1HAT4FocusW4FinetuneEnvCfg(G1HAT4FocusFinetuneEnvCfg):
    """F3w4: complete method with a four-times FOCUS wrist weight."""

    focus_weight: float = 4.0


@configclass
class G1HAT4FocusHeadFinetuneEnvCfg(G1HAT4FocusFinetuneEnvCfg):
    """F3h: also up-weight torso, the rigid G1 head proxy, in FOCUS."""

    phase_weighted_body_names: list[str] = WRIST_LINKS + ["torso_link"]


@configclass
class G1HAT4PhaseObsOnlyFinetuneEnvCfg(G1HAT4FinetuneEnvCfg):
    """F2o: HAT-4 plus phase observation, without phase reward weighting."""

    observations: HAT4FocusObservationCfg = HAT4FocusObservationCfg()


@configclass
class G1HAT4OnlyFinetuneEnvCfg(G1HAT4FinetuneEnvCfg):
    """F1m: HAT-4-only mode sampling, the catastrophic-forgetting upper bound."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.mode_sampling_weights = {
            name: float(name == HAT4_NAME)
            for name in self.commands.motion.mode_candidates
        }
