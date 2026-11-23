"""Environment variants for same-physics IK replay and ablations."""

from isaaclab.utils import configclass

from scaletrack.robots.inspire_hand import LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES
from scaletrack.tasks.dexhand_tracking import mdp
from scaletrack.tasks.dexhand_tracking.dexhand_env_cfg import DexHandTrackingEnvCfg


@configclass
class IKReplayActionsCfg:
    right_hand = mdp.ReferenceJointPositionActionCfg(
        asset_name="right_hand",
        joint_names=RIGHT_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
        hand="rh",
    )
    left_hand = mdp.ReferenceJointPositionActionCfg(
        asset_name="left_hand",
        joint_names=LEFT_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
        hand="lh",
    )


@configclass
class DexHandIKReplayEnvCfg(DexHandTrackingEnvCfg):
    actions: IKReplayActionsCfg = IKReplayActionsCfg()
    motion_split: str = "heldout"


@configclass
class DexHandGMTHeldoutEnvCfg(DexHandTrackingEnvCfg):
    motion_split: str = "heldout"


@configclass
class DexHandNoObjectEnvCfg(DexHandTrackingEnvCfg):
    objects_enabled: bool = False
    scene_assets_enabled: bool = False
