"""Manager-based Isaac Lab environment for bimanual hand GMT."""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from scaletrack.robots.inspire_hand import (
    INSPIRE_ACTION_SCALE,
    INSPIRE_HAND_LEFT_CFG,
    INSPIRE_HAND_RIGHT_CFG,
    LEFT_CONTACT_BODY_NAMES,
    LEFT_JOINT_NAMES,
    LEFT_TIP_BODY_NAMES,
    RIGHT_CONTACT_BODY_NAMES,
    RIGHT_JOINT_NAMES,
    RIGHT_TIP_BODY_NAMES,
)
from scaletrack.tasks.dexhand_tracking import mdp
from scaletrack.tasks.dexhand_tracking.asset_spawner import (
    DeterministicMultiAssetSpawnerCfg,
    DummyObjectSpawnerCfg,
    HeterogeneousRigidObject,
)
from scaletrack.tasks.dexhand_tracking.data import (
    deterministic_motion_assignment,
    load_manifest,
    select_manifest_shard,
)


def _dummy_object_cfg() -> DummyObjectSpawnerCfg:
    return DummyObjectSpawnerCfg(
        cube_cfg=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        )
    )


def _heterogeneous_object_cfg(entries, assignment, hand: str, objects_enabled: bool) -> RigidObjectCfg:
    assets = [_dummy_object_cfg()]
    unique: dict[tuple[str, float], int] = {}
    indices = []
    for motion_id in assignment:
        entry = entries[int(motion_id)]
        active = objects_enabled and (entry.has_rh_object if hand == "rh" else entry.has_lh_object)
        usd = entry.object_rh_usd if hand == "rh" else entry.object_lh_usd
        scale = entry.object_rh_scale if hand == "rh" else entry.object_lh_scale
        if not active or not usd:
            indices.append(0)
            continue
        key = (usd, scale)
        if key not in unique:
            unique[key] = len(assets)
            assets.append(
                sim_utils.UsdFileCfg(
                    usd_path=usd,
                    scale=(scale, scale, scale),
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_linear_velocity=100.0,
                        max_angular_velocity=100.0,
                        max_depenetration_velocity=2.0,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=2,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=200.0),
                )
            )
        indices.append(unique[key])
    cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Object{hand.upper()}",
        spawn=DeterministicMultiAssetSpawnerCfg(assets_cfg=assets, asset_indices=indices),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    cfg.class_type = HeterogeneousRigidObject
    return cfg


def _heterogeneous_scene_object_cfg(
    entries,
    assignment,
    slot: int,
    scene_assets_enabled: bool,
) -> RigidObjectCfg:
    """Build one fixed task-scene slot (FAVOR primitive or HuMoTo USD)."""

    assets = [_dummy_object_cfg()]
    unique: dict[tuple, int] = {}
    indices = []
    for motion_id in assignment:
        if not scene_assets_enabled:
            indices.append(0)
            continue
        scene_objects = entries[int(motion_id)].scene_objects
        if slot >= len(scene_objects):
            indices.append(0)
            continue
        value = scene_objects[slot]
        shape = value["shape"]
        if shape == "cuboid":
            size = tuple(float(component) for component in value["size"])
            color = tuple(float(component) for component in value["color"])
            key = (shape, size, color)
        elif shape == "usd":
            usd_path = str(value["usd_path"])
            kinematic = bool(value.get("kinematic", True))
            disable_gravity = bool(value.get("disable_gravity", kinematic))
            key = (shape, usd_path, kinematic, disable_gravity)
        else:
            raise ValueError(f"unsupported task scene object shape {shape!r}")
        if key not in unique:
            unique[key] = len(assets)
            if shape == "cuboid":
                assets.append(
                    DummyObjectSpawnerCfg(
                        cube_cfg=sim_utils.CuboidCfg(
                            size=size,
                            visible=True,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                kinematic_enabled=True,
                                disable_gravity=True,
                            ),
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                        )
                    )
                )
            else:
                assets.append(
                    sim_utils.UsdFileCfg(
                        usd_path=usd_path,
                        activate_contact_sensors=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=kinematic,
                            disable_gravity=disable_gravity,
                            max_linear_velocity=100.0,
                            max_angular_velocity=100.0,
                            max_depenetration_velocity=2.0,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=2,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(density=200.0),
                    )
                )
        indices.append(unique[key])
    cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/SceneObject{slot}",
        spawn=DeterministicMultiAssetSpawnerCfg(assets_cfg=assets, asset_indices=indices),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    cfg.class_type = HeterogeneousRigidObject
    return cfg


@configclass
class DexHandSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )
    right_hand: ArticulationCfg = MISSING
    left_hand: ArticulationCfg = MISSING
    object_rh: RigidObjectCfg = MISSING
    object_lh: RigidObjectCfg = MISSING
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.6, 0.03),
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.1,
                dynamic_friction=0.1,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.18)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )
    scene_object_0: RigidObjectCfg = MISSING
    scene_object_1: RigidObjectCfg = MISSING
    scene_object_2: RigidObjectCfg = MISSING
    scene_object_3: RigidObjectCfg = MISSING
    scene_object_4: RigidObjectCfg = MISSING
    scene_object_5: RigidObjectCfg = MISSING
    scene_object_6: RigidObjectCfg = MISSING
    scene_object_7: RigidObjectCfg = MISSING
    scene_object_8: RigidObjectCfg = MISSING
    scene_object_9: RigidObjectCfg = MISSING
    scene_object_10: RigidObjectCfg = MISSING
    scene_object_11: RigidObjectCfg = MISSING
    right_tip_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RightHand/(" + "|".join(RIGHT_CONTACT_BODY_NAMES) + ")",
        history_length=3,
        track_air_time=False,
    )
    left_tip_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeftHand/(" + "|".join(LEFT_CONTACT_BODY_NAMES) + ")",
        history_length=3,
        track_air_time=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    motion = mdp.DexHandMotionCommandCfg(
        manifest_file="",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        right_tip_body_names=RIGHT_TIP_BODY_NAMES,
        left_tip_body_names=LEFT_TIP_BODY_NAMES,
        right_joint_names=RIGHT_JOINT_NAMES,
        left_joint_names=LEFT_JOINT_NAMES,
    )


@configclass
class ActionsCfg:
    right_hand = mdp.JointPositionActionCfg(
        asset_name="right_hand",
        joint_names=RIGHT_JOINT_NAMES,
        preserve_order=True,
        scale=INSPIRE_ACTION_SCALE,
        use_default_offset=True,
    )
    left_hand = mdp.JointPositionActionCfg(
        asset_name="left_hand",
        joint_names=LEFT_JOINT_NAMES,
        preserve_order=True,
        scale=INSPIRE_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.actor_proprio,
            params={"command_name": "motion"},
            history_length=11,
            flatten_history_dim=False,
        )

        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class PolicyTaskCfg(ObsGroup):
        future_keypoints = ObsTerm(func=mdp.future_keypoint_targets, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        object_and_contact = ObsTerm(func=mdp.critic_privileged, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class BCTargetCfg(ObsGroup):
        action = ObsTerm(func=mdp.bc_target_action, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    policy_task: PolicyTaskCfg = PolicyTaskCfg()
    critic: CriticCfg = CriticCfg()
    bc_target: BCTargetCfg = BCTargetCfg()


@configclass
class RewardsCfg:
    fingertip_tracking = RewTerm(
        func=mdp.fingertip_tracking,
        weight=4.0,
        params={
            "command_name": "motion",
            "coefficient_start": -20.0,
            "coefficient_end": -150.0,
            "curriculum_steps": 250_000,
        },
    )
    joint_position_tracking = RewTerm(
        func=mdp.joint_position_tracking,
        weight=2.0,
        params={"command_name": "motion", "coefficient": -0.15},
    )
    joint_velocity_tracking = RewTerm(
        func=mdp.joint_velocity_tracking,
        weight=0.2,
        params={"command_name": "motion", "coefficient": -0.01},
    )
    alive = RewTerm(func=mdp.is_alive, weight=5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(func=mdp.joint_velocity_l2, weight=-1.0e-5)
    right_joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("right_hand", joint_names=RIGHT_JOINT_NAMES)},
    )
    left_joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("left_hand", joint_names=LEFT_JOINT_NAMES)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_finished = DoneTerm(
        func=mdp.motion_finished,
        time_out=True,
        params={"command_name": "motion", "disable_flag": False},
    )
    fingertip_tracking = DoneTerm(
        func=mdp.bad_fingertip_tracking,
        params={"command_name": "motion", "contact_relaxation": 1.5, "disable_flag": False},
    )
    wrist_velocity = DoneTerm(
        func=mdp.bad_wrist_velocity,
        params={"command_name": "motion", "disable_flag": False},
    )
    object_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={"command_name": "motion", "disable_flag": False},
    )


@configclass
class EventsCfg:
    object_properties = EventTerm(
        func=mdp.randomize_object_properties,
        mode="reset",
        params={"command_name": "motion", "scale_range": (0.8, 1.2)},
    )
    drive_references = EventTerm(
        func=mdp.drive_wrist_roots_and_soft_attach,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"command_name": "motion", "disable_flag": False},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class DexHandTrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: DexHandSceneCfg = DexHandSceneCfg(
        # 2048 is the measured safe single-RTX-4090 capacity for the full
        # heterogeneous scene. 4096 exhausts 24 GB during PhysX solver setup;
        # larger runs must pass --num_envs explicitly on higher-memory GPUs.
        num_envs=2048,
        env_spacing=1.5,
        replicate_physics=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    manifest_file: str = ""
    motion_split: str = "train"
    objects_enabled: bool = True
    scene_assets_enabled: bool = True
    assignment_seed: int = 20260728
    exact_motion_repeats: bool = False
    manifest_shard_count: int = 1
    manifest_shard_index: int = 0
    manifest_shard_seed: int = 20260728

    def configure_manifest(
        self,
        manifest_file: str,
        split: str | None = None,
        num_envs: int | None = None,
        exact_motion_repeats: bool | None = None,
        manifest_shard_count: int = 1,
        manifest_shard_index: int = 0,
        manifest_shard_seed: int = 20260728,
    ) -> None:
        if split is not None:
            self.motion_split = split
        if num_envs is not None:
            self.scene.num_envs = num_envs
        if exact_motion_repeats is not None:
            self.exact_motion_repeats = exact_motion_repeats
        self.manifest_shard_count = manifest_shard_count
        self.manifest_shard_index = manifest_shard_index
        self.manifest_shard_seed = manifest_shard_seed
        # Size buffers from measured bimanual demand.  The lower bounds cover
        # 250-env evaluation; the caps keep large training scenes bounded.
        # On a 24 GB RTX 4090, use the validated 2048-env default. A 4096-env
        # scene actually needs the larger values below and therefore requires
        # a higher-memory GPU.
        self.sim.physx.gpu_max_rigid_patch_count = max(
            2**18, min(2**22, self.scene.num_envs * 1024)
        )
        self.sim.physx.gpu_max_rigid_contact_count = max(
            2**20, min(2**24, self.scene.num_envs * 4096)
        )
        self.manifest_file = os.path.abspath(os.path.expanduser(manifest_file))
        _, entries = load_manifest(self.manifest_file, split=self.motion_split)
        entries = select_manifest_shard(
            entries,
            self.manifest_shard_count,
            self.manifest_shard_index,
            self.manifest_shard_seed,
        )
        assignment = deterministic_motion_assignment(
            entries,
            self.scene.num_envs,
            self.assignment_seed,
            exact_repeats=self.exact_motion_repeats,
        )
        self.scene.object_rh = _heterogeneous_object_cfg(
            entries, assignment, "rh", self.objects_enabled
        )
        self.scene.object_lh = _heterogeneous_object_cfg(
            entries, assignment, "lh", self.objects_enabled
        )
        for slot in range(12):
            setattr(
                self.scene,
                f"scene_object_{slot}",
                _heterogeneous_scene_object_cfg(
                    entries,
                    assignment,
                    slot,
                    self.scene_assets_enabled,
                ),
            )
        self.commands.motion.manifest_file = self.manifest_file
        self.commands.motion.split = self.motion_split
        self.commands.motion.assignment_seed = self.assignment_seed
        self.commands.motion.exact_motion_repeats = self.exact_motion_repeats
        self.commands.motion.manifest_shard_count = self.manifest_shard_count
        self.commands.motion.manifest_shard_index = self.manifest_shard_index
        self.commands.motion.manifest_shard_seed = self.manifest_shard_seed
        self.commands.motion.objects_enabled = self.objects_enabled
        self.commands.motion.scene_assets_enabled = self.scene_assets_enabled

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.viewer.eye = (1.2, 1.2, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.8)

        self.scene.right_hand = INSPIRE_HAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/RightHand")
        self.scene.left_hand = INSPIRE_HAND_LEFT_CFG.replace(prim_path="{ENV_REGEX_NS}/LeftHand")
        manifest = self.manifest_file or os.getenv("SCALETRACK_DEXHAND_MANIFEST", "")
        split = os.getenv("SCALETRACK_DEXHAND_SPLIT", self.motion_split)
        if manifest:
            self.configure_manifest(manifest, split=split)
