"""Static-bound dexhand motion command and reference cache."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv

from scaletrack.tasks.dexhand_tracking.data import (
    FUTURE_STEPS,
    MotionEntry,
    deterministic_motion_assignment,
    load_manifest,
    select_manifest_shard,
)


class DexHandMotionCommand(CommandTerm):
    """HDF5-cache reference with immutable env-to-motion/object binding."""

    cfg: "DexHandMotionCommandCfg"

    def __init__(self, cfg: "DexHandMotionCommandCfg", env):
        super().__init__(cfg, env)
        if not cfg.manifest_file:
            raise ValueError("DexHandMotionCommandCfg.manifest_file must be set before creating the environment")
        _, manifest_entries = load_manifest(cfg.manifest_file, split=cfg.split)
        manifest_entries = select_manifest_shard(
            manifest_entries,
            cfg.manifest_shard_count,
            cfg.manifest_shard_index,
            cfg.manifest_shard_seed,
        )
        original_assignment = deterministic_motion_assignment(
            manifest_entries,
            self.num_envs,
            cfg.assignment_seed,
            exact_repeats=cfg.exact_motion_repeats,
        )
        # Small smoke/debug environments may use only a tiny weighted subset of
        # a large manifest. Load only caches actually assigned to an env while
        # preserving the exact original env->entry binding used by the spawner.
        selected_ids = np.unique(original_assignment)
        compact_id = {
            int(original_id): compact
            for compact, original_id in enumerate(selected_ids)
        }
        entries = [manifest_entries[int(index)] for index in selected_ids]
        assignment = np.asarray(
            [compact_id[int(index)] for index in original_assignment],
            dtype=np.int64,
        )
        self.entries: list[MotionEntry] = entries
        self.motion_ids = torch.as_tensor(assignment, dtype=torch.long)
        self.num_motion = len(entries)
        self.has_test_set = False
        self.is_evaluating = cfg.split != "train"

        self.right_hand: Articulation = env.scene[cfg.right_hand_asset_name]
        self.left_hand: Articulation = env.scene[cfg.left_hand_asset_name]
        self.object_rh: RigidObject = env.scene[cfg.right_object_asset_name]
        self.object_lh: RigidObject = env.scene[cfg.left_object_asset_name]
        self.table: RigidObject = env.scene[cfg.table_asset_name]
        self.scene_objects: list[RigidObject] = [
            env.scene[name] for name in cfg.scene_object_asset_names
        ]
        self.right_tip_ids = torch.as_tensor(
            self.right_hand.find_bodies(cfg.right_tip_body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.left_tip_ids = torch.as_tensor(
            self.left_hand.find_bodies(cfg.left_tip_body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.right_joint_ids = torch.as_tensor(
            self.right_hand.find_joints(cfg.right_joint_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.left_joint_ids = torch.as_tensor(
            self.left_hand.find_joints(cfg.left_joint_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        fields = (
            "q_rh",
            "q_lh",
            "dq_rh",
            "dq_lh",
            "state_rh",
            "state_lh",
            "fingertip_pos_rh",
            "fingertip_pos_lh",
            "state_manip_obj_rh",
            "state_manip_obj_lh",
            "tip_force_rh",
            "tip_force_lh",
            "bc_action",
        )
        chunks: dict[str, list[torch.Tensor]] = {field: [] for field in fields}
        lengths = []
        for entry in entries:
            with np.load(entry.cache_path, allow_pickle=False) as data:
                if int(data["fps"]) != 50:
                    raise ValueError(f"{entry.name}: expected 50 FPS")
                length = len(data["q_rh"])
                if length < 2:
                    raise ValueError(f"{entry.name}: cache is too short")
                lengths.append(length)
                for field in fields:
                    if field not in data:
                        raise ValueError(f"{entry.name}: cache is missing {field!r}")
                    chunks[field].append(torch.from_numpy(np.asarray(data[field], dtype=np.float32)))
        self.time_totals = torch.as_tensor(lengths, dtype=torch.long)
        self.time_offsets = torch.zeros(len(lengths), dtype=torch.long)
        if len(lengths) > 1:
            self.time_offsets[1:] = torch.cumsum(self.time_totals[:-1], dim=0)
        for field, values in chunks.items():
            setattr(self, f"cat_{field}", torch.cat(values, dim=0))

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.attach_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.contact_established = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_active_rh = torch.as_tensor(
            [cfg.objects_enabled and entries[index].has_rh_object for index in assignment],
            dtype=torch.bool,
            device=self.device,
        )
        self.object_active_lh = torch.as_tensor(
            [cfg.objects_enabled and entries[index].has_lh_object for index in assignment],
            dtype=torch.bool,
            device=self.device,
        )
        self.hand_active_rh = torch.as_tensor(
            [entries[index].uses_rh for index in assignment], dtype=torch.bool, device=self.device
        )
        self.hand_active_lh = torch.as_tensor(
            [entries[index].uses_lh for index in assignment], dtype=torch.bool, device=self.device
        )
        def resolved_masses(hand: str, obj: RigidObject, active: torch.Tensor) -> torch.Tensor:
            requested = torch.as_tensor(
                [
                    getattr(entries[index], f"object_{hand}_mass")
                    if getattr(entries[index], f"object_{hand}_mass") is not None
                    else float("nan")
                    for index in assignment
                ],
                dtype=torch.float32,
                device=self.device,
            )
            imported = obj.root_physx_view.get_masses().squeeze(-1).to(self.device)
            # DexHandManipBiH caps an imported object's computed mass at 0.5 kg
            # unless the OakInk override table specifies an exact mass.
            fallback = torch.clamp(imported, max=cfg.fallback_object_mass_cap)
            resolved = torch.where(torch.isfinite(requested), requested, fallback)
            return torch.where(active, resolved, torch.full_like(resolved, 0.001))

        self.object_mass_rh = resolved_masses("rh", self.object_rh, self.object_active_rh)
        self.object_mass_lh = resolved_masses("lh", self.object_lh, self.object_active_lh)
        self.object_friction_rh = torch.as_tensor(
            [entries[index].object_rh_friction for index in assignment], dtype=torch.float32, device=self.device
        )
        self.object_friction_lh = torch.as_tensor(
            [entries[index].object_lh_friction for index in assignment], dtype=torch.float32, device=self.device
        )
        self.table_heights = torch.as_tensor(
            [
                entries[index].table_height
                if cfg.scene_assets_enabled and entries[index].table_height is not None
                else -10.0
                for index in assignment
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.table_center_x = torch.as_tensor(
            [entries[index].table_center_x for index in assignment],
            dtype=torch.float32,
            device=self.device,
        )
        self.has_table = self.table_heights > -5.0
        self.scene_object_active: list[torch.Tensor] = []
        self.scene_object_states: list[torch.Tensor] = []
        parked = torch.tensor(
            [0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        for slot in range(len(self.scene_objects)):
            active = torch.as_tensor(
                [
                    cfg.scene_assets_enabled
                    and slot < len(entries[index].scene_objects)
                    for index in assignment
                ],
                dtype=torch.bool,
                device=self.device,
            )
            states = torch.stack(
                [
                    torch.as_tensor(
                        entries[index].scene_objects[slot]["state"]
                        if slot < len(entries[index].scene_objects)
                        else parked,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for index in assignment
                ]
            )
            self.scene_object_active.append(active)
            self.scene_object_states.append(states)

        for name in (
            "fingertip_error_rh",
            "fingertip_error_lh",
            "joint_error_rh",
            "joint_error_lh",
            "object_drop",
            "object_contact",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _global_index(self, frame_offset: int | torch.Tensor = 0) -> torch.Tensor:
        motion_ids = self.motion_ids
        step = self.time_steps
        offset = torch.as_tensor(frame_offset, dtype=torch.long)
        if offset.ndim == 0:
            selected = torch.minimum(step + offset, self.time_totals[motion_ids] - 1)
            return self.time_offsets[motion_ids] + selected
        selected = torch.minimum(
            step[:, None] + offset[None, :],
            self.time_totals[motion_ids, None] - 1,
        )
        return self.time_offsets[motion_ids, None] + selected

    def _get(self, field: str, frame_offset: int | torch.Tensor = 0) -> torch.Tensor:
        index = self._global_index(frame_offset)
        return getattr(self, f"cat_{field}")[index].to(self.device)

    def _world_state(self, field: str, frame_offset: int = 0) -> torch.Tensor:
        state = self._get(field, frame_offset).clone()
        state[..., :3] += self._env.scene.env_origins
        return state

    @property
    def command(self) -> torch.Tensor:
        return self.reference_action

    @property
    def reference_action(self) -> torch.Tensor:
        return self._get("bc_action")

    @property
    def q_rh(self) -> torch.Tensor:
        return self._get("q_rh")

    @property
    def q_lh(self) -> torch.Tensor:
        return self._get("q_lh")

    @property
    def dq_rh(self) -> torch.Tensor:
        return self._get("dq_rh")

    @property
    def dq_lh(self) -> torch.Tensor:
        return self._get("dq_lh")

    @property
    def wrist_state_rh(self) -> torch.Tensor:
        return self._world_state("state_rh")

    @property
    def wrist_state_lh(self) -> torch.Tensor:
        return self._world_state("state_lh")

    @property
    def object_state_rh(self) -> torch.Tensor:
        return self._world_state("state_manip_obj_rh")

    @property
    def object_state_lh(self) -> torch.Tensor:
        return self._world_state("state_manip_obj_lh")

    @property
    def fingertip_pos_rh(self) -> torch.Tensor:
        value = self._get("fingertip_pos_rh")
        return value + self._env.scene.env_origins[:, None, :]

    @property
    def fingertip_pos_lh(self) -> torch.Tensor:
        value = self._get("fingertip_pos_lh")
        return value + self._env.scene.env_origins[:, None, :]

    def future_wrist_states(self, hand: str, future_steps: Sequence[int] = FUTURE_STEPS) -> torch.Tensor:
        value = self._get(f"state_{hand}", torch.as_tensor(future_steps, dtype=torch.long)).clone()
        value[..., :3] += self._env.scene.env_origins[:, None, :]
        return value

    def future_fingertips(self, hand: str, future_steps: Sequence[int] = FUTURE_STEPS) -> torch.Tensor:
        value = self._get(f"fingertip_pos_{hand}", torch.as_tensor(future_steps, dtype=torch.long))
        return value + self._env.scene.env_origins[:, None, None, :]

    def _write_objects(self, env_ids: torch.Tensor, reference: bool = True) -> None:
        for hand, obj, active in (
            ("rh", self.object_rh, self.object_active_rh),
            ("lh", self.object_lh, self.object_active_lh),
        ):
            state = self._world_state(f"state_manip_obj_{hand}") if reference else obj.data.root_state_w.clone()
            inactive = ~active[env_ids]
            if inactive.any():
                state[env_ids[inactive], :3] = self._env.scene.env_origins[env_ids[inactive]]
                state[env_ids[inactive], 2] -= 10.0
                state[env_ids[inactive], 3:7] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device
                )
                state[env_ids[inactive], 7:] = 0.0
            obj.write_root_state_to_sim(state[env_ids], env_ids=env_ids)

    def _write_scene_objects(self, env_ids: torch.Tensor) -> None:
        """Place fixed FAVOR/HuMoTo scene assets; inactive slots stay parked."""

        for obj, active, local_state in zip(
            self.scene_objects, self.scene_object_active, self.scene_object_states
        ):
            state = local_state[env_ids].clone()
            state[:, :3] += self._env.scene.env_origins[env_ids]
            inactive = ~active[env_ids]
            if inactive.any():
                state[inactive, :3] = self._env.scene.env_origins[env_ids[inactive]]
                state[inactive, 2] -= 10.0
            obj.write_root_state_to_sim(state, env_ids=env_ids)

    def _write_wrist_roots(self, env_ids: torch.Tensor) -> None:
        self.right_hand.write_root_state_to_sim(self.wrist_state_rh[env_ids], env_ids=env_ids)
        self.left_hand.write_root_state_to_sim(self.wrist_state_lh[env_ids], env_ids=env_ids)

    def joint_pos(self, hand: str) -> torch.Tensor:
        asset = self.right_hand if hand == "rh" else self.left_hand
        ids = self.right_joint_ids if hand == "rh" else self.left_joint_ids
        return asset.data.joint_pos[:, ids]

    def joint_vel(self, hand: str) -> torch.Tensor:
        asset = self.right_hand if hand == "rh" else self.left_hand
        ids = self.right_joint_ids if hand == "rh" else self.left_joint_ids
        return asset.data.joint_vel[:, ids]

    def default_joint_pos(self, hand: str) -> torch.Tensor:
        asset = self.right_hand if hand == "rh" else self.left_hand
        ids = self.right_joint_ids if hand == "rh" else self.left_joint_ids
        return asset.data.default_joint_pos[:, ids]

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids_cpu = env_ids.cpu()
        lengths = self.time_totals[self.motion_ids[env_ids_cpu]]
        if self.cfg.random_start and not self.is_evaluating:
            # Reserve the last frame for a complete transition target.
            self.time_steps[env_ids_cpu] = (
                torch.rand(len(env_ids_cpu)) * torch.clamp(lengths - 1, min=1)
            ).long()
        else:
            self.time_steps[env_ids_cpu] = 0

        self.right_hand.write_joint_state_to_sim(
            self.q_rh[env_ids],
            self.dq_rh[env_ids],
            joint_ids=self.right_joint_ids,
            env_ids=env_ids,
        )
        self.left_hand.write_joint_state_to_sim(
            self.q_lh[env_ids],
            self.dq_lh[env_ids],
            joint_ids=self.left_joint_ids,
            env_ids=env_ids,
        )
        self._write_wrist_roots(env_ids)
        self._write_objects(env_ids)
        self._write_scene_objects(env_ids)

        table_state = self.table.data.default_root_state[env_ids].clone()
        table_state[:, :3] += self._env.scene.env_origins[env_ids]
        table_state[:, 0] += self.table_center_x[env_ids]
        table_state[:, 2] = self.table_heights[env_ids] - self.cfg.table_thickness * 0.5
        self.table.write_root_state_to_sim(table_state, env_ids=env_ids)
        self.attach_steps_remaining[env_ids] = self.cfg.soft_attach_steps
        self.contact_established[env_ids] = False

    def _update_command(self) -> None:
        self.time_steps += 1

    def _update_metrics(self) -> None:
        actual_rh = self.right_hand.data.body_pos_w[:, self.right_tip_ids]
        actual_lh = self.left_hand.data.body_pos_w[:, self.left_tip_ids]
        error_rh = torch.linalg.vector_norm(
            actual_rh - self.fingertip_pos_rh, dim=-1
        ).mean(-1)
        error_lh = torch.linalg.vector_norm(
            actual_lh - self.fingertip_pos_lh, dim=-1
        ).mean(-1)
        self.metrics["fingertip_error_rh"][:] = torch.where(
            self.hand_active_rh, error_rh, torch.zeros_like(error_rh)
        )
        self.metrics["fingertip_error_lh"][:] = torch.where(
            self.hand_active_lh, error_lh, torch.zeros_like(error_lh)
        )
        self.metrics["joint_error_rh"][:] = torch.abs(self.joint_pos("rh") - self.q_rh).mean(-1)
        self.metrics["joint_error_lh"][:] = torch.abs(self.joint_pos("lh") - self.q_lh).mean(-1)
        self.metrics["object_contact"][:] = self.contact_established.float()

    def update_contact_established(self, threshold: float = 2.0) -> torch.Tensor:
        forces = torch.cat(
            (
                self._env.scene["right_tip_contact"].data.net_forces_w,
                self._env.scene["left_tip_contact"].data.net_forces_w,
            ),
            dim=1,
        )
        touching = torch.linalg.vector_norm(forces, dim=-1).amax(-1) > threshold
        has_object = self.object_active_rh | self.object_active_lh
        self.contact_established |= touching & has_object
        return self.contact_established

    def resample_motions(self) -> None:
        """Compatibility hook; static asset binding deliberately forbids resampling."""

    def switch_motion_set(self, train_to_test: bool = True) -> None:
        raise RuntimeError(
            "Dexhand assets are statically bound at scene construction; create a heldout environment "
            "instead of switching motion sets at runtime."
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass


@configclass
class DexHandMotionCommandCfg(CommandTermCfg):
    class_type: type = DexHandMotionCommand
    manifest_file: str = MISSING
    split: str = "train"
    assignment_seed: int = 20260728
    exact_motion_repeats: bool = False
    manifest_shard_count: int = 1
    manifest_shard_index: int = 0
    manifest_shard_seed: int = 20260728
    random_start: bool = True
    soft_attach_steps: int = 5
    objects_enabled: bool = True
    scene_assets_enabled: bool = True
    fallback_object_mass_cap: float = 0.5
    table_thickness: float = 0.03
    right_hand_asset_name: str = "right_hand"
    left_hand_asset_name: str = "left_hand"
    right_object_asset_name: str = "object_rh"
    left_object_asset_name: str = "object_lh"
    table_asset_name: str = "table"
    scene_object_asset_names: list[str] = [
        "scene_object_0",
        "scene_object_1",
        "scene_object_2",
        "scene_object_3",
        "scene_object_4",
        "scene_object_5",
        "scene_object_6",
        "scene_object_7",
        "scene_object_8",
        "scene_object_9",
        "scene_object_10",
        "scene_object_11",
    ]
    right_tip_body_names: list[str] = MISSING
    left_tip_body_names: list[str] = MISSING
    right_joint_names: list[str] = MISSING
    left_joint_names: list[str] = MISSING
