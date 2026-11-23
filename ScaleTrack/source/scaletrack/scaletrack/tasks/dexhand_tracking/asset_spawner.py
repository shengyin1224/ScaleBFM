"""Deterministic per-environment heterogeneous asset spawner.

Isaac Lab's MultiAssetSpawner supports deterministic round-robin selection, but
requires one prototype entry per environment to express an arbitrary weighted
assignment.  This variant keeps one prototype per unique asset and accepts an
explicit prototype index for every matched environment.
"""

from __future__ import annotations

import re
from dataclasses import MISSING

import carb
import omni.log
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import configclass


class HeterogeneousRigidObject(RigidObject):
    """Rigid object view whose one-level rigid-root name may differ per env.

    Isaac Lab's stock implementation derives the PhysX view suffix from env 0
    (for example ``/base``). Imported HuMoTo USDs instead retain asset-specific
    link names, so a mixed view silently covered only the environments sharing
    env 0's suffix. Every spawner here guarantees exactly one rigid body one
    level below the asset root; a wildcard view covers all such names.
    """

    def _initialize_impl(self):
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(
                f"Failed to find prim for expression: {self.cfg.prim_path!r}"
            )
        template_prim_path = template_prim.GetPath().pathString
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
        )
        if len(root_prims) != 1:
            raise RuntimeError(
                f"{template_prim_path}: expected exactly one rigid body, "
                f"found {len(root_prims)}"
            )
        relative = root_prims[0].GetPath().pathString[len(template_prim_path) :]
        if relative.count("/") != 1:
            raise RuntimeError(
                f"{template_prim_path}: heterogeneous rigid root must be one "
                f"level below the asset root, got {relative!r}"
            )
        articulation_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
        )
        if articulation_prims and articulation_prims[0].GetAttribute(
            "physxArticulation:articulationEnabled"
        ).Get():
            raise RuntimeError(
                f"{template_prim_path}: articulation root is enabled for a "
                "heterogeneous rigid object"
            )

        root_prim_path_expr = self.cfg.prim_path + "/*"
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(
            root_prim_path_expr.replace(".*", "*")
        )
        if self._root_physx_view._backend is None:
            raise RuntimeError(
                f"Failed to create heterogeneous rigid-body view: "
                f"{root_prim_path_expr}"
            )
        expected_count = len(sim_utils.find_matching_prim_paths(self.cfg.prim_path))
        if self._root_physx_view.count != expected_count:
            raise RuntimeError(
                f"{root_prim_path_expr}: PhysX view contains "
                f"{self._root_physx_view.count} bodies, expected {expected_count}"
            )
        omni.log.info(
            f"Heterogeneous rigid body initialized at {root_prim_path_expr} "
            f"with {self._root_physx_view.count} instances"
        )
        self._data = RigidObjectData(self.root_physx_view, self.device)
        self._create_buffers()
        self._process_cfg()


def spawn_dummy_object(
    prim_path: str,
    cfg: DummyObjectSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a disabled cube below a stable ``base`` rigid-body path.

    Imported object USDs expose their sole rigid body at ``<asset>/base``.
    PhysX tensor views require that path topology to be identical in every
    environment, so inactive slots cannot use a rigid cube at ``<asset>``.
    """

    prim_utils.create_prim(
        prim_path,
        "Xform",
        translation=translation,
        orientation=orientation,
    )
    cfg.cube_cfg.func(f"{prim_path}/base", cfg.cube_cfg)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class DummyObjectSpawnerCfg(RigidObjectSpawnerCfg):
    func = spawn_dummy_object
    cube_cfg: SpawnerCfg = MISSING


def spawn_deterministic_multi_asset(
    prim_path: str,
    cfg: DeterministicMultiAssetSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    stage = get_current_stage()
    root_path, asset_path = prim_path.rsplit("/", 1)
    is_regex = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None
    source_paths = sim_utils.find_matching_prim_paths(root_path) if is_regex and root_path else [root_path]
    if not source_paths:
        raise RuntimeError(f"no source prims match {root_path!r}")
    if len(cfg.asset_indices) != len(source_paths):
        raise ValueError(
            f"asset_indices has {len(cfg.asset_indices)} entries but {len(source_paths)} environments matched"
        )
    if not cfg.assets_cfg:
        raise ValueError("assets_cfg must not be empty")
    bad = [value for value in cfg.asset_indices if value < 0 or value >= len(cfg.assets_cfg)]
    if bad:
        raise ValueError(f"asset_indices contains out-of-range values: {bad[:10]}")

    template_path = stage_utils.get_next_free_path("/World/DexHandAssetTemplates")
    prim_utils.create_prim(template_path, "Scope")
    prototype_paths = []
    try:
        for index, asset_cfg in enumerate(cfg.assets_cfg):
            prototype_path = f"{template_path}/Asset_{index:04d}"
            asset_cfg.func(
                prototype_path,
                asset_cfg,
                translation=translation,
                orientation=orientation,
                clone_in_fabric=clone_in_fabric,
                replicate_physics=replicate_physics,
            )
            prototype_paths.append(prototype_path)

        prim_paths = [f"{source}/{asset_path}" for source in source_paths]
        with Sdf.ChangeBlock():
            for destination, prototype_index in zip(prim_paths, cfg.asset_indices):
                spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), destination)
                Sdf.CopySpec(
                    spec.layer,
                    Sdf.Path(prototype_paths[prototype_index]),
                    spec.layer,
                    Sdf.Path(destination),
                )
    finally:
        prim_utils.delete_prim(template_path)

    carb.settings.get_settings().set_bool("/isaaclab/spawn/multi_assets", True)
    return prim_utils.get_prim_at_path(prim_paths[0])


@configclass
class DeterministicMultiAssetSpawnerCfg(RigidObjectSpawnerCfg):
    func = spawn_deterministic_multi_asset
    assets_cfg: list[SpawnerCfg] = MISSING
    asset_indices: list[int] = MISSING
