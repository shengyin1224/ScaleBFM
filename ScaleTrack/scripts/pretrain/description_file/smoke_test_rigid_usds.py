#!/usr/bin/env python3
"""Run each converted object for one simulated second and blacklist instability."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("usd_root", type=str)
parser.add_argument("--report", type=str, required=True)
parser.add_argument("--blacklist", type=str, required=True)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--max-assets", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument(
    "--fast-exit",
    action="store_true",
    help="Exit the isolated worker after reports are flushed; avoids a known multi-ICD Kit shutdown stall",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid  # noqa: E402


def smoke_batch(
    sim: sim_utils.SimulationContext,
    usd_paths: list[Path],
    *,
    cleanup: bool = True,
) -> dict[str, dict]:
    if sim.is_playing():
        sim.stop()
    root_path = "/World/TestObjects"
    if is_prim_path_valid(root_path):
        delete_prim(root_path)

    objects = []
    origins = []
    spacing = 10.0
    try:
        for index, usd_path in enumerate(usd_paths):
            x = spacing * (index % 8)
            y = spacing * (index // 8)
            cfg = RigidObjectCfg(
                prim_path=f"{root_path}/Object_{index:03d}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(usd_path),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=2.0,
                    ),
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.5)),
            )
            objects.append(RigidObject(cfg))
            origins.append(torch.tensor((x, y, 0.0), device=args.device))

        # A hard reset performs two renderer warm-up frames. On this headless
        # multi-ICD machine those frames can stall for minutes even though the
        # physics tensors are already initialized by play().
        print(f"[SETUP] play assets={len(objects)}", flush=True)
        sim.play()
        print(f"[SETUP] initialize_physics assets={len(objects)}", flush=True)
        sim.initialize_physics()
        print(f"[SETUP] soft_reset assets={len(objects)}", flush=True)
        sim.reset(soft=True)
        if not sim.is_playing():
            sim.play()
        print(f"[PHYSICS] playing={sim.is_playing()} assets={len(objects)}", flush=True)
        physics_steps = round(1.0 / sim.get_physics_dt())
        for step in range(physics_steps):
            sim.step(render=False)
            if (step + 1) % 50 == 0:
                print(f"[PHYSICS] {step + 1}/{physics_steps} assets={len(objects)}", flush=True)

        results = {}
        print(f"[READBACK] begin assets={len(objects)}", flush=True)
        for usd_path, obj, origin in zip(usd_paths, objects, origins):
            obj.update(sim.get_physics_dt())
            state = obj.data.root_state_w[0]
            finite = bool(torch.isfinite(state).all())
            speed = float(torch.linalg.vector_norm(state[7:10]))
            local_position_norm = float(torch.linalg.vector_norm(state[:3] - origin))
            stable = finite and speed < 100.0 and local_position_norm < 10.0 and math.isfinite(speed)
            results[str(usd_path)] = {
                "stable": stable,
                "final_state": state.detach().cpu().tolist(),
                "speed": speed,
                "local_position_norm": local_position_norm,
            }
        print(f"[READBACK] done assets={len(objects)}", flush=True)
        return results
    finally:
        if cleanup:
            print(f"[CLEANUP] begin assets={len(objects)}", flush=True)
            objects.clear()
            print("[CLEANUP] objects_cleared", flush=True)
            if sim.is_playing():
                sim.stop()
            print("[CLEANUP] simulation_stopped", flush=True)
            if is_prim_path_valid(root_path):
                delete_prim(root_path)
            print("[CLEANUP] done", flush=True)


def smoke_with_isolation(
    sim: sim_utils.SimulationContext,
    usd_paths: list[Path],
    *,
    cleanup: bool = True,
) -> dict[str, dict]:
    """Split failed batches so one malformed USD does not hide good neighbors."""

    try:
        return smoke_batch(sim, usd_paths, cleanup=cleanup)
    except Exception as exc:
        if len(usd_paths) == 1:
            return {
                str(usd_paths[0]): {
                    "stable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            }
        middle = len(usd_paths) // 2
        return {
            **smoke_with_isolation(sim, usd_paths[:middle]),
            **smoke_with_isolation(sim, usd_paths[middle:], cleanup=cleanup),
        }


def main() -> None:
    # UrdfConverter writes four implementation sublayers below
    # ``configuration/`` for every requested output USD. Only the requested
    # top-level USD is a spawnable asset and belongs in stability coverage.
    paths = sorted(
        path
        for path in Path(args.usd_root).expanduser().resolve().rglob("*.usd")
        if "configuration" not in path.parts
    )
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    paths = paths[args.start_index :]
    if args.max_assets is not None:
        paths = paths[: args.max_assets]
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    print(f"[MAIN] assets={len(paths)}", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device)
    # IsaacLab's defaults reserve hundreds of MB for millions of contacts,
    # which is unnecessary for this one-object stability check.
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**16
    sim_cfg.physx.gpu_max_rigid_contact_count = 2**17
    sim = sim_utils.SimulationContext(sim_cfg)
    print("[MAIN] simulation_context_ready", flush=True)
    sim_utils.spawn_ground_plane("/World/Ground", sim_utils.GroundPlaneCfg())
    print("[MAIN] ground_ready", flush=True)
    results = {}
    for start in range(0, len(paths), args.batch_size):
        batch = paths[start : start + args.batch_size]
        is_last_batch = start + len(batch) == len(paths)
        results.update(smoke_with_isolation(sim, batch, cleanup=not is_last_batch))
        print(f"[SMOKE] {min(start + len(batch), len(paths))}/{len(paths)}", flush=True)
    blacklist = [path for path, result in results.items() if not result["stable"]]
    Path(args.report).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    Path(args.blacklist).write_text(json.dumps({"usd_paths": blacklist}, indent=2) + "\n", encoding="utf-8")
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if blacklist else 0)
    if blacklist:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
