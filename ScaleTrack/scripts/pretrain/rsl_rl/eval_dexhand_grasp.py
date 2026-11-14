#!/usr/bin/env python3
"""Same-environment held-out comparison: IK replay versus deterministic GMT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument(
    "--checkpoint",
    default="",
    help="GMT checkpoint (not required for the M1 --ik-only gate)",
)
parser.add_argument(
    "--ik-only",
    action="store_true",
    help="Run the pre-training M1 IK replay go/no-go without loading a policy",
)
parser.add_argument(
    "--bc-checkpoint",
    default="",
    help="Optional BC-only checkpoint; when set, include the auxiliary IK/BC/main free-air comparison.",
)
parser.add_argument("--seeds-per-motion", type=int, default=5)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--free-air", action="store_true")
parser.add_argument(
    "--bc-gate",
    action="store_true",
    help=(
        "Evaluate --checkpoint as the BC policy in free air and write the "
        "<checkpoint>.online_gate.json artifact required by PPO training"
    ),
)
parser.add_argument("--seed", type=int, default=20260728)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import csv  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import scaletrack.tasks  # noqa: F401,E402
from my_rsl_rl.modules import ActorCriticDexHand  # noqa: E402
from scaletrack.tasks.dexhand_tracking.data import load_manifest  # noqa: E402
from scaletrack.tasks.dexhand_tracking.mdp.terminations import object_dropped  # noqa: E402


def _orientation_error(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(torch.sum(reference * actual, dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _category(primitive: str) -> str:
    lower = primitive.lower()
    if any(word in lower for word in ("rotate", "twist", "turn", "orient", "shake")):
        return "orientation"
    if any(word in lower for word in ("press", "insert", "write", "pen", "button")):
        return "precision"
    return "grasp"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _disable_auto_resets(cfg) -> None:
    cfg.terminations.motion_finished.params["disable_flag"] = True
    cfg.terminations.fingertip_tracking.params["disable_flag"] = True
    cfg.terminations.wrist_velocity.params["disable_flag"] = True
    cfg.terminations.object_dropped.params["disable_flag"] = True
    cfg.episode_length_s = 120.0


def _make_policy(obs, checkpoint: str, device: str) -> ActorCriticDexHand:
    policy = ActorCriticDexHand(
        obs,
        {"policy": ["policy", "policy_task"], "critic": ["policy", "policy_task", "critic"]},
        24,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        motion_latent_dim=128,
        activation="silu",
        init_noise_std=float(np.exp(-2)),
        noise_std_type="log",
        actor_obs_normalization=True,
    ).to(device)
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(raw["model_state_dict"], strict=True)
    policy.eval()
    return policy


def run(executor: str, entries, num_envs: int) -> dict:
    task = "ScaleTrack-DexHand-IKReplay-v0" if executor == "ik" else "ScaleTrack-DexHand-GMTHeldout-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.objects_enabled = not args.free_air
    cfg.scene_assets_enabled = not args.free_air
    cfg.configure_manifest(
        args.manifest,
        split="heldout",
        num_envs=num_envs,
        exact_motion_repeats=True,
    )
    _disable_auto_resets(cfg)
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset(seed=args.seed)
    command = env.unwrapped.command_manager.get_term("motion")
    checkpoint = args.checkpoint if executor == "gmt" else args.bc_checkpoint
    policy = _make_policy(obs, checkpoint, env.unwrapped.device) if executor != "ik" else None
    max_motion_steps = int(command.time_totals.max())
    hold_steps = 50
    total_steps = max_motion_steps + hold_steps
    progress_interval = max(total_steps // 10, 1)
    print(
        f"[eval] executor={executor} envs={num_envs} "
        f"steps={total_steps} start",
        flush=True,
    )
    ever_dropped = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
    tip_errors = []
    tip_error_motion_ids = []
    joint_errors = []
    joint_error_motion_ids = []

    with torch.inference_mode():
        for step in range(total_steps):
            action = (
                torch.zeros(num_envs, 24, device=env.unwrapped.device)
                if executor == "ik"
                else policy.act_inference(obs)
            )
            obs, _, _, _, _ = env.step(action)
            # The termination term is disabled to permit the one-second hold,
            # so call the exact predicate manually and accumulate it.
            ever_dropped |= object_dropped(env.unwrapped, disable_flag=False)
            active_frame = step < command.time_totals[command.motion_ids].to(env.unwrapped.device)
            if active_frame.any():
                error_rh = torch.linalg.vector_norm(
                    command.right_hand.data.body_pos_w[:, command.right_tip_ids] - command.fingertip_pos_rh,
                    dim=-1,
                ).mean(-1)
                error_lh = torch.linalg.vector_norm(
                    command.left_hand.data.body_pos_w[:, command.left_tip_ids] - command.fingertip_pos_lh,
                    dim=-1,
                ).mean(-1)
                active_count = (
                    command.hand_active_rh.float() + command.hand_active_lh.float()
                ).clamp_min(1.0)
                error = (
                    error_rh * command.hand_active_rh.float()
                    + error_lh * command.hand_active_lh.float()
                ) / active_count
                tip_errors.append(error[active_frame].cpu())
                tip_error_motion_ids.append(command.motion_ids[active_frame.cpu()].clone())
                joint_error_rh = torch.abs(
                    command.joint_pos("rh") - command.q_rh
                ).mean(-1)
                joint_error_lh = torch.abs(
                    command.joint_pos("lh") - command.q_lh
                ).mean(-1)
                joint_error = (
                    joint_error_rh * command.hand_active_rh.float()
                    + joint_error_lh * command.hand_active_lh.float()
                ) / active_count
                joint_errors.append(joint_error[active_frame].cpu())
                joint_error_motion_ids.append(
                    command.motion_ids[active_frame.cpu()].clone()
                )
            if (step + 1) % progress_interval == 0 or step + 1 == total_steps:
                print(
                    f"[eval] executor={executor} progress="
                    f"{step + 1}/{total_steps}",
                    flush=True,
                )

    support = torch.where(command.has_table, command.table_heights, torch.zeros_like(command.table_heights))
    wrist_rh = command.right_hand.data.root_pos_w
    wrist_lh = command.left_hand.data.root_pos_w
    has_object = command.object_active_rh | command.object_active_lh
    success = has_object & ~ever_dropped
    orientation_error = torch.zeros(num_envs, device=env.unwrapped.device)
    position_error = torch.zeros_like(orientation_error)
    for obj, active, reference in (
        (command.object_rh, command.object_active_rh, command.object_state_rh),
        (command.object_lh, command.object_active_lh, command.object_state_lh),
    ):
        distance = torch.minimum(
            torch.linalg.vector_norm(obj.data.root_pos_w - wrist_rh, dim=-1),
            torch.linalg.vector_norm(obj.data.root_pos_w - wrist_lh, dim=-1),
        )
        held = distance < 0.25
        held &= obj.data.root_pos_w[:, 2] > support - 0.05
        success &= ~active | held
        orientation_error = torch.maximum(
            orientation_error,
            torch.where(
                active,
                _orientation_error(reference[:, 3:7], obj.data.root_quat_w),
                torch.zeros_like(orientation_error),
            ),
        )
        position_error = torch.maximum(
            position_error,
            torch.where(
                active,
                torch.linalg.vector_norm(reference[:, :3] - obj.data.root_pos_w, dim=-1),
                torch.zeros_like(position_error),
            ),
        )

    records = []
    for env_id in range(num_envs):
        motion_id = int(command.motion_ids[env_id])
        entry = entries[motion_id]
        category = _category(entry.primitive)
        passed = bool(success[env_id])
        if category == "orientation":
            passed &= float(orientation_error[env_id]) < np.deg2rad(20.0)
        elif category == "precision":
            passed &= float(position_error[env_id]) < 0.03
            passed &= float(orientation_error[env_id]) < np.deg2rad(20.0)
        records.append(
            {
                "executor": executor,
                "env_id": env_id,
                "motion_id": motion_id,
                "seed_replica": sum(
                    1
                    for prior_env_id in range(env_id)
                    if int(command.motion_ids[prior_env_id]) == motion_id
                ),
                "motion": entry.name,
                "category": category,
                "success": int(passed),
                "dropped": int(ever_dropped[env_id]),
                "orientation_error_deg": float(torch.rad2deg(orientation_error[env_id])),
                "position_error_m": float(position_error[env_id]),
            }
        )
    env.close()
    print(f"[eval] executor={executor} complete", flush=True)
    return {
        "records": records,
        "tip_errors": tip_errors,
        "tip_motion_ids": tip_error_motion_ids,
        "joint_errors": joint_errors,
        "joint_motion_ids": joint_error_motion_ids,
    }


def main() -> None:
    if not args.ik_only and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --ik-only is set")
    if args.bc_gate and (args.ik_only or not args.free_air):
        raise ValueError("--bc-gate requires --free-air and cannot be combined with --ik-only")
    _, entries = load_manifest(args.manifest, split="heldout")
    num_envs = len(entries) * args.seeds_per_motion
    if args.bc_gate:
        # The gate is deterministic free-air tracking. IK does not contribute
        # to its pass/fail value and would double the simulation cost.
        executors = ["gmt"]
    else:
        executors = ["ik"]
    if not args.ik_only and not args.bc_gate:
        if args.bc_checkpoint:
            executors.append("bc_only")
        executors.append("gmt")
    outputs = [run(executor, entries, num_envs) for executor in executors]
    records = [record for output in outputs for record in output["records"]]
    summary = {}
    for executor, output in zip(executors, outputs):
        selected = [row for row in records if row["executor"] == executor]
        by_category = defaultdict(list)
        for row in selected:
            by_category[row["category"]].append(row["success"])
        errors = (
            torch.cat(output["tip_errors"]).numpy()
            if output["tip_errors"]
            else np.asarray([], dtype=np.float32)
        )
        error_motion_ids = (
            torch.cat(output["tip_motion_ids"]).numpy()
            if output["tip_motion_ids"]
            else np.asarray([], dtype=np.int64)
        )
        joint_errors = (
            torch.cat(output["joint_errors"]).numpy()
            if output["joint_errors"]
            else np.asarray([], dtype=np.float32)
        )
        source_tracking = {}
        for source in sorted({entry.source for entry in entries}):
            source_motion_ids = np.asarray(
                [index for index, entry in enumerate(entries) if entry.source == source],
                dtype=np.int64,
            )
            selected_source = np.isin(error_motion_ids, source_motion_ids)
            values = errors[selected_source]
            source_tracking[source] = {
                "count": int(len(values)),
                "mean_mm": float(1000.0 * np.mean(values)) if len(values) else None,
                "p95_mm": float(1000.0 * np.percentile(values, 95)) if len(values) else None,
            }
        summary[executor] = {
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "count": len(selected),
            "by_category": {key: float(np.mean(value)) for key, value in sorted(by_category.items())},
            "wrist_frame_fingertip_mean_mm": float(1000.0 * np.mean(errors)) if len(errors) else None,
            "wrist_frame_fingertip_p95_mm": float(1000.0 * np.percentile(errors, 95)) if len(errors) else None,
            "joint_tracking_mean_rad": float(np.mean(joint_errors)) if len(joint_errors) else None,
            "joint_tracking_p95_rad": (
                float(np.percentile(joint_errors, 95)) if len(joint_errors) else None
            ),
            "tracking_by_source": source_tracking,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ik_success_rate = summary["ik"]["success_rate"] if "ik" in summary else None
    if ik_success_rate is None:
        m1_decision = None
    elif ik_success_rate < 0.5:
        m1_decision = "GO"
    elif ik_success_rate > 0.8:
        m1_decision = "NO_GO"
    else:
        m1_decision = "REVIEW"
    resolved_manifest = str(Path(args.manifest).resolve())
    resolved_checkpoint = str(Path(args.checkpoint).resolve()) if args.checkpoint else None
    gate = None
    if args.bc_gate:
        error_mm = summary["gmt"]["wrist_frame_fingertip_mean_mm"]
        gate = {
            "schema_version": 2,
            "checkpoint": resolved_checkpoint,
            "checkpoint_sha256": _sha256(resolved_checkpoint),
            "manifest": resolved_manifest,
            "manifest_sha256": _sha256(resolved_manifest),
            "split": "heldout",
            "metric": "wrist_frame_fingertip_mean_mm",
            "value_mm": error_mm,
            "threshold_mm": 15.0,
            "passed": error_mm is not None and error_mm < 15.0,
            "seed": args.seed,
            "seeds_per_motion": args.seeds_per_motion,
        }
        gate_path = Path(resolved_checkpoint + ".online_gate.json")
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = gate_path.with_name(gate_path.name + ".tmp")
        temporary.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        temporary.replace(gate_path)
        checkpoint_report_path = Path(resolved_checkpoint).with_suffix(".json")
        if checkpoint_report_path.is_file():
            checkpoint_report = json.loads(
                checkpoint_report_path.read_text(encoding="utf-8")
            )
            checkpoint_report["online_free_air_gate"] = gate
            report_temporary = checkpoint_report_path.with_name(
                checkpoint_report_path.name + ".tmp"
            )
            report_temporary.write_text(
                json.dumps(checkpoint_report, indent=2) + "\n",
                encoding="utf-8",
            )
            report_temporary.replace(checkpoint_report_path)
    report = {
        "manifest": resolved_manifest,
        "checkpoint": resolved_checkpoint,
        "bc_checkpoint": str(Path(args.bc_checkpoint).resolve()) if args.bc_checkpoint else None,
        "free_air": args.free_air,
        "seeds_per_motion": args.seeds_per_motion,
        "m1_go_no_go": (
            {
                "decision": m1_decision,
                "ik_success_rate": ik_success_rate,
                "go_threshold": "< 0.50",
                "no_go_threshold": "> 0.80",
            }
            if args.ik_only and not args.free_air
            else None
        ),
        "bc_online_gate": gate,
        "accepted_physics_approximations": [
            "URDF-to-USD collision decomposition uses the configured finite VHACD resolution",
            "rolling/torsion friction is approximated by PhysX static/dynamic friction",
        ],
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    try:
        import matplotlib.pyplot as plt

        names = [{"ik": "IK replay", "bc_only": "BC only", "gmt": "GMT"}[name] for name in executors]
        values = [summary[name]["success_rate"] for name in executors]
        figure, axis = plt.subplots(figsize=(4.5, 4.0))
        axis.bar(names, values, color=("#777777", "#3366cc"))
        axis.set_ylim(0, 1)
        axis.set_ylabel("Held-out success rate")
        for index, value in enumerate(values):
            axis.text(index, value + 0.02, f"{100 * value:.1f}%", ha="center")
        figure.tight_layout()
        figure.savefig(args.output_dir / "ik_vs_gmt_success.png", dpi=200)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
