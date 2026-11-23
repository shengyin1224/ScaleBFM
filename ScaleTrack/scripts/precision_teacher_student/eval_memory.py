"""Run the established MuJoCo rollout with metrics kept entirely in memory."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
import numpy as np


def evaluate(checkpoint, reference, mode="4", hold_s=3.0, device="cuda:0", jitter_mm=0.0):
    track = Path(__file__).resolve().parents[2]
    network = track / "source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py"
    spec = importlib.util.spec_from_file_location("precision_network", network)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = types.ModuleType("my_rsl_rl")
    networks = types.ModuleType("my_rsl_rl.networks")
    networks.HumanoidTransformer = module.HumanoidTransformer
    networks.TaskEmbedder = module.TaskEmbedder
    package.networks = networks
    sys.modules["my_rsl_rl"] = package
    sys.modules["my_rsl_rl.networks"] = networks
    contract = json.loads((Path(__file__).with_name("deployment_contract.json")).read_text())

    class Reference(dict):
        @property
        def files(self):
            return list(self)

    with np.load(reference) as f:
        ref = Reference({k: f[k].copy() for k in f.files})
    original_length = len(ref["joint_pos"])
    extra = round(hold_s * 50)
    for key, value in list(ref.items()):
        if value.ndim and len(value) == original_length:
            tail = np.repeat(value[-1:], extra, axis=0)
            if key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                tail[:] = 0
            ref[key] = np.concatenate((value, tail))
    metadata = {"joint_names": contract["joint_names"],
                "action_offset": contract["default_dof_pos"],
                "action_scale": contract["action_scale"]}
    args = argparse.Namespace(
        checkpoint=Path(checkpoint), reference=Path(reference), tracking_mode=mode,
        device=device, isaac_result_npz=None, focus_cache=None, focus_motion_name=None,
        xml=track.parent.parent / "TWIST/assets/g1inspire/g1_29dof_inspire_hand.xml",
        no_video=True, hand_reference_ik=None, future_offset=5,
        root_drift_cm=0.0, reanchor_mode="on", metrics_out=Path("/dev/null"),
        video_out=Path("/dev/null"), trace_out=None, video_fps=30,
        video_width=1280, video_height=720)
    source = (Path(__file__).with_name("_legacy_mujoco.py")).read_text()
    source = source.replace('key.removeprefix("actor.")', 'key[len("actor."): ]')
    source = source.replace('key.removeprefix("actor_task_embedder.")', 'key[len("actor_task_embedder."): ]')
    source = source.replace("reference = np.load(args.reference)", "reference = _reference")
    source = source.replace("isaac_arrays = np.load(args.isaac_result_npz)", "isaac_arrays = _metadata")
    source = source.replace("args.metrics_out.parent.mkdir(parents=True, exist_ok=True)", "pass")
    source = source.replace("    with torch.inference_mode():\n        for sim_step", "    _q_trace = []\n    _a_trace = []\n    with torch.inference_mode():\n        for sim_step")
    source = source.replace("                previous_action = action", "                _q_trace.append(dof_pos.cpu().numpy().copy())\n                _a_trace.append(action[0].cpu().numpy().copy())\n                previous_action = action")
    source = source.replace("                task_input = torch.cat(",
        "                _bias = torch.as_tensor(_position_bias[frame], dtype=torch.float32, device=device).view(1,1,1,3).expand(1,6,14,3).flatten(2)\n"
        "                task_obs[..., :42] += _bias\n"
        "                task_obs[..., 42:84] += _bias\n"
        "                task_input = torch.cat(")
    source = source.replace('    args.metrics_out.write_text(json.dumps(report, indent=2) + "\\n")', "    _result.update(report)\n    _result['_q'] = np.asarray(_q_trace)\n    _result['_a'] = np.asarray(_a_trace)")
    source = source.replace("    print(json.dumps(report, indent=2))", "    pass")
    result = {}
    rng = np.random.RandomState(42)
    bias = np.zeros((len(ref["joint_pos"]),3))
    for i in range(1,len(bias)):
        bias[i] = .8*bias[i-1] + np.sqrt(1-.8**2)*jitter_mm/1000*rng.randn(3)
    namespace = {"__name__": "_precision_eval", "__file__": str(Path(__file__).with_name("_legacy_mujoco.py")),
                 "_reference": ref, "_metadata": metadata, "_result": result, "_position_bias": bias}
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    namespace["parse_args"] = lambda: args
    namespace["main"]()
    q, a = result.pop("_q"), result.pop("_a")
    # Compare full-motion smoothness, and last 2s of an appended static hold.
    arm = [i for i,n in enumerate(contract["joint_names"]) if any(t in n for t in ("shoulder", "elbow", "wrist"))]
    tail = q[-100:, arm]
    from scipy.signal import welch
    f, density = welch(tail, fs=50, axis=0, nperseg=min(100,len(tail)))
    band = (f >= 3) & (f <= 15)
    hf = float(np.sqrt(np.trapz(density[band], f[band], axis=0).mean()))
    output = {"checkpoint": str(checkpoint), "reference": str(reference), "mode": mode,
              "success_rate_frames": result["success_rate_frames"],
              "first_failure_frame": result["first_failure_frame"],
              "static_arm_hf_rms_rad": hf,
              "action_second_difference_rms": float(np.sqrt(np.mean(np.diff(a[150:], n=2, axis=0)**2))),
              "frames": len(q), "hold_s": hold_s, "localization_jitter_rms_mm_per_axis": jitter_mm}
    # Preserve established global and pelvis-local definitions in printed results.
    output["tracking"] = {k:v for k,v in result.items() if k in
                           ("active_link_pose_vs_reference", "active_link_pose_pelvis_frame_local_twist")}
    if not output["tracking"]:
        output["tracking"] = {k:v for k,v in result.items() if isinstance(v, dict) and
                               any(w in k for w in ("tracking", "metrics", "position"))}
    # Training validation separates arm commands from legs/waist, in physical units.
    scale = np.asarray(contract["action_scale"])
    groups = {"arms": arm, "legs_waist": [i for i in range(29) if i not in arm]}
    output["dynamic_by_group"] = {
        name: {
            "action_second_difference_rms": float(np.sqrt(np.mean(np.diff(a[150:, ids], n=2, axis=0)**2))),
            "target_angle_second_difference_rms_rad": float(np.sqrt(np.mean(
                np.diff(a[150:, ids] * scale[ids], n=2, axis=0)**2))),
            "joint_acceleration_control_rate_rms_rad_s2": float(np.sqrt(np.mean(
                (np.diff(q[150:, ids], n=2, axis=0) * 50**2)**2))),
        } for name, ids in groups.items()
    }
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--mode", default="4", choices=("4", "14"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--jitter-mm", type=float, default=0.0)
    args = p.parse_args()
    print("PRECISION_EVAL " + json.dumps(evaluate(args.checkpoint, args.reference, args.mode,
                                                 device=args.device, jitter_mm=args.jitter_mm)), flush=True)
