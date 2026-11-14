#!/usr/bin/env python3
"""Launch the three requested HAT4 focus comparisons from the official checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
TRACK = REPO / "ScaleTrack"
PYTHON = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
CHECKPOINT = TRACK / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt"
OFFICIAL_LOCAL_SHA256 = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
GROUPS = (
    ("v3_focus_only", "hat4_unifolm_focus_v3", False),
    ("v3_new_settings", "hat4_unifolm_focus_v3", True),
    ("v2_new_settings", "hat4_unifolm_focus_v2", True),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-list", type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--run-prefix", default="official_threeway_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--startup-stagger-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if len(set(args.gpu_list)) != 3:
        raise ValueError("Use three distinct physical GPUs")
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != OFFICIAL_LOCAL_SHA256:
        raise ValueError("Official starting checkpoint changed; inspect its provenance before training")
    log_dir = REPO / (args.run_prefix + "_launch")
    jobs = []
    for (label, corpus, settings), gpu in zip(GROUPS, args.gpu_list):
        train = TRACK / "data" / corpus / "manifests/unifolm_hat4_train.yaml"
        heldout = TRACK / "data" / corpus / "manifests/unifolm_hat4_heldout.yaml"
        for path in (PYTHON, CHECKPOINT, train, heldout):
            if not path.is_file():
                raise FileNotFoundError(path)
        run_name = args.run_prefix + "_" + label
        if (TRACK / "logs/rsl_rl/g1_hat4_focus_finetune" / run_name).exists():
            raise FileExistsError("Training run already exists: " + run_name)
        command = [
            str(PYTHON), str(TRACK / "scripts/pretrain/rsl_rl/train.py"),
            "--task", "G1-HAT4-F3H-Finetune",
            "--motion_file", str(train), "--test_motion_file", str(heldout),
            "--pretrained_checkpoint", str(CHECKPOINT),
            "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
            "--run_name", run_name, "--seed", "42", "--device", "cuda:0", "--headless",
        ]
        if not settings:
            command.append("--disable-new-settings")
        jobs.append({
            "group": label, "physical_gpu": gpu, "run_name": run_name,
            "checkpoint_sha256": digest, "command": command,
            "settings": {"playback": settings, "wrist_precision": settings,
                         "focus_weight": 2.0, "mode_replay": "50% HAT4 + 50% factory modes",
                         "schedule_end_control_step": 640000 if settings else None},
            "log": str(log_dir / (label + ".log")),
        })
    print(json.dumps(jobs, indent=2), flush=True)
    if not args.execute:
        return
    log_dir.mkdir(exist_ok=False)
    launched = []
    for index, job in enumerate(jobs):
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(job["physical_gpu"]),
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "PYTHONUNBUFFERED": "1",
            "LD_LIBRARY_PATH": str(REPO / "libshim") + ":" + environment.get("LD_LIBRARY_PATH", ""),
        })
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "4"
        with open(job["log"], "w") as log:
            process = subprocess.Popen(job["command"], cwd=TRACK, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        job["pid"] = process.pid
        launched.append(job)
        (log_dir / "launch_record.json").write_text(json.dumps(launched, indent=2) + "\n")
        print(f"STARTED {job['group']} GPU={job['physical_gpu']} PID={process.pid}", flush=True)
        if index + 1 < len(jobs):
            time.sleep(args.startup_stagger_seconds)
    print("LAUNCH_RECORD " + str(log_dir / "launch_record.json"), flush=True)


if __name__ == "__main__":
    main()
