#!/usr/bin/env python3
"""Print or launch a selected set of one-GPU HAT-4 finetuning ablations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRACK = REPO / "ScaleTrack"
PYTHON = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
TRAIN = TRACK / "scripts" / "pretrain" / "rsl_rl" / "train.py"
CHECKPOINT = (
    TRACK
    / "logs"
    / "rsl_rl"
    / "g1_bfm_tracking_exp"
    / "humanoid_transformer_m"
    / "model_22200.pt"
)
MANIFESTS = TRACK / "data" / "hat4_unifolm_focus_v3" / "manifests"
VARIANTS = (
    ("F1", "G1-HAT4-F1-Finetune", 0),
    ("F2", "G1-HAT4-F2-Finetune", 1),
    ("F3", "G1-HAT4-F3-Finetune", 2),
    ("F3w4", "G1-HAT4-F3W4-Finetune", 3),
    ("F3h", "G1-HAT4-F3H-Finetune", 4),
    ("F2o", "G1-HAT4-F2O-Finetune", 5),
    ("F1m", "G1-HAT4-F1M-Finetune", 6),
    ("F3s", "G1-HAT4-F3S-Finetune", 7),
    ("F3hw4", "G1-HAT4-F3HW4-Finetune", 1),
    ("F3hw4m", "G1-HAT4-F3HW4M-Finetune", 2),
)
VARIANT_BY_NAME = {item[0]: item for item in VARIANTS}
DEFAULT_VULKAN_ICD = Path("/etc/vulkan/icd.d/nvidia_icd.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--train-manifest", type=Path, default=MANIFESTS / "unifolm_hat4_train.yaml")
    parser.add_argument("--test-manifest", type=Path, default=MANIFESTS / "unifolm_hat4_heldout.yaml")
    parser.add_argument("--log-dir", type=Path, default=REPO / "hat4_launch_logs")
    parser.add_argument(
        "--run-prefix",
        default="",
        help="Prefix runner log names (for example smoke10_); empty keeps paper names.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_BY_NAME),
        help="Variants to launch; defaults to all eight.",
    )
    parser.add_argument(
        "--gpu-list",
        type=int,
        nargs="+",
        help="Physical GPUs in the same order as --variants.",
    )
    parser.add_argument(
        "--vulkan-icd",
        type=Path,
        default=DEFAULT_VULKAN_ICD,
        help="Single Vulkan ICD JSON exported as VK_ICD_FILENAMES.",
    )
    parser.add_argument(
        "--startup-stagger-seconds",
        type=float,
        default=0.0,
        help="Delay between launches to avoid simultaneous Isaac/data-loader startup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.checkpoint,
        args.train_manifest,
        args.test_manifest,
        args.vulkan_icd,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    commands = []
    selected_variants = (
        [VARIANT_BY_NAME[name] for name in args.variants]
        if args.variants
        else list(VARIANTS)
    )
    gpu_assignment = args.gpu_list or [item[2] for item in selected_variants]
    if len(gpu_assignment) != len(selected_variants):
        raise ValueError(
            "--gpu-list length must equal selected variant count: "
            f"{len(gpu_assignment)} != {len(selected_variants)}"
        )
    if len(set(gpu_assignment)) != len(gpu_assignment):
        raise ValueError(f"Physical GPU assignments must be unique: {gpu_assignment}")
    run_names = [f"{args.run_prefix}{item[0]}" for item in selected_variants]
    if len(set(run_names)) != len(run_names):
        raise ValueError(f"Runner names must be unique: {run_names}")
    for (name, task, _), gpu in zip(selected_variants, gpu_assignment):
        command = [
            str(PYTHON),
            str(TRAIN),
            "--task",
            task,
            "--motion_file",
            str(args.train_manifest.resolve()),
            "--test_motion_file",
            str(args.test_manifest.resolve()),
            "--pretrained_checkpoint",
            str(args.checkpoint.resolve()),
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.max_iterations),
            "--run_name",
            f"{args.run_prefix}{name}",
            "--device",
            f"cuda:{gpu}",
            "--headless",
        ]
        commands.append({"variant": name, "gpu": gpu, "command": command})

    print(json.dumps(commands, indent=2))
    if not args.execute:
        print("Dry run only; pass --execute to launch.")
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    launch_record = args.log_dir / "launch_record.json"
    launched = []
    for index, item in enumerate(commands):
        log_path = args.log_dir / f"{item['variant']}.log"
        log_handle = log_path.open("w")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
        environment["VK_ICD_FILENAMES"] = str(args.vulkan_icd.resolve())
        environment["PYTHONUNBUFFERED"] = "1"
        # Five independent Isaac processes otherwise create hundreds of host
        # BLAS/OpenMP threads while concatenating the same 3000-motion corpus.
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[variable] = "4"
        # With CUDA_VISIBLE_DEVICES set, the selected card is cuda:0 in-process.
        command = [
            "cuda:0" if token == f"cuda:{item['gpu']}" else token
            for token in item["command"]
        ]
        process = subprocess.Popen(
            command,
            cwd=TRACK,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        launched.append(
            {
                "variant": item["variant"],
                "physical_gpu": item["gpu"],
                "pid": process.pid,
                "log": str(log_path.resolve()),
                "command": command,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(item["gpu"]),
                    "VK_ICD_FILENAMES": str(args.vulkan_icd.resolve()),
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "NUMEXPR_NUM_THREADS": "4",
                },
            }
        )
        log_handle.close()
        launch_record.write_text(json.dumps(launched, indent=2) + "\n")
        if index + 1 < len(commands) and args.startup_stagger_seconds > 0:
            time.sleep(args.startup_stagger_seconds)
    print(f"Launched {len(launched)} jobs; record={launch_record}")


if __name__ == "__main__":
    main()
