#!/usr/bin/env python3
"""Render singleton IsaacLab videos for representative failed motions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--load-run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode-index", type=int, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-video-length", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    python = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
    play = root / "scripts/pretrain/rsl_rl/play.py"
    entries = yaml.safe_load(args.manifest.read_text())
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for index, (name, motion_value) in enumerate(entries.items()):
        motion = Path(motion_value).resolve()
        with np.load(motion) as reference:
            transitions = int(len(reference["joint_pos"]) - 1)
        if args.max_video_length is not None:
            transitions = min(transitions, args.max_video_length)
        singleton = manifest_dir / f"{name}.yaml"
        singleton.write_text(yaml.safe_dump({name: str(motion)}, sort_keys=False))
        jobs.append((index, name, singleton, transitions))

    def run(job):
        index, name, singleton, transitions = job
        video_dir = (args.output_dir / "videos" / name).resolve()
        existing = sorted(video_dir.glob("*.mp4"))
        if args.skip_existing and existing:
            return {
                "motion": name,
                "transitions": transitions,
                "gpu": None,
                "exit_code": 0,
                "video": str(existing[-1]),
                "log": None,
                "status": "skipped_existing",
            }
        command = [
            str(python),
            str(play),
            "--task",
            args.task,
            "--motion_file",
            str(singleton.resolve()),
            "--num_envs",
            "1",
            "--mode_index",
            str(args.mode_index),
            "--load_run",
            args.load_run,
            "--checkpoint",
            args.checkpoint,
            "--video",
            "--video_length",
            str(transitions),
            "--video_folder",
            str(video_dir),
            "--headless",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
        environment["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log = args.output_dir / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout)
        videos = sorted(video_dir.glob("*.mp4"))
        return {
            "motion": name,
            "transitions": transitions,
            "gpu": gpus[index % len(gpus)],
            "exit_code": completed.returncode,
            "video": str(videos[-1]) if videos else None,
            "log": str(log.resolve()),
            "status": "rendered",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs))
    report = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "mode_index": args.mode_index,
        "results": results,
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = [result for result in results if result["exit_code"] or not result["video"]]
    if failures:
        raise RuntimeError(f"Failed renders: {failures}")
    print(args.output_dir / "render_summary.json")


if __name__ == "__main__":
    main()
