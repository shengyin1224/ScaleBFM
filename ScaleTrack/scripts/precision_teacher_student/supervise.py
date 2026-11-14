"""Train, select a stable precision teacher, distill, and gate HAT-4 candidates.

Training logs/checkpoints are retained. Simulation arrays and reports stay in
memory; only concise lifecycle status is printed. Deployment is never replaced.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
PYTHON = Path("/root/miniconda3/envs/isaaclab_sonic/bin/python")
MUJOCO_PYTHON = Path("/root/miniconda3/envs/twist/bin/python")
INITIAL = TRACK / "releases/hat4_focus_release_v1/model_20000.pt"
MANIFESTS = TRACK / "data/hat4_unifolm_focus_v3/manifests"
RUNS = TRACK / "logs/rsl_rl/precision_teacher_student"
REFERENCE = TRACK / "data/hat4_unifolm_focus_v3/processed/G1_WBT_Inspire_Pick_Up_Drinks_ep_0006.npz"


def rollout(checkpoint, mode):
    completed = subprocess.run([str(MUJOCO_PYTHON), str(HERE / "eval_memory.py"),
        "--checkpoint", str(checkpoint), "--reference", str(REFERENCE),
        "--mode", mode, "--jitter-mm", "1"], cwd=TRACK, env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:])
    line = next(x for x in completed.stdout.splitlines() if x.startswith("PRECISION_EVAL "))
    return json.loads(line[len("PRECISION_EVAL "):])


def wrists(result, scope):
    metrics = result["tracking"][scope]
    return [metrics[n]["position_m"]["mean"] for n in
            ("left_wrist_yaw_link", "right_wrist_yaw_link")]


def stable(result, baseline):
    if result["first_failure_frame"] is not None or result["success_rate_frames"] != 1.0:
        return False
    if result["static_arm_hf_rms_rad"] > 1.2*baseline["static_arm_hf_rms_rad"] + 2e-5:
        return False
    if result["action_second_difference_rms"] > 1.2*baseline["action_second_difference_rms"]:
        return False
    # Do not buy world-frame accuracy by losing body/arm coordination.
    for value, old in zip(wrists(result,"active_link_pose_pelvis_frame_local_twist"),
                          wrists(baseline,"active_link_pose_pelvis_frame_local_twist")):
        if value > 1.2*old + .002:
            return False
    return True


def train(script, command, logfile):
    with logfile.open("a", buffering=1) as stream:
        result = subprocess.run([str(PYTHON), str(HERE/script)] + command,
                                cwd=TRACK, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{script} exited {result.returncode}; see {logfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=float, choices=(1.,2.), required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    prefix = args.prefix
    teacher_run = prefix + "_teacher"
    student_run = prefix + "_student"
    launch_dir = TRACK / "precision_launch_20261013"
    launch_dir.mkdir(exist_ok=True)
    teacher_log = launch_dir / (teacher_run + ".log")
    student_log = launch_dir / (student_run + ".log")
    common = ["--motion-file", str(MANIFESTS/"unifolm_hat4_train.yaml"),
              "--heldout-file", str(MANIFESTS/"unifolm_hat4_heldout.yaml"),
              "--initial-checkpoint", str(INITIAL), "--headless", "--device", "cuda:0"]
    print("TRAIN_TEACHER", teacher_run, "cap", args.cap, flush=True)
    train("train_teacher.py", common + ["--run-name", teacher_run, "--cap", str(args.cap),
                                       "--iterations", "20000", "--num-envs", "4096"], teacher_log)
    baseline = rollout(INITIAL, "14")
    best = None
    for step in range(10000,20001,2000):
        checkpoint = RUNS / teacher_run / f"model_{step}.pt"
        result = rollout(checkpoint,"14")
        errors = wrists(result,"active_link_pose_vs_reference")
        old = wrists(baseline,"active_link_pose_vs_reference")
        if stable(result,baseline) and sum(errors) <= .9*sum(old) and max(errors) <= max(old):
            score = sum(errors)
            if best is None or score < best[0]:
                best = (score, checkpoint)
    if best is None:
        print("NO_STABLE_IMPROVED_TEACHER", teacher_run, "student not started", flush=True)
        return
    selected = best[1]
    print("DISTILL_FROM_SELECTED_TEACHER", selected, flush=True)
    train("distill_student.py", common + ["--teacher-checkpoint", str(selected),
                                         "--run-name", student_run, "--iterations", "5000",
                                         "--num-envs", "1024", "--steps", "16"], student_log)
    baseline = rollout(INITIAL,"4")
    candidates = []
    for step in range(500,5001,500):
        checkpoint = RUNS / student_run / f"model_{step}.pt"
        result = rollout(checkpoint,"4")
        errors = wrists(result,"active_link_pose_vs_reference")
        if stable(result,baseline) and max(errors) <= .02:
            candidates.append((sum(errors),checkpoint))
    if not candidates:
        print("NO_QUALIFIED_STUDENT", student_run, "deployment not changed", flush=True)
        return
    selected = min(candidates,key=lambda x:x[0])[1]
    print("SIMULATION_QUALIFIED_CHECKPOINT", selected,
          "requires broader validation before real-world recommendation", flush=True)


if __name__ == "__main__":
    main()
