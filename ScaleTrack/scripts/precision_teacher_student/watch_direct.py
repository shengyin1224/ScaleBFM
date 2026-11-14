"""After direct training finishes, screen saved checkpoints in memory."""
import json
import os
from pathlib import Path
import time
from supervise import rollout, stable, wrists

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
LAUNCH = TRACK / "hat4_direct_launch_20261013"
LABELS = ("base", "speed", "wrist1", "wrist2", "speed_wrist1", "speed_wrist2")
INITIAL = TRACK / "releases/hat4_focus_release_v1/model_20000.pt"
REFERENCE = TRACK / "data/hat4_unifolm_focus_v3/processed/G1_WBT_Inspire_Pick_Up_Drinks_ep_0006.npz"


def main():
    pending = set(LABELS)
    while pending:
        try:
            jobs = json.loads((LAUNCH/"launch_record.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(300)
            continue
        for job in jobs:
            if job["label"] not in pending:
                continue
            proc = Path(f"/proc/{job['pid']}/cmdline")
            try:
                command_line = proc.read_bytes()
            except FileNotFoundError:
                command_line = b""
            if job["run_name"].encode() in command_line:
                continue
            prefix = job["run_name"]
            folder = TRACK/"logs/rsl_rl/hat4_direct_ablation"/prefix
            status = LAUNCH/f"{prefix}_pipeline.log"
            if not (folder/"model_20000.pt").is_file():
                status.write_text("TRAINING_FAILED "+prefix+"; inspect training log\n")
                pending.remove(job["label"])
                continue
            # The finished training GPU can now run MuJoCo checks without contention.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(job["physical_gpu"])
            baseline = rollout(INITIAL, "4")
            selected = None
            for step in range(2000,20001,2000):
                checkpoint = folder/f"model_{step}.pt"
                result = rollout(checkpoint, "4")
                errors = wrists(result, "active_link_pose_vs_reference")
                if stable(result,baseline) and max(errors) <= .02:
                    score = sum(errors)
                    if selected is None or score < selected[0]:
                        selected = (score, checkpoint)
            if selected:
                status.write_text("SIMULATION_QUALIFIED_CHECKPOINT "+str(selected[1])+"\n")
            else:
                status.write_text("NO_QUALIFIED_STUDENT "+prefix+"; deployment not changed\n")
            pending.remove(job["label"])
        if pending:
            time.sleep(300)


if __name__ == "__main__":
    main()
