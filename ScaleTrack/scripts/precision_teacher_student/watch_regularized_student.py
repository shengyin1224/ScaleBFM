"""Validate student checkpoints during training, keeping arrays only in memory."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
LAUNCH = TRACK/"student_20261023_launch"
PYTHON = "/root/miniconda3/envs/twist/bin/python"
REFERENCE = TRACK/"data/hat4_unifolm_focus_v3/processed/G1_WBT_Inspire_Pick_Up_Drinks_ep_0006.npz"


def evaluate(checkpoint, jitter):
    # Do not occupy a GPU acquired by someone else after this watcher started.
    values = subprocess.check_output(["nvidia-smi","--id=4","--query-gpu=memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"],text=True)
    memory, utilization = (int(v.strip()) for v in values.split(","))
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "PYTHONDONTWRITEBYTECODE":"1"})
    device = "cpu"
    if memory < 100 and utilization < 5:
        environment["CUDA_VISIBLE_DEVICES"] = "4"
        device = "cuda:0"
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [PYTHON,str(HERE/"eval_memory.py"),"--checkpoint",str(checkpoint),
               "--reference",str(REFERENCE),"--mode","4","--jitter-mm",str(jitter),"--device",device]
    result = subprocess.run(command,cwd=TRACK,env=environment,stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    line = next(s for s in result.stdout.splitlines() if s.startswith("PRECISION_EVAL "))
    metrics = json.loads(line[len("PRECISION_EVAL "):])
    print("STUDENT_VALIDATION "+json.dumps(metrics),flush=True)


def main():
    jobs = json.loads((LAUNCH/"launch_record.json").read_text())
    baseline = TRACK/"releases/hat4_focus_release_v1/model_20000.pt"
    for jitter in (0,1,3):
        evaluate(baseline,jitter)
    done = set()
    while True:
        alive = False
        for job in jobs:
            try:
                running = job["run_name"].encode() in Path(f"/proc/{job['pid']}/cmdline").read_bytes()
            except FileNotFoundError:
                running = False
            alive |= running
            folder = TRACK/"logs/rsl_rl/student_regularized"/job["run_name"]
            for step in range(500,5001,500):
                checkpoint = folder/f"model_{step}.pt"
                if not checkpoint.is_file():
                    continue
                for jitter in ((0,1,3) if step == 5000 else (1,)):
                    key = (job["run_name"],step,jitter)
                    if key in done:
                        continue
                    try:
                        evaluate(checkpoint,jitter)
                        done.add(key)
                    except Exception as error:
                        print("VALIDATION_ERROR "+str(key)+" "+str(error),flush=True)
            if not running and not (folder/"model_5000.pt").is_file():
                print("TRAINING_STOPPED_BEFORE_COMPLETION "+job["run_name"],flush=True)
        if not alive:
            print("STUDENT_VALIDATION_FINISHED; deployment unchanged",flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
