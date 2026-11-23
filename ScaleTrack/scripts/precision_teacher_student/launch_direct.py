"""Launch exactly the six approved direct-PPO groups on physical GPUs 2–7."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
PYTHON = "/root/miniconda3/envs/isaaclab_sonic/bin/python"
GROUPS = ((2, "base", False, 0), (3, "speed", True, 0),
          (4, "wrist1", False, 1), (5, "wrist2", False, 2),
          (6, "speed_wrist1", True, 1), (7, "speed_wrist2", True, 2))


def main():
    available = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                        "--format=csv,noheader,nounits"], text=True)
    memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in available.splitlines()}
    for gpu, _, _, _ in GROUPS:
        if memory[gpu] > 100:
            raise RuntimeError(f"GPU {gpu} is occupied ({memory[gpu]} MiB)")
    launch = TRACK / "hat4_direct_launch_20261013"
    if launch.exists():
        raise FileExistsError("Inspect the existing launch before restarting")
    initial = TRACK / "releases/hat4_focus_release_v1/model_20000.pt"
    manifests = TRACK / "data/hat4_unifolm_focus_v3/manifests"
    digest = hashlib.sha256(initial.read_bytes()).hexdigest()
    source_hashes = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                     for name in ("train_direct.py", "env_cfg.py", "policy.py", "eval_memory.py")}
    jobs = []
    # Validate all inputs and run-directory collisions before starting any job.
    for name in ("unifolm_hat4_train.yaml", "unifolm_hat4_heldout.yaml"):
        if not (manifests/name).is_file():
            raise FileNotFoundError(manifests/name)
    for _, label, _, _ in GROUPS:
        if (TRACK/"logs/rsl_rl/hat4_direct_ablation"/f"direct_20261013_{label}").exists():
            raise FileExistsError(label)
    launch.mkdir()
    for gpu, label, speed, cap in GROUPS:
        run_name = f"direct_20261013_{label}"
        command = [PYTHON, str(HERE/"train_direct.py"), "--run-name", run_name,
                   "--motion-file", str(manifests/"unifolm_hat4_train.yaml"),
                   "--heldout-file", str(manifests/"unifolm_hat4_heldout.yaml"),
                   "--initial-checkpoint", str(initial), "--cap", str(cap),
                   "--iterations", "20000", "--num-envs", "4096", "--headless", "--device", "cuda:0"]
        if speed:
            command.append("--playback")
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
                            "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                            "NUMEXPR_NUM_THREADS": "4", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
                            "LD_LIBRARY_PATH": str(TRACK.parent/"libshim")+":"+environment.get("LD_LIBRARY_PATH", "")})
        log = launch/f"{label}.log"
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=TRACK, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({"label": label, "physical_gpu": gpu, "pid": child.pid, "run_name": run_name,
                     "playback": speed, "cap_combined": cap, "focus": False,
                     "command": command, "log": str(log), "initial_sha256": digest,
                     "code_sha256": source_hashes, "seed": 42,
                     "save_interval": 1000, "eval_interval": 2000})
        (launch/"launch_record.json").write_text(json.dumps(jobs, indent=2)+"\n")
        print("STARTED", run_name, "GPU", gpu, "PID", child.pid, flush=True)
        if gpu != 7:
            time.sleep(12)


if __name__ == "__main__":
    main()
