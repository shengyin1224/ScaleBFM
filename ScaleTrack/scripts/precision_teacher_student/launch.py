"""Launch the authorized cap=1 and cap=2 teacher/student pipelines."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
PYTHON = "/root/miniconda3/envs/isaaclab_sonic/bin/python"


def main():
    launch = TRACK / "precision_launch_20261013"
    launch.mkdir(exist_ok=True)
    record = launch / "launch_record.json"
    if record.exists():
        raise FileExistsError("Launch already recorded; inspect running jobs before restarting")
    initial = TRACK / "releases/hat4_focus_release_v1/model_20000.pt"
    jobs = []
    for gpu, cap in ((0,1),(1,2)):
        prefix = f"precision_20261013_w{cap}"
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                    "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
                    "LD_LIBRARY_PATH": str(TRACK.parent / "libshim") + ":" + env.get("LD_LIBRARY_PATH","")})
        command = [PYTHON,str(HERE / "supervise.py"),"--cap",str(cap),"--prefix",prefix]
        log = launch / f"{prefix}_pipeline.log"
        with log.open("x") as stream:
            child = subprocess.Popen(command,cwd=TRACK,env=env,stdout=stream,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({"cap_combined":cap,"physical_gpu":gpu,"pid":child.pid,"prefix":prefix,
                     "command":command,"pipeline_log":str(log),"initial_sha256":hashlib.sha256(initial.read_bytes()).hexdigest(),
                     "teacher_updates":20000,"student_updates":5000,"focus":False,
                     "teacher_mode":"WholeBody-14","student_mode":"HAT-4",
                     "contract":{"history":3,"future_idx":[0,1,2,3,4,5],"actor_task_dim":269,"action_dim":29},
                     "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}})
        record.write_text(json.dumps(jobs,indent=2)+"\n")
        print("STARTED",prefix,"gpu",gpu,"pid",child.pid,flush=True)
        if gpu == 0:
            time.sleep(10)


if __name__ == "__main__":
    main()
