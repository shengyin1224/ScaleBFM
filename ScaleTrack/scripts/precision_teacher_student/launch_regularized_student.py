"""Launch only the approved student A/B/C experiments on idle GPUs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
TRACK = HERE.parents[1]
PYTHON = "/root/miniconda3/envs/isaaclab_sonic/bin/python"
PREFIX = "student_20261023"


def environment(gpu):
    result = os.environ.copy()
    result.update({"CUDA_VISIBLE_DEVICES":str(gpu), "PYTHONUNBUFFERED":"1",
        "PYTHONDONTWRITEBYTECODE":"1", "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4",
        "OPENBLAS_NUM_THREADS":"4", "NUMEXPR_NUM_THREADS":"4",
        "VK_ICD_FILENAMES":"/etc/vulkan/icd.d/nvidia_icd.json",
        "LD_LIBRARY_PATH":str(TRACK.parent/"libshim")+":"+result.get("LD_LIBRARY_PATH","")})
    return result


def main():
    initial = TRACK/"releases/hat4_focus_release_v1/model_20000.pt"
    teacher = TRACK/"logs/rsl_rl/precision_teacher_student/precision_20261013_w1_teacher/model_12000.pt"
    manifests = TRACK/"data/hat4_unifolm_focus_v3/manifests"
    for path in (initial,teacher,manifests/"unifolm_hat4_train.yaml",manifests/"unifolm_hat4_heldout.yaml"):
        if not path.is_file():
            raise FileNotFoundError(path)
    launch = TRACK/(PREFIX+"_launch")
    if launch.exists():
        raise FileExistsError(launch)
    for variant in ("A","B","C"):
        if (TRACK/"logs/rsl_rl/student_regularized"/(PREFIX+"_"+variant)).exists():
            raise FileExistsError(variant)
    initial_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
    assert initial_hash == "c9c65cfb60bea8e4a254abaa8a60496e8fd0b9703751a0d70c3d322789766393"
    teacher_hash = hashlib.sha256(teacher.read_bytes()).hexdigest()
    source_hashes = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
        ("train_regularized_student.py","student_losses.py","student_validation.py",
         "student_lifecycle.py","env_cfg.py","policy.py")}
    launch.mkdir()
    jobs = []
    for gpu,variant in enumerate(("A","B","C")):
        current = subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.used,utilization.gpu",
                                          "--format=csv,noheader,nounits"],text=True)
        memory, utilization = (int(v.strip()) for v in current.split(","))
        if memory > 100 or utilization > 5:
            raise RuntimeError(f"GPU {gpu} is no longer idle: {current}")
        name = PREFIX+"_"+variant
        command = [PYTHON,str(HERE/"train_regularized_student.py"),"--variant",variant,
            "--run-name",name,"--motion-file",str(manifests/"unifolm_hat4_train.yaml"),
            "--heldout-file",str(manifests/"unifolm_hat4_heldout.yaml"),
            "--initial-checkpoint",str(initial),"--teacher-checkpoint",str(teacher),
            "--iterations","5000","--num-envs","1024","--steps","16","--minibatch","1024",
            "--calibration-update","100","--ramp-updates","500","--save-interval","500",
            "--headless","--device","cuda:0"]
        log = launch/(variant+".log")
        with log.open("x") as stream:
            process = subprocess.Popen(command,cwd=TRACK,env=environment(gpu),stdout=stream,
                                       stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({"variant":variant,"physical_gpu":gpu,"pid":process.pid,"run_name":name,
            "command":command,"log":str(log),"teacher_sha256":teacher_hash,
            "initial_sha256":initial_hash,"source_sha256":source_hashes,
            "teacher":str(teacher),"initial":str(initial),"deployment_changed":False})
        (launch/"launch_record.json").write_text(json.dumps(jobs,indent=2)+"\n")
        print("STARTED "+json.dumps(jobs[-1]),flush=True)


if __name__ == "__main__":
    main()
