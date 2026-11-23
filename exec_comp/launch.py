"""Start isolated jobs only on GPUs that are currently idle."""
import json
import os
from pathlib import Path
import subprocess
import time
from exec_comp.common import ROOT,PROJECT,log,save_json

PYTHON="/root/miniconda3/envs/isaaclab_sonic/bin/python"


def idle_gpus():
    result=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    return [int(parts[0]) for line in result.splitlines() if (parts:=[int(x.strip()) for x in line.split(",")])[1]<100 and parts[2]<5]


def environment(gpu):
    env=os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONUNBUFFERED":"1","PYTHONDONTWRITEBYTECODE":"1",
                "OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4",
                "VK_ICD_FILENAMES":"/etc/vulkan/icd.d/nvidia_icd.json",
                "LD_LIBRARY_PATH":str(PROJECT/"libshim")+":"+env.get("LD_LIBRARY_PATH","")})
    return env


def start(module,args,gpu,label,python=None):
    # Driver utilization samples can lag a process's exit. Allow a short grace
    # period only when memory is already below our idle threshold; a card with
    # resident work is rejected immediately.
    for _ in range(10):
        if gpu in idle_gpus():break
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in rows.splitlines()}
        if memory[gpu]>=100:break
        time.sleep(.5)
    if gpu not in idle_gpus():raise RuntimeError(f"GPU {gpu} is occupied")
    logs=ROOT/"runs/logs";logs.mkdir(parents=True,exist_ok=True)
    destination=logs/(label+".log")
    if destination.exists():raise FileExistsError(destination)
    executable=python or PYTHON
    command=[executable,"-m",module,*map(str,args)]
    with destination.open("x") as stream:
        process=subprocess.Popen(command,cwd=PROJECT,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    job={"pid":process.pid,"gpu":gpu,"module":module,"args":args,"label":label,"log":str(destination),"started":time.time(),'python':executable}
    save_json(logs/(label+"_job.json"),job)
    log("job_started",**job)
    return process,job


def wait(process,job):
    code=process.wait()
    if code:raise RuntimeError(f"{job['label']} failed ({code}); {job['log']}")
    log("job_complete",label=job["label"],elapsed_s=time.time()-job["started"])
