"""Continue from completed shared collection; train and validate both routes."""
import json
from pathlib import Path
import time
from exec_comp.common import ROOT,log,save_json
from exec_comp.launch import start,idle_gpus
from exec_comp.report import make_report


def alive(job):
    try:return job["module"].encode() in Path(f"/proc/{job['pid']}/cmdline").read_bytes()
    except FileNotFoundError:return False


def wait_collection():
    expected=[("train",i) for i in range(3)]+[("validation",3)]
    while True:
        ready=[(ROOT/"runs/data"/f"{split}_worker{worker:02d}_manifest.json").exists() for split,worker in expected]
        if all(ready):return
        for complete,(split,worker) in zip(ready,expected):
            label="collect_train_"+str(worker) if split=="train" else "collect_validation"
            record=max((ROOT/"runs/logs").glob(label+"*_job.json"),key=lambda p:p.stat().st_mtime)
            job=json.loads(record.read_text())
            if not complete and not alive(job):raise RuntimeError(f"Collection failed: {label}; inspect {job['log']}")
        make_report();time.sleep(30)


def jobs(tasks):
    tasks=list(tasks);running=[]
    while tasks or running:
        for process,job in list(running):
            if process.poll() is not None:
                if process.returncode:raise RuntimeError(f"{job['label']} failed; inspect {job['log']}")
                log("pipeline_job_complete",label=job["label"]);running.remove((process,job));make_report()
        reserved={j["gpu"] for _,j in running}
        for gpu in idle_gpus():
            if not tasks:break
            if gpu in reserved:continue
            module,arguments,label=tasks.pop(0)
            running.append(start(module,arguments,gpu,label));reserved.add(gpu)
        time.sleep(5)


def main():
    save_json(ROOT/"runs/pipeline_status.json",{"stage":"waiting_collection","no_existing_training_restarted":True})
    wait_collection();make_report()
    count=sum(json.loads(p.read_text())["transitions"] for p in (ROOT/"runs/data").glob("train_worker*_manifest.json"))
    if count<5_000_000:raise RuntimeError(f"Insufficient shared data: {count}")
    save_json(ROOT/"runs/pipeline_status.json",{"stage":"training_A1_F_B1","train_transitions":count})
    jobs([
        ("exec_comp.train_inverse",["--updates","10000","--batch","1024"],"train_inverse"),
        ("exec_comp.train_forward_kp",["--updates","10000","--batch","1024"],"train_forward"),
        ("exec_comp.train_body_wm",["--updates","12000","--batch","2048","--name","body_default"],"train_body_default"),
        ("exec_comp.train_body_wm",["--updates","12000","--batch","2048","--name","body_privileged","--privileged"],"train_body_privileged"),
        ("exec_comp.eval_closed_loop",["--method","raw","--headless","--device","cuda:0"],"eval_raw"),
    ])
    jobs([
        ("exec_comp.train_adapter",["--updates","5000","--batch","1024"],"train_adapter"),
        ("exec_comp.eval_body_wm",["--checkpoint",str(ROOT/"runs/models/body_default_best.pt")],"validate_body_default"),
        ("exec_comp.eval_body_wm",["--checkpoint",str(ROOT/"runs/models/body_privileged_best.pt")],"validate_body_privileged"),
        ("exec_comp.eval_closed_loop",["--method","A1","--headless","--device","cuda:0"],"eval_A1"),
    ])
    tasks=[("exec_comp.eval_closed_loop",["--method","A2","--headless","--device","cuda:0"],"eval_A2")]
    body=json.loads((ROOT/"runs/body_privileged_validation.json").read_text())
    if body["gate_passed"]:
        for n in (1,5,10):tasks.append(("exec_comp.eval_closed_loop",["--method","B","--iterations",str(n),"--headless","--device","cuda:0"],"eval_B_N"+str(n)))
        tasks.append(("exec_comp.eval_closed_loop",["--method","B","--iterations","5","--compile-body","--headless","--device","cuda:0"],"eval_B_compile"))
    else:log("B_ONLINE_SKIPPED",reason="B1 world prediction gate failed",result=body)
    jobs(tasks);make_report()
    save_json(ROOT/"runs/pipeline_status.json",{"stage":"complete","B_online_permitted":body["gate_passed"]})
    log("pipeline_complete")


if __name__=="__main__":
    try:main()
    except BaseException as error:
        save_json(ROOT/"runs/pipeline_status.json",{"stage":"failed","error":str(error)})
        make_report()
        raise
