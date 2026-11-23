"""Finalize the common-start benchmark from committed metrics, or expose failure."""
import json
import time
from exec_comp.common import ROOT,save_json,log
from exec_comp.run_pipeline import alive
from exec_comp.report import make_report


def main():
    folder=ROOT/"runs/closed_loop_startup_guard"
    while True:
        completed=[]
        for method in ("raw","A1","A2"):
            if (folder/method/"metrics.json").exists():
                completed.append(method)
                continue
            job=json.loads((ROOT/f"runs/logs/eval_{method}_startup_guard_job.json").read_text())
            if not alive(job):raise RuntimeError(f"{method} benchmark stopped without metrics; see {job['log']}")
        save_json(ROOT/"runs/pipeline_status.json",{"stage":"final_startup_guard_validation","completed_methods":completed,"startup_guard_frames":3})
        if len(completed)==3:break
        time.sleep(15)
    make_report()
    save_json(ROOT/"runs/pipeline_status.json",{"stage":"complete","completed_methods":completed,"heldout_clips":300,"startup_guard_frames":3,"B_online_permitted":False})
    log("final_benchmark_complete",methods=completed)


if __name__=="__main__":
    try:main()
    except BaseException as e:
        save_json(ROOT/"runs/pipeline_status.json",{"stage":"failed_final_validation","error":str(e)})
        make_report()
        raise
