"""Retain the initial timing diagnostic and run the aligned implementation once."""
import json
import time
from exec_comp.common import ROOT,log
from exec_comp.launch import start,idle_gpus


def main():
    source=ROOT/"runs/closed_loop/A1"
    while not (source/"metrics.json").exists():time.sleep(5)
    result=json.loads((source/"metrics.json").read_text())
    if result.get("response_target_offset_control_steps")==1:return
    archive=ROOT/"runs/diagnostics/A1_before_response_alignment"
    archive.parent.mkdir(exist_ok=True)
    source.rename(archive)
    while 7 not in idle_gpus():time.sleep(5)
    process,job=start("exec_comp.eval_closed_loop",["--method","A1","--headless","--device","cuda:0"],7,"eval_A1_aligned")
    code=process.wait()
    if code:raise RuntimeError(f"Aligned A1 evaluation failed: {job['log']}")
    from exec_comp.report import make_report
    make_report();log("aligned_A1_complete")


if __name__=="__main__":main()
