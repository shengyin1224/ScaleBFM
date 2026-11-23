"""Order the A1-5 evaluations onto idle GPUs; A1 work always goes first.

Stages run in dependency order and each stage fills every free card before the
next one starts, so the long 300-clip runs overlap instead of queueing. B1 is a
separate entry point and is only allowed to take cards this script is not using.
"""
import argparse
import json
import time
from pathlib import Path
from exec_comp.common import ROOT,save_json,log
from exec_comp.launch import start,idle_gpus

OUTPUT="runs/round3/closed_loop"
BASE=["--headless","--device","cuda:0","--output-root",OUTPUT]
# Stage 1 settles which call mode and which lookahead setting to carry forward;
# stage 2 only re-runs the winner under randomization and the guard variant.
STAGE1=(
    ("raw_stall-on",["--method","raw","--anti-stall","on"]),
    ("raw_stall-off",["--method","raw","--anti-stall","off"]),
    ("A1_stepwise_stall-on",["--method","A1","--mode","stepwise","--anti-stall","on"]),
    ("A1_stepwise_stall-off",["--method","A1","--mode","stepwise","--anti-stall","off"]),
    ("A1_event_stall-on",["--method","A1","--mode","switch_event","--anti-stall","on"]),
    ("A1_event_stall-off",["--method","A1","--mode","switch_event","--anti-stall","off"]),
)


def run(jobs,reserve=()):
    """Resume completed results and recycle a GPU only after its child exits."""
    pending=[];running=[]
    def completed(label):
        if label=='hat_export15':
            path=ROOT/'runs/round3/hat_predictions/manifest.json'
            return json.loads(path.read_text()) if path.exists() else None
        return read(label)
    for label,extra in jobs:
        metric=completed(label)
        if metric is not None:
            valid=(not metric.get('smoke') and len(metric['episodes'])==25) if label=='hat_export15' else (not metric.get('truncated_smoke') and metric['metrics']['all']['clips']==300)
            if not valid:
                raise RuntimeError('Incomplete result cannot be resumed: '+label)
            log('round3_reuse_completed',label=label);continue
        saved=ROOT/f'runs/logs/r3_v4_{label}_job.json'
        if saved.exists():
            job=json.loads(saved.read_text())
            running.append((None,job,job['gpu'],label));continue
        pending.append((label,extra))
    while pending or running:
        for item in list(running):
            process,record,gpu,label=item
            if process is not None:code=process.poll()
            else:
                proc=Path(f"/proc/{record['pid']}/cmdline")
                alive=proc.exists() and record['module'].encode() in proc.read_bytes().split(b'\0')
                code=None if alive else (0 if completed(label) else 1)
            if code is None:continue
            if code or not completed(label):raise RuntimeError(f"round3 failed: {record['log']}")
            running.remove(item);log('round3_job_completed',label=label)
        free=[g for g in idle_gpus() if g not in reserve and g not in [j[2] for j in running]]
        for gpu in free:
            if not pending:break
            label,extra=pending.pop(0)
            if label=='hat_export15':
                process,record=start('exec_comp.export_hat',extra,gpu,'r3_v4_'+label,
                    python='/home/aigc/miniconda/envs/human_policy/bin/python')
            else:
                process,record=start('exec_comp.round3.eval_switching',
                    [*extra,*BASE,'--label',label],gpu,'r3_v4_'+label)
            running.append((process,record,gpu,label))
        if pending or running:time.sleep(2)
    return [label for label,_ in jobs]


def read(label):
    path=ROOT/OUTPUT/label/"metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def ratios(a1,raw):
    """The A1-5 acceptance numbers, all measured against raw."""
    def pair(key):
        top=a1["jitter"][key];bottom=raw["jitter"][key]
        return None if not bottom else top/bottom
    wrist=a1["summary"]["wrist_mean_cm"];base=raw["summary"]["wrist_mean_cm"]
    return {"actual_above_5hz_ratio":pair("actual_above_5hz_energy_fraction"),
            "action_difference_ratio":pair("action_difference"),
            "joint_velocity_rms_ratio":pair("joint_velocity_rms_rad_s"),
            "command_above_5hz_ratio":pair("command_above_5hz_energy_fraction"),
            "plan_seam_excess_p95_cm":(a1["summary"]["plan_seam_p95_cm"]-raw["summary"]["plan_seam_p95_cm"]
                                      if a1["summary"]["plan_seam_p95_cm"] is not None and raw["summary"]["plan_seam_p95_cm"] is not None else None),
            "wrist_error_reduction":1.-wrist/base,
            "wrist_mean_cm":wrist,"raw_wrist_mean_cm":base}


def gates(values):
    return {"actual_high_frequency<=1.5x":values["actual_above_5hz_ratio"] is not None and values["actual_above_5hz_ratio"]<=1.5,
            "action_difference<=1.5x":values["action_difference_ratio"] is not None and values["action_difference_ratio"]<=1.5,
            "joint_velocity_rms<=1.5x":values["joint_velocity_rms_ratio"] is not None and values["joint_velocity_rms_ratio"]<=1.5,
            "seam_excess_p95<=0.5cm":values["plan_seam_excess_p95_cm"] is not None and values["plan_seam_excess_p95_cm"]<=.5,
            "wrist_error_reduction>=40%":values["wrist_error_reduction"]>=.40}


def select_candidate(motion,brain):
    return max(motion,key=lambda k:(sum(gates(motion[k]).values())+sum(gates(brain[k]).values()),
               (motion[k]['wrist_error_reduction']+brain[k]['wrist_error_reduction'])/2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reserve",type=int,nargs="*",default=[],
                        help="GPU indices left alone for B1")
    parser.add_argument("--stage",choices=("1","2","all"),default="all")
    arguments=parser.parse_args()
    reserve=set(arguments.reserve)
    status={"started":time.time(),"stage":"A1-5 stage 1: call mode and lookahead"}
    save_json(ROOT/"runs/round3/status.json",status)
    if arguments.stage in ("1","all"):
        contract=json.loads((ROOT/'configs/online_execution_contract.json').read_text())
        export=('hat_export15',['--repo','/mnt/nvme0n1/human_policy_ckpt/isolated_20261021_no_left_wrist/src',
            '--bundle',str(ROOT/'assets/hat_brain_oct22'),'--fps','15',
            '--replan-interval-s',str(contract['replan_interval_s']['p50']),
            '--output','runs/round3/hat_predictions'])
        run((*STAGE1,export),reserve)
    scores={}
    for label,_ in STAGE1:
        metrics=read(label)
        if metrics is None:raise RuntimeError(f"missing metrics for {label}")
        if metrics["method"]=="A1":
            scores[label]=ratios(metrics,read("raw_stall-"+label.split("stall-")[1]))
    # Choose using BOTH mandatory heldout populations. Library-only selection
    # can miss the slow HAT-output regime that failed the physical HF gate.
    brain_folder=ROOT/'runs/round3/brain_evaluation'
    brain_labels=[k for k in scores]+['raw_stall-on','raw_stall-off']
    while not all((brain_folder/k/'metrics.json').exists() for k in brain_labels):time.sleep(2)
    brains={k:json.loads((brain_folder/k/'metrics.json').read_text()) for k in brain_labels}
    if any(v.get('truncated_smoke') or v['robot_episodes']!=10 for v in brains.values()):raise RuntimeError('Incomplete brain evaluation cannot select a release')
    brain_scores={k:ratios(brains[k],brains['raw_stall-'+k.split('stall-')[1]]) for k in scores}
    best=select_candidate(scores,brain_scores)
    save_json(ROOT/"runs/round3/stage1.json",
              {"per_configuration":scores,"gates":{k:gates(v) for k,v in scores.items()},"selected":best,
               "brain_per_configuration":brain_scores,"brain_gates":{k:gates(v) for k,v in brain_scores.items()},
               "selection_rule":"maximize passed gates across motion300 and robot-brain10; ties maximize average relative wrist improvement across those populations"})
    log("round3_stage1",selected=best,**scores[best])
    if arguments.stage=="1":return
    mode="stepwise" if "stepwise" in best else "switch_event"
    stall="on" if best.endswith("stall-on") else "off"
    status["stage"]="A1-5 stage 2: randomization and cap-guard sensitivity for "+best
    save_json(ROOT/"runs/round3/status.json",status)
    run((("raw_dr",["--method","raw","--anti-stall",stall,"--randomization"]),
         (best+"_dr",["--method","A1","--mode",mode,"--anti-stall",stall,"--randomization"]),
         (best+"_capguard-limited",["--method","A1","--mode",mode,"--anti-stall",stall,
                                    "--cap-guard","limited"])),reserve)
    randomized=read(best+"_dr");baseline=read("raw_dr")
    save_json(ROOT/"runs/round3/stage2.json",
              {"selected":best,"randomized":ratios(randomized,baseline),
               "randomized_gates":gates(ratios(randomized,baseline)),
               "cap_guard_sensitivity":{
                   "residual":read(best)["executor"]["fallback_events"],
                   "limited":read(best+"_capguard-limited")["executor"]["fallback_events"]}})
    log("round3_stage2",selected=best)


if __name__=='__main__':
    import fcntl
    lock_path=ROOT/'runs/round3/a1_pipeline.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            path=ROOT/'runs/round3/status.json'
            save_json(path,{'failed':True,'stage':'A1流程因明确错误停止','error':repr(exc)})
            raise

