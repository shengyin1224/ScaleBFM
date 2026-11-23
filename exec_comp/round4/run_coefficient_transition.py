"""R4-5 targeted switch intervention, then all3 heldout populations."""
import fcntl
import json
from exec_comp.common import ROOT,save_json,log
from exec_comp.launch import idle_gpus,start,wait
from exec_comp.round4.run import task,queue,B1,ROUND


def main():
    evidence=json.loads((ROUND/'fallback_analysis.json').read_text())
    if not evidence.get('parameter_transition_required'):raise RuntimeError('No recorded switch concentration supporting this intervention')
    smoke=task('b1_coeffblend_smoke','brain','B1',checkpoint=B1,coefficient_transition=True,
        max_steps=20,output_root='runs/round4/smoke')
    path=ROUND/'smoke/b1_coeffblend_smoke/metrics.json'
    if not path.exists():
        free=idle_gpus()
        if not free:raise RuntimeError('No idle GPU for smoke; no existing process stopped')
        process,job=start('exec_comp.round4.evaluate',smoke['args'],free[0],'r4_coeffblend_smoke')
        wait(process,job)
    if not path.exists():raise RuntimeError('Smoke ended without metrics')
    jobs=[]
    for population in ('brain','motion','dr'):
        for disabled in (False,True):
            label='b1_lowdim_coeffblend_v1'+('_fallback_off' if disabled else '')
            jobs.append(task(label,population,'B1',checkpoint=B1,coefficient_transition=True,fallback_off=disabled))
    save_json(ROUND/'coefficient_transition_status.json',dict(stage='6 heldout evaluations',
        intervention='interpolate native world offset/slope before spherical cap, align old offset to new time origin; duration120ms; refresh target each50Hz step',
        explanation='brain3/3 jumps near switch; does not assert most motion/DR jumps have same cause',real_robot_run=False))
    queue(jobs,list(range(8)))
    save_json(ROUND/'coefficient_transition_status.json',dict(stage='evaluations complete',real_robot_run=False))
    log('r4_coeffblend_evaluations_complete')


if __name__=='__main__':
    with (ROUND/'coefficient_transition.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            save_json(ROUND/'coefficient_transition_status.json',dict(stage='failed',error=repr(exc),real_robot_run=False));raise
