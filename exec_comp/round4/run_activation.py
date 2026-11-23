"""Test the startup ramp with the SAME original B1, strict safety thresholds."""
import fcntl
from exec_comp.common import ROOT,save_json,log
from exec_comp.launch import idle_gpus,start,wait
from exec_comp.round4.run import task,queue,B1,ROUND


def main():
    smoke=task('b1_activation_smoke','brain','B1',checkpoint=B1,activation_ramp=True,
        max_steps=20,output_root='runs/round4/smoke')
    path=ROUND/'smoke/b1_activation_smoke/metrics.json'
    if not path.exists():
        free=idle_gpus()
        if not free:raise RuntimeError('No idle GPU; no process stopped')
        process,job=start('exec_comp.round4.evaluate',smoke['args'],free[0],'r4_activation_smoke')
        wait(process,job)
    if not path.exists():raise RuntimeError('Smoke missing metrics')
    jobs=[task('b1_lowdim_activation_v1'+('_fallback_off' if off else ''),population,'B1',
        checkpoint=B1,activation_ramp=True,fallback_off=off)
        for population in ('brain','motion','dr') for off in (False,True)]
    save_json(ROUND/'activation_status.json',dict(stage='6 heldout evaluations',
        intervention='initial zero-to-live150ms using existing release ramp; original B1 weights, alpha=.5LP, strict cap and1cm guard unchanged; no coefficient blending',real_robot_run=False))
    queue(jobs,list(range(8)))
    save_json(ROUND/'activation_status.json',dict(stage='evaluations complete',real_robot_run=False))
    log('r4_activation_evaluations_complete')


if __name__=='__main__':
    with (ROUND/'activation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            save_json(ROUND/'activation_status.json',dict(stage='failed',error=repr(exc),real_robot_run=False));raise
