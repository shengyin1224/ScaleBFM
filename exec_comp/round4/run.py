"""Resumable R4 simulator queue. Only currently idle GPUs are admitted."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from exec_comp.common import ROOT,log,save_json,digest
from exec_comp.launch import idle_gpus,start
from exec_comp.round3.metrics import acceptance,verdict
from exec_comp.round4.basis import NAMES

ROUND=ROOT/'runs/round4'
B1=ROOT/'runs/models/b1_r3_v4_lowdim_online_2_h32_best.pt'
B1_FRAME=ROOT/'runs/models/b1_r3_v4_perframe_h32_best.pt'


def result(population,label):
    folder='brain_evaluation' if population=='brain' else 'closed_loop'
    path=ROUND/folder/label/'metrics.json'
    return json.loads(path.read_text()) if path.exists() else None


def task(label,population,method='A1',mode='stepwise',basis=None,checkpoint=None,**options):
    actual=label+('_dr' if population=='dr' else '')
    args=['--population','brain' if population=='brain' else 'motion','--method',method,
          '--mode',mode,'--anti-stall','on','--label',actual,'--headless','--device','cuda:0']
    if population=='dr':args+=['--randomization']
    if basis:args+=['--basis',basis]
    if checkpoint:args+=['--checkpoint',str(checkpoint)]
    for key,value in options.items():
        if isinstance(value,bool):
            if value:args+=['--'+key.replace('_','-')]
        else:args+=['--'+key.replace('_','-'),str(value)]
    return dict(label=actual,population=population,args=args)


def complete(task):
    record=result(task['population'],task['label'])
    if not record:return False
    if not record.get('round4_execution'):
        # The first wave completed valid physics/aggregate counters, but native
        # Kit exit skipped its outer Python finally. Preserve that original
        # record; do not invent the lost individual events or traces.
        if not record.get('executor',{}).get('round4'):return False
        import shutil
        folder='brain_evaluation' if task['population']=='brain' else 'closed_loop'
        path=ROUND/folder/task['label']/'metrics.json'
        archive=ROUND/'diagnostics/pre_native_exit_audit_fix'/task['population']/task['label']
        archive.mkdir(parents=True,exist_ok=True)
        if not (archive/'metrics.json').exists():shutil.copy2(path,archive/'metrics.json')
        record['round4_execution']=dict(record['executor']['round4_options'],
            audit_trace_status='N/A: native Kit exit skipped finally; original aggregate counters retained',
            per_event_trace_available=False)
        save_json(path,record)
        save_json(path.with_name('fallback_statistics.json'),record['executor']['round4'])
    if record.get('truncated_smoke'):raise RuntimeError('Smoke result cannot count: '+task['label'])
    size=record['robot_episodes'] if task['population']=='brain' else record['metrics']['all']['clips']
    if size!=(10 if task['population']=='brain' else 300):raise RuntimeError('Wrong heldout count')
    return True


def queue(tasks,gpus):
    pending=[];running=[]
    for item in tasks:
        if complete(item):continue
        label='r4_'+item['population']+'_'+item['label']
        path=ROOT/'runs/logs'/f'{label}_job.json'
        if path.exists():
            job=json.loads(path.read_text());running.append((None,job,item))
        else:pending.append((label,item))
    while pending or running:
        for process,job,item in list(running):
            if process is not None:code=process.poll()
            else:
                proc=Path(f"/proc/{job['pid']}/cmdline")
                alive=proc.exists() and job['module'].encode() in proc.read_bytes().split(b'\0')
                code=None if alive else 0 if complete(item) else 1
            if code is not None:
                if code or not complete(item):raise RuntimeError('Evaluation failed; not restarted: '+job['log'])
                running.remove((process,job,item));log('r4_job_complete',label=item['label'],population=item['population'])
        occupied=[job['gpu'] for _,job,_ in running]
        for gpu in [g for g in idle_gpus() if g in gpus and g not in occupied]:
            if not pending:break
            label,item=pending.pop(0)
            process,job=start('exec_comp.round4.evaluate',item['args'],gpu,label)
            running.append((process,job,item))
        if running or pending:time.sleep(3)


def scores(label):
    values={}
    for population in ('motion','brain','dr'):
        key=label+('_dr' if population=='dr' else '')
        row=result(population,key)
        folder='brain_evaluation' if population=='brain' else 'closed_loop'
        raw='raw_dr' if population=='dr' else 'raw_stall-on'
        baseline=json.loads((ROOT/'runs/round3'/folder/raw/'metrics.json').read_text())
        value=acceptance(row,baseline);checks=verdict(value)
        fraction=row['executor']['round4']['fallback_fraction']
        checks['fallback_fraction']=fraction is not None and fraction<=(.05 if population=='dr' else .02)
        values[population]=dict(metrics=value,gates=checks,fallback_fraction=fraction)
    return values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    parser.add_argument('--stage',choices=('projection','audit','all'),default='all');args=parser.parse_args()
    ROUND.mkdir(parents=True,exist_ok=True)
    labels=[f'a1_{basis}_{mode}' for basis in NAMES for mode in ('stepwise','switch_event')]
    if args.stage in ('projection','all'):
        jobs=[task(f'a1_{basis}_{mode}',population,mode=mode,basis=basis)
              for population in ('brain','motion','dr') for basis in NAMES for mode in ('stepwise','switch_event')]
        priority=[task('b1_lowdim','brain','B1',checkpoint=B1),
                  task('a1_original_wrists_only','brain',wrists_only=True)]
        for population in ('brain','motion','dr'):
            for label,checkpoint in [('b1_lowdim',B1),('b1_perframe',B1_FRAME)]:
                if not (population=='brain' and label=='b1_lowdim'):
                    priority.append(task(label,population,'B1',checkpoint=checkpoint))
            priority.append(task('b1_lowdim_fallback_off',population,'B1',checkpoint=B1,fallback_off=True))
            priority.append(task('b1_lowdim_limited',population,'B1',checkpoint=B1,cap_guard='limited'))
        jobs=priority+jobs
        save_json(ROUND/'status.json',dict(stage='R4-1 inference projection12 configurations x3 populations',
            priority='B1 R4 evaluations + A1 wrists-only; existing projection jobs continue without interruption',
            skipped=['R4-2 per user instruction'],deferred=['CAPS per user instruction'],
            jobs=len(jobs),started=time.time(),real_robot_run=False))
        queue(jobs,args.gpus)
        measured={label:scores(label) for label in labels}
        best=max(labels,key=lambda label:(sum(v is True for pop in measured[label].values() for v in pop['gates'].values()),
                 -measured[label]['brain']['metrics']['wrist_mean_cm']))
        save_json(ROUND/'projection_selection.json',dict(selected=best,configurations=measured,
            rule='maximize passed physical/precision/fallback gates across all3 populations; then minimize brain wrist error',
            projection='position xyz and SO3-log orientation xyz, original A1 weights unchanged'))
        log('r4_projection_complete',selected=best)
    if args.stage=='projection':return
    selection=json.loads((ROUND/'projection_selection.json').read_text());best=selection['selected']
    best_basis=next(b for b in NAMES if best.startswith('a1_'+b+'_'))
    best_mode=best[len('a1_'+best_basis+'_'):]
    jobs=[]
    configurations=[('a1_original_stepwise','A1','stepwise',None,None),
        ('a1_original_event','A1','switch_event',None,None),
        ('b1_lowdim','B1','stepwise',None,B1),('b1_perframe','B1','stepwise',None,B1_FRAME)]
    for population in ('brain','motion','dr'):
        for label,method,mode,basis,checkpoint in configurations:
            jobs.append(task(label,population,method,mode,basis,checkpoint))
        for label,method,mode,basis,checkpoint in [('b1_lowdim','B1','stepwise',None,B1),
                (best,'A1',best_mode,best_basis,None)]:
            jobs.append(task(label+'_fallback_off',population,method,mode,basis,checkpoint,fallback_off=True))
            jobs.append(task(label+'_limited',population,method,mode,basis,checkpoint,cap_guard='limited'))
    save_json(ROUND/'status.json',dict(stage='R4-5 fallback baseline, off and limited diagnostics',jobs=len(jobs),real_robot_run=False))
    queue(jobs,args.gpus)
    save_json(ROUND/'audit_evaluations_complete.json',dict(selected=best,real_robot_run=False))
    log('r4_audit_evaluations_complete',selected=best)


if __name__=='__main__':
    ROUND.mkdir(parents=True,exist_ok=True)
    with (ROUND/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            save_json(ROUND/'status.json',dict(failed=True,error=repr(exc),real_robot_run=False));raise
