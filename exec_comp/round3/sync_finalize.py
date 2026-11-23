"""Finish local artifacts when the long remote evaluations reach their gates.

Read-only remote status polling; copies only small evaluation outputs and final
G weights. Local benchmarks wait until there is no GPU compute process. This
never starts a robot or edits any live ScaleBridge file/configuration/weight.
"""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from exec_comp.common import ROOT,PROJECT,save_json,log,digest

HOST='4090-168'
REMOTE='/root/shengyin/ScaleBFM/exec_comp'
CEREBELLUM=PROJECT/'MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt'
CEREBELLUM_SHA='24d10e69d30d253f87d65465c9fa2a0be756d67d17b3a4a08299b0084576aacf'


def remote_status():
    code=f'''import json
from pathlib import Path
root=Path({REMOTE!r})/'runs/round3'
def read(name):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else None
print(json.dumps(dict(a1=read('a1_evaluation_complete.json'),a1_status=read('status.json'),b1=read('b1_summary.json'))))'''
    return json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=15',HOST,
        shlex.join(['/usr/bin/python3','-c',code])],text=True,timeout=45))


def run(args):
    subprocess.run(list(map(str,args)),cwd=PROJECT,check=True)


def fetch_small():
    run(['rsync','-a','--exclude=diagnostics/','--exclude=b1_cache/','--include=*/',
        '--include=*.json','--include=*.npz','--exclude=*',f'{HOST}:{REMOTE}/runs/round3/',ROOT/'runs/round3/'])
    run(['rsync','-a',f'{HOST}:{REMOTE}/releases/a1_round3/latency_4090.json',ROOT/'releases/a1_round3/'])


def wait_local_gpu():
    while True:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        utilization=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True)
        if not apps and all(int(v.strip())<5 for v in utilization.splitlines()):return
        log('round3_local_wait_gpu',reason='existing compute process or GPU utilization');time.sleep(30)


def update_reports():
    run([sys.executable,'-m','exec_comp.round3.report'])
    # Share only the local benchmark JSONs needed by the remote report. The
    # deployment manifest/config remain the local user's reviewable artifacts.
    for p in (ROOT/'releases/a1_round3').glob('latency_4080s.json'):
        run(['rsync','-a',p,f'{HOST}:{REMOTE}/releases/a1_round3/'])
    for p in (ROOT/'runs/round3').glob('B1_*_latency_4080s.json'):
        run(['rsync','-a',p,f'{HOST}:{REMOTE}/runs/round3/'])
    run(['ssh',HOST,shlex.join(['bash','-c',
        'cd /root/shengyin/ScaleBFM && /root/miniconda3/envs/isaaclab_sonic/bin/python -m exec_comp.round3.report'])])


def finalize_a1():
    fetch_small()
    from exec_comp.round3.vendor import verify
    verify(strict_against_source=True)
    if digest(CEREBELLUM)!=CEREBELLUM_SHA:raise RuntimeError('Deployment cerebellum changed; refuse to bind this experiment to another artifact')
    selected=json.loads((ROOT/'runs/round3/stage1.json').read_text())['selected']
    mode='switch_event' if '_event_' in selected else 'stepwise'
    wait_local_gpu()
    run([sys.executable,'-m','exec_comp.round3.package_release','--refresh-metadata','--mode',mode,
        '--cerebellum-artifact',CEREBELLUM])
    run([sys.executable,'-m','exec_comp.round3.package_deployment','--mode',mode])
    run([sys.executable,'-m','unittest','discover','-s','exec_comp/tests'])
    save_json(ROOT/'runs/round3/a1_local_finalized.json',dict(selected=selected,mode=mode,
        real_robot_run=False,enable=False,completed_at=time.time()))
    update_reports()


def main():
    a1_done=(ROOT/'runs/round3/a1_local_finalized.json').exists()
    while True:
        try:status=remote_status()
        except (subprocess.SubprocessError,json.JSONDecodeError) as exc:
            log('round3_sync_retry',error=repr(exc));time.sleep(30);continue
        if status['a1_status'] and status['a1_status'].get('failed'):
            raise RuntimeError('A1 remote pipeline stopped: '+repr(status['a1_status']))
        if status['a1'] and not a1_done:
            finalize_a1();a1_done=True
        b1=status['b1']
        if b1 and (b1['status'].startswith(('stopped_','failed_')) or b1['status']=='evaluation_complete'):
            # A1's final package remains necessary even when B1 stops early.
            if not a1_done:time.sleep(30);continue
            fetch_small()
            if b1['status']=='evaluation_complete':
                for kind,key in [('lowdim','controller'),('perframe','perframe_controller')]:
                    name=Path(b1[key]).name
                    run(['rsync','-a',f'{HOST}:{REMOTE}/runs/models/{name}',ROOT/'runs/models/'])
                    wait_local_gpu()
                    run([sys.executable,'-m','exec_comp.round3.benchmark_b1','--checkpoint',ROOT/'runs/models'/name,
                        '--name','B1_'+kind])
            update_reports()
            save_json(ROOT/'runs/round3/local_completion.json',dict(a1_finalized=True,b1_status=b1['status'],
                real_robot_run=False,completed_at=time.time(),report=str(ROOT/'REPORT.md')))
            log('round3_local_complete',b1_status=b1['status']);return
        time.sleep(30)


if __name__=='__main__':
    lock_path=ROOT/'runs/round3/local_finalize.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            traceback.print_exc()
            save_json(ROOT/'runs/round3/local_finalize_error.json',dict(error=repr(exc),time=time.time(),real_robot_run=False))
            try:
                status=remote_status()
                if status.get('a1_status'):save_json(ROOT/'runs/round3/status.json',status['a1_status'])
                if status.get('b1'):save_json(ROOT/'runs/round3/b1_summary.json',status['b1'])
                run([sys.executable,'-m','exec_comp.round3.report'])
            except Exception:traceback.print_exc()
            raise
