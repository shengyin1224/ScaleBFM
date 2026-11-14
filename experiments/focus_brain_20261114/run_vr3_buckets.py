"""Use only idle GPUs for baseline/native-label VR-3 stage statistics."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import datetime

ROOT=Path('/root/shengyin/ScaleBFM_focus_brain_20261114')
RUN=ROOT/'experiments/focus_brain_20261114'
OUT=RUN/'vr3_buckets'
OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('queue_settings',RUN/'run_queue.py')
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
OLD=Path('/root/shengyin/ScaleBFM/ScaleTrack')
BASE=OLD/'logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt'
JULY=OLD/'logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F3h/model_20000.pt'
AUGUST=OLD/'logs/rsl_rl/g1_hat4_focus_finetune/formal_focusv2_20260824_F3h/model_20000.pt'
V2=OLD/'data/hat4_unifolm_focus_v2/manifests/unifolm_hat4_heldout.yaml'
NATIVE=OLD/'data/hat4_unifolm/manifests/unifolm_hat4_heldout.yaml'
jobs=[
 [('official_v2',BASE,V2,'corrected August v2 labels; official model has no phase training'),
  ('july_native',JULY,NATIVE,'original July training labels and phase input')],
 [('july_v2',JULY,V2,'same corrected August v2 labels and input'),
  ('august_v2',AUGUST,V2,'original August labels, same corrected v2 definition')],
]
state={}
lock=threading.Lock()
def update(key,**values):
    with lock:
        state[key]=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        temp=OUT/'status.tmp'
        temp.write_text(json.dumps(state,indent=2))
        temp.replace(OUT/'status.json')
        print(key,json.dumps(state[key]),flush=True)

def worker(group):
    gpu,locks=q.reserve(1)
    try:
        env=q.base_env.copy()
        env['CUDA_VISIBLE_DEVICES']=str(gpu[0])
        for name,checkpoint,manifest,definition in group:
            output=OUT/(name+'.json')
            if output.exists():
                update(name,stage='complete',gpu=gpu[0],cached=True)
                continue
            command=[q.PY,str(q.SCRIPT/'eval_vr3_focus_buckets.py'),
                     '--checkpoint',str(checkpoint),'--motion-file',str(manifest),
                     '--output',str(output),'--condition',name,'--label-definition',definition,
                     '--max-steps','1000','--num-envs','600','--device','cuda:0','--headless']
            (OUT/(name+'.launch.json')).write_text(json.dumps(dict(command=command,gpu=gpu[0]),indent=2))
            with open(OUT/(name+'.log'),'w') as log:
                child=subprocess.Popen(command,cwd=ROOT/'ScaleTrack',env=env,stdout=log,stderr=subprocess.STDOUT)
                update(name,stage='running',gpu=gpu[0],pid=child.pid)
                code=child.wait()
            update(name,stage='complete' if code==0 else 'failed',gpu=gpu[0],exit_code=code)
            if code:
                raise RuntimeError(f'{name} failed with exit={code}')
    finally:
        for handle in locks:
            handle.close()

with ThreadPoolExecutor(max_workers=2) as executor:
    futures=[executor.submit(worker,group) for group in jobs]
    for future in futures:
        future.result()
spec=importlib.util.spec_from_file_location('summarize_vr3_buckets',RUN/'summarize_vr3_buckets.py')
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
summary.main(OUT)
