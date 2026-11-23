"""Recover the official model's historical-label VR-3 stage statistics."""
import importlib.util
import json
import subprocess
from pathlib import Path

r=Path('/root/shengyin/ScaleBFM_focus_brain_20261114/experiments/focus_brain_20261114')
spec=importlib.util.spec_from_file_location('queue_settings',r/'run_queue.py')
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
gpus,locks=q.reserve(1)
out=r/'vr3_buckets/f1_v2.json'
checkpoint='/root/shengyin/ScaleBFM/ScaleTrack/logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F1/model_20000.pt'
manifest='/root/shengyin/ScaleBFM/ScaleTrack/data/hat4_unifolm_focus_v2/manifests/unifolm_hat4_heldout.yaml'
command=[q.PY,str(q.SCRIPT/'eval_vr3_focus_buckets.py'),'--checkpoint',checkpoint,
         '--motion-file',manifest,'--output',str(out),'--condition','f1_v2',
         '--label-definition','corrected August v2 labels; F1 trained without focus reward or phase input',
         '--device','cuda:0','--headless']
env=q.base_env.copy()
env['CUDA_VISIBLE_DEVICES']=str(gpus[0])
with open(out.with_suffix('.log'),'w') as log:
    child=subprocess.Popen(command,cwd=q.ROOT/'ScaleTrack',env=env,stdout=log,stderr=subprocess.STDOUT)
    state=dict(pid=child.pid,gpus=gpus,command=command)
    (out.parent/'f1_v2_status.json').write_text(json.dumps(state,indent=2))
    print('START',json.dumps(state),flush=True)
    code=child.wait()
    state['exit_code']=code
    (out.parent/'f1_v2_status.json').write_text(json.dumps(state,indent=2))
    print('EXIT',code,flush=True)
    if code:
        raise SystemExit(code)
