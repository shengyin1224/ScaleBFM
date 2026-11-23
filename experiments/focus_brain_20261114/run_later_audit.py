"""Read-only checkpoint comparison using the existing VR-3 and stability tests."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path('/root/shengyin/ScaleBFM_focus_brain_20261114')
RUN = ROOT / 'experiments/focus_brain_20261114'
OUT = RUN / 'later_audit'
OLD = Path('/root/shengyin/ScaleBFM/ScaleTrack')
GROUP = OLD / 'logs/rsl_rl/g1_hat4_focus_finetune'
MODELS = {
    'july': GROUP / 'formal_core5_20260728b_F3h/model_20000.pt',
    'august': GROUP / 'formal_focusv2_20260824_F3h/model_20000.pt',
    'w4': GROUP / 'F3hw4/model_20000.pt',
    'w4_only_hat4': GROUP / 'F3hw4m/model_20000.pt',
    'oct_v3_focus': GROUP / 'official_threeway_20261007_1510_v3_focus_only/model_20000.pt',
    'oct_v3_precision': GROUP / 'official_threeway_20261007_1510_v3_new_settings/model_20000.pt',
    'oct_v2_precision': GROUP / 'official_threeway_20261007_1510_v2_new_settings/model_20000.pt',
}
spec = importlib.util.spec_from_file_location('queue_settings', RUN / 'run_queue.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
OUT.mkdir(exist_ok=True)
lock = threading.Lock()
state = {}

def execute(name, command, gpu, env):
    with lock:
        state[name] = dict(stage='running', gpu=gpu, command=command, start=time.time())
        (OUT / 'status.json').write_text(json.dumps(state, indent=2))
    with (OUT / (name + '.log')).open('w') as stream:
        result = subprocess.run(command, cwd=ROOT / 'ScaleTrack', env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    with lock:
        state[name].update(stage='complete' if result.returncode == 0 else 'failed',
                           exit_code=result.returncode, end=time.time())
        (OUT / 'status.json').write_text(json.dumps(state, indent=2))
    print(name, result.returncode, flush=True)

def vr3():
    gpus, locks = q.reserve(1)
    try:
        env = q.base_env.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpus[0])
        for name, checkpoint in MODELS.items():
            if name in ('july', 'august'):
                continue
            output = OUT / (name + '_vr3.json')
            if output.exists():
                continue
            execute(name + '_vr3', [q.PY, str(q.SCRIPT / 'eval_vr3_focus_buckets.py'),
                    '--checkpoint', str(checkpoint), '--motion-file',
                    str(OLD / 'data/hat4_unifolm_focus_v2/manifests/unifolm_hat4_heldout.yaml'),
                    '--output', str(output), '--condition', name,
                    '--label-definition', 'same corrected August v2 labels for all checkpoints',
                    '--max-steps', '1000', '--num-envs', '600', '--device', 'cuda:0', '--headless'],
                    gpus[0], env)
    finally:
        for handle in locks:
            handle.close()

def stability():
    gpus, locks = q.reserve(1)
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpus[0]), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1')
        for name, checkpoint in MODELS.items():
            for mode in ('3', '4'):
                key = name + '_mode' + mode + '_jitter1'
                if (OUT / (key + '.json')).exists():
                    continue
                execute(key, ['/root/miniconda3/envs/twist/bin/python',
                        str(RUN / 'stability_one.py'), str(checkpoint), mode, '1',
                        str(OUT / (key + '.json'))], gpus[0], env)
    finally:
        for handle in locks:
            handle.close()

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(vr3), pool.submit(stability)]
        for future in futures:
            future.result()
