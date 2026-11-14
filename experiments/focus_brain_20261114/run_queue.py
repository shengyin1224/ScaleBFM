"""Wait for genuinely idle GPUs; evaluate phase ablations, benchmark, then train."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path('/root/shengyin/ScaleBFM_focus_brain_20261114')
RUN = ROOT / 'experiments/focus_brain_20261114'
PY = '/root/miniconda3/envs/isaaclab_sonic/bin/python'
ORIGINAL = Path('/root/shengyin/ScaleBFM')
CORPUS = RUN / 'corpus'
SCRIPT = ROOT / 'ScaleTrack/scripts/pretrain/rsl_rl'
RUN.mkdir(parents=True, exist_ok=True)
base_env = os.environ.copy()
base_env.update(PYTHONPATH=':'.join(str(ROOT / p) for p in ['ScaleTrack/source/scaletrack','ScaleTrack/source/my_rsl_rl','ScaleTrack/scripts/pretrain/rsl_rl','']),
                LD_LIBRARY_PATH=str(ORIGINAL / 'libshim') + ':/root/miniconda3/envs/isaaclab_sonic/lib:' + base_env.get('LD_LIBRARY_PATH',''),
                VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json', PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                FOCUS_LR_ROOT=str(CORPUS))

def status(stage, **extra):
    data = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage=stage, pid=os.getpid(), **extra)
    temp = RUN / 'queue_status.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(RUN / 'queue_status.json')
    print(json.dumps(data), flush=True)

def idle_gpus():
    lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).splitlines()
    idle = []
    for line in lines:
        index, memory, utilization = [int(x.strip()) for x in line.split(',')]
        if memory < 1000 and utilization < 10:
            idle.append(index)
    return sorted(idle, reverse=True)

def reserve(count):
    previous = set()
    while True:
        free = idle_gpus()
        stable = [g for g in free if g in previous]
        if len(stable) >= count:
            locks = []
            chosen = []
            for gpu in stable:
                handle = open(f'/tmp/scalebfm_focus_20261114_gpu_{gpu}.lock','w')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                locks.append(handle)
                chosen.append(gpu)
                if len(chosen) == count:
                    return chosen, locks
            for lock in locks:
                lock.close()
        previous = set(free)
        time.sleep(15)

def run_process(name, command, gpus):
    env = base_env.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpus))
    record = dict(command=command, physical_gpus=gpus, environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','FOCUS_LR_ROOT']})
    (RUN / f'{name}.launch.json').write_text(json.dumps(record, indent=2))
    with open(RUN / f'{name}.log','w') as log:
        child = subprocess.Popen(command, cwd=ROOT / 'ScaleTrack', env=env, stdout=log, stderr=subprocess.STDOUT)
        status(name, child_pid=child.pid, physical_gpus=gpus)
        code = child.wait()
    if code:
        raise RuntimeError(f'{name} failed: exit={code}; see {RUN / (name + ".log")}')

def train_command(world_size, iterations, run_name, benchmark=False):
    script = str(SCRIPT / 'train.py')
    command = [PY]
    if world_size > 1:
        command += ['-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={world_size}']
    command += [script,'--task','G1-HAT4-BrainFocusLR-Finetune',
                '--motion_file',str(CORPUS / 'manifests/unifolm_hat4_train.yaml'),
                '--test_motion_file',str(CORPUS / 'manifests/unifolm_hat4_heldout.yaml'),
                '--pretrained_checkpoint',str(ORIGINAL / 'ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt'),
                '--num_envs',str(4096 // world_size),'--max_iterations',str(iterations),
                '--run_name',run_name,'--device','cuda:0','--headless']
    if world_size > 1:
        command.append('--distributed')
    if benchmark:
        command += ['agent.eval_during_training=false']
    return command

def main():
    status('waiting_for_ablation_gpu')
    gpus, locks = reserve(1)
    if not (RUN / 'ablation/comparison.json').is_file():
        run_process('focus_input_ablation', [PY,str(SCRIPT / 'eval_focus_input_ablation.py'),
        '--output-dir',str(RUN / 'ablation'),
        '--motion-file',str(ORIGINAL / 'ScaleTrack/data/hat4_unifolm_focus_v2/manifests/unifolm_hat4_heldout.yaml'),
        '--july-checkpoint',str(ORIGINAL / 'ScaleTrack/logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F3h/model_20000.pt'),
        '--august-checkpoint',str(ORIGINAL / 'ScaleTrack/logs/rsl_rl/g1_hat4_focus_finetune/formal_focusv2_20260824_F3h/model_20000.pt'),
        '--num-envs','600','--device','cuda:0','--headless'], gpus)
    for lock in locks:
        lock.close()
    status('waiting_for_validated_training_code_and_labels')
    while not (RUN / 'CODE_READY').is_file() or not (CORPUS / 'READY').is_file():
        time.sleep(15)
    status('waiting_for_training_smoke_gpu')
    gpus, locks = reserve(1)
    smoke = train_command(1,5,'brain_focus_lr_smoke_20261114')
    import yaml
    source_manifest_dir = CORPUS / 'manifests'
    smoke_dir = RUN / 'smoke_manifests'
    smoke_dir.mkdir(exist_ok=True)
    for split, per_family in [('train',2),('heldout',1)]:
        full = yaml.safe_load((source_manifest_dir / f'unifolm_hat4_{split}.yaml').read_text())
        selected = {}
        counts = {}
        for name, path in full.items():
            family = name.rsplit('_ep_',1)[0]
            if counts.get(family,0) < per_family:
                selected[name] = path
                counts[family] = counts.get(family,0) + 1
        assert len(counts) == 10
        dest = smoke_dir / f'{split}.yaml'
        dest.write_text(yaml.safe_dump(selected))
        flag = '--motion_file' if split == 'train' else '--test_motion_file'
        smoke[smoke.index(flag)+1] = str(dest)
    smoke[smoke.index('--num_envs')+1] = '64'
    smoke += ['agent.eval_interval=5','agent.eval_max_steps=50']
    run_process('training_smoke',smoke,gpus)
    for lock in locks:
        lock.close()
    results = {}
    # Compare equal global batch sizes: 4096 environments in all cases.
    for world in (1,2,4):
        status(f'waiting_for_{world}_gpu_benchmark')
        gpus, locks = reserve(world)
        name = f'benchmark_{world}gpu'
        run_process(name, train_command(world,10,f'{name}_20261114',True), gpus)
        text = (RUN / f'{name}.log').read_text(errors='replace')
        times = [float(a)+float(b) for a,b in re.findall(r'collection: ([\d.]+)s, learning ([\d.]+)s', text)]
        if len(times) < 5:
            raise RuntimeError(f'{name}: insufficient measured iterations')
        results[world] = dict(median_iteration_seconds=statistics.median(times[2:]), measured_iterations=times,
                              global_envs=4096, envs_per_rank=4096//world)
        (RUN / 'multi_gpu_benchmark.json').write_text(json.dumps(results,indent=2))
        for lock in locks:
            lock.close()
    chosen = min(results, key=lambda n: results[n]['median_iteration_seconds'])
    # Require a measured 5% gain before using extra GPUs.
    if results[chosen]['median_iteration_seconds'] > .95 * results[1]['median_iteration_seconds']:
        chosen = 1
    status('waiting_for_formal_training', selected_gpu_count=chosen, benchmarks=results)
    gpus, locks = reserve(chosen)
    run_process('formal_training', train_command(chosen,20000,'brain_focus_lr_20261114'), gpus)
    status('complete', selected_gpu_count=chosen)

if __name__ == '__main__':
    with open(RUN / 'queue.lock','w') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as error:
            status('failed', error=repr(error))
            raise
