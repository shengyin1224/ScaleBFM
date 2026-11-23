"""Evaluate the three last saved October students when an audit GPU is released."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/shengyin/ScaleBFM_focus_brain_20261114')
RUN=ROOT/'experiments/focus_brain_20261114'
OUT=RUN/'later_audit'
OLD=Path('/root/shengyin/ScaleBFM/ScaleTrack')
spec=importlib.util.spec_from_file_location('queue_settings',RUN/'run_queue.py')
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

def worker(variants):
    gpus,locks=q.reserve(1)
    try:
        env=q.base_env.copy()
        env['CUDA_VISIBLE_DEVICES']=str(gpus[0])
        for variant in variants:
            key='oct_student_'+variant.lower() if variant in ('A','B','C') else variant
            output=OUT/(key+'_vr3_fixed5.json')
            if output.exists():continue
            if variant in ('A','B','C'):
                checkpoint=OLD/'logs/rsl_rl/student_regularized'/('student_20261023_'+variant)/'model_1000.pt'
            else:
                run='formal_core5_20260728b_F3h' if variant=='july' else 'formal_focusv2_20260824_F3h'
                checkpoint=OLD/'logs/rsl_rl/g1_hat4_focus_finetune'/run/'model_20000.pt'
            stability_output=OUT/(key+'_mode3_jitter1.json')
            if not stability_output.exists():
                stability_env=env.copy()
                stability_env.pop('PYTHONPATH',None)
                stability_env.pop('LD_LIBRARY_PATH',None)
                stability_env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                                     PYTHONDONTWRITEBYTECODE='1')
                with (OUT/(key+'_mode3_jitter1.log')).open('w') as stream:
                    probe=subprocess.run(['/root/miniconda3/envs/twist/bin/python',
                        str(RUN/'stability_one.py'),str(checkpoint),'3','1',str(stability_output)],
                        cwd=ROOT/'ScaleTrack',env=stability_env,stdout=stream,stderr=subprocess.STDOUT)
                print(key,'stability',probe.returncode,flush=True)
            command=[q.PY,str(q.SCRIPT/'eval_vr3_focus_buckets.py'),'--checkpoint',str(checkpoint),
                '--motion-file',str(OLD/'data/hat4_unifolm_focus_v2/manifests/unifolm_hat4_heldout.yaml'),
                '--output',str(output),'--condition',key,
                '--label-definition',('same corrected August v2 labels; student phase weights are disabled'
                    if variant in ('A','B','C') else 'same corrected August v2 labels and phase input'),
                '--max-steps','1000','--num-envs','600','--fixed-future-offset','5',
                '--device','cuda:0','--headless']
            start=time.time()
            with (OUT/(key+'_vr3_fixed5.log')).open('w') as stream:
                result=subprocess.run(command,cwd=ROOT/'ScaleTrack',env=env,stdout=stream,stderr=subprocess.STDOUT)
            (OUT/(key+'_status.json')).write_text(json.dumps(dict(gpu=gpus[0],exit_code=result.returncode,
                elapsed_s=time.time()-start,command=command),indent=2))
            print(key,result.returncode,flush=True)
    finally:
        for handle in locks:handle.close()

with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(worker,variants) for variants in [('A','C','july'),('B','august')]]
    for future in futures:future.result()
