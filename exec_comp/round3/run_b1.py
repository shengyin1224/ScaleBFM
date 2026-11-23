"""Resumable B1 loop on spare GPUs; never occupies A1's three-card queue.

Every launched command and PID is saved by exec_comp.launch. A failed stage
stops the pipeline; incomplete artifacts are never counted as completed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path
from exec_comp.common import ROOT,PROJECT,save_json,log
from exec_comp.launch import start,idle_gpus


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def job(module,args,label,done,allowed):
    def finished():
        if not done.exists():return False
        if module=='exec_comp.round3.prepare_b1':return 'validation' in read(done).get('splits',{})
        return True
    if finished():return
    record_path=ROOT/'runs/logs'/f'{label}_job.json'
    process=None
    if record_path.exists():
        record=read(record_path)
    else:
        while True:
            free=[g for g in idle_gpus() if g in allowed]
            if free:break
            time.sleep(5)
        process,record=start(module,args,free[0],label)
    while True:
        if process is None:
            proc=Path(f"/proc/{record['pid']}/cmdline")
            alive=proc.exists() and module.encode() in proc.read_bytes().split(b'\0')
            code=None if alive else 0 if finished() else 1
        else:code=process.poll()
        if code is not None:
            if code or not finished():raise RuntimeError(f'Stage failed: {record["log"]}')
            return
        time.sleep(5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',type=int,nargs='+',default=[4]);p.add_argument('--parallel-after-a1',action='store_true');args=p.parse_args()
    primary=args.gpus[:1]
    if args.parallel_after_a1 and len(args.gpus)<3:raise ValueError('Need primary plus2 spare GPU indices')
    out=ROOT/'runs/round3';summary={'rounds':[],'status':'waiting for reference cache','conclusion':'B1尚未完成，不作性能结论。'}
    def status(message):
        summary['status']=message;save_json(out/'b1_summary.json',summary);log('b1_stage',stage=message)
    cache=out/'b1_cache/initial_v4'
    while not (cache/'manifest.json').exists() or 'validation' not in read(cache/'manifest.json').get('splits',{}):time.sleep(5)
    if (cache/'manifest.json').stat().st_size==0:raise RuntimeError('Incomplete cache')
    body=ROOT/'runs/models/body_b_online_r2_best.pt'
    initial=ROOT/'runs/models/b1_r3_v4_lowdim_h17_best.pt'
    status('H17低维G训练，原始full-accept缓存，零初始化')
    job('exec_comp.round3.train_b1',['--cache',str(cache),'--name','b1_r3_v4_lowdim_h17','--updates','1500','--batch','64'],
        'r3_v4_b1_lowdim_h17',ROOT/'runs/models/b1_r3_v4_lowdim_h17_training.json',primary)
    controller=initial;online_roots=[]
    for iteration in (1,2):
        name=f'round3_b1_online_{iteration}_v4';data=ROOT/'runs/data'/name
        online_roots.append(data)
        status(f'在线轮次{iteration}: TRAIN动作中采集约1M，sigma=0.05')
        base=['--method','B1','--checkpoint',str(controller),'--data-label',name,'--limit','300','--headless','--device','cuda:0']
        if iteration==2 and args.parallel_after_a1:
            # A1's formal/DR/brain jobs finish BEFORE borrowing its cards.
            while not (out/'a1_evaluation_complete.json').exists():time.sleep(5)
            def collect_piece(split,n,seed,worker,gpu):
                label=f'B1_collect_r{iteration}_{split}_w{worker}_v4'
                job('exec_comp.round3.eval_switching',base+['--limit',str(300 if split=='train' else 128),
                    '--collect-transitions',str(n),'--data-split',split,'--seed',str(seed),'--worker',str(worker),'--label',label],
                    f'r3_v4_b1_collect_r{iteration}_{split}_w{worker}',
                    data/f'{split}_worker{worker:02d}_manifest.json',[gpu])
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(collect_piece,'train',500000,102,0,args.gpus[0]),
                         pool.submit(collect_piece,'train',500000,1102,1,args.gpus[1]),
                         pool.submit(collect_piece,'validation',100000,202,0,args.gpus[2])]
                for future in futures:future.result()
        else:
            for split,n,seed in [('train',1000000,100+iteration),('validation',100000,200+iteration)]:
                job('exec_comp.round3.eval_switching',base+['--limit',str(300 if split=='train' else 128),'--collect-transitions',str(n),'--data-split',split,'--seed',str(seed),
                    '--label',f'B1_collect_r{iteration}_{split}_v4'],f'r3_v4_b1_collect_r{iteration}_{split}',
                    data/f'{split}_worker00_manifest.json',primary)
        auditname=f'b1_r{iteration}_pre_refine_v4'
        auditargs=['--body',str(body),'--data',name,'--name',auditname,'--iteration',str(iteration),'--controller',str(controller)]
        job('exec_comp.round3.audit_b1_model',auditargs,f'r3_v4_b1_audit_pre_r{iteration}',out/f'{auditname}.json',primary)
        before=read(out/f'{auditname}.json')
        if before['optimism_stop']:
            summary['rounds'].append(before);summary['conclusion']='停止：同初态/同目标/实际G动作下，世界模型想象误差至少低一倍且差距≥1cm；触发模型乐观偏差停止条件。未继续调参。'
            status('stopped_model_optimism');return
        status(f'在线轮次{iteration}: 冻结G，用新数据微调68维世界模型17步')
        newbody=f'body_b1_r3_v4_online_{iteration}'
        trainargs=['--privileged','--width','1024','--initialize',str(body),'--body-horizon','17','--project-gravity',
                   '--updates','1500','--batch','512','--eval-interval','250','--balance-data-roots','--name',newbody]
        for folder in [ROOT/'runs/data/round2_b_online_1',ROOT/'runs/data/round2_b_online_2',*online_roots]:
            trainargs+=['--extra-train-data',str(folder),'--extra-validation-data',str(folder)]
        job('exec_comp.train_body_wm',trainargs,f'r3_v4_b1_body_r{iteration}',ROOT/f'runs/models/{newbody}_training.json',primary)
        body=ROOT/f'runs/models/{newbody}_best.pt'
        auditname=f'b1_r{iteration}_post_refine_v4';auditargs=['--body',str(body),'--data',name,'--name',auditname,'--iteration',str(iteration),'--controller',str(controller)]
        job('exec_comp.round3.audit_b1_model',auditargs,f'r3_v4_b1_audit_post_r{iteration}',out/f'{auditname}.json',primary)
        result=read(out/f'{auditname}.json');summary['rounds'].append(result)
        if not result['gate_passed']:
            summary['conclusion']='停止：世界模型在G轨迹上的误差超过普通分布2倍，或出现明显乐观偏差。未继续调参。';status('stopped_world_model_gate');return
        status(f'在线轮次{iteration}: 从新旧replay混合准备G训练参考缓存')
        cache_name=f'online_{iteration}_v4';cache=out/'b1_cache'/cache_name
        prep=['--name',cache_name,'--train','2048','--validation','256','--balance-data-roots',
              '--extra-data',str(ROOT/'runs/data/round2_b_online_1'),'--extra-data',str(ROOT/'runs/data/round2_b_online_2'),
              *[arg for folder in online_roots for arg in ('--extra-data',str(folder))]]
        job('exec_comp.round3.prepare_b1',prep,f'r3_v4_b1_cache_r{iteration}',cache/'manifest.json',primary)
        horizon=32;modelname=f'b1_r3_v4_lowdim_online_{iteration}_h32'
        status(f'在线轮次{iteration}: 使用更新世界模型继续训练G H32')
        job('exec_comp.round3.train_b1',['--cache',str(cache),'--body',str(body),'--initialize',str(controller),'--horizon','32',
            '--name',modelname,'--updates','1500','--batch','64'],f'r3_v4_b1_G_r{iteration}',ROOT/f'runs/models/{modelname}_training.json',primary)
        controller=ROOT/f'runs/models/{modelname}_best.pt';summary['controller']=str(controller)
    status('低维在线循环完成；训练逐帧输出对照')
    frame='b1_r3_v4_perframe_h32'
    job('exec_comp.round3.train_b1',['--cache',str(cache),'--body',str(body),'--parameterization','per_frame','--horizon','32',
        '--name',frame,'--updates','1500','--batch','64'],'r3_v4_b1_perframe',ROOT/f'runs/models/{frame}_training.json',primary)
    summary['perframe_controller']=str(ROOT/f'runs/models/{frame}_best.pt')
    summary['conclusion']='在线循环完成；全量300条/10大脑/DR比较仍待完成，不能据训练误差认定B1优于A1。'
    status('training_complete_evaluation_pending')


if __name__=='__main__':
    import fcntl
    lock_path=ROOT/'runs/round3/b1_pipeline.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            path=ROOT/'runs/round3/b1_summary.json';summary=read(path) or {'rounds':[]}
            summary.update(status='failed_pipeline',conclusion='流程因明确错误停止；不自动重启实际训练。'+repr(exc))
            save_json(path,summary)
            raise

