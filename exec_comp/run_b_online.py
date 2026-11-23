"""Two gated simulation-only online data/refinement rounds, then held-out replay."""
import json
import traceback
import time
import fcntl
from pathlib import Path
from exec_comp.common import ROOT,save_json,log
from exec_comp.launch import start as launch_start,wait
from exec_comp.report_round2 import make_report,passes_jitter


def read(path):return json.loads(path.read_text())


class ExistingProcess:
    def __init__(self,job):self.job=job
    def wait(self):
        job=self.job;proc=Path(f"/proc/{job['pid']}")
        while proc.exists():
            try:
                cmd=(proc/'cmdline').read_bytes().split(b'\0')
                if job['module'].encode() not in cmd:break
                if (proc/'stat').read_text().split(') ')[1].startswith('Z'):break
            except FileNotFoundError:break
            time.sleep(1)
        text=Path(job['log']).read_text(errors='replace')
        completed=any(f'"event": "{event}"' in text for event in ('collection_complete','model_training_complete','body_exec_validation','closed_loop_complete'))
        return 0 if completed and 'Traceback (most recent call last)' not in text else 1


def start(module,args,gpu,label):
    saved=ROOT/f'runs/logs/{label}_job.json'
    if saved.exists():
        job=read(saved)
        if job['module']!=module or list(map(str,job['args']))!=list(map(str,args)):raise ValueError('Resume job arguments differ: '+label)
        log('resume_existing_job',label=label,pid=job['pid']);return ExistingProcess(job),job
    return launch_start(module,args,gpu,label)


def stage(text,gate=None):
    path=ROOT/'runs/round2/stages.json';s=read(path);s['stage']=text
    if gate:s['gates'].update(gate)
    save_json(path,s);make_report();log('b_pipeline_stage',stage=text)


def validate(checkpoint,name,gpu,folder=None,split='validation'):
    args=['--checkpoint',str(checkpoint),'--name',name,'--split',split]
    if folder:args+=['--data-root',str(folder)]
    return start('exec_comp.eval_body_exec',args,gpu,'r2_'+name)


def endpoint(result):
    world=result['horizons']['17']['world_keypoints_cm']
    return {k:world[k]['median'] for k in ('left_wrist','right_wrist')}


def main():
    models=ROOT/'runs/models';roundroot=ROOT/'runs/round2'
    checkpoint=models/'body_privileged_best.pt';gate=roundroot/'body_validation/original_privileged.json'
    if not read(gate)['world_gate_passed']:raise RuntimeError('Initial privileged model gate failed')
    candidates=[('initial',checkpoint,gate)];roots=[];rounds=[]
    stage('B-4：仅通过门槛的68维无噪声特权模型，两轮训练片段在线循环',{'B-2':'近真机分支未通过：8→17步各12000更新完成，64维世界平移N/A；68维根速度±0.1m/s的左右腕中位数0.988/1.138cm，停止该分支。B-4只评估已过门槛的无噪声特权分支。','B-3':'特权17步优化及两种初始化完成；CUDA Graph前向/梯度一致，N=1约133–136ms，N=5/10超过333ms预算；后续仿真用N=1'})
    for iteration in (1,2):
        stage(f'B-4第{iteration}轮：在训练片段采集约1M transitions与10万独立回放')
        folder=ROOT/f'runs/data/round2_b_online_{iteration}';roots.append(folder)
        common=['--headless','--mode','collect','--split','train','--startup-prefix','validated',
                '--label',f'round2_b_online_{iteration}','--execution-checkpoint',str(checkpoint),
                '--execution-gate',str(gate),'--execution-init','mixed','--execution-iterations','1','--action-noise','.05']
        if iteration==1:
            trains=[start('exec_comp.collect',common+['--num-envs','128','--transitions','1000064','--worker','10'],5,'r2_b4_round1_train')]
        else:
            # Same controller/data protocol; independent workers share the
            # round's1M budget, so the third idle GPU contributes collection.
            trains=[start('exec_comp.collect',common+['--num-envs','128','--transitions','500032','--worker',str(worker)],gpu,f'r2_b4_round2_train_{worker}') for gpu,worker in ((5,20),(7,22))]
        validation=start('exec_comp.collect',common+['--num-envs','64','--transitions','100032','--worker',str(iteration*10+1),'--data-split','validation'],6,f'r2_b4_round{iteration}_validation')
        wait(*validation)
        before=validate(checkpoint,f'b_round{iteration}_before_on_B',6,folder)
        wait(*before)
        for train in trains:wait(*train)
        name=f'body_b_online_r{iteration}';args=['--privileged','--updates','6000','--batch','512','--body-horizon','17','--project-gravity','--initialize',str(checkpoint),'--name',name,'--balance-data-roots']
        for root in roots:args+=['--extra-train-data',str(root),'--extra-validation-data',str(root)]
        stage(f'B-4第{iteration}轮：约1M训练transitions采集完成，微调17步世界模型')
        training=start('exec_comp.train_body_wm',args,5,f'r2_b4_round{iteration}_refine');wait(*training)
        new_checkpoint=models/f'{name}_best.pt'
        ordinary=validate(new_checkpoint,f'b_round{iteration}_ordinary_train',7,split='train')
        after=validate(new_checkpoint,f'b_round{iteration}_after_on_B',6,folder)
        standard=validate(new_checkpoint,f'b_round{iteration}_standard',5)
        for job in (ordinary,after,standard):wait(*job)
        base=roundroot/'body_validation';normal=read(base/f'b_round{iteration}_ordinary_train.json');on_b=read(base/f'b_round{iteration}_after_on_B.json')
        n=endpoint(normal);b=endpoint(on_b);ratios={k:b[k]/max(n[k],1e-8) for k in n}
        new_gate=base/f'b_round{iteration}_standard.json';passed=read(new_gate)['world_gate_passed']
        rounds.append({'iteration':iteration,'train_folder':str(folder),'checkpoint':str(new_checkpoint),
                       'ordinary_train_median_cm':n,'B_trajectory_median_cm':b,'B_over_ordinary':ratios,'standard_gate_passed':passed})
        save_json(roundroot/'b_online_progress.json',{'rounds':rounds,'close_definition':'both wrists B/ordinary median <=2; this operational threshold was fixed before online collection', 'simulation_privileged':True})
        if not passed:
            stage(f'B-4第{iteration}轮停止：微调模型标准验证17步门槛失效',{'B-4':'未通过：新模型世界系腕部17步中位数超过1cm；不进入B-5，保留诊断数据'})
            return
        checkpoint=new_checkpoint;gate=new_gate;candidates.append((f'round{iteration}',checkpoint,gate))
    if any(v>2 for v in rounds[-1]['B_over_ordinary'].values()):
        stage('B-4停止：两轮后B轨迹误差仍超过普通训练分布2倍',{'B-4':'未通过：两轮后B轨迹世界系腕部误差仍远高于普通训练分布；B-5按要求停止'})
        return
    stage('B-5：两轮在线循环通过分布差距检查，开始300片段闭环与抖动评估',{'B-4':'通过仿真循环检查：两轮各约1M train transitions，最终B轨迹两腕误差均不超过普通训练分布2倍；仅特权分支'})
    cases=[(label,checkpoint,gate,initial) for label,checkpoint,gate in candidates for initial in ('zero','A1')]
    for offset in range(0,len(cases),3):
        jobs=[]
        for gpu,(label,checkpoint,gate,initial) in zip((5,6,7),cases[offset:offset+3]):
            jobs.append(start('exec_comp.eval_closed_loop',['--headless','--method','B_exec','--limit','300','--iterations','1','--detailed',
                         '--checkpoint',str(checkpoint),'--execution-gate',str(gate),'--execution-init',initial,
                         '--output-root','runs/round2/b_closed_loop','--label',f'B_{label}_{initial}'],gpu,f'r2_b5_{label}_{initial}'))
        for job in jobs:wait(*job)
        make_report()
    raw=read(roundroot/'a1_jitter/raw/metrics.json')
    results=[read(p) for p in (roundroot/'b_closed_loop').glob('*/metrics.json')]
    stable=all(passes_jitter(d,raw) for d in results)
    verdict='通过抖动口径检查（仍仅特权仿真）' if stable else '未通过稳定性：实际腕部高频或命令二阶差分/HF超过raw的1.5倍'
    stage('第二轮离线实验执行完成：原始及两轮B模型各完成两种初始化、每组300片段；A1及近真机B仍未通过全部门槛',{'B-5':verdict+'；六组完整闭环与误差/抖动/失败率均已保存'})


if __name__=='__main__':
    lock=(ROOT/'runs/round2/b_pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception as exc:
        traceback.print_exc();stage('B流程因明确错误停止，未自动重启：'+repr(exc));raise
