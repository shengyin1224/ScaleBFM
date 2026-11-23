"""Ordered A1 jitter remedies and paired robustness tests on idle GPUs only."""
import json
import time
from pathlib import Path
from exec_comp.common import ROOT,save_json,log
from exec_comp.launch import start,idle_gpus
from exec_comp.run_pipeline import alive
from exec_comp.report_round2 import ROUND,read,make_report,passes_jitter,jitter_ratios,wrist


def status(stage,**gates):
    state=read(ROUND/'stages.json') or {'gates':{}}
    state['stage']=stage;state['gates'].update(gates)
    save_json(ROUND/'stages.json',state);make_report();log('round2_stage',**state)


def wait_existing(method):
    path=ROUND/f'a1_jitter/{method}/metrics.json'
    job=read(ROOT/f'runs/logs/r2_jitter_{method}_job.json')
    while not path.exists():
        if not alive(job):raise RuntimeError(f"{method} jitter replay failed: {job['log']}")
        time.sleep(10)
    return read(path)


def run(module,args,label):
    while not idle_gpus():time.sleep(10)
    process,job=start(module,args,idle_gpus()[0],label)
    code=process.wait()
    if code:raise RuntimeError(f"{label} failed: {job['log']}")
    make_report()


def evaluate(method,label,folder,checkpoint=None,lowpass=False,dr=False,smoke=False):
    args=['--method',method,'--label',label,'--output-root',f'runs/round2/{folder}','--detailed','--headless','--device','cuda:0']
    if checkpoint:args+=['--checkpoint',str(checkpoint)]
    if lowpass:args+=['--residual-lowpass']
    if dr:args+=['--randomization']
    if smoke:args+=['--limit','2','--max-steps','30']
    run('exec_comp.eval_closed_loop',args,'r2_'+folder+'_'+label)
    return read(ROUND/f'{folder}/{label}/metrics.json')


def main():
    status('A1-1：原始权重全量抖动验收')
    raw=wait_existing('raw');original=wait_existing('A1')
    candidates=[{'label':'A1','checkpoint':None,'lowpass':False,'result':original}]
    if not passes_jitter(original,raw):
        status('A1-1(a)：20ms残差低通',**{'A1-1':'未通过：原始A1有抖动指标超过raw的1.5倍；依次执行(a)/(b)'})
        filtered=evaluate('A1','A1_lowpass','a1_smoothing',lowpass=True)
        candidates.append({'label':'A1_lowpass','checkpoint':None,'lowpass':True,'result':filtered})
        status('A1-1(b)：带残差时间平滑损失重训')
        run('exec_comp.train_inverse',['--updates','10000','--batch','1024','--name','inverse_smooth_r2','--residual-smooth-weight','.1'],'r2_train_inverse_smooth')
        checkpoint=ROOT/'runs/models/inverse_smooth_r2_best.pt'
        trained=evaluate('A1','A1_smooth_loss','a1_smoothing',checkpoint=checkpoint)
        candidates.append({'label':'A1_smooth_loss','checkpoint':str(checkpoint),'lowpass':False,'result':trained})
        if not passes_jitter(trained,raw):
            both=evaluate('A1','A1_smooth_loss_lowpass','a1_smoothing',checkpoint=checkpoint,lowpass=True)
            candidates.append({'label':'A1_smooth_loss_lowpass','checkpoint':str(checkpoint),'lowpass':True,'result':both})
    eligible=[c for c in candidates if passes_jitter(c['result'],raw)]
    selected=min(eligible or candidates,key=lambda c:wrist(c['result']))
    save_json(ROUND/'selected_a1.json',{k:v for k,v in selected.items() if k!='result'})
    gate=('通过：'+selected['label']+'全部三项≤1.5倍' if eligible else '未通过：原始/低通/重训方案仍无全部指标≤1.5倍的版本；不视为可上真机')
    status('A1-2：域随机化环境短测',**{'A1-1':gate})
    evaluate('raw','raw','dr_smoke',dr=True,smoke=True)
    status('A1-2：相同随机参数300段raw对照')
    robust_raw=evaluate('raw','raw','a1_robustness',dr=True)
    status('A1-2：当前选择的A1权重300段对照')
    robust_a1=evaluate('A1',selected['label'],'a1_robustness',checkpoint=selected['checkpoint'],lowpass=selected['lowpass'],dr=True)
    for key in ('stiffness','damping','torso_com_b','action_delay_control_steps','material_friction_min','material_friction_max'):
        assert robust_raw['domain_randomization'][key]==robust_a1['domain_randomization'][key],f'Unpaired randomization: {key}'
    reduction=1-wrist(selected['result'])/wrist(raw)
    robust_reduction=1-wrist(robust_a1)/wrist(robust_raw)
    passed=robust_reduction>=.5*reduction
    save_json(ROUND/'robustness_gate.json',{'passed':passed,'nominal_reduction':reduction,'randomized_reduction':robust_reduction,'selected':selected['label']})
    if passed:status('A1-3：等待大脑输出评估',**{'A1-2':f'通过：无DR降幅{reduction:.2%}，DR降幅{robust_reduction:.2%}，保留超过一半'})
    else:status('A1-2：需补采DR数据并重训',**{'A1-2':f'未通过：无DR降幅{reduction:.2%}，DR降幅{robust_reduction:.2%}；需要补采2M–5M'})


if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('停止：'+str(e));raise
