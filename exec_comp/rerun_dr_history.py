"""Correct actuator-delay history semantics and rerun the paired robustness gate."""
import time
from exec_comp.common import ROOT,save_json
from exec_comp.launch import start,idle_gpus
from exec_comp.report_round2 import ROUND,read,wrist
from exec_comp.run_round2_a1 import status


def main():
    old=ROUND/'a1_robustness';archive=ROUND/'diagnostics/a1_robustness_delayed_history'
    archive.parent.mkdir(exist_ok=True)
    if old.exists():old.rename(archive)
    status('A1-2：修正动作历史契约后复验',**{'A1-2':'待复验：延迟只影响物理动作，policy历史保留即时发送指令；旧DR结果归档'})
    selected=read(ROUND/'selected_a1.json')
    while len(idle_gpus())<2:time.sleep(10)
    available=idle_gpus()[:2];jobs=[]
    for method,gpu in zip(('raw','A1'),available):
        label='raw' if method=='raw' else selected['label']
        args=['--method',method,'--label',label,'--output-root','runs/round2/a1_robustness','--detailed','--randomization','--headless','--device','cuda:0']
        if method=='A1':
            if selected['checkpoint']:args+=['--checkpoint',selected['checkpoint']]
            if selected['lowpass']:args+=['--residual-lowpass']
        jobs.append(start('exec_comp.eval_closed_loop',args,gpu,'r2_dr_sent_history_'+method))
    for process,job in jobs:
        if process.wait():raise RuntimeError(job['log'])
    raw=read(ROUND/'a1_robustness/raw/metrics.json');a1=read(ROUND/'a1_robustness'/selected['label']/'metrics.json')
    for key in ('stiffness','damping','torso_com_b','action_delay_control_steps'):
        assert raw['domain_randomization'][key]==a1['domain_randomization'][key],key
    nominal=next(read(p) for p in (ROUND/'a1_jitter'/selected['label']/'metrics.json',ROUND/'a1_smoothing'/selected['label']/'metrics.json') if p.exists())
    baseline=read(ROUND/'a1_jitter/raw/metrics.json')
    reduction=1-wrist(nominal)/wrist(baseline);robust=1-wrist(a1)/wrist(raw);passed=robust>=.5*reduction
    save_json(ROUND/'robustness_gate.json',{'passed':passed,'nominal_reduction':reduction,'randomized_reduction':robust,'selected':selected['label'],'history_contract':'sent actions before physical delay'})
    status('A1-3：待评估大脑输出' if passed else 'A1-2：需补采DR训练数据',**{'A1-2':('通过' if passed else '未通过')+f'：无DR降幅{reduction:.2%}，正确延迟契约DR降幅{robust:.2%}'})


if __name__=='__main__':
    try:main()
    except BaseException as e:status('DR复验停止：'+str(e));raise
