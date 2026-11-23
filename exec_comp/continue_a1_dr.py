"""Conditional2M DR augmentation after the corrected paired robustness test."""
import time
from exec_comp.common import ROOT,save_json,log
from exec_comp.report_round2 import ROUND,read,wrist,passes_jitter
from exec_comp.run_round2_a1 import run,evaluate,status


def main():
    while True:
        gate=read(ROUND/'robustness_gate.json')
        if gate and gate.get('history_contract')=='sent actions before physical delay':break
        state=read(ROUND/'stages.json') or {}
        if state.get('stage','').startswith('DR复验停止'):raise RuntimeError(state['stage'])
        time.sleep(10)
    if gate['passed']:
        log('dr_augmentation_not_triggered',gate=gate);return
    selected=read(ROUND/'selected_a1.json')
    status('A1-2：补采2M训练DR transition',**{'A1-2':'未通过初次正确契约DR门槛，按文档补采并重训'})
    for split,worker,n,transitions in (('train',10,1024,2_000_000),('validation',11,300,100_200)):
        run('exec_comp.collect',['--mode','collect','--split',split,'--worker',str(worker),'--num-envs',str(n),
            '--transitions',str(transitions),'--label','round2_dr','--startup-prefix','validated','--randomization',
            '--headless','--device','cuda:0'],'r2_collect_dr_correct_'+split)
    status('A1-2：混合原始/DR数据继续训练A1')
    data=ROOT/'runs/data/round2_dr';initialize=selected['checkpoint'] or str(ROOT/'runs/models/inverse_best.pt')
    run('exec_comp.train_inverse',['--updates','10000','--batch','1024','--name','inverse_dr_r2','--initialize',initialize,
        '--residual-smooth-weight','.1','--extra-train-data',str(data),'--extra-validation-data',str(data),'--balance-data-roots'],'r2_train_inverse_dr')
    checkpoint=ROOT/'runs/models/inverse_dr_r2_best.pt';name='A1_dr_lowpass' if selected['lowpass'] else 'A1_dr'
    nominal=evaluate('A1',name,'a1_dr_retrain',checkpoint=checkpoint,lowpass=selected['lowpass'])
    robust=evaluate('A1',name,'a1_robustness',checkpoint=checkpoint,lowpass=selected['lowpass'],dr=True)
    raw=read(ROUND/'a1_jitter/raw/metrics.json');dr_raw=read(ROUND/'a1_robustness/raw/metrics.json')
    reduction=1-wrist(nominal)/wrist(raw);dr_reduction=1-wrist(robust)/wrist(dr_raw)
    passed=dr_reduction>=.5*reduction;jitter_ok=passes_jitter(nominal,raw)
    save_json(ROUND/'dr_retrain_gate.json',{'passed':passed,'nominal_reduction':reduction,'randomized_reduction':dr_reduction,'jitter_passed':jitter_ok})
    # This is a second independent candidate, never an overwrite of original A1.
    chosen={'label':name,'checkpoint':str(checkpoint),'lowpass':selected['lowpass']}
    save_json(ROUND/'selected_a1_after_dr.json',chosen)
    status('A1-3：DR补采重训完成，准备大脑输出验证',**{
        'A1-1':('通过' if jitter_ok else '未通过')+'：DR新权重重新做全量抖动验收',
        'A1-2':('通过' if passed else '未通过')+f'：补采后无DR降幅{reduction:.2%}，DR降幅{dr_reduction:.2%}；失败比例另列，不能只凭误差宣称稳定'})


if __name__=='__main__':
    try:main()
    except BaseException as e:status('DR补采/重训停止：'+str(e));raise
