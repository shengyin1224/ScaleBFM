"""Ordered A1-3 held-out HAT export and frozen-controller replay, never a robot run."""
import time
from exec_comp.common import ROOT,save_json
from exec_comp.launch import start,idle_gpus
from exec_comp.run_round2_a1 import status
from exec_comp.report_round2 import ROUND,read


def job(module,args,label,python=None):
    while not idle_gpus():time.sleep(10)
    process,record=start(module,args,idle_gpus()[0],label,python=python)
    if process.wait():raise RuntimeError(record['log'])


def main():
    stage=(read(ROUND/'stages.json') or {}).get('stage','')
    if not (stage.startswith('A1-3') or stage.startswith('大脑输出评估停止')):raise RuntimeError('A1-2 must finish before HAT validation')
    selected=read(ROUND/'selected_a1_after_dr.json') or read(ROUND/'selected_a1.json')
    status('A1-3：25条留出大脑输出，先做单帧链路核验')
    common=['--repo','/mnt/nvme0n1/human_policy_ckpt/isolated_20261021_no_left_wrist/src',
            '--bundle',str(ROOT/'assets/hat_brain_oct22'),'--fps','30']
    human_python='/home/aigc/miniconda/envs/human_policy/bin/python'
    job('exec_comp.export_hat',common+['--limit','1','--max-queries','1','--output','runs/round2/hat_predictions_smoke_v2'],'r2_hat_export_smoke_v2',python=human_python)
    status('A1-3：导出25条完整留出预测，禁止训练使用')
    job('exec_comp.export_hat',common,'r2_hat_export_full',python=human_python)
    status('A1-3：冻结小脑闭环回放raw/A1')
    while len(idle_gpus())<2:time.sleep(10)
    jobs=[]
    for method,gpu in zip(('raw','A1'),idle_gpus()[:2]):
        label='raw' if method=='raw' else selected['label']
        args=['--method',method,'--label',label,'--headless','--device','cuda:0']
        if method=='A1':
            if selected['checkpoint']:args+=['--checkpoint',selected['checkpoint']]
            if selected['lowpass']:args+=['--residual-lowpass']
        jobs.append(start('exec_comp.eval_hat',args,gpu,'r2_hat_eval_'+method))
    for process,record in jobs:
        if process.wait():raise RuntimeError(record['log'])
    raw=read(ROUND/'hat_evaluation/raw/metrics.json');a1=read(ROUND/'hat_evaluation'/selected['label']/'metrics.json')
    save_json(ROUND/'hat_gate.json',{'tracking_evaluation_complete':True,'episodes':25,'grasp_window_available':False,
                                  'deployment_rate_discrepancy':'Most recent online recording used15Hz; this experiment follows explicitly requested100@30Hz,3Hz replanning',
                                  'raw_initialization_failure_rate':raw['initialization_failure_rate'],'a1_initialization_failure_rate':a1['initialization_failure_rate']})
    status('A1-4：大脑输出跟踪评估完成，准备离线打包',**{'A1-3':'跟踪评估完成；抓取/FOCUS窗口N/A：25条原HDF无标签；不推算抓取成功率'})


if __name__=='__main__':
    try:main()
    except BaseException as e:status('大脑输出评估停止：'+str(e));raise
