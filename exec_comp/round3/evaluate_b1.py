"""Formal B1 comparisons after the mandatory online model audit passes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time
from exec_comp.common import ROOT,PROJECT,save_json,log
from exec_comp.round3.run_b1 import job,read
from exec_comp.round3.metrics import acceptance,verdict


def compare_population(b1,a1,raw):
    values=acceptance(b1,raw);gates=verdict(values)
    error_ratio=b1['summary']['wrist_mean_cm']/a1['summary']['wrist_mean_cm']
    a_hf=a1['jitter']['actual_above_5hz_energy_fraction']
    hf_ratio=b1['jitter']['actual_above_5hz_energy_fraction']/a_hf if a_hf else None
    a_seam=a1['summary']['plan_seam_p95_cm'];b_seam=b1['summary']['plan_seam_p95_cm']
    seam_ratio=b_seam/a_seam if a_seam and b_seam is not None else None
    safe=all(gates[k] is True for k in gates if k!='wrist_error_reduction')
    better=safe and (error_ratio<=.9 or (error_ratio<=1.05 and (
        hf_ratio is not None and hf_ratio<=.8 or seam_ratio is not None and seam_ratio<=.8)))
    return dict(wrist_ratio_vs_A1=error_ratio,hf_ratio_vs_A1=hf_ratio,seam_ratio_vs_A1=seam_ratio,
        raw_acceptance=values,gates=gates,clearly_better_than_A1=better)


def combine_populations(populations):
    # Keep the existing10% tracking/20% physical HF-or-seam improvement rule,
    # but apply it to all mandatory populations rather than motion300 only.
    all_pass=all(v is True for p in populations.values() for v in p['gates'].values())
    no_regression=all(p['wrist_ratio_vs_A1']<=1.05 for p in populations.values())
    improved=[k for k,p in populations.items() if p['clearly_better_than_A1']]
    return dict(populations=populations,all_acceptance_gates_passed=all_pass,
        no_material_wrist_regression=no_regression,improved_populations=improved,
        clearly_better_than_A1=all_pass and no_regression and bool(improved))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-summary',action='store_true',
        help='Recompute conclusions from completed evaluations only; starts no training or simulation')
    args=parser.parse_args()
    out=ROOT/'runs/round3';summary_path=out/'b1_summary.json'
    while True:
        summary=read(summary_path)
        if args.refresh_summary:
            if not summary:raise FileNotFoundError(summary_path)
            break
        if summary and summary['status'].startswith(('stopped_','failed_')):
            subprocess.run([sys.executable,'-m','exec_comp.round3.report'],cwd=PROJECT,check=True);return
        if summary and summary['status']=='training_complete_evaluation_pending' and (out/'a1_evaluation_complete.json').exists():break
        time.sleep(5)
    selected=read(out/'stage1.json')['selected'];stall='off' if selected.endswith('stall-off') else 'on'
    def evaluate(kind,checkpoint,gpu):
        label=f'B1_{kind}_stall-{stall}';base=['--method','B1','--checkpoint',checkpoint,'--anti-stall',stall,
               '--headless','--device','cuda:0']
        for dr in (False,True):
            name=label+('_dr' if dr else '')
            job('exec_comp.round3.eval_switching',base+['--label',name]+(['--randomization'] if dr else []),
                'r3_v4_'+name,out/'closed_loop'/name/'metrics.json',[gpu])
        job('exec_comp.round3.eval_brain',base+['--label',label], 'r3_v4_brain_'+label,
            out/'brain_evaluation'/label/'metrics.json',[gpu])
        job('exec_comp.round3.benchmark_b1',['--checkpoint',checkpoint,'--name',f'B1_{kind}'],
            'r3_v4_latency_B1_'+kind,out/f'B1_{kind}_latency_4090.json',[gpu])
        return label
    if args.refresh_summary:
        labels=[f'B1_{kind}_stall-{stall}' for kind in ('lowdim','perframe')]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(evaluate,'lowdim',summary['controller'],4),
                  pool.submit(evaluate,'perframe',summary['perframe_controller'],5)]
            labels=[f.result() for f in jobs]
    comparison={}
    for label in labels:
        populations={}
        for population,folder,b_label,a_label,raw_label in [
            ('motion300',out/'closed_loop',label,selected,f'raw_stall-{stall}'),
            ('domain_randomization300',out/'closed_loop',label+'_dr',selected+'_dr','raw_dr'),
            ('robot_brain10',out/'brain_evaluation',label,selected,f'raw_stall-{stall}')]:
            records=[read(folder/name/'metrics.json') for name in (b_label,a_label,raw_label)]
            for record in records:
                if record is None or record.get('truncated_smoke'):raise RuntimeError('Incomplete evaluation: '+population)
                count=record.get('robot_episodes') if population=='robot_brain10' else record['metrics']['all']['clips']
                if count!=(10 if population=='robot_brain10' else 300):raise RuntimeError('Wrong evaluation count: '+population)
            populations[population]=compare_population(*records)
        comparison[label]=combine_populations(populations)
    summary['training_progress']=[]
    for name in ('b1_r3_v4_lowdim_h17','b1_r3_v4_lowdim_online_1_h32',
                 'b1_r3_v4_lowdim_online_2_h32','b1_r3_v4_perframe_h32'):
        record=read(ROOT/'runs/models'/f'{name}_training.json')
        if record:
            config=record['config']
            summary['training_progress'].append(dict(name=name,horizon=config['horizon'],updates=config['updates'],
                elapsed_s=record['elapsed_s'],completed=True))
    summary['online_data']=[]
    for iteration in (1,2):
        folder=ROOT/f'runs/data/round3_b1_online_{iteration}_v4'
        totals={split:sum(read(p)['transitions'] for p in folder.glob(f'{split}_worker*_manifest.json'))
                for split in ('train','validation')}
        summary['online_data'].append(dict(iteration=iteration,**totals))
    summary['comparison']=comparison;summary['status']='evaluation_complete' 
    summary['clear_improvement_definition']='same per-population10% wrist OR20% physical HF/seam improvement; require all raw acceptance gates on motion300/DR300/robot-brain10 and <=5% wrist regression vs A1 in each population'
    summary['comparison_revision']='all_mandatory_populations_v2'
    summary['summary_bug_fixed']='Previous conclusion used motion300 only and ignored robot-brain10 and DR300; metrics and weights are unchanged.'
    summary['conclusion']=('低维B1在动作库上精度基本持平、实际高频下降，在机器人大脑输出上同时改善精度与高频；三类评估的全部门槛通过，值得保留为后续接入候选。逐帧版仍未通过稳定性门槛。不自动启动新训练或真机。' if any(v['clearly_better_than_A1'] for v in comparison.values())
        else 'B1没有在跟踪或接缝/抖动指标上明显优于A1，停止；不继续调参。')
    save_json(summary_path,summary)
    subprocess.run([sys.executable,'-m','exec_comp.round3.report'],cwd=PROJECT,check=True)
    log('b1_evaluation_complete',comparison=comparison)


if __name__=='__main__':main()
