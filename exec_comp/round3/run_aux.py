"""A1 brain checks take priority on a newly free card, then allow B1 there."""
import sys
import subprocess
from exec_comp.common import ROOT,PROJECT,save_json
from exec_comp.round3.run_b1 import job


def main():
    for method,mode,stall in [('raw','stepwise','on'),('raw','stepwise','off'),
            ('A1','stepwise','on'),('A1','stepwise','off'),('A1','switch_event','on'),('A1','switch_event','off')]:
        label=f'raw_stall-{stall}' if method=='raw' else f'A1_{"event" if mode=="switch_event" else mode}_stall-{stall}'
        job('exec_comp.round3.eval_brain',['--method',method,'--mode',mode,'--anti-stall',stall,
            '--label',label,'--headless','--device','cuda:0'],'r3_v4_brain_'+label,
            ROOT/'runs/round3/brain_evaluation'/label/'metrics.json',[4])
    save_json(ROOT/'runs/round3/brain_stage_complete.json',{'configurations':6,'robot_episodes_each':10})
    result=subprocess.run([sys.executable,'-m','exec_comp.round3.run_b1','--gpus','4'],cwd=PROJECT)
    if result.returncode:raise RuntimeError('B1 driver failed; inspect this log and last job log')


if __name__=='__main__':main()
