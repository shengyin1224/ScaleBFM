"""Update the R4 report on new completed artifacts; never waits on a GPU."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from exec_comp.common import ROOT,PROJECT,save_json


def main():
    root=ROOT/'runs/round4';root.mkdir(parents=True,exist_ok=True);last=None
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            complete=root/'audit_evaluations_complete.json';analysis=root/'fallback_analysis.json'
            b1_baseline_ready=all((root/folder/label/'fallback_events.jsonl').exists() for folder,label in (
                ('brain_evaluation','b1_lowdim'),('closed_loop','b1_lowdim'),('closed_loop','b1_lowdim_dr')))
            if (complete.exists() or b1_baseline_ready) and (not analysis.exists() or not json.loads(analysis.read_text()).get('complete')):
                subprocess.run([sys.executable,'-m','exec_comp.round4.analyze_fallback'],cwd=PROJECT,check=True)
            b1_all_ready=all((root/folder/(label+suffix)/'metrics.json').exists()
                for folder,suffix in (('brain_evaluation',''),('closed_loop',''),('closed_loop','_dr'))
                for label in ('b1_lowdim','b1_perframe','b1_lowdim_fallback_off','b1_lowdim_limited'))
            if b1_all_ready and not (root/'b1_r4_evaluations_complete.json').exists():
                save_json(root/'b1_r4_evaluations_complete.json',dict(complete=True,
                    note='B1 evaluations available independently of remaining A1 jobs; inspect gates/transition diagnostic before deployment',real_robot_run=False))
            files=[*root.glob('*/**/metrics.json'),*root.glob('*selection.json'),*root.glob('*analysis.json'),root/'status.json']
            key=tuple((str(p),p.stat().st_mtime_ns) for p in files if p.exists())
            if key!=last:
                subprocess.run([sys.executable,'-m','exec_comp.round4.report'],cwd=PROJECT,check=True);last=key
            time.sleep(30)


if __name__=='__main__':main()
