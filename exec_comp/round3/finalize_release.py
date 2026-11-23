"""Finalize the disabled package after A1's formal selection and DR checks."""
import json
import sys
import subprocess
from exec_comp.common import ROOT,PROJECT,save_json


def main():
    stage=json.loads((ROOT/'runs/round3/stage2.json').read_text())
    selected=stage['selected'];mode='switch_event' if '_event_' in selected else 'stepwise'
    command=[sys.executable,'-m','exec_comp.round3.package_release','--refresh-metadata','--mode',mode]
    subprocess.run(command,cwd=PROJECT,check=True)
    save_json(ROOT/'runs/round3/a1_remote_package_complete.json',{'selected':selected,'mode':mode})
    subprocess.run([sys.executable,'-m','exec_comp.round3.report'],cwd=PROJECT,check=True)


if __name__=='__main__':main()
