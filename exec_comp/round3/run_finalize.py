"""Finalize A1 independently of the longer B1 loop; no robot is constructed."""
import json
import time
import subprocess
import sys
from exec_comp.common import ROOT,PROJECT,save_json
from exec_comp.round3.run_b1 import job


def main():
    out=ROOT/'runs/round3'
    while not (out/'stage2.json').exists():
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if status.get('failed'):raise RuntimeError(status['error'])
        time.sleep(5)
    job('exec_comp.round3.finalize_release',[], 'r3_v4_a1_finalize_package',
        out/'a1_remote_package_complete.json',[5,6,7])
    while not (out/'brain_stage_complete.json').exists():time.sleep(5)
    subprocess.run([sys.executable,'-m','exec_comp.round3.report'],cwd=PROJECT,check=True)
    save_json(out/'a1_evaluation_complete.json',{'stage2':json.loads((out/'stage2.json').read_text()),
        'brain_configurations':6,'real_robot_run':False,
        'local_remaining':'sync results, select mode and refresh4080S manifest/config/latency; no auto-deploy'})


if __name__=='__main__':main()
