"""Mirror remote R4 progress without using the occupied local GPU."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from exec_comp.common import ROOT,PROJECT,log

REMOTE='/root/shengyin/ScaleBFM/exec_comp/runs/round4'


def fingerprint():
    source=f'''from pathlib import Path
import json
root=Path({REMOTE!r})
files=[*root.glob('**/metrics.json'),*root.glob('*analysis.json'),*root.glob('*selection.json'),root/'status.json']
print(json.dumps([(str(p.relative_to(root)),p.stat().st_mtime_ns) for p in files if p.exists() and 'diagnostics' not in p.parts and 'smoke' not in p.parts]))'''
    return subprocess.check_output(['ssh','-o','ConnectTimeout=15','4090-168',shlex.join(['python3','-c',source])],text=True,timeout=45)


def main():
    destination=ROOT/'runs/round4';destination.mkdir(parents=True,exist_ok=True);last=None
    with (destination/'sync_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                current=fingerprint()
                if current!=last:
                    subprocess.run(['rsync','-a','--exclude=diagnostics/','--exclude=smoke/','--include=*/',
                        '--include=*.json','--exclude=*','4090-168:'+REMOTE+'/',str(destination)+'/'],check=True)
                    subprocess.run([sys.executable,'-m','exec_comp.round4.report'],cwd=PROJECT,check=True);last=current
            except (subprocess.SubprocessError,json.JSONDecodeError) as exc:log('r4_sync_retry',error=repr(exc))
            time.sleep(45)


if __name__=='__main__':main()
