"""Wait for an idle4080S; benchmark only our artifact, never stop any process."""
import fcntl
import subprocess
import sys
import time
from exec_comp.common import ROOT,PROJECT,save_json,log
from exec_comp.round4.benchmark import idle


def main():
    folder=ROOT/'runs/round4'
    with (folder/'latency_watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for variant in ('resident','graph'):
            if (folder/f'b1_latency_{variant}_4080s.json').exists():continue
            while not idle():
                save_json(folder/'latency_watch_status.json',dict(stage='waiting for idle4080S',variant=variant,stopped_user_process=False))
                time.sleep(15)
            save_json(folder/'latency_watch_status.json',dict(stage='benchmark',variant=variant,stopped_user_process=False))
            subprocess.run([sys.executable,'-m','exec_comp.round4.benchmark','--artifact',str(ROOT/'releases/round4/b1/b1_correct_kp.pt'),'--variant',variant],cwd=PROJECT,check=True)
            subprocess.run([sys.executable,'-m','exec_comp.round4.report'],cwd=PROJECT,check=True)
        save_json(folder/'latency_watch_status.json',dict(stage='completed',stopped_user_process=False))
        log('r4_latency_watch_complete')


if __name__=='__main__':main()
