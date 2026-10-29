"""Hardware-free checks for silent controller-linked shutdown."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from scalebridge.online.controller_watchdog import (
    ControllerWatchdog, _alive, _collect_children, _processes,
)


ROOT = Path(__file__).resolve().parents[1]


class ControllerWatchdogTests(unittest.TestCase):
    def test_pid_reuse_is_an_exit(self):
        self.assertFalse(_alive(12, 100, {12: (1, 101, 'controller')}))
        self.assertTrue(_alive(12, 100, {12: (1, 100, 'controller')}))

    def test_collects_only_own_descendants_and_retains_orphans(self):
        tracked = {}
        snapshot = {20: (10, 100, ''), 30: (20, 101, ''), 40: (1, 102, '')}
        _collect_children(snapshot, tracked, 10)
        self.assertEqual(tracked, {20: 100, 30: 101})
        _collect_children({30: (1, 101, '')}, tracked, 10)
        self.assertEqual(tracked[30], 101)

    def test_start_stop_is_silent_without_controller(self):
        watcher = ControllerWatchdog().start()
        watcher.stop()
        self.assertFalse(watcher.shutdown_requested.is_set())

    def test_two_apps_exit_silently_and_kill_stubborn_children(self):
        if any(Path(info[2]).name == 'g1_29dof_controller'
               for info in _processes().values()):
            self.skipTest('A real controller is running')
        apps = []
        child_pids = []
        controller = unrelated = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                executable = root / 'g1_29dof_controller'
                executable.symlink_to('/bin/sleep')
                unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                # Apps can be started before the controller exists.
                for index in range(2):
                    ready = root / ('ready%d' % index)
                    finalized = root / ('finalized%d' % index)
                    code = '''
import signal, subprocess, sys, time
from pathlib import Path
from scalebridge.online.controller_watchdog import ControllerWatchdog
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'],
    stdout=subprocess.PIPE, text=True)
child.stdout.readline()
watcher = ControllerWatchdog().start()
def handle(signum, frame):
    raise KeyboardInterrupt
if sys.argv[3] == '1':
    signal.signal(signal.SIGTERM, handle)
Path(sys.argv[1]).write_text(str(child.pid))
try:
    while True:
        time.sleep(.05)
except KeyboardInterrupt:
    pass
finally:
    Path(sys.argv[2]).write_text('closed')
'''
                    app = subprocess.Popen(
                        [sys.executable, '-B', '-c', code, str(ready), str(finalized), str(index)],
                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True,
                    )
                    apps.append(app)
                    deadline = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(ready.exists())
                    child_pids.append(int(ready.read_text()))
                controller = subprocess.Popen([str(executable), '60'])
                time.sleep(.4)
                self.assertTrue(all(app.poll() is None for app in apps))
                controller.kill()
                controller.wait()
                for app in apps:
                    stdout, stderr = app.communicate(timeout=8)
                    self.assertEqual((app.returncode, stdout, stderr), (0, '', ''))
                self.assertTrue(all((root / ('finalized%d' % i)).exists() for i in range(2)))
                deadline = time.monotonic() + 2
                while any(pid in _processes() for pid in child_pids) and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(all(pid not in _processes() for pid in child_pids))
                self.assertIsNone(unrelated.poll())
        finally:
            for proc in apps + [controller, unrelated]:
                if proc is not None:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
            for pid in child_pids:
                if pid in _processes():
                    os.kill(pid, signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
