"""Silent process-local shutdown when the observed G1 controller exits."""

import os
from pathlib import Path
import signal
import threading
import time


def _processes():
    """Return PID -> (parent PID, start ticks, argv[0]) for our user's tasks."""
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            argv0 = (path / 'cmdline').read_bytes().split(b'\0', 1)[0]
            result[int(path.name)] = (
                int(stat[1]), int(stat[19]), os.fsdecode(argv0),
            )
        except (OSError, ValueError, IndexError):
            continue
    return result


def _alive(pid, started, snapshot):
    return pid in snapshot and snapshot[pid][1] == started


def _collect_children(snapshot, tracked, owner_pid):
    parents = {owner_pid}
    parents.update(pid for pid, started in tracked.items()
                   if _alive(pid, started, snapshot))
    while True:
        added = set()
        for pid, (ppid, started, _) in snapshot.items():
            if ppid in parents and pid not in parents:
                tracked[pid] = started
                added.add(pid)
        if not added:
            return
        parents.update(added)


def _signal_children(tracked, sig):
    snapshot = _processes()
    for pid, started in tracked.items():
        if _alive(pid, started, snapshot):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def _exit_on_sigterm(signum, frame):
    raise SystemExit(0)


class ControllerWatchdog:
    """Arm after seeing one controller; its PID/start time define the session.

    No controller at startup is allowed, preserving independent/offline use.
    This object writes nothing to stdout/stderr or logging.
    """

    def __init__(self, poll_s=0.1, grace_s=2.0):
        self.poll_s = poll_s
        self.grace_s = grace_s
        self.shutdown_requested = threading.Event()
        self._stopped = threading.Event()
        self._children = {}
        self._owner_pid = os.getpid()
        self._thread = threading.Thread(
            target=self._watch, name='g1-controller-watchdog', daemon=True,
        )

    def start(self):
        # SystemExit lets Python wait for the non-daemon shutdown thread,
        # which also cleans up children that ignore SIGTERM.
        if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
            signal.signal(signal.SIGTERM, _exit_on_sigterm)
        self._thread.start()
        return self

    def stop(self):
        self._stopped.set()
        self._thread.join()

    def _watch(self):
        controller = None
        while not self._stopped.is_set():
            snapshot = _processes()
            _collect_children(snapshot, self._children, self._owner_pid)
            if controller is None:
                matches = [(pid, info[1]) for pid, info in snapshot.items()
                           if Path(info[2]).name == 'g1_29dof_controller']
                if len(matches) == 1:
                    controller = matches[0]
            elif not _alive(controller[0], controller[1], snapshot):
                self.shutdown_requested.set()
                threading.Thread(
                    target=self._shutdown, name='g1-controller-shutdown',
                    daemon=False,
                ).start()
                return
            self._stopped.wait(self.poll_s)

    def _shutdown(self):
        _collect_children(_processes(), self._children, self._owner_pid)
        _signal_children(self._children, signal.SIGTERM)
        os.kill(self._owner_pid, signal.SIGTERM)
        # Allow application finalizers (including video/recording) to run.
        deadline = time.monotonic() + self.grace_s
        while time.monotonic() < deadline:
            _collect_children(_processes(), self._children, self._owner_pid)
            time.sleep(self.poll_s)
        _signal_children(self._children, signal.SIGKILL)
        # Also handles a main thread stuck in native inference or I/O.
        os._exit(0)
