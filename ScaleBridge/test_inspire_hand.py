#!/usr/bin/env python3
"""Standalone Inspire hand test: isolates whether the 0.02 measured-anchored
step limit is why the hands never move during policy runs.

Run with the robot powered on and hands connected (no controller needed):
    /home/nerv/miniconda3/envs/twist2/bin/python test_inspire_hand.py

Phase A replicates the worker exactly (command = measured + 0.02 toward target).
Phase B uses a free integrator (command += 0.02 toward target, ignores measured).
If A does not move but B does, the measured-anchored limiter is the bug.
"""
import sys
import time

import numpy as np

SDK_PATH = "/home/nerv/qingyaoxu/TWIST2/unitree_sdk2_python"
NET = "enp4s0"
RATE_HZ = 50.0
STEP = 0.02  # same as hand_max_step in real_world_inspire.yaml

sys.path.insert(0, SDK_PATH)
from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # noqa: E402


def read_q12(subscriber, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = subscriber.Read()
        if msg is not None and len(msg.states) >= 12:
            return np.clip(
                np.array([msg.states[i].q for i in range(12)], dtype=np.float64), 0.0, 1.0
            )
        time.sleep(0.01)
    return None


def publish_q12(publisher, command, q12):
    for i in range(12):
        command.cmds[i].q = float(q12[i])
    publisher.Write(command)


def run_phase(name, publisher, command, subscriber, target, seconds, anchored):
    """anchored=True replicates the worker; anchored=False lets the command integrate."""
    measured0 = read_q12(subscriber)
    if measured0 is None:
        print(f"[{name}] no state; aborting phase")
        return
    cmd = measured0.copy()
    n = int(seconds * RATE_HZ)
    stalled_cmd = None
    for k in range(n):
        if anchored:
            m = read_q12(subscriber, timeout_s=0.05)
            if m is not None:
                cmd = m
        cmd = cmd + np.clip(target - cmd, -STEP, STEP)
        publish_q12(publisher, command, cmd)
        if k == n - 1:
            stalled_cmd = cmd.copy()
        time.sleep(1.0 / RATE_HZ)
    time.sleep(0.3)
    measured1 = read_q12(subscriber)
    moved = np.abs(measured1 - measured0).max() if measured1 is not None else float("nan")
    print(f"[{name}] target={target[0]:.2f}  measured before={np.round(measured0, 3).tolist()}")
    print(f"[{name}] measured after ={np.round(measured1, 3).tolist()}")
    print(f"[{name}] last command   ={np.round(stalled_cmd, 3).tolist()}")
    print(f"[{name}] max measured change = {moved:.3f}  ({'MOVED' if moved > 0.05 else 'DID NOT MOVE'})")
    print()


def main():
    ChannelFactoryInitialize(0, NET)
    publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
    publisher.Init()
    subscriber = ChannelSubscriber("rt/inspire/state", MotorStates_)
    subscriber.Init()
    command = MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_() for _ in range(12)])

    q0 = read_q12(subscriber, timeout_s=8.0)
    if q0 is None:
        print("No rt/inspire/state received. Check robot power / inspire_g1 service / enp4s0.")
        sys.exit(1)
    print("Initial measured q12 (right[0:6], left[6:12]):", np.round(q0, 3).tolist())
    print()

    lo = np.zeros(12)   # one extreme of the normalized range
    hi = np.ones(12)    # the other extreme

    # Phase A: exact replica of the worker's measured-anchored limiter.
    run_phase("A worker-style -> 0", publisher, command, subscriber, lo, 6.0, anchored=True)
    run_phase("A worker-style -> 1", publisher, command, subscriber, hi, 6.0, anchored=True)

    # Phase B: same 0.02 step but free integrator (windup allowed).
    run_phase("B integrator  -> 0", publisher, command, subscriber, lo, 6.0, anchored=False)
    run_phase("B integrator  -> 1", publisher, command, subscriber, hi, 6.0, anchored=False)

    print("Done. If A did not move but B did, the measured-anchored limit is the bug.")


if __name__ == "__main__":
    main()
