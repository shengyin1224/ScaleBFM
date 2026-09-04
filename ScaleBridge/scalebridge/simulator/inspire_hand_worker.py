#!/usr/bin/env python3
"""DDS worker for ScaleBridge Inspire hands.

The ScaleBridge environment uses Python 3.11, while the tested Unitree DDS
stack on this machine lives in the TWIST2 Python 3.8 environment.  This small
process keeps that ABI boundary explicit.  It accepts JSON lines containing
12-DOF URDF IK targets, converts them with TWIST2's calibrated lookup table,
and publishes the resulting six motor commands per hand.

Nothing is published until the parent sends the first target.
"""

import argparse
import json
import sys
import time

import numpy as np


# MyScaleBFM/Pinocchio IK order:
#   index(2), middle(2), pinky(2), ring(2), thumb(yaw,pitch,intermediate,distal)
# TWIST2 dofpos2cmd order:
#   thumb(yaw,pitch,intermediate,distal), index(2), middle(2), ring(2), pinky(2)
IK_TO_TWIST2 = np.array([8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5], dtype=np.int64)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True)
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument("--mapping-path", required=True)
    parser.add_argument("--max-step", type=float, default=0.02)
    parser.add_argument("--state-timeout", type=float, default=8.0)
    return parser.parse_args()


def _validate_target(value, name):
    target = np.asarray(value, dtype=np.float64)
    if target.shape != (12,):
        raise ValueError("{} must contain 12 joint angles, got {}".format(name, target.shape))
    if not np.isfinite(target).all():
        raise ValueError("{} contains NaN or infinity".format(name))
    return target


def main():
    args = _parse_args()
    sys.path.insert(0, args.sdk_path)
    sys.path.insert(0, args.mapping_path)

    from dofpos2cmd import dofpos12_to_q6
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

    ChannelFactoryInitialize(0, args.net)
    publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
    publisher.Init()
    subscriber = ChannelSubscriber("rt/inspire/state", MotorStates_)
    subscriber.Init()
    command = MotorCmds_(
        cmds=[unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
    )

    deadline = time.monotonic() + args.state_timeout
    state = None
    while time.monotonic() < deadline:
        candidate = subscriber.Read()
        if candidate is not None and len(candidate.states) >= 12:
            state = candidate
            break
        time.sleep(0.01)
    if state is None:
        raise RuntimeError(
            "No rt/inspire/state received; verify inspire_g1 and network interface {}"
            .format(args.net)
        )

    # DDS layout is right[0:6], left[6:12].  inspire_g1 already contains this
    # robot's SWAPPED_20260802 serial mapping, so no additional side swap belongs
    # here.
    current_right = np.array([state.states[i].q for i in range(6)], dtype=np.float64)
    current_left = np.array([state.states[i + 6].q for i in range(6)], dtype=np.float64)
    current_left = np.clip(current_left, 0.0, 1.0)
    current_right = np.clip(current_right, 0.0, 1.0)
    print("READY", flush=True)
    print(
        "STATE " + json.dumps(
            {"left": current_left.tolist(), "right": current_right.tolist()},
            separators=(",", ":"),
        ),
        flush=True,
    )

    max_step = max(0.0, float(args.max_step))
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            left12 = _validate_target(payload["left"], "left")
            right12 = _validate_target(payload["right"], "right")
            target_left = np.clip(
                dofpos12_to_q6(left12[IK_TO_TWIST2]), 0.0, 1.0
            )
            target_right = np.clip(
                dofpos12_to_q6(right12[IK_TO_TWIST2]), 0.0, 1.0
            )

            # Rate-limit the COMMAND integrator only. current_* starts from the
            # measured pose (safe first ramp) and then advances max_step per
            # command regardless of how fast the hand follows. Anchoring each
            # step to the measured pose instead pinned the command 0.02 above
            # the actual position, which stalled or slow-crawled every grasp.
            if max_step > 0.0:
                current_left += np.clip(target_left - current_left, -max_step, max_step)
                current_right += np.clip(target_right - current_right, -max_step, max_step)
            else:
                current_left = target_left
                current_right = target_right

            for i in range(6):
                command.cmds[i].q = float(current_right[i])
                command.cmds[i + 6].q = float(current_left[i])
            publisher.Write(command)
            measured = subscriber.Read()
            if measured is not None and len(measured.states) >= 12:
                measured_right = np.clip(
                    np.array([measured.states[i].q for i in range(6)], dtype=np.float64),
                    0.0,
                    1.0,
                )
                measured_left = np.clip(
                    np.array([measured.states[i + 6].q for i in range(6)], dtype=np.float64),
                    0.0,
                    1.0,
                )
            else:
                measured_right = current_right
                measured_left = current_left
            print(
                "STATE " + json.dumps(
                    {"left": measured_left.tolist(), "right": measured_right.tolist()},
                    separators=(",", ":"),
                ),
                flush=True,
            )
        except Exception as exc:
            print("ERROR {}".format(exc), flush=True)


if __name__ == "__main__":
    main()
