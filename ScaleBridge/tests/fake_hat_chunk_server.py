"""Deterministic no-camera HAT peer for online ScaleBridge integration tests."""

import argparse
import time

import numpy as np

from scalebridge.online.protocol import make_hat_chunk, monotonic_ns, validate_robot_state
from scalebridge.online.transport import LatestPublisher, LatestSubscriber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--chunk-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--period", type=float, default=0.2)
    args = parser.parse_args()
    states = LatestSubscriber(args.state_endpoint, bind=False)
    chunks = LatestPublisher(args.chunk_endpoint, bind=True)
    sequence = 0
    try:
        while True:
            state = states.receive_blocking(timeout_ms=1000)
            if state is None:
                continue
            state = validate_robot_state(state)
            n = 64
            root = np.repeat(state["root_pos_world"][None], n, axis=0)
            root_quat = np.repeat(state["root_quat_world_wxyz"][None], n, axis=0)
            head = root + np.array([0.0, 0.0, 0.6], dtype=np.float32)
            left = root + np.array([0.15, 0.25, 0.35], dtype=np.float32)
            right = root + np.array([0.15, -0.25, 0.35], dtype=np.float32)
            # A finite, spread open-hand target used only to exercise online IK.
            tips = np.array([
                [0.02, -0.05, 0.01], [0.09, -0.03, 0.0],
                [0.10, -0.01, 0.0], [0.09, 0.01, 0.0], [0.08, 0.03, 0.0],
            ], dtype=np.float32)
            tips = np.repeat(tips[None], n, axis=0)
            chunks.send(make_hat_chunk(
                sequence_id=sequence,
                source_state_sequence_id=state["sequence_id"],
                generated_at_ns=monotonic_ns(), fps=30.0, frame_count=n,
                root_pos=root, root_quat_wxyz=root_quat,
                head_pos=head, head_quat_wxyz=root_quat,
                left_wrist_pos=left, left_wrist_quat_wxyz=root_quat,
                right_wrist_pos=right, right_wrist_quat_wxyz=root_quat,
                left_fingertip_local=tips, right_fingertip_local=tips,
            ))
            sequence += 1
            time.sleep(args.period)
    finally:
        states.close()
        chunks.close()


if __name__ == "__main__":
    main()
