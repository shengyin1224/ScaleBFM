from __future__ import annotations

import struct
import threading
import time
from typing import Optional

import numpy as np
from loguru import logger

from scalebridge.utils.xsens_dataloader.data_buffer import XsensDataBuffer

# Wire format mirrors GR00T-WholeBodyControl/integrations/pico_scalebfm/
# pico_link_protocol.py. The two processes run in different conda environments,
# so the layout is restated here rather than imported across repositories. Keep
# the two in sync; the magic string is versioned for exactly that reason.
_MAGIC = b"PSB1"
_HEADER = struct.Struct("<4sIId")


def _decode(payload: bytes) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if len(payload) < _HEADER.size:
        raise ValueError(f"payload too short: {len(payload)} bytes")
    magic, nbody, nhand, t_mono = _HEADER.unpack_from(payload, 0)
    if magic != _MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {_MAGIC!r}")

    pos_bytes = nbody * 3 * 4
    quat_bytes = nbody * 4 * 4
    expected = _HEADER.size + pos_bytes + quat_bytes + nhand * 4
    if len(payload) != expected:
        raise ValueError(f"payload is {len(payload)} bytes, expected {expected}")

    off = _HEADER.size
    pos = np.frombuffer(payload, dtype=np.float32, count=nbody * 3, offset=off).reshape(nbody, 3)
    off += pos_bytes
    quat = np.frombuffer(payload, dtype=np.float32, count=nbody * 4, offset=off).reshape(nbody, 4)
    off += quat_bytes
    hand = np.frombuffer(payload, dtype=np.float32, count=nhand, offset=off)
    return float(t_mono), pos.copy(), quat.copy(), hand.copy()


class PicoOnlineClient:
    """Drop-in replacement for ``XsensOnlineClient`` fed by the PICO sender.

    Exposes the same surface ``MotionTrackingXsensEnv`` uses: a ``buffer`` with
    ``latest()`` / ``interpolate_batch()``, a ``frame_count``, ``start()``,
    ``stop()`` and ``wait_first_frame()``.

    Unlike the Xsens client there is no processor stage. The sender already
    publishes robot-link poses in the robot's own frame, obtained by forward
    kinematics on the GMR-retargeted G1 pose, so there is nothing left to
    retarget, rescale or reorder on this side.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5571",
        expected_num_bodies: int = 14,
        num_hand_qpos: int = 8,
        max_buffer_frames: int = 5000,
        recv_timeout_ms: int = 200,
        stamp_on_recv: bool = False,
    ):
        self.endpoint = str(endpoint)
        self.expected_num_bodies = int(expected_num_bodies)
        self.num_hand_qpos = int(num_hand_qpos)
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.stamp_on_recv = bool(stamp_on_recv)

        self.buffer = XsensDataBuffer(max_frames=max_buffer_frames)
        self.frame_count = 0

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ctx = None
        self._sock = None
        self._warned_shape = False

    def start(self) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.RCVHWM, 1000)
        self._sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._sock.connect(self.endpoint)

        self._running.set()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="PicoOnlineRecv")
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        # Context is the process-wide instance(); leave it for other users.
        self._ctx = None

    def wait_first_frame(self, timeout_s: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.frame_count >= 1:
                return True
            time.sleep(0.02)
        return False

    def _recv_loop(self) -> None:
        import zmq

        hand_pad = np.zeros(self.num_hand_qpos, dtype=np.float32)
        while self._running.is_set():
            try:
                payload = self._sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                if self._running.is_set():
                    logger.debug(f"[PICO] socket closed: {exc}")
                break

            try:
                t_mono, pos, quat, hand = _decode(payload)
            except ValueError as exc:
                logger.debug(f"[PICO] dropped frame: {exc}")
                continue

            if pos.shape[0] != self.expected_num_bodies:
                if not self._warned_shape:
                    logger.error(
                        f"[PICO] sender publishes {pos.shape[0]} links but the policy "
                        f"expects {self.expected_num_bodies}. Check SELECTED_BODY_NAMES "
                        f"in pico_scalebfm_sender.py against the checkpoint metadata."
                    )
                    self._warned_shape = True
                continue

            if hand.shape[0] != self.num_hand_qpos:
                hand = hand_pad

            if self.stamp_on_recv:
                t_mono = time.monotonic()

            self.buffer.append(t_mono, pos, quat, hand)
            self.frame_count += 1
