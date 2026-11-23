"""Small latest-only ZMQ endpoints used by the two online processes."""

from __future__ import annotations

import threading

from scalebridge.online.protocol import decode_message, encode_message


def _zmq():
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "Online HAT transport requires pyzmq. Install it in both the "
            "ScaleBridge and HAT environments."
        ) from exc
    return zmq


class LatestPublisher:
    def __init__(self, endpoint, bind):
        zmq = _zmq()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        (self.socket.bind if bind else self.socket.connect)(str(endpoint))

    def send(self, message):
        self.socket.send(encode_message(message), flags=0)

    def close(self):
        self.socket.close(linger=0)


class LatestSubscriber:
    def __init__(self, endpoint, bind):
        zmq = _zmq()
        self.zmq = zmq
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        (self.socket.bind if bind else self.socket.connect)(str(endpoint))

    def receive_latest(self):
        latest = None
        while True:
            try:
                latest = self.socket.recv(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
        return None if latest is None else decode_message(latest)

    def receive_blocking(self, timeout_ms=None):
        if timeout_ms is not None and self.socket.poll(int(timeout_ms)) == 0:
            return None
        return decode_message(self.socket.recv())

    def close(self):
        self.socket.close(linger=0)


class BackgroundLatestSubscriber:
    """Decode and validate a conflated stream outside the control loop.

    A ZeroMQ socket belongs to the worker thread for its entire lifetime.  The
    consumer only swaps an already-decoded Python object under a short lock, so
    a large action chunk cannot put JSON parsing directly on the 50 Hz path.
    """

    def __init__(self, endpoint, bind, validator=None):
        self.endpoint = str(endpoint)
        self.bind = bool(bind)
        self.validator = validator
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._latest_error = None
        self._thread = threading.Thread(
            target=self._receive_loop,
            name="hat-chunk-receiver",
            daemon=True,
        )
        self._thread.start()

    def _receive_loop(self):
        zmq = _zmq()
        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        (socket.bind if self.bind else socket.connect)(self.endpoint)
        try:
            while not self._stop.is_set():
                if socket.poll(50) == 0:
                    continue
                payload = socket.recv()
                # CONFLATE normally leaves one message, but drain defensively.
                while True:
                    try:
                        payload = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                try:
                    value = decode_message(payload)
                    if self.validator is not None:
                        value = self.validator(value)
                except Exception as exc:
                    with self._lock:
                        self._latest_error = str(exc)
                    continue
                with self._lock:
                    self._latest = value
        finally:
            socket.close(linger=0)

    def receive_latest(self):
        with self._lock:
            latest = self._latest
            self._latest = None
            return latest

    def receive_error(self):
        with self._lock:
            error = self._latest_error
            self._latest_error = None
            return error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
