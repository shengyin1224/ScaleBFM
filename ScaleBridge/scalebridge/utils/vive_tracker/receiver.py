import json
import logging
import socket
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

RAW_TRACKER_ROW_SIZE = 14

def _validate_payload(payload) -> List[List[float]]:
    """Validate that the decoded JSON payload is an (n, 14) array."""
    if not isinstance(payload, list):
        raise ValueError("Expected the payload to be a JSON array.")
    for index, row in enumerate(payload):
        if not isinstance(row, list):
            raise ValueError(f"Row {index} is not a JSON array.")
        if len(row) != RAW_TRACKER_ROW_SIZE:
            raise ValueError(f"Row {index} has length {len(row)}, expected {RAW_TRACKER_ROW_SIZE}.")
    return payload


def _decode_payload(payload) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Decode the versioned sender format while retaining legacy array support."""
    if isinstance(payload, dict):
        rows = _validate_payload(payload.get("rows"))
        device_names = payload.get("device_names", [])
        requested_names = payload.get("requested_device_names", [])
        if not isinstance(device_names, list) or not all(isinstance(name, str) for name in device_names):
            raise ValueError("device_names must be an array of strings.")
        if len(device_names) != len(rows):
            raise ValueError(
                f"device_names has length {len(device_names)}, expected {len(rows)}."
            )
        if not isinstance(requested_names, list) or not all(isinstance(name, str) for name in requested_names):
            raise ValueError("requested_device_names must be an array of strings.")
        metadata = {
            "schema_version": int(payload.get("schema_version", 2)),
            "source_timestamp": float(payload.get("timestamp", 0.0)),
            "device_names": list(device_names),
            "requested_device_names": list(requested_names),
        }
        return rows, metadata

    rows = _validate_payload(payload)
    source_timestamp = float(rows[0][-1]) if rows else 0.0
    return rows, {
        "schema_version": 1,
        "source_timestamp": source_timestamp,
        "device_names": [],
        "requested_device_names": [],
    }


class ViveTrackerReceiver:

    def __init__(self, host: str = "0.0.0.0", port: int = 5001, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout

        self.data: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.frame_metadata: Dict[str, Any] = {}
        self._frame_lock = Lock()

        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.receiver_thread: Optional[Thread] = None
        self.running = False

    def listen(self) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

    def accept_blocking(self) -> Tuple[str, int]:
        assert self.server_socket is not None
        self.client_socket, addr = self.server_socket.accept()
        return str(addr[0]), int(addr[1])

    def _parse_json_lines(self, buffer: bytes) -> bytes:

        while b"\n" in buffer:
            raw_message, buffer = buffer.split(b"\n", 1)
            raw_message = raw_message.strip()
            if not raw_message:
                continue
            try:
                payload = json.loads(raw_message.decode("utf-8"))
                rows, metadata = _decode_payload(payload)
                with self._frame_lock:
                    self.data = np.array(rows, dtype=np.float64)
                    self.frame_metadata = metadata
                    self.frame_count += 1
            except (json.JSONDecodeError, ValueError) as e:
                logging.debug(f"Vive tracker parse error: {e}")
        return buffer

    def latest_frame(self) -> Optional[Dict[str, Any]]:
        """Return one internally consistent copy of the newest received frame."""
        with self._frame_lock:
            if self.data is None:
                return None
            return {
                "frame_count": int(self.frame_count),
                "data": self.data.copy(),
                **self.frame_metadata,
            }

    def socket_receiver(self) -> None:
        assert self.client_socket is not None
        buffer = b""
        while self.running:
            try:
                chunk = self.client_socket.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                buffer = self._parse_json_lines(buffer)
            except (socket.error, OSError, BrokenPipeError) as e:
                logging.info(f"Vive tracker socket stopped: {e}")
                self.running = False
                break

    def start_receiving(self) -> None:
        if self.receiver_thread is None or not self.receiver_thread.is_alive():
            self.running = True
            self.receiver_thread = Thread(target=self.socket_receiver, daemon=True)
            self.receiver_thread.start()

    def stop_receiving(self) -> None:
        self.running = False
        if self.receiver_thread is not None:
            self.receiver_thread.join(timeout=2.0)
            self.receiver_thread = None

    def close(self) -> None:
        self.stop_receiving()
        try:
            if self.client_socket is not None:
                self.client_socket.close()
        except OSError:
            pass
        try:
            if self.server_socket is not None:
                self.server_socket.close()
        except OSError:
            pass
        self.client_socket = None
        self.server_socket = None
