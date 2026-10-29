"""HTTP client for the SONIC-compatible Vive tracker recorder service."""

from __future__ import annotations

import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PicoTrackerRecorderClient:
    def __init__(self, base_url: str, dataset_name: str, hz: float = 120.0, timeout: float = 0.8):
        self.base_url = base_url.rstrip("/")
        self.dataset_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("._-") or "pico_scalebfm"
        self.hz = float(hz)
        self.timeout = max(0.1, float(timeout))
        self._session: str | None = None

    def _post(self, path: str, payload: dict) -> dict | None:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return None
        return result if isinstance(result, dict) and result.get("ok") else None

    def start(self, episode_index: int) -> bool:
        self._session = f"{self.dataset_name}_episode_{int(episode_index):06d}"
        result = self._post(
            "/start",
            {"session": self._session, "hz": self.hz, "start_unix": time.time()},
        )
        if result is None:
            self._session = None
            return False
        return True

    def stop(self) -> bool:
        session, self._session = self._session, None
        if session is None:
            return False
        return self._post("/stop", {"session": session}) is not None

    def discard(self) -> bool:
        session, self._session = self._session, None
        if session is None:
            return False
        return self._post("/discard", {"session": session}) is not None

