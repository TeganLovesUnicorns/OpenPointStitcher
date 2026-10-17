"""Talking to a running bench over HTTP, for QA, the tests and the measurement scripts."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def _url(base, path: str) -> str:
    """base is a port on this machine or a URL; path is appended."""
    return (f"http://127.0.0.1:{base}" if isinstance(base, int) else base) + path


def request(base, method: str, path: str = "", body=None, timeout: float = 120.0) -> tuple[int, dict]:
    """(status, JSON reply). An error status comes back as its status and reply, not as an exception."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(_url(base, path), data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def call(base, method: str, path: str = "", body=None, timeout: float = 120.0) -> dict:
    """The JSON reply; an error status raises urllib.error.HTTPError."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(_url(base, path), data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def wait_for_http(url: str, timeout_s: float, label: str) -> float:
    """Poll a URL until it answers 200. Returns elapsed seconds. Raises on timeout."""
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return time.time() - t0
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_err = exc
        time.sleep(0.5)
    raise TimeoutError(f"{label}: {url} never answered within {timeout_s:.0f}s ({last_err})")
