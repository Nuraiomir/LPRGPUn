"""
Tests the HTTP server's protective behaviour with fake workers (no GPU):
request size limit, input validation, worker failures, per-session locking
and session expiry.

Run:
    python3 tests/test_api_server_limits.py
"""

import json
import socket
import ssl
import subprocess
import tempfile
import urllib.request
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import lpr_api_server  # noqa: E402
from gpu_workers_client import WorkerError  # noqa: E402

JPEG = cv2.imencode(".jpg", np.zeros((120, 160, 3), np.uint8))[1].tobytes()


class ConcurrencyWorkers:
    """Fake workers that record how many detect() calls overlap."""

    def __init__(self, delay=0.05, fail_first=0):
        self.delay = delay
        self.fail_first = fail_first
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self.ocr_variant_mode = "full"
        self.last_timing = None

    def detect(self, frame):
        with self._lock:
            self.calls += 1
            call = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if call <= self.fail_first:
                raise WorkerError("OCR worker timed out")
            time.sleep(self.delay)
            return None
        finally:
            with self._lock:
                self.active -= 1

    def ocr(self, crop, mode):
        return {"text": "", "conf": 0.0}


def serve(workers):
    lpr_api_server.WORKERS = workers
    lpr_api_server.SESSIONS = lpr_api_server.SessionStore()
    server = ThreadingHTTPServer(("127.0.0.1", 0), lpr_api_server.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def raw_request(port, head, body=b"", timeout=5.0):
    """Sends a hand-built HTTP request and returns (status, json)."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(head.encode("ascii") + body)
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    header, _, payload = data.partition(b"\r\n\r\n")
    status = int(header.split(b" ", 2)[1])
    return status, json.loads(payload)


def post(port, query, body=JPEG):
    head = (f"POST /frame{query} HTTP/1.1\r\nHost: x\r\n"
            f"Content-Type: image/jpeg\r\nContent-Length: {len(body)}\r\n\r\n")
    return raw_request(port, head, body)


def test_oversized_frame_is_rejected_without_reading_it():
    server, port = serve(ConcurrencyWorkers())
    try:
        size = lpr_api_server.MAX_FRAME_BYTES + 1
        head = (f"POST /frame?session_id=a HTTP/1.1\r\nHost: x\r\n"
                f"Content-Length: {size}\r\n\r\n")
        t0 = time.monotonic()
        status, body = raw_request(port, head)       # the body is never sent
        assert status == 413 and body["error_code"] == "payload_too_large", (status, body)
        assert time.monotonic() - t0 < 3, "server waited for a body it should not read"
    finally:
        server.shutdown()
    print("[OK] frame over the size limit -> 413 at once, body is not read")


def test_invalid_input_is_a_client_error():
    server, port = serve(ConcurrencyWorkers())
    try:
        head = "POST /frame?session_id=a HTTP/1.1\r\nHost: x\r\n\r\n"
        status, body = raw_request(port, head)
        assert status == 400 and body["error_code"] == "bad_request", (status, body)

        status, body = post(port, "?session_id=" + "x" * 200)
        assert status == 400 and body["error_code"] == "bad_request", (status, body)

        status, body = post(port, "?session_id=bad%20id")
        assert status == 400 and body["error_code"] == "bad_request", (status, body)

        status, body = post(port, "?session_id=a&t=abc")
        assert status == 400 and body["error_code"] == "bad_request", (status, body)

        status, body = post(port, "?session_id=a", body=b"not an image")
        assert status == 400 and body["error_code"] == "decode_failed", (status, body)
    finally:
        server.shutdown()
    print("[OK] missing length, bad session_id, bad t and bad image -> 400")


def test_worker_failure_is_503_and_server_keeps_working():
    server, port = serve(ConcurrencyWorkers(fail_first=1))
    try:
        status, body = post(port, "?session_id=a")
        assert status == 503 and body["error_code"] == "worker_unavailable", (status, body)
        status, body = post(port, "?session_id=a")
        assert status == 200 and body["ok"] is True, (status, body)
    finally:
        server.shutdown()
    print("[OK] worker failure -> 503, the next request succeeds")


def test_same_session_is_serialized_other_sessions_run_in_parallel():
    workers = ConcurrencyWorkers(delay=0.1)
    server, port = serve(workers)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: post(port, "?session_id=same"), range(6)))
        assert all(s == 200 for s, _ in results), results
        assert workers.max_active == 1, f"one session ran {workers.max_active} frames at once"

        workers.max_active = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda i: post(port, f"?session_id=s{i}"), range(6)))
        assert all(s == 200 for s, _ in results), results
        assert workers.max_active >= 2, "different sessions did not run in parallel"
    finally:
        server.shutdown()
    print(f"[OK] one session: frames one at a time; different sessions: up to "
          f"{workers.max_active} in parallel")


def test_idle_sessions_expire_and_count_is_capped():
    store = lpr_api_server.SessionStore(ttl_sec=0.2, max_sessions=3)
    first = store.get("a")
    assert store.get("a") is first, "same id must return the same session"
    time.sleep(0.3)
    store.get("b")
    assert store.count() == 1, "idle session was not dropped"
    assert store.get("a") is not first, "expired session came back with old state"

    for i in range(10):
        store.get(f"s{i}")
    assert store.count() == 3, f"session cap not enforced: {store.count()}"
    print("[OK] idle sessions expire; number of sessions is capped")



def serve_with_keys(keys, certfile=None, keyfile=None):
    lpr_api_server.WORKERS = ConcurrencyWorkers()
    lpr_api_server.SESSIONS = lpr_api_server.SessionStore()
    lpr_api_server.API_KEYS = set(keys)
    server = ThreadingHTTPServer(("127.0.0.1", 0), lpr_api_server.Handler)
    if certfile:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def post_with_header(port, header=None, body=JPEG):
    head = (f"POST /frame?session_id=a HTTP/1.1\r\nHost: x\r\n"
            f"Content-Type: image/jpeg\r\nContent-Length: {len(body)}\r\n"
            + (f"Authorization: {header}\r\n" if header else "") + "\r\n")
    return raw_request(port, head, body)


def test_access_key_is_required_only_when_configured():
    server, port = serve_with_keys([])                 # no keys: open service
    try:
        status, _ = post_with_header(port)
        assert status == 200, status
    finally:
        server.shutdown()
    print("[OK] with no keys configured the service stays open")


def test_wrong_or_missing_key_is_rejected():
    server, port = serve_with_keys(["s3cret-key", "second-key"])
    try:
        for header, why in [(None, "no header"),
                            ("Bearer wrong-key", "wrong key"),
                            ("Bearer ", "empty key"),
                            ("s3cret-key", "key without the Bearer scheme"),
                            ("Basic s3cret-key", "wrong scheme")]:
            status, body = post_with_header(port, header)
            assert status == 401 and body["error_code"] == "unauthorized", (why, status, body)

        for key in ("s3cret-key", "second-key"):
            status, body = post_with_header(port, f"Bearer {key}")
            assert status == 200 and body["ok"] is True, (key, status, body)

        # a health check stays open, so monitoring needs no key
        status, body = raw_request(port, "GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        assert status == 200 and body["ok"] is True, (status, body)
    finally:
        server.shutdown()
    print("[OK] missing, wrong and malformed keys give 401; valid keys pass; health stays open")


def test_https_serves_the_same_answers():
    cert = Path(tempfile.mkdtemp()) / "cert.pem"
    key = cert.with_name("key.pem")
    made = subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-subj", "/CN=localhost", "-keyout", str(key), "-out", str(cert)],
        capture_output=True)
    if made.returncode != 0:
        print("[SKIP] openssl not available, cannot test https")
        return

    server, port = serve_with_keys(["k"], certfile=str(cert), keyfile=str(key))
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(f"https://127.0.0.1:{port}/frame?session_id=a",
                                     data=JPEG, method="POST",
                                     headers={"Content-Type": "image/jpeg",
                                              "Authorization": "Bearer k"})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
            body = json.loads(r.read())
        assert r.status == 200 and body["ok"] is True, body
    finally:
        server.shutdown()
    print("[OK] over https the service answers the same, and still checks the key")

if __name__ == "__main__":
    test_oversized_frame_is_rejected_without_reading_it()
    test_invalid_input_is_a_client_error()
    test_worker_failure_is_503_and_server_keeps_working()
    test_same_session_is_serialized_other_sessions_run_in_parallel()
    test_idle_sessions_expire_and_count_is_capped()
    test_access_key_is_required_only_when_configured()
    test_wrong_or_missing_key_is_rejected()
    test_https_serves_the_same_answers()
    print("\nAll tests passed.")
