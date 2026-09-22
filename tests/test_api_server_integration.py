"""
Integration test for app/lpr_api_server.py and client/camera_client.py.
No GPU needed.

The GPU workers are replaced by a fake that replays recorded OCR readings
(the same ones as tests/test_lpr_recognizer.py). The test checks the whole
HTTP path: query parsing, session isolation, response JSON, error codes, and
that camera_client.py runs against a live server.

Run:
    python3 tests/test_api_server_integration.py
"""

import sys
import os
import threading
import subprocess
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import lpr_api_server  # noqa: E402


REAL_SQUARE_SEQUENCE = [
    (0.1, "KZ979", 0.862, "UZCBB", 0.829),
    (0.4, "Kz 979", 0.876, "UZCBB", 0.906),
    (0.7, "Kz 979", 0.866, "02CBB", 0.988),
    (1.0, "Kz 919", 0.802, "UZCBB", 0.86),
    (1.3, "Kz 979", 0.747, "UZCBB", 0.896),
    (1.6, "Kz 979", 0.871, "UZCBB", 0.875),
    (2.0, "Kz 9/9", 0.698, "UZCBB", 0.879),
    (2.4, "kz 979", 0.828, "02CBB", 0.98),
    (3.1, "5", 0.773, "9U", 0.19),
]
REAL_NORMAL_FOLLOWUP = [(3.5, "221ZVZ05", 0.867), (3.7, "221ZVZ05", 0.777)]


class FakeWorkers:
    """Deterministic stand-in for the real GPU workers. detect() shape is
    controlled explicitly via set_phase() so the test can drive the SAME
    OCR_EVERY_N_DETECTIONS=3 square-frame sub-sampling the real server
    applies (only every 3rd square-classified detection actually reaches
    OCR) -- this is real, faithfully-preserved v19 behavior, not a test
    artifact, so the test has to work with it rather than around it."""

    def __init__(self, ocr_script):
        self._script = list(ocr_script)  # list of ("square"|"normal", payload)
        self._i = 0
        self._phase = "square"
        self.yolo_info = {"type": "ready", "providers": ["FAKE"]}
        self.ocr_info = {"type": "ready", "backend": "FAKE"}

    def set_phase(self, phase):
        self._phase = phase

    def detect(self, frame_bgr):
        if self._phase == "square":
            return (0, 0, 200, 150, 0.90)  # aspect 1.33, 200x150 -> "square"
        return (0, 0, 300, 60, 0.90)  # aspect 5.0 -> "normal"

    def ocr(self, crop_bgr, mode):
        if self._i >= len(self._script):
            return {"mode": mode, "text": "", "conf": 0.0} if mode == "normal" else \
                   {"mode": "square", "top_text": "", "top_conf": 0.0, "bottom_text": "", "bottom_conf": 0.0}
        expected_mode, payload = self._script[self._i]
        self._i += 1
        return payload

    def close(self):
        pass


def build_script():
    script = []
    for _t, top, tc, bot, bc in REAL_SQUARE_SEQUENCE:
        script.append(("square", {"mode": "square", "top_text": top, "top_conf": tc,
                                   "bottom_text": bot, "bottom_conf": bc}))
    for _t, plate, conf in REAL_NORMAL_FOLLOWUP:
        script.append(("normal", {"mode": "normal", "text": plate, "conf": conf}))
    return script


def test_full_http_stack_reproduces_real_v19_switch_sequence():
    lpr_api_server.WORKERS = FakeWorkers(build_script())
    lpr_api_server.SESSIONS = lpr_api_server.SessionStore()

    server = ThreadingHTTPServer(("127.0.0.1", 0), lpr_api_server.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        ok, buf = cv2.imencode(".jpg", np.zeros((150, 200, 3), np.uint8))
        jpeg = buf.tobytes()
        base = f"http://127.0.0.1:{port}/frame"

        # OCR_EVERY_N_DETECTIONS=3 (real v19 behavior, preserved on purpose)
        # means only every 3rd square-classified detection actually reaches
        # OCR. We already proved the exact switch arithmetic in isolation in
        # tests/test_lpr_recognizer.py; this test's job is to prove the HTTP
        # + session + JSON wiring around that arithmetic, so we send enough
        # square-classified requests for the real sub-sampling to consume
        # all 9 canned readings and reach the same 979CBB02 confirmation --
        # by hand-tracing the real thresholds, that happens on the 8th
        # sampled OCR call, i.e. request #22 (22 % 3 == 1).
        lpr_api_server.WORKERS.set_phase("square")
        seen_switch_to_979 = False
        last_plate = ""
        for _ in range(27):
            resp = requests.post(base + "?session_id=camA", data=jpeg, timeout=5)
            assert resp.status_code == 200
            r = resp.json()
            assert r["ok"] is True
            assert r["session_id"] == "camA"
            assert set(["plate", "confirmed", "bbox", "confidence", "changed",
                        "ocr_confidence", "plate_type"]).issubset(r.keys())
            assert r["plate_type"] == "square"
            if r["changed"] and r["plate"] == "979CBB02":
                seen_switch_to_979 = True
            last_plate = r["plate"]

        assert seen_switch_to_979, "server never reported the 979CBB02 switch over real HTTP"
        assert last_plate == "979CBB02"
        print("[OK] full HTTP stack (server + sessions + JSON) reproduces the real v19 979CBB02 confirmation")

        # Session isolation: a second session must start from a clean state,
        # not see camA's confirmed 979CBB02.
        resp = requests.post(base + "?session_id=camB", data=jpeg, timeout=5)
        r = resp.json()
        assert r["session_id"] == "camB"
        assert r["plate"] != "979CBB02", (
            "session camB appears to share state with camA -- session isolation failed"
        )
        print("[OK] a second session_id does not inherit the first session's confirmed plate")

        # Health check
        health = requests.get(f"http://127.0.0.1:{port}/", timeout=5).json()
        assert health["ok"] is True
        assert health["session_count"] == 2
        assert "active_sessions" not in health, "health check must not expose session ids"
        print("[OK] GET / reports the session count without exposing session ids")

        # Wrong path -> 404 with error_code
        bad = requests.post(f"http://127.0.0.1:{port}/wrong", data=jpeg, timeout=5)
        assert bad.status_code == 404
        assert bad.json()["error_code"] == "not_found"
        print("[OK] wrong path -> 404 with error_code=not_found")

        # Garbage bytes -> 400 with error_code=decode_failed (a client error)
        bad2 = requests.post(base, data=b"not a jpeg", timeout=5)
        assert bad2.status_code == 400
        assert bad2.json()["error_code"] == "decode_failed"
        print("[OK] undecodable body -> 400 with error_code=decode_failed")

    finally:
        server.shutdown()
        server.server_close()


def test_camera_client_runs_against_live_server():
    """Smoke test: does client/camera_client.py itself run cleanly end to
    end (CLI parsing, video reading, HTTP posting, JSON parsing) against a
    live server? Uses a tiny synthetic video; the FakeWorkers behind the
    server doesn't need to line up with this video's actual content."""
    lpr_api_server.WORKERS = FakeWorkers(build_script())
    lpr_api_server.SESSIONS = lpr_api_server.SessionStore()
    server = ThreadingHTTPServer(("127.0.0.1", 0), lpr_api_server.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    video_path = "/tmp/_lpr_client_smoke_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (200, 150))
    for i in range(8):
        frame = np.full((150, 200, 3), i * 20, np.uint8)
        writer.write(frame)
    writer.release()

    try:
        result = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(__file__), "..", "client", "camera_client.py"),
             "--video", video_path,
             "--server", f"http://127.0.0.1:{port}",
             "--fps", "20", "--max-frames", "5", "--session-id", "smoketest"],
            capture_output=True, text=True, timeout=30,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        assert result.returncode == 0, "camera_client.py exited non-zero"
        assert "Отправлено кадров:" in result.stdout
        assert "ЗАПРОС НЕ ПРОШЁЛ" not in result.stdout
        assert "РЕАЛЬНАЯ ПРОПУСКНАЯ СПОСОБНОСТЬ" in result.stdout
        print("[OK] client/camera_client.py runs end-to-end against a live server without crashing")
    finally:
        server.shutdown()
        server.server_close()
        try:
            os.remove(video_path)
        except OSError:
            pass


if __name__ == "__main__":
    test_full_http_stack_reproduces_real_v19_switch_sequence()
    test_camera_client_runs_against_live_server()
    print("\nAll integration tests passed.")
