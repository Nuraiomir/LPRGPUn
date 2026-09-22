"""
Tests app/gpu_workers_client.py against real worker subprocesses.

The workers here are small stand-ins for the GPU workers: they speak the same
protocol but need no GPU, Paddle or ONNX Runtime. Environment variables make
them misbehave on purpose:

    FAKE_HANG_ON=N          never answer the Nth request of this process
    FAKE_CRASH_ON=N         exit on the Nth request of this process
    FAKE_STALE_FIRST=1      send a reply with a wrong id before the real one

Run:
    python3 tests/test_gpu_workers_client.py
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from gpu_workers_client import Workers, WorkerError  # noqa: E402


FAKE_COMMON = r'''
import os, sys, time
from multiprocessing.connection import Client
HANG = int(os.environ.get("FAKE_HANG_ON", "0"))
CRASH = int(os.environ.get("FAKE_CRASH_ON", "0"))
STALE = os.environ.get("FAKE_STALE_FIRST") == "1"
host, port, auth = sys.argv[1], int(sys.argv[2]), sys.argv[3]
conn = Client((host, port), authkey=bytes.fromhex(auth))
n = 0
'''

FAKE_YOLO = FAKE_COMMON + r'''
conn.send({"type": "ready", "providers": ["FAKE"]})
while True:
    msg = conn.recv()
    if msg["type"] == "stop":
        break
    n += 1
    if n == CRASH:
        os._exit(1)
    if n == HANG:
        time.sleep(3600)
    fid = msg["fid"]
    if STALE:
        conn.send({"type": "result", "fid": fid - 1000, "det": "STALE"})
    conn.send({"type": "result", "fid": fid, "det": (0, 0, 10, 10, 0.9), "ms": 1.0})
'''

FAKE_OCR = FAKE_COMMON + r'''
conn.send({"type": "booting", "stage": "connected"})
conn.send({"type": "ready", "device": "FAKE", "mode": sys.argv[4]})
while True:
    msg = conn.recv()
    if msg["type"] == "stop":
        break
    n += 1
    if n == CRASH:
        os._exit(1)
    if n == HANG:
        time.sleep(3600)
    payload = {"mode": msg["mode"], "text": "545BDR05", "conf": 0.97,
               "jid": msg["jid"], "fid": msg["fid"], "ms": 1.0}
    if STALE:
        conn.send({"type": "result", "payload": dict(payload, jid=msg["jid"] - 1000, text="STALE")})
    conn.send({"type": "result", "payload": payload})
'''

FRAME = np.zeros((120, 160, 3), np.uint8)


def make_project():
    root = Path(tempfile.mkdtemp(prefix="lpr_workers_test_"))
    (root / "workers").mkdir()
    (root / "model").mkdir()
    (root / "workers" / "yolo_gpu_worker.py").write_text(FAKE_YOLO)
    (root / "workers" / "ocr_gpu_worker.py").write_text(FAKE_OCR)
    (root / "model" / "best_512.onnx").write_bytes(b"")
    return root


def start(root, **env):
    for key in ("FAKE_HANG_ON", "FAKE_CRASH_ON", "FAKE_STALE_FIRST"):
        os.environ.pop(key, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    return Workers(root, sys.executable, sys.executable, root / "model" / "best_512.onnx",
                   "", "", ocr_variant_mode="no-enhanced",
                   yolo_timeout=1.0, ocr_timeout=1.0, startup_timeout=20.0)


def test_round_trip():
    w = start(make_project())
    try:
        assert w.detect(FRAME) == (0, 0, 10, 10, 0.9)
        assert w.ocr(FRAME, "normal")["text"] == "545BDR05"
        assert w.ocr_info.get("mode") == "no-enhanced", "OCR mode was not passed to the worker"
        assert w.last_timing["mode"] == "normal"
    finally:
        w.close()
    print("[OK] detect and ocr round trip; OCR mode reaches the worker")


def test_hang_is_bounded_and_worker_recovers():
    w = start(make_project(), FAKE_HANG_ON=2)
    try:
        w.detect(FRAME)
        t0 = time.monotonic()
        try:
            w.detect(FRAME)
            raise AssertionError("a hung worker did not raise")
        except WorkerError:
            pass
        waited = time.monotonic() - t0
        assert waited < 10, f"hang was not bounded: {waited:.1f} s"
        assert w.detect(FRAME) == (0, 0, 10, 10, 0.9), "worker did not recover after restart"
        assert w.restarts["yolo"] == 1
    finally:
        w.close()
    print(f"[OK] hung worker fails after {waited:.1f} s, is restarted, next call works")


def test_crash_is_detected_and_worker_recovers():
    w = start(make_project(), FAKE_CRASH_ON=2)
    try:
        assert w.ocr(FRAME, "normal")["text"] == "545BDR05"
        try:
            w.ocr(FRAME, "normal")
            raise AssertionError("a crashed worker did not raise")
        except WorkerError:
            pass
        assert w.ocr(FRAME, "normal")["text"] == "545BDR05", "worker did not recover"
        assert w.restarts["ocr"] == 1
    finally:
        w.close()
    print("[OK] crashed worker is detected, restarted, next call works")


def test_reply_with_wrong_id_is_discarded():
    w = start(make_project(), FAKE_STALE_FIRST=1)
    try:
        for _ in range(3):
            assert w.detect(FRAME) != "STALE", "YOLO returned a reply meant for another request"
            assert w.ocr(FRAME, "normal")["text"] != "STALE", "OCR returned another request's reply"
    finally:
        w.close()
    print("[OK] replies carrying another request's id are discarded")


def test_parallel_threads_keep_their_own_timing():
    w = start(make_project())
    errors = []

    def run(kind):
        try:
            for _ in range(30):
                if kind == "yolo":
                    w.detect(FRAME)
                    assert "mode" not in w.last_timing
                else:
                    w.ocr(FRAME, "square")
                    assert w.last_timing["mode"] == "square"
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=run, args=(k,)) for k in ("yolo", "ocr", "yolo", "ocr")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, errors
    finally:
        w.close()
    print("[OK] parallel YOLO and OCR calls complete, each thread sees its own timing")


if __name__ == "__main__":
    test_round_trip()
    test_hang_is_bounded_and_worker_recovers()
    test_crash_is_detected_and_worker_recovers()
    test_reply_with_wrong_id_is_discarded()
    test_parallel_threads_keep_their_own_timing()
    print("\nAll tests passed.")
