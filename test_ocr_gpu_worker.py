#!/usr/bin/env python3
"""
Starts the OCR worker for real and times its round trips. Needs the GPU.

A hand-run check, not part of the suite: tests/ replaces the workers with
stand-ins so it can run anywhere, and this does the opposite. It spawns the
actual subprocess, sends it crops over the same connection the pipeline uses,
and prints the average, minimum and maximum time for an answer to come back,
then the worker's exit code.

Use it when the question is about the machine rather than the logic: whether
the GPU environment is intact after a change, whether paddle still loads,
whether a worker that looked slow really is.

Usage:
    .venv_gpu/bin/python test_ocr_gpu_worker.py
"""

import os
import sys
import time
import socket
import secrets
import subprocess
from multiprocessing.connection import Listener

import cv2


HOST = "127.0.0.1"
PORT = 0
AUTHKEY = secrets.token_bytes(32)

# Берём реальный crop/кадр из нашего видео.
VIDEO = "videos/20260909_171120.mp4"

# Worker ожидает JPEG.
cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError(f"Cannot read video: {VIDEO}")

ok, jpeg = cv2.imencode(
    ".jpg",
    frame,
    [cv2.IMWRITE_JPEG_QUALITY, 90],
)

if not ok:
    raise RuntimeError("JPEG encoding failed")

jpeg_bytes = jpeg.tobytes()

# Выбираем свободный локальный порт.
sock = socket.socket()
sock.bind((HOST, 0))
port = sock.getsockname()[1]
sock.close()

listener = Listener(
    (HOST, port),
    authkey=AUTHKEY,
)

env = os.environ.copy()

worker = subprocess.Popen(
    [
        sys.executable,
        "workers/ocr_gpu_worker.py",
        HOST,
        str(port),
        AUTHKEY.hex(),
        "full",
    ],
    env=env,
)

print(f"Worker PID: {worker.pid}")
print("Waiting for OCR worker...")

conn = listener.accept()
print("Worker connected.")

# booting
msg = conn.recv()
print("BOOT:", msg)

# paddle_imported
msg = conn.recv()
print("BOOT:", msg)

# ready
msg = conn.recv()
print("READY:", msg)

if msg.get("type") != "ready":
    raise RuntimeError(f"Worker did not become ready: {msg}")

# Реальный OCR запрос.
t0 = time.perf_counter()

conn.send({
    "type": "ocr",
    "mode": "normal",
    "jpeg": jpeg_bytes,
    "jid": 1,
    "fid": 1,
})

result = conn.recv()

elapsed = (time.perf_counter() - t0) * 1000

print()
print("=== OCR RESULT ===")
print(result)
print(f"Round trip: {elapsed:.2f} ms")

# Ещё несколько прогонов, чтобы исключить cold start.
print()
print("=== WARMUP / BENCHMARK ===")

times = []

for i in range(10):
    t0 = time.perf_counter()

    conn.send({
        "type": "ocr",
        "mode": "normal",
        "jpeg": jpeg_bytes,
        "jid": i + 2,
        "fid": i + 2,
    })

    result = conn.recv()

    dt = (time.perf_counter() - t0) * 1000
    times.append(dt)

    payload = result.get("payload", {})
    print(
        f"{i+1:02d}: "
        f"text={payload.get('text')!r} "
        f"conf={payload.get('conf')} "
        f"worker_ms={payload.get('ms')} "
        f"roundtrip={dt:.2f} ms "
        f"variants={payload.get('variant_count')}"
    )

print()
print(f"Average round trip: {sum(times)/len(times):.2f} ms")
print(f"Min: {min(times):.2f} ms")
print(f"Max: {max(times):.2f} ms")

conn.send({"type": "stop"})

conn.close()
listener.close()

worker.wait(timeout=10)

print()
print(f"Worker exit code: {worker.returncode}")
