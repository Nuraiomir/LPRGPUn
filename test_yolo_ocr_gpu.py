import os
import sys
import time
import socket
import secrets
import subprocess
from multiprocessing.connection import Listener

import cv2
import numpy as np
import onnxruntime as ort


HOST = "127.0.0.1"
VIDEO = "videos/20260909_171120.mp4"
MODEL = "model/best_512.onnx"


# ============================================================
# 1. YOLO GPU
# ============================================================

print("=== YOLO GPU ===")

sess = ort.InferenceSession(
    MODEL,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print("Providers:", sess.get_providers())

input_name = sess.get_inputs()[0].name

cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Cannot read video")

original_h, original_w = frame.shape[:2]
print("Frame:", original_w, "x", original_h)


# YOLO preprocessing
img = cv2.resize(frame, (512, 512))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[None]


t0 = time.perf_counter()
output = sess.run(None, {input_name: img})
yolo_ms = (time.perf_counter() - t0) * 1000

print(f"YOLO inference: {yolo_ms:.2f} ms")
print("Output:", output[0].shape)


# ============================================================
# 2. Найдём наиболее вероятный bbox
# ============================================================

pred = output[0][0]  # [5, N]

boxes = pred[:4].T
scores = pred[4]

idx = int(np.argmax(scores))
score = float(scores[idx])

cx, cy, w, h = boxes[idx]

# coordinates 512 -> original frame
x1 = int((cx - w / 2) * original_w / 512)
y1 = int((cy - h / 2) * original_h / 512)
x2 = int((cx + w / 2) * original_w / 512)
y2 = int((cy + h / 2) * original_h / 512)

x1 = max(0, min(original_w - 1, x1))
y1 = max(0, min(original_h - 1, y1))
x2 = max(x1 + 1, min(original_w, x2))
y2 = max(y1 + 1, min(original_h, y2))

print()
print("=== DETECTION ===")
print("Score:", score)
print("BBox:", x1, y1, x2, y2)

crop = frame[y1:y2, x1:x2]

if crop.size == 0:
    raise RuntimeError("Empty crop")

print("Crop:", crop.shape[1], "x", crop.shape[0])


# save crop so we can visually inspect it
cv2.imwrite("test_ocr_crop.jpg", crop)
print("Saved: test_ocr_crop.jpg")


# JPEG
ok, jpeg = cv2.imencode(
    ".jpg",
    crop,
    [cv2.IMWRITE_JPEG_QUALITY, 95],
)

if not ok:
    raise RuntimeError("JPEG encoding failed")

jpeg_bytes = jpeg.tobytes()


# ============================================================
# 3. Start our REAL OCR worker
# ============================================================

print()
print("=== OCR WORKER ===")

AUTHKEY = secrets.token_bytes(32)

sock = socket.socket()
sock.bind((HOST, 0))
port = sock.getsockname()[1]
sock.close()

listener = Listener(
    (HOST, port),
    authkey=AUTHKEY,
)

worker = subprocess.Popen(
    [
        sys.executable,
        "workers/ocr_gpu_worker.py",
        HOST,
        str(port),
        AUTHKEY.hex(),
        "full",
    ],
    env=os.environ.copy(),
)

print("Worker PID:", worker.pid)

conn = listener.accept()

print("Connected")

print("BOOT:", conn.recv())
print("BOOT:", conn.recv())

ready = conn.recv()
print("READY:", ready)

if ready.get("type") != "ready":
    raise RuntimeError(f"OCR worker failed: {ready}")


# ============================================================
# 4. OCR benchmark on REAL license-plate crop
# ============================================================

def run(mode, jpeg_bytes, n=10):
    print()
    print(f"=== OCR MODE: {mode} ===")

    results = []
    times = []

    for i in range(n):
        t0 = time.perf_counter()

        conn.send({
            "type": "ocr",
            "mode": mode,
            "jpeg": jpeg_bytes,
            "jid": i + 1,
            "fid": i + 1,
        })

        result = conn.recv()

        rt = (time.perf_counter() - t0) * 1000
        times.append(rt)

        payload = result.get("payload", {})

        print(
            f"{i+1:02d}: "
            f"text={payload.get('text')!r} "
            f"conf={payload.get('conf')} "
            f"worker={payload.get('ms', 0):.2f} ms "
            f"roundtrip={rt:.2f} ms "
            f"variants={payload.get('variant_count')}"
        )

        if i == 0:
            print("Variants:")
            for v in payload.get("variants", []):
                print(" ", v)

        results.append(payload)

    print()
    print(f"Average: {sum(times)/len(times):.2f} ms")
    print(f"Median:  {sorted(times)[len(times)//2]:.2f} ms")
    print(f"Min:     {min(times):.2f} ms")
    print(f"Max:     {max(times):.2f} ms")

    return results


results = run("normal", jpeg_bytes)


# ============================================================
# 5. Stop
# ============================================================

conn.send({"type": "stop"})
conn.close()
listener.close()

worker.wait(timeout=10)

print()
print("Worker exit code:", worker.returncode)
