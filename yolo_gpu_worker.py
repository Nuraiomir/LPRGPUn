
import os, sys, time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# CUDA 12 / cuDNN 9 stack for ONNX Runtime
CUDA12 = "/opt/conda/lib/python3.11/site-packages/nvidia"
paths = [
    f"{CUDA12}/cuda_runtime/lib",
    f"{CUDA12}/cublas/lib",
    f"{CUDA12}/cudnn/lib",
    f"{CUDA12}/cuda_nvrtc/lib",
    f"{CUDA12}/curand/lib",
    f"{CUDA12}/cufft/lib",
]
os.environ["LD_LIBRARY_PATH"] = ":".join(paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "best_512.onnx"

CONF = 0.40

def letterbox(img, size=512):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx = (size - nw) // 2
    dy = (size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas, scale, dx, dy

def detect(session, frame):
    img, scale, dx, dy = letterbox(frame, 512)
    x = img[:, :, ::-1].astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    inp = session.get_inputs()[0].name
    pred = session.run(None, {inp: x})[0]
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 10:
        pred = pred.T

    best = None
    fh, fw = frame.shape[:2]
    for row in pred:
        if len(row) < 5:
            continue
        cx, cy, bw, bh = map(float, row[:4])
        conf = float(row[4]) if len(row) == 5 else float(np.max(row[4:]))
        if conf < CONF:
            continue
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
            cx *= 512; cy *= 512; bw *= 512; bh *= 512
        x1 = max(0, min(fw-1, int((cx-bw/2-dx)/scale)))
        y1 = max(0, min(fh-1, int((cy-bh/2-dy)/scale)))
        x2 = max(1, min(fw, int((cx+bw/2-dx)/scale)))
        y2 = max(1, min(fh, int((cy+bh/2-dy)/scale)))
        if x2 <= x1 or y2 <= y1:
            continue
        cand = (x1,y1,x2,y2,conf)
        if best is None or conf > best[4]:
            best = cand
    return best

def main(in_q, out_q):
    session = ort.InferenceSession(
        str(MODEL),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    out_q.put({
        "type": "ready",
        "providers": session.get_providers()
    })

    while True:
        item = in_q.get()
        if item is None:
            break
        frame_id, frame = item
        t0 = time.perf_counter()
        det = detect(session, frame)
        ms = (time.perf_counter()-t0)*1000
        out_q.put((frame_id, det, ms))

if __name__ == "__main__":
    import multiprocessing as mp
    main(mp.Queue(), mp.Queue())
