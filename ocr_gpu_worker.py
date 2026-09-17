
import os, time, re
from pathlib import Path

# CUDA 11.8 / cuDNN 8.6 stack for Paddle 2.6.2
VENV = os.environ.get("VIRTUAL_ENV", "")
SITE = Path(VENV) / "lib/python3.11/site-packages/nvidia"
paths = [
    str(SITE / "cuda_runtime/lib"),
    str(SITE / "cudnn/lib"),
    str(SITE / "cublas/lib"),
]
os.environ["LD_LIBRARY_PATH"] = ":".join(paths)

import cv2
import numpy as np
from paddleocr import PaddleOCR

def clean(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

def once(ocr, image):
    if image is None or image.size == 0:
        return "", 0.0
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h = image.shape[0]
    if h < 32:
        image = cv2.resize(image, None, fx=max(2, 64/h), fy=max(2, 64/h),
                           interpolation=cv2.INTER_CUBIC)
    try:
        result = ocr.ocr(image, cls=False)
        lines = result[0] if result else []
    except Exception as e:
        return "", 0.0
    best_t, best_c = "", 0.0
    for item in lines or []:
        try:
            t, c = item[1][0], float(item[1][1])
            if t and c > best_c:
                best_t, best_c = str(t).strip(), c
        except Exception:
            pass
    return best_t, best_c

def run_ocr(ocr, crop):
    variants = [crop]
    up = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants += [up]
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    variants += [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]
    try:
        variants += [cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)]
    except Exception:
        pass
    best = ("", 0.0)
    for im in variants:
        t, c = once(ocr, im)
        if c > best[1]:
            best = (t, c)
    return best

def main(in_q, out_q):
    ocr = PaddleOCR(lang="en", use_gpu=True, show_log=True)
    out_q.put({
        "type": "ready",
        "device": "GPU",
        "backend": "PaddleOCR 2.10"
    })

    while True:
        item = in_q.get()
        if item is None:
            break
        job_id, mode, crop = item
        t0 = time.perf_counter()
        if mode == "normal":
            text, conf = run_ocr(ocr, crop)
            payload = {"text": text, "conf": conf}
        else:
            h, w = crop.shape[:2]
            py, px = max(2,int(h*.05)), max(2,int(w*.03))
            sq = crop[py:max(py+1,h-py), px:max(px+1,w-px)]
            sh = sq.shape[0]
            split = int(sh*.48)
            gap = max(1,int(sh*.04))
            top = sq[:split]
            bottom = sq[min(sh,split+gap):]
            tt, tc = run_ocr(ocr, top)
            bt, bc = run_ocr(ocr, bottom)
            payload = {"top_text":tt,"top_conf":tc,"bottom_text":bt,"bottom_conf":bc}
        payload["ms"] = (time.perf_counter()-t0)*1000
        out_q.put((job_id, payload))

if __name__ == "__main__":
    import multiprocessing as mp
    main(mp.Queue(), mp.Queue())
