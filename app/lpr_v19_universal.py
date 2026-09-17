
import cv2
import json
import re
import os
import sys
import time
import subprocess
import tempfile
import threading
import queue
from pathlib import Path
from collections import defaultdict
from multiprocessing.connection import Listener, Client

import numpy as np


ROOT = Path(__file__).resolve().parent

# v18: video file is now a command-line argument instead of hardcoded,
# so the same validated pipeline can be pointed at any test video without
# touching recognition logic. Falls back to the original video if none is
# given, so old invocations still work unchanged.
DEFAULT_VIDEO_NAME = "20260909_171120.mp4"
VIDEO_ARG = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_NAME
VIDEO = (ROOT / VIDEO_ARG) if not Path(VIDEO_ARG).is_absolute() else Path(VIDEO_ARG)
VIDEO_STEM = VIDEO.stem

ONNX_MODEL = ROOT.parent / "model" / "best_512.onnx"
# Results/video output are namespaced by the input video's filename, so
# running against a second video never overwrites the first run's results.
RUN_DIR = ROOT / "runs" / f"real_video_v18_{VIDEO_STEM}"
OUT = RUN_DIR / "results_vehicle_switch_GPU.json"
VIDEO_OUT = RUN_DIR / "result_vehicle_switch_GPU.mp4"

YOLO_VENV = ROOT / ".venv_kz_gpu"
OCR_VENV = ROOT / ".venv_paddlex_gpu"
YOLO_PYTHON = YOLO_VENV / "bin" / "python"
OCR_PYTHON = OCR_VENV / "bin" / "python"

CUDA12 = ":".join([
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib",
])

CUDA11 = ":".join([
    str(YOLO_VENV / "lib/python3.11/site-packages/nvidia/cuda_runtime/lib"),
    str(YOLO_VENV / "lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"),
    str(YOLO_VENV / "lib/python3.11/site-packages/nvidia/cudnn/lib"),
    str(YOLO_VENV / "lib/python3.11/site-packages/nvidia/cublas/lib"),
])

TMP = Path(tempfile.gettempdir()) / "lpr_gpu_v9_exact"
TMP.mkdir(parents=True, exist_ok=True)

# False (default here): reading from a recorded FILE for benchmarking.
# The YOLO submit queue uses BACKPRESSURE — it blocks until the worker
# is free, so every one of the FRAME_STEP-selected frames is guaranteed
# to actually reach YOLO. This is what keeps vote counts comparable to
# the fully-synchronous v9 run while still overlapping frame read/decode
# with GPU inference.
#
# True: reading from a LIVE camera in real time. The queue instead keeps
# only the newest submitted frame (drop-stale) — losing an intermediate
# frame is fine, minimizing latency to "now" matters more.
LIVE_MODE = False

FRAME_STEP = 6
OCR_EVERY_N_DETECTIONS = 3

YOLO_CONF = 0.40
SQUARE_ASPECT_MAX = 1.80
MIN_SQUARE_W = 130
MIN_SQUARE_H = 85

WINDOW_SEC = 3.0
MIN_TOP_WEIGHT = 1.60
MIN_BOTTOM_WEIGHT = 2.00
MIN_FINAL_WEIGHT = 2.50

# Vehicle switching: keep the currently confirmed plate until a different
# complete KZ plate is independently read at least twice in a short window.
SWITCH_WINDOW_SEC = 1.5
SWITCH_CONFIRM_READS = 2
# A single very strong valid read can confirm a new vehicle.
SWITCH_STRONG_CONF = 0.95


class FrameRelay:
    """
    Hands a single in-flight item from a producer (main loop) to one
    consumer worker thread. Two modes, chosen by `live_mode`:

    - live_mode=True  (DROP-STALE): submit() always overwrites whatever
      is currently waiting and never blocks. Correct for a live camera:
      an unprocessed old frame is worthless once a newer one exists, and
      the producer must never stall behind the GPU.

    - live_mode=False (BACKPRESSURE): submit() blocks until the previous
      item has been picked up by take(). No item is ever silently
      dropped — required when benchmarking against a recorded file,
      where losing frames would silently change vote counts vs. a
      synchronous run. The producer can still do useful work (decode,
      draw, write) up until the point the queue is actually full, so
      this is still faster than calling the worker inline.
    """

    def __init__(self, live_mode):
        self.live_mode = live_mode
        self._cv = threading.Condition()
        self._item = None
        self._stopped = False
        # True from the moment take() hands an item to the worker until
        # the worker calls mark_done() -- i.e. "GPU is still crunching
        # this frame". pending() alone is NOT enough to know the worker
        # is idle: once take() removes the item from the slot, pending()
        # goes False even though inference on it is still running.
        self._in_flight = False

    def submit(self, item):
        with self._cv:
            if self._stopped:
                return
            if self.live_mode:
                self._item = item
                self._cv.notify()
                return
            # Backpressure: wait for the slot to be free first.
            while self._item is not None and not self._stopped:
                self._cv.wait()
            if self._stopped:
                return
            self._item = item
            self._cv.notify()

    def take(self, timeout=0.5):
        with self._cv:
            if self._item is None and not self._stopped:
                self._cv.wait(timeout=timeout)
            if self._item is None:
                return None
            item, self._item = self._item, None
            self._in_flight = True
            self._cv.notify_all()
            return item

    def mark_done(self):
        # Worker calls this in a finally: after it's actually finished
        # processing (successfully or not) the item take() gave it.
        with self._cv:
            self._in_flight = False
            self._cv.notify_all()

    def pending(self):
        with self._cv:
            return self._item is not None

    def busy(self):
        # True while there's either an unclaimed item waiting, OR the
        # worker is still crunching the one it already took. This is
        # the correct check for "is it safe to stop the worker now".
        with self._cv:
            return self._item is not None or self._in_flight

    def stop(self):
        with self._cv:
            self._stopped = True
            self._cv.notify_all()


def valid_kz_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return bool(re.fullmatch(r"\d{3}[A-Z]{3}\d{2}", text))


print("=" * 72)
print("REAL VIDEO TEST — v9 SQUARE TEMPORAL ROW VOTING")
print("=" * 72)
print(f"Видео: {VIDEO}")
print("Цель: устойчиво подтвердить все считанные номера, любой формы")
print("Метод: независимое временное голосование верхней/нижней строки")
print()


YOLO_WORKER = TMP / 'v9_gpu_yolo_worker.py'
OCR_WORKER = TMP / 'v9_gpu_paddlex_worker.py'

YOLO_WORKER.write_text('\nimport sys\nimport time\nfrom multiprocessing.connection import Client\nimport cv2\nimport numpy as np\nimport onnxruntime as ort\n\nhost, port, auth_hex, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]\nconn = Client((host, port), authkey=bytes.fromhex(auth_hex))\n\nsession = ort.InferenceSession(\n    model,\n    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],\n)\nconn.send({"type": "ready", "providers": session.get_providers()})\ninp = session.get_inputs()[0].name\n\ndef letterbox(im, size=512):\n    h,w=im.shape[:2]\n    scale=min(size/w,size/h)\n    nw,nh=int(round(w*scale)),int(round(h*scale))\n    r=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)\n    canvas=np.full((size,size,3),114,np.uint8)\n    dx=(size-nw)//2; dy=(size-nh)//2\n    canvas[dy:dy+nh,dx:dx+nw]=r\n    return canvas,scale,dx,dy\n\ndef detect(im):\n    x,scale,dx,dy=letterbox(im)\n    a=x[:,:,::-1].astype(np.float32)/255.0\n    a=np.transpose(a,(2,0,1))[None]\n    pred=session.run(None,{inp:a})[0]\n    if pred.ndim==3: pred=pred[0]\n    if pred.ndim==2 and pred.shape[0]<pred.shape[1] and pred.shape[0]<=10:\n        pred=pred.T\n    best=None\n    fh,fw=im.shape[:2]\n    for row in pred:\n        if len(row)<5: continue\n        cx,cy,bw,bh=map(float,row[:4])\n        conf=float(row[4]) if len(row)==5 else float(np.max(row[4:]))\n        if conf<0.40: continue\n        if max(abs(cx),abs(cy),abs(bw),abs(bh))<=2:\n            cx*=512; cy*=512; bw*=512; bh*=512\n        x1=max(0,min(fw-1,int((cx-bw/2-dx)/scale)))\n        y1=max(0,min(fh-1,int((cy-bh/2-dy)/scale)))\n        x2=max(1,min(fw,int((cx+bw/2-dx)/scale)))\n        y2=max(1,min(fh,int((cy+bh/2-dy)/scale)))\n        if x2<=x1 or y2<=y1: continue\n        if best is None or conf>best[4]:\n            best=(x1,y1,x2,y2,conf)\n    return best\n\nwhile True:\n    msg=conn.recv()\n    if msg["type"]=="frame":\n        fid=msg["fid"]\n        arr=np.frombuffer(msg["jpeg"],np.uint8)\n        im=cv2.imdecode(arr,cv2.IMREAD_COLOR)\n        t=time.perf_counter()\n        try:\n            det=detect(im)\n            conn.send({"type":"result","fid":fid,"det":det,\n                       "ms":(time.perf_counter()-t)*1000.0})\n        except Exception as e:\n            conn.send({"type":"error","fid":fid,"error":repr(e)})\n    elif msg["type"]=="stop":\n        break\nconn.close()\n', encoding='utf-8')
OCR_WORKER.write_text('import sys\nimport time\nimport traceback\nfrom multiprocessing.connection import Client\n\nhost, port, auth_hex = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n\n# Connect before importing Paddle/PaddleX so startup failures are visible.\nconn = Client((host, port), authkey=bytes.fromhex(auth_hex))\nconn.send({"type": "booting", "stage": "connected"})\n\ntry:\n    import cv2\n    import numpy as np\n    import paddle\n    from paddlex.inference import create_predictor\n\n    conn.send({\n        "type": "booting",\n        "stage": "paddle_imported",\n        "paddle": paddle.__version__,\n    })\n\n    ocr = create_predictor("en_PP-OCRv5_mobile_rec", device="gpu:0")\n\n    conn.send({\n        "type": "ready",\n        "device": "GPU",\n        "backend": "PaddleX en_PP-OCRv5_mobile_rec",\n        "paddle": paddle.__version__,\n    })\nexcept Exception as e:\n    conn.send({\n        "type": "startup_error",\n        "error": repr(e),\n        "traceback": traceback.format_exc(),\n    })\n    conn.close()\n    raise\n\ndef _ocr_once(image):\n    if image is None or image.size == 0:\n        return "", 0.0\n    if len(image.shape) == 2:\n        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)\n    h, w = image.shape[:2]\n    if h < 32:\n        scale = max(2.0, 64.0 / max(1, h))\n        image = cv2.resize(image, None, fx=scale, fy=scale,\n                           interpolation=cv2.INTER_CUBIC)\n    try:\n        results = list(ocr(image))\n    except Exception:\n        return "", 0.0\n    best_text, best_conf = "", 0.0\n    for item in results:\n        text = getattr(item, "rec_text", None)\n        conf = getattr(item, "rec_score", None)\n        if text is None and isinstance(item, dict):\n            text = item.get("rec_text") or item.get("text")\n            conf = item.get("rec_score") or item.get("score")\n        if text is None:\n            continue\n        try:\n            conf = float(conf or 0.0)\n        except Exception:\n            conf = 0.0\n        text = str(text).strip()\n        if text and conf > best_conf:\n            best_text, best_conf = text, conf\n    return best_text, best_conf\n\n# v17: early-exit once a variant is confident enough. This never changes\n# WHICH variant wins when several are tried (still strict best-by-conf,\n# same tie-break order original->upscaled->gray->enhanced) -- it only\n# skips trying MORE variants once one is already good enough, which is\n# the exact case where trying more could not plausibly help. detailEnhance\n# (the most expensive, CPU-bound variant) still runs as a last resort for\n# genuinely hard crops, same as before -- it just stops being run\n# unconditionally on every single call.\nOCR_EARLY_EXIT_CONF = 0.92\n\ndef run_ocr(crop):\n    best_text, best_conf = "", 0.0\n\n    def _try(image):\n        nonlocal best_text, best_conf\n        text, conf = _ocr_once(image)\n        if text and conf > best_conf:\n            best_text, best_conf = text, conf\n        return best_conf >= OCR_EARLY_EXIT_CONF\n\n    if _try(crop):\n        return best_text, best_conf\n\n    up = cv2.resize(crop, None, fx=2.0, fy=2.0,\n                    interpolation=cv2.INTER_CUBIC)\n    if _try(up):\n        return best_text, best_conf\n\n    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)\n    if _try(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)):\n        return best_text, best_conf\n\n    try:\n        enhanced = cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)\n        _try(enhanced)\n    except Exception:\n        pass\n\n    return best_text, best_conf\n\ndef square_ocr(crop):\n    h, w = crop.shape[:2]\n    py, px = max(2, int(h*.05)), max(2, int(w*.03))\n    s = crop[py:h-py, px:w-px]\n    sh = s.shape[0]\n    split, gap = int(sh*.48), max(1, int(sh*.04))\n    top = s[:split]\n    bottom = s[min(sh, split+gap):]\n    tt, tc = run_ocr(top)\n    bt, bc = run_ocr(bottom)\n    return tt, tc, bt, bc\n\nwhile True:\n    msg = conn.recv()\n    if msg["type"] == "ocr":\n        arr = np.frombuffer(msg["jpeg"], np.uint8)\n        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)\n        t = time.perf_counter()\n        try:\n            if msg["mode"] == "normal":\n                txt, cf = run_ocr(im)\n                payload = {"mode":"normal", "text":txt, "conf":cf}\n            else:\n                tt, tc, bt, bc = square_ocr(im)\n                payload = {"mode":"square", "top_text":tt, "top_conf":tc,\n                           "bottom_text":bt, "bottom_conf":bc}\n            payload.update({"jid":msg["jid"], "fid":msg["fid"],\n                            "ms":(time.perf_counter()-t)*1000.0})\n            conn.send({"type":"result", "payload":payload})\n        except Exception as e:\n            conn.send({"type":"error", "jid":msg["jid"], "fid":msg["fid"],\n                       "error":repr(e), "traceback":traceback.format_exc()})\n    elif msg["type"] == "stop":\n        break\nconn.close()\n', encoding='utf-8')

def _env(ld):
    e=os.environ.copy()
    e["LD_LIBRARY_PATH"]=ld
    return e

def _spawn(script, python_exe, ld, args, listener):
    host,port=listener.address
    return subprocess.Popen(
        [str(python_exe),str(script),host,str(port),
         listener._authkey.hex(),*map(str,args)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=_env(ld),
    )

class GPUYOLO:
    def __init__(self, conn):
        self.conn=conn
        self.fid=0
    def detect(self, frame):
        self.fid+=1
        ok,enc=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,90])
        if not ok: return None
        self.conn.send({"type":"frame","fid":self.fid,"jpeg":enc.tobytes()})
        while True:
            m=self.conn.recv()
            if m.get("type")=="error":
                raise RuntimeError("YOLO ERROR: "+str(m))
            if m.get("type")=="result":
                return m["det"]

class GPUOCR:
    def __init__(self, conn):
        self.conn=conn
        self.jid=0
    def __call__(self, image):
        self.jid+=1
        ok,enc=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok: return []
        self.conn.send({"type":"ocr","jid":self.jid,"fid":self.jid,
                        "mode":"normal","jpeg":enc.tobytes()})
        while True:
            m=self.conn.recv()
            if m.get("type")=="error":
                raise RuntimeError("OCR ERROR: "+str(m))
            if m.get("type")=="result":
                p=m["payload"]
                # Return a dict-shaped result understood by _ocr_once replacement.
                return [{"rec_text":p["text"],"rec_score":p["conf"]}]

GPU_YOLO = None
GPU_OCR = None
YOLO_READY = None
OCR_READY = None


def _ocr_once(image):
    if GPU_OCR is None or image is None or image.size == 0:
        return "", 0.0
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if h < 32:
        scale = max(2.0, 64.0 / max(1, h))
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    try:
        results = GPU_OCR(image)
    except Exception as e:
        print("OCR ERROR:", repr(e), flush=True)
        return "", 0.0
    best_text = ""
    best_conf = 0.0
    for item in results:
        text = item.get("rec_text") if isinstance(item, dict) else getattr(item, "rec_text", None)
        conf = item.get("rec_score") if isinstance(item, dict) else getattr(item, "rec_score", None)
        try:
            conf = float(conf or 0.0)
        except Exception:
            conf = 0.0
        if text and conf > best_conf:
            best_text = str(text).strip()
            best_conf = conf
    return best_text, best_conf



def clean_text(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def normalize_top(s):
    """
    Square top row should be exactly 3 digits.
    We deliberately do NOT aggressively convert arbitrary letters to digits.
    This avoids turning APAJERO into a false numeric candidate.
    """
    s = clean_text(s)

    m = re.search(r"(\d{3})", s)
    if m:
        return m.group(1)

    # Common OCR confusion only when the result is exactly three symbols.
    if len(s) == 3:
        trans = str.maketrans({
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "E": "3",
            "A": "4",
            "S": "5",
            "G": "6",
            "T": "7",
            "B": "8",
            "P": "9",
        })
        mapped = s.translate(trans)
        if mapped.isdigit() and len(mapped) == 3:
            return mapped

    return ""


def normalize_bottom(s):
    """
    Returns bottom row as RRLLL, e.g. 02BBT.
    """
    s = clean_text(s).replace("KZ", "")

    m = re.fullmatch(r"(\d{2})([A-Z]{3})", s)
    if m:
        return m.group(1) + m.group(2)

    m = re.fullmatch(r"([A-Z]{3})(\d{2})", s)
    if m:
        return m.group(2) + m.group(1)

    # Search inside noisy OCR.
    m = re.search(r"(\d{2})([A-Z]{3})", s)
    if m:
        return m.group(1) + m.group(2)

    m = re.search(r"([A-Z]{3})(\d{2})", s)
    if m:
        return m.group(2) + m.group(1)

    return ""


def run_ocr(crop):
    """
    v6.6-style multi-pass OCR:
    original -> upscale -> gray -> enhanced.
    """
    if crop is None or crop.size == 0:
        return "", 0.0

    variants = []

    variants.append(("original", crop))

    up = cv2.resize(
        crop, None, fx=2.0, fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )
    variants.append(("upscaled", up))

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    variants.append(("gray", gray3))

    enhanced = cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)
    variants.append(("enhanced", enhanced))

    best_text = ""
    best_conf = 0.0
    best_source = ""

    for source, image in variants:
        text, conf = _ocr_once(image)

        if text:
            print(
                f"BEST RAW OCR: {text} "
                f"conf={conf:.3f} source={source}"
            )

        if text and conf > best_conf:
            best_text = text
            best_conf = conf
            best_source = source

    if best_text:
        print(
            f"BEST FAST OCR: {best_text} "
            f"conf={best_conf:.3f} source={best_source}"
        )

    return best_text, best_conf


def letterbox(img, size=512):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    dx = (size - nw) // 2
    dy = (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized

    return canvas, scale, dx, dy


def detect_plate_simple(frame):
    if GPU_YOLO is None:
        raise RuntimeError("GPU YOLO worker is not initialized")
    return GPU_YOLO.detect(frame)


def add_vote(votes, value, conf, t):
    if not value or conf < 0.50:
        return

    weight = float(conf)

    # Strong readings receive extra weight.
    if conf >= 0.90:
        weight += 0.40

    votes.append({
        "time": float(t),
        "value": value,
        "weight": weight,
        "conf": float(conf),
    })


def aggregate(votes, now):
    agg = defaultdict(lambda: {
        "weight": 0.0,
        "count": 0,
        "best_conf": 0.0,
    })

    for v in votes:
        if now - v["time"] <= WINDOW_SEC:
            a = agg[v["value"]]
            a["weight"] += v["weight"]
            a["count"] += 1
            a["best_conf"] = max(a["best_conf"], v["conf"])

    result = []
    for value, a in agg.items():
        result.append(
            (
                value,
                a["weight"],
                a["count"],
                a["best_conf"],
            )
        )

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def best_top(votes, now):
    agg = aggregate(votes, now)

    if agg and agg[0][1] >= MIN_TOP_WEIGHT:
        return agg[0][0], agg[0][1], agg[0][2], agg

    # Character-level temporal fallback.
    recent = [
        v for v in votes
        if now - v["time"] <= WINDOW_SEC
        and len(v["value"]) == 3
        and v["value"].isdigit()
    ]

    if not recent:
        return "", 0.0, 0, agg

    pos = [defaultdict(float) for _ in range(3)]

    for v in recent:
        for i, ch in enumerate(v["value"]):
            pos[i][ch] += v["weight"]

    if any(not p for p in pos):
        return "", 0.0, 0, agg

    candidate = "".join(
        max(p.items(), key=lambda kv: kv[1])[0]
        for p in pos
    )

    weight = sum(pos[i][candidate[i]] for i in range(3))

    if weight >= MIN_TOP_WEIGHT:
        return candidate, weight, len(recent), agg

    return "", 0.0, 0, agg


def best_bottom(votes, now):
    agg = aggregate(votes, now)

    if not agg:
        return "", 0.0, 0, agg

    return agg[0][0], agg[0][1], agg[0][2], agg



def aggregate_all(votes):
    groups = {}
    for v in votes:
        value = v["value"]
        groups.setdefault(value, {
            "weight": 0.0,
            "count": 0,
            "best": 0.0,
        })
        groups[value]["weight"] += float(v["weight"])
        groups[value]["count"] += 1
        groups[value]["best"] = max(
            groups[value]["best"], float(v["conf"])
        )
    rows = [
        (k, d["weight"], d["count"], d["best"])
        for k, d in groups.items()
    ]
    rows.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return rows

def main():
    global GPU_YOLO, GPU_OCR, YOLO_READY, OCR_READY

    # V16 profiling only: these counters do not affect recognition logic.
    profile_t0 = time.perf_counter()
    profile_yolo_ms = 0.0
    profile_yolo_count = 0
    profile_ocr_ms = 0.0
    profile_ocr_count = 0
    profile_writer_ms = 0.0
    profile_writer_count = 0
    profile_ocr_ready_sec = None

    yl = Listener(("127.0.0.1", 0), authkey=os.urandom(32))
    ol = Listener(("127.0.0.1", 0), authkey=os.urandom(32))

    yp = _spawn(YOLO_WORKER, YOLO_PYTHON, CUDA12, (ONNX_MODEL,), yl)

    ocr_ld = ":".join([
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cublas/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cudnn/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/curand/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cufft/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"),
        os.environ.get("LD_LIBRARY_PATH", ""),
    ])
    op = _spawn(OCR_WORKER, OCR_PYTHON, ocr_ld, (), ol)

    yc = yl.accept()
    YOLO_READY = yc.recv()
    print("YOLO GPU:", YOLO_READY, flush=True)
    if YOLO_READY.get("type") != "ready":
        raise RuntimeError(YOLO_READY)

    oc = ol.accept()
    print("OCR worker connected.", flush=True)
    while True:
        OCR_READY = oc.recv()
        print("OCR worker:", OCR_READY, flush=True)
        if OCR_READY.get("type") == "startup_error":
            raise RuntimeError(
                "OCR GPU STARTUP ERROR:\n" + OCR_READY.get("traceback", str(OCR_READY))
            )
        if OCR_READY.get("type") == "ready":
            profile_ocr_ready_sec = time.perf_counter() - profile_t0
            break

    GPU_YOLO = GPUYOLO(yc)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames_total / fps if fps else 0.0

    VIDEO_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h, out_fps = 1280, 720, 60.0
    writer = cv2.VideoWriter(
        str(VIDEO_OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать видео: {VIDEO_OUT}")

    # OCR also uses FrameRelay, with separate NORMAL and SQUARE slots.
    # This preserves the original NORMAL-first priority while preventing
    # submitted OCR crops from being overwritten under async YOLO load.
    #
    # In recorded-video mode (LIVE_MODE=False), each relay applies
    # backpressure, so every OCR submission is guaranteed to reach the
    # OCR worker. In live mode, each relay keeps only its newest pending
    # item, which minimizes latency.
    ocr_normal_relay = FrameRelay(live_mode=LIVE_MODE)
    ocr_square_relay = FrameRelay(live_mode=LIVE_MODE)
    ocr_result_q = queue.Queue()
    ocr_stop = False
    jid_counter = 0

    def ocr_thread():
        while True:
            # Preserve the original priority: NORMAL is always preferred
            # because it is important for detecting a vehicle switch.
            item = ocr_normal_relay.take(timeout=0.05)
            relay_used = ocr_normal_relay

            if item is None:
                item = ocr_square_relay.take(timeout=0.05)
                relay_used = ocr_square_relay

            if item is None:
                if ocr_stop:
                    return
                continue

            jid, mode, image, t, det = item
            try:
                ok, enc = cv2.imencode(
                    ".jpg", image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if not ok:
                    raise RuntimeError("JPEG encode failed")

                oc.send({
                    "type": "ocr",
                    "jid": jid,
                    "fid": jid,
                    "mode": mode,
                    "jpeg": enc.tobytes(),
                })

                while True:
                    msg = oc.recv()
                    if msg.get("type") == "error":
                        raise RuntimeError(
                            f"OCR ERROR: {msg.get('error')}"
                        )
                    if msg.get("type") == "result":
                        ocr_result_q.put(
                            (mode, msg["payload"], t, det)
                        )
                        break
            except Exception as exc:
                ocr_result_q.put(
                    ("error", {"error": repr(exc)}, t, det)
                )
            finally:
                # The OCR relay becomes free only after the PaddleX worker
                # has actually returned its result.
                relay_used.mark_done()

    threading.Thread(target=ocr_thread, daemon=True).start()

    def submit_ocr(mode, image, t, det):
        nonlocal jid_counter
        jid_counter += 1
        item = (jid_counter, mode, image.copy(), t, det)

        if mode == "normal":
            ocr_normal_relay.submit(item)
        else:
            ocr_square_relay.submit(item)

    # --- Async YOLO worker via FrameRelay. The main loop no longer
    # blocks on the YOLO IPC round-trip (jpeg encode -> send -> inference
    # -> recv). Frame capture and video writing keep running while
    # detection happens in the background; results (with the exact frame
    # they belong to) come back through yolo_result_q.
    #
    # LIVE_MODE=False (this file's default, for the recorded-video
    # benchmark): FrameRelay blocks submit_yolo() until the worker is
    # free, so every FRAME_STEP-selected frame is guaranteed to reach
    # YOLO -- vote counts stay comparable to a synchronous v9 run.
    # LIVE_MODE=True (live camera): FrameRelay drops stale frames instead.
    yolo_relay = FrameRelay(live_mode=LIVE_MODE)
    yolo_result_q = queue.Queue()

    def yolo_thread():
        while True:
            item = yolo_relay.take(timeout=0.5)
            if item is None:
                if yolo_relay._stopped:
                    return
                continue
            t, frame_copy = item
            try:
                yolo_t0 = time.perf_counter()
                det = GPU_YOLO.detect(frame_copy)
                yolo_result_q.put((t, frame_copy, det, None, (time.perf_counter()-yolo_t0)*1000.0))
            except Exception as exc:
                yolo_result_q.put((t, frame_copy, None, repr(exc), (time.perf_counter()-yolo_t0)*1000.0))
            finally:
                # Only now is the worker actually free -- taking an item
                # off the queue and finishing the GPU call are two
                # different moments, and busy() below depends on this.
                yolo_relay.mark_done()

    threading.Thread(target=yolo_thread, daemon=True).start()

    def submit_yolo(frame, t):
        # In LIVE_MODE=False this call can block until the GPU worker
        # frees the slot -- that's the whole point (no dropped frames).
        yolo_relay.submit((t, frame.copy()))

    frame_idx = 0
    visual_box = None
    visual_box_time = -999.0
    visual_text = ""
    visual_status = "Ожидание"
    checked = 0
    detections = 0
    ocr_attempts = 0
    square_ocr_frames = 0

    top_votes = []
    bottom_votes = []
    final_votes = []
    square_readings = []
    normal_readings = []

    confirmed_plate = ""
    confirmed_history = []
    switch_events = []
    # Async OCR results can finish out of video-time order.  A result whose
    # video timestamp is older than a decision already made must not be
    # allowed to rewrite confirmed_plate retroactively.
    last_decision_t = -1.0
    out_of_order_decisions = 0

    def consider_plate(candidate, conf, t, source):
        nonlocal confirmed_plate, confirmed_history
        nonlocal last_decision_t, out_of_order_decisions

        if not candidate or not valid_kz_plate(candidate):
            return False

        if t < last_decision_t - 1e-6:
            # The OCR result belongs to an older video moment but finished
            # after a newer decision. Keep it in diagnostics/votes, but do
            # not let it change confirmed_plate retroactively.
            out_of_order_decisions += 1
            print(
                f"[OUT-OF-ORDER DECISION IGNORED] "
                f"t={t:.2f}s < last_decision_t={last_decision_t:.2f}s "
                f"candidate={candidate} source={source}",
                flush=True,
            )
            return False

        last_decision_t = max(last_decision_t, t)

        confirmed_history = [
            x for x in confirmed_history
            if t - x["time"] <= SWITCH_WINDOW_SEC
        ]
        confirmed_history.append({
            "plate": candidate,
            "time": t,
            "source": source,
            "conf": conf
        })

        if candidate == confirmed_plate:
            return False

        reads = [x for x in confirmed_history if x["plate"] == candidate]
        # Keep the original 2-read protection for ordinary OCR results,
        # but allow one exceptionally strong valid read to confirm a switch.
        strong_single = float(conf) >= SWITCH_STRONG_CONF
        if len(reads) >= SWITCH_CONFIRM_READS or strong_single:
            old = confirmed_plate
            confirmed_plate = candidate
            confirmed_history = []
            switch_events.append({
                "time": round(t, 2),
                "from": old,
                "to": candidate,
                "source": source,
                "confidence": round(float(conf), 3),
            })
            print(
                f"*** VEHICLE SWITCH: {old or '-'} -> "
                f"{candidate} ({source}) ***",
                flush=True
            )
            return True
        return False

    def process_ocr_results():
        nonlocal visual_status, visual_text
        nonlocal profile_ocr_ms, profile_ocr_count
        while True:
            try:
                mode, payload, t, det = ocr_result_q.get_nowait()
            except queue.Empty:
                return

            if isinstance(payload, dict) and "ms" in payload:
                profile_ocr_ms += float(payload.get("ms", 0.0))
                profile_ocr_count += 1

            if mode == "error":
                print("OCR worker error:", payload, flush=True)
                continue

            if mode == "normal":
                raw_text = str(payload.get("text", ""))
                raw_conf = float(payload.get("conf", 0.0))
                # Diagnostic only: expose what PP-OCRv5 actually returned.
                print(
                    f"[NORMAL OCR RAW] t={t:5.2f}s "
                    f"RAW={raw_text!r} conf={raw_conf:.3f}",
                    flush=True
                )
                text = clean_text(raw_text)
                conf = raw_conf
                if valid_kz_plate(text):
                    normal_readings.append({
                        "time": round(t, 2),
                        "plate": text,
                        "confidence": round(conf, 3),
                    })
                    consider_plate(text, conf, t, "normal")
                visual_status = "NORMAL OCR"
                visual_text = confirmed_plate or text or "..."

            else:
                top_text = payload.get("top_text", "")
                top_conf = float(payload.get("top_conf", 0.0))
                bottom_text = payload.get("bottom_text", "")
                bottom_conf = float(payload.get("bottom_conf", 0.0))

                top = normalize_top(top_text)
                bottom = normalize_bottom(bottom_text)

                square_readings.append({
                    "time": round(t, 2),
                    "top_raw": top_text,
                    "top": top,
                    "top_conf": round(top_conf, 3),
                    "bottom_raw": bottom_text,
                    "bottom": bottom,
                    "bottom_conf": round(bottom_conf, 3),
                })

                add_vote(top_votes, top, top_conf, t)
                add_vote(bottom_votes, bottom, bottom_conf, t)

                best_t, top_weight, _, _ = best_top(top_votes, t)
                best_b, bottom_weight, _, _ = best_bottom(
                    bottom_votes, t
                )

                candidate = ""
                if best_t and best_b:
                    candidate = best_t + best_b[2:] + best_b[:2]

                if (
                    candidate
                    and valid_kz_plate(candidate)
                    and top_weight >= MIN_TOP_WEIGHT
                    and bottom_weight >= MIN_BOTTOM_WEIGHT
                ):
                    final_conf = min(
                        0.99,
                        (top_weight + bottom_weight) / 4.0
                    )
                    add_vote(final_votes, candidate, final_conf, t)
                    # v19: top_weight/bottom_weight are themselves built up
                    # from sustained temporal evidence (that's what makes
                    # them cross MIN_TOP_WEIGHT/MIN_BOTTOM_WEIGHT in the
                    # first place). Requiring this SAME simultaneous
                    # crossing to additionally recur inside a second,
                    # independent final_votes window was redundant --
                    # MIN_TOP_WEIGHT + MIN_BOTTOM_WEIGHT already exceeds
                    # MIN_FINAL_WEIGHT by construction (1.60 + 2.00 > 2.50).
                    # On a vehicle only briefly visible at a square angle,
                    # that redundant second gate could mean a correctly
                    # read plate never gets confirmed at all, because the
                    # vehicle leaves view before a second qualifying
                    # moment can occur. consider_plate() below still
                    # applies its own SWITCH_CONFIRM_READS /
                    # SWITCH_STRONG_CONF gate before this can actually
                    # change confirmed_plate -- this only removes the
                    # extra, unjustified recurrence requirement upstream
                    # of that.
                    consider_plate(candidate, final_conf, t, "square")

                # Reporting-only from here: still shown in the end-of-run
                # summary and JSON, no longer gates the online decision.
                final_summary = aggregate(final_votes, t)

                visual_status = "SQUARE / OCR"
                visual_text = candidate or (
                    f"TOP={best_t or '-'}  BOTTOM={best_b or '-'}"
                )

                print(
                    f"[SQUARE ROW ASYNC] t={t:5.2f}s "
                    f"TOP='{top_text}' -> '{top}' ({top_conf:.2f}) "
                    f"BOTTOM='{bottom_text}' -> '{bottom}' ({bottom_conf:.2f}) "
                    f"FINAL='{candidate}'",
                    flush=True
                )

    def process_yolo_results():
        # Same logic that used to run inline right after the blocking
        # GPU_YOLO.detect(frame) call — now it runs whenever a detection
        # result comes back, on whichever frame it was computed for.
        nonlocal visual_box, visual_box_time, visual_status
        nonlocal detections, ocr_attempts, square_ocr_frames
        nonlocal profile_yolo_ms, profile_yolo_count
        while True:
            try:
                t, frame_copy, det, err, yolo_ms = yolo_result_q.get_nowait()
                # yolo_ms is the full GPU_YOLO.detect() round-trip measured
                # in yolo_thread() (jpeg encode + IPC + inference + decode) --
                # counted here regardless of success/error, since it's still
                # real time spent per submitted frame.
                profile_yolo_ms += yolo_ms
                profile_yolo_count += 1

            except queue.Empty:
                return

            if err is not None:
                print("YOLO worker error:", err, flush=True)
                continue

            if det is None:
                visual_status = "YOLO: no detection"
                continue

            detections += 1
            x1, y1, x2, y2, det_conf = det
            visual_box = det
            visual_box_time = t
            crop = frame_copy[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            h, w = crop.shape[:2]
            aspect = w / max(1, h)

            if aspect > SQUARE_ASPECT_MAX or (
                w < MIN_SQUARE_W or h < MIN_SQUARE_H
            ):
                ocr_attempts += 1
                print(
                    f"[NORMAL OCR SUBMIT] t={t:5.2f}s "
                    f"bbox={w}x{h} aspect={aspect:.2f}",
                    flush=True
                )
                submit_ocr("normal", crop, t, det)
                visual_status = "NORMAL OCR: GPU busy / latest"
            else:
                square_ocr_frames += 1
                if square_ocr_frames % OCR_EVERY_N_DETECTIONS == 1:
                    ocr_attempts += 1
                    py = max(2, int(h * .05))
                    px = max(2, int(w * .03))
                    sq = crop[
                        py:max(py + 1, h - py),
                        px:max(px + 1, w - px)
                    ]

                    submit_ocr("square", sq, t, det)

                    # v19: was hardcoded to one specific test plate
                    # (633BBT02) from the original single-video demo --
                    # generalized to fire for whichever vehicle is
                    # currently confirmed, so it helps catch the next
                    # transition on any video, not just that one.
                    if confirmed_plate:
                        print(
                            f"[NORMAL FALLBACK SUBMIT] t={t:5.2f}s "
                            f"bbox={w}x{h} aspect={aspect:.2f}",
                            flush=True
                        )
                        submit_ocr("normal", crop, t, det)

                    visual_status = "SQUARE OCR: GPU busy / latest"
                else:
                    visual_status = "SQUARE: waiting OCR"

    try:
        while True:
            process_ocr_results()
            process_yolo_results()

            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            t = frame_idx / fps

            # Submit every FRAME_STEP-th frame to the async YOLO worker.
            # This no longer blocks: capture and writing keep going while
            # the GPU works on the previous submitted frame in the
            # background thread. Detection results for these frames are
            # picked up by process_yolo_results() above, whenever they
            # actually finish (which may be a frame or two later).
            if frame_idx % FRAME_STEP == 0:
                checked += 1
                submit_yolo(frame, t)

            # Unified drawing/writing path for every frame, whether or not
            # it was submitted this iteration: the overlay always reflects
            # the most recent detection/OCR state, same as v9's "skipped
            # frame" branch did.
            display = cv2.resize(
                frame, (out_w, out_h),
                interpolation=cv2.INTER_AREA
            )
            if visual_box is not None and t - visual_box_time <= 0.5:
                x1, y1, x2, y2, _ = visual_box
                sx, sy = out_w / frame.shape[1], out_h / frame.shape[0]
                cv2.rectangle(
                    display,
                    (int(x1 * sx), int(y1 * sy)),
                    (int(x2 * sx), int(y2 * sy)),
                    (0, 255, 0), 3
                )
            cv2.putText(
                display, visual_status, (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2, cv2.LINE_AA
            )
            if confirmed_plate:
                visual_text = confirmed_plate
            if visual_text:
                cv2.putText(
                    display, visual_text, (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
            cv2.putText(
                display, f"t={t:.2f}s", (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            _writer_t0 = time.perf_counter()
            writer.write(display)
            profile_writer_ms += (time.perf_counter() - _writer_t0) * 1000.0
            profile_writer_count += 1

    finally:
        # Give already-running YOLO + OCR work a short bounded drain so
        # the last few in-flight detections/reads still get counted.
        deadline = time.time() + 8.0
        while time.time() < deadline:
            process_yolo_results()
            process_ocr_results()

            ocr_busy_now = (
                ocr_normal_relay.busy()
                or ocr_square_relay.busy()
            )

            if not yolo_relay.busy() and not ocr_busy_now:
                break
            time.sleep(0.02)

        ocr_stop = True
        ocr_normal_relay.stop()
        ocr_square_relay.stop()
        yolo_relay.stop()
        try:
            oc.send({"type":"stop"})
        except Exception:
            pass
        try:
            yc.send({"type":"stop"})
        except Exception:
            pass
        cap.release()
        writer.release()
        try:
            oc.close()
            yc.close()
            ol.close()
            yl.close()
        except Exception:
            pass
        for p in (op, yp):
            try:
                p.terminate()
            except Exception:
                pass

    final_now = duration
    # Reporting only: use the complete accumulated vote history.
    # Online confirmation still uses the original v9 3-second window.
    top_summary = aggregate_all(top_votes)
    bottom_summary = aggregate_all(bottom_votes)
    final_summary = aggregate_all(final_votes)

    confirmed_square = ""
    if final_summary and final_summary[0][1] >= MIN_FINAL_WEIGHT:
        confirmed_square = final_summary[0][0]

    print()
    print("=" * 72)
    print("SQUARE ROW TEMPORAL VOTING RESULT — v17 OCR EARLY-EXIT")
    print("=" * 72)

    print("TOP ROW VOTES:")
    for value, weight, count, best in top_summary[:10]:
        print(
            f"  {value:8s} weight={weight:.2f} "
            f"count={count} best={best:.3f}"
        )

    print("\nBOTTOM ROW VOTES:")
    for value, weight, count, best in bottom_summary[:10]:
        print(
            f"  {value:8s} weight={weight:.2f} "
            f"count={count} best={best:.3f}"
        )

    print("\nFINAL SQUARE PLATE VOTES:")
    for value, weight, count, best in final_summary[:10]:
        print(
            f"  {value:10s} weight={weight:.2f} "
            f"count={count} best={best:.3f}"
        )

    print(
        f"\n*** SQUARE CONFIRMED: "
        f"{confirmed_square or 'НЕТ'} ***"
    )

    profile_total_sec = time.perf_counter() - profile_t0
    profile = {
        "total_wall_sec": profile_total_sec,
        "ocr_ready_sec": profile_ocr_ready_sec,
        "yolo": {
            "count": profile_yolo_count,
            "sum_ms": profile_yolo_ms,
            "avg_ms": (profile_yolo_ms / profile_yolo_count) if profile_yolo_count else 0.0,
        },
        "ocr": {
            "count": profile_ocr_count,
            "sum_ms": profile_ocr_ms,
            "avg_ms": (profile_ocr_ms / profile_ocr_count) if profile_ocr_count else 0.0,
        },
        "writer": {
            "count": profile_writer_count,
            "sum_ms": profile_writer_ms,
            "avg_ms": (profile_writer_ms / profile_writer_count) if profile_writer_count else 0.0,
        },
    }

    result = {
        "version": "v16_timing_profile_694BPT05",
        "timing_profile": profile,
        "video": str(VIDEO),
        "fps": fps,
        "duration_sec": duration,
        "frames_total": frames_total,
        "checked_frames": checked,
        "detections": detections,
        "ocr_attempts": ocr_attempts,
        "square_ocr_frames": square_ocr_frames,
        "confirmed_square": confirmed_square,
        "confirmed_plate_final": confirmed_plate,
        "switch_events": switch_events,
        "out_of_order_decisions_ignored": out_of_order_decisions,
        "last_decision_t": last_decision_t,
        "ocr_backend": OCR_READY,
        "yolo_backend": YOLO_READY,
        "top_votes": [
            {"value":v,"weight":w,"count":c,"best_conf":b}
            for v,w,c,b in top_summary
        ],
        "bottom_votes": [
            {"value":v,"weight":w,"count":c,"best_conf":b}
            for v,w,c,b in bottom_summary
        ],
        "final_votes": [
            {"value":v,"weight":w,"count":c,"best_conf":b}
            for v,w,c,b in final_summary
        ],
        "square_readings": square_readings,
        "normal_readings": normal_readings,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("ИТОГ V17 — OCR early-exit (voting не менялся)")
    print("=" * 72)
    print(f"FPS:                 {fps:.2f}")
    print(f"Длительность:        {duration:.2f} сек")
    print(f"Detection:           {detections}")
    print(f"OCR попыток:         {ocr_attempts}")
    print(f"Square OCR кадров:   {square_ocr_frames}")
    print(f"ПОДТВЕРЖДЁННЫЙ SQUARE: {confirmed_square or 'НЕТ'}")
    print(f"ПОДТВЕРЖДЁННЫЙ FINAL:  {confirmed_plate or 'НЕТ'}")
    print(f"OUT-OF-ORDER DECISIONS IGNORED: {out_of_order_decisions}")
    print("SWITCH EVENTS:")
    for event in switch_events:
        print(
            f"  {event['time']:.2f}s: "
            f"{event['from'] or '-'} -> {event['to']} "
            f"({event['source']})"
        )
    print(f"JSON: {OUT}")
    print(f"VIDEO: {VIDEO_OUT}")
    print("TIMING PROFILE:")
    print(f"  OCR ready:        {profile_ocr_ready_sec:.3f}s" if profile_ocr_ready_sec is not None else "  OCR ready:        n/a")
    print(f"  YOLO:             {profile_yolo_count} calls, {profile_yolo_ms:.1f} ms total, "
          f"{(profile_yolo_ms/profile_yolo_count):.1f} ms avg" if profile_yolo_count else "  YOLO:             0 calls")
    print(f"  OCR:              {profile_ocr_count} calls, {profile_ocr_ms:.1f} ms total, "
          f"{(profile_ocr_ms/profile_ocr_count):.1f} ms avg" if profile_ocr_count else "  OCR:              0 calls")
    print(f"  writer.write():    {profile_writer_count} calls, {profile_writer_ms:.1f} ms total, "
          f"{(profile_writer_ms/profile_writer_count):.1f} ms avg" if profile_writer_count else "  writer.write():    0 calls")
    print(f"  Wall time:        {profile_total_sec:.3f}s")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        for _conn in (globals().get("yc"), globals().get("oc")):
            try:
                if _conn is not None:
                    _conn.send({"type": "stop"})
            except Exception:
                pass
        for _proc, _timeout in (
            (globals().get("yp"), 20),
            (globals().get("op"), 30),
        ):
            try:
                if _proc is not None:
                    _proc.wait(timeout=_timeout)
            except Exception:
                try:
                    if _proc is not None:
                        _proc.kill()
                        _proc.wait(timeout=5)
                except Exception:
                    pass
