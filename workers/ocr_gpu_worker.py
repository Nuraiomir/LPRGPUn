"""
Plate text recognition with PaddleX PP-OCRv5 on the GPU, run as a subprocess.

Used by app/lpr_v19_universal.py and, through app/gpu_workers_client.py, by
the HTTP server.

Each crop is tried in up to four versions: original, 2x upscaled, grayscale,
and cv2.detailEnhance. The first version that reaches OCR_EARLY_EXIT_CONF
stops the search. detailEnhance runs on the CPU and is by far the most
expensive step; mode "no-enhanced" skips it. Square plates are split into a
top and a bottom row, and each row is read separately.

Protocol (multiprocessing.connection, authkey-secured):
  argv: host, port, authkey_hex, [mode]   mode: "full" (default) or "no-enhanced"
  -> sends {"type":"booting", ...} then {"type":"ready", ...} once PaddleX
     is loaded on GPU (or {"type":"startup_error", ...} and exits on failure)
  recv {"type":"ocr","mode":"normal"|"square","jpeg":<bytes>,"jid":...,"fid":...}
    -> "normal": {"type":"result","payload":{"mode":"normal","text":str,"conf":float,"jid":...,"fid":...,"ms":float}}
    -> "square": {"type":"result","payload":{"mode":"square","top_text":str,"top_conf":float,
                  "bottom_text":str,"bottom_conf":float,"jid":...,"fid":...,"ms":float}}
       Both payloads also carry timing: decode_ms, prep_total_ms, infer_total_ms,
       variant_count and a per-version "variants" list.
    -> or {"type":"error","jid":...,"fid":...,"error":repr(exc),"traceback":str}
  recv {"type":"stop"} -> exits cleanly
"""
import os
import sys
import time
import traceback
from multiprocessing.connection import Client

host, port, auth_hex = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# A/B experiment switch (argv[4], optional):
#   full        = original + upscaled + gray + enhanced   (default)
#   no-enhanced = original + upscaled + gray
OCR_VARIANT_MODE = sys.argv[4] if len(sys.argv) > 4 else "full"
if OCR_VARIANT_MODE not in ("full", "no-enhanced"):
    raise SystemExit(f"unknown OCR_VARIANT_MODE: {OCR_VARIANT_MODE!r}")
ENABLE_ENHANCED = OCR_VARIANT_MODE != "no-enhanced"
print(f"OCR VARIANT MODE: {OCR_VARIANT_MODE}", flush=True)

# Connect before importing Paddle/PaddleX so startup failures are visible.
conn = Client((host, port), authkey=bytes.fromhex(auth_hex))
conn.send({"type": "booting", "stage": "connected"})

try:
    import cv2
    import numpy as np
    import paddle
    from paddlex.inference import create_predictor

    conn.send({
        "type": "booting",
        "stage": "paddle_imported",
        "paddle": paddle.__version__,
    })

    # The model name is an environment variable so another recogniser can be
    # measured against the same benchmark without editing code, and so a
    # failed experiment is undone by unsetting one variable rather than by a
    # revert. PP-OCRv6 is the reason it exists: its tiny recogniser claims the
    # same accuracy several times faster on CPU, which is worth a run on
    # bench/ocr_dataset_eval.py before it is believed. Whether PaddleX knows a
    # given name depends on its version, and an unknown one fails here, at
    # startup, where the message is plain.
    OCR_MODEL = os.environ.get("LPR_OCR_MODEL", "en_PP-OCRv5_mobile_rec")
    ocr = create_predictor(OCR_MODEL, device="gpu:0")

    conn.send({
        "type": "ready",
        "device": "GPU",
        "backend": f"PaddleX {OCR_MODEL}",
        "paddle": paddle.__version__,
    })
except Exception as e:
    conn.send({
        "type": "startup_error",
        "error": repr(e),
        "traceback": traceback.format_exc(),
    })
    conn.close()
    raise

def _ocr_once(image):
    if image is None or image.size == 0:
        return "", 0.0
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if h < 32:
        scale = max(2.0, 64.0 / max(1, h))
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    try:
        results = list(ocr(image))
    except Exception:
        return "", 0.0
    best_text, best_conf = "", 0.0
    for item in results:
        text = getattr(item, "rec_text", None)
        conf = getattr(item, "rec_score", None)
        if text is None and isinstance(item, dict):
            text = item.get("rec_text") or item.get("text")
            conf = item.get("rec_score") or item.get("score")
        if text is None:
            continue
        try:
            conf = float(conf or 0.0)
        except Exception:
            conf = 0.0
        text = str(text).strip()
        if text and conf > best_conf:
            best_text, best_conf = text, conf
    return best_text, best_conf

# v17: early-exit once a variant is confident enough. This never changes
# WHICH variant wins when several are tried (still strict best-by-conf,
# same tie-break order original->upscaled->gray->enhanced) -- it only
# skips trying MORE variants once one is already good enough, which is
# the exact case where trying more could not plausibly help. detailEnhance
# (the most expensive, CPU-bound variant) still runs as a last resort for
# genuinely hard crops, same as before -- it just stops being run
# unconditionally on every single call.
OCR_EARLY_EXIT_CONF = 0.92

# A detector box much taller than a plate holds the plate in a band with
# bodywork above and below it. Measured on 20260923_152319, 163 single-row
# readings: boxes flatter than 2.0 gave a plate 0 times out of 9, boxes from
# 3.5 to 4.5 gave one 47 times out of 51. The two cars that video never
# confirmed sat at 1.8 to 2.1 the whole time they were visible, in boxes
# 1800 pixels wide. Nothing was too small; the shape was wrong.
#
# So for a flat box, one more variant: the middle band, cut to the shape a
# single-row plate actually has. It is tried LAST and still wins only by
# being strictly more confident than every other variant, so a band that
# slices through the characters cannot displace a good full-crop read.
# LPR_OCR_BAND=0 turns it off, which is how the A/B is run.
BAND_ENABLED = os.environ.get("LPR_OCR_BAND", "1") != "0"
BAND_MAX_ASPECT = 3.0      # only boxes flatter than this get a band
BAND_TARGET_ASPECT = 4.5   # what a single-row plate looks like
BAND_MIN_HEIGHT = 16       # thinner than this is not worth an OCR call


def _band(crop):
    """The middle band of a too-tall crop, or None when the crop is fine."""
    if not BAND_ENABLED:
        return None
    h, w = crop.shape[:2]
    if h <= 0 or w / max(1, h) >= BAND_MAX_ASPECT:
        return None
    band_h = int(round(w / BAND_TARGET_ASPECT))
    if band_h < BAND_MIN_HEIGHT or band_h >= h:
        return None
    y0 = (h - band_h) // 2
    return crop[y0:y0 + band_h]


# PROFILING ONLY: per-variant timings are appended here by run_ocr and
# drained by the message loop. Nothing reads this to make a decision --
# it does not affect which variant wins, the early-exit threshold, or any
# recognition outcome. Behaviour is byte-identical to the version validated
# on the three reference videos.
_VARIANT_LOG = []

def run_ocr(crop):
    best_text, best_conf = "", 0.0

    def _try(image, variant_name, prep_ms):
        nonlocal best_text, best_conf
        t0 = time.perf_counter()
        text, conf = _ocr_once(image)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        if text and conf > best_conf:
            best_text, best_conf = text, conf
        stop = best_conf >= OCR_EARLY_EXIT_CONF
        _VARIANT_LOG.append({
            "variant": variant_name,
            "prep_ms": round(prep_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "conf": round(float(conf or 0.0), 3),
            "early_exit": bool(stop),
            "size": [int(image.shape[1]), int(image.shape[0])],
        })
        return stop

    if _try(crop, "original", 0.0):
        return best_text, best_conf

    t0 = time.perf_counter()
    up = cv2.resize(crop, None, fx=2.0, fy=2.0,
                    interpolation=cv2.INTER_CUBIC)
    prep = (time.perf_counter() - t0) * 1000.0
    if _try(up, "upscaled", prep):
        return best_text, best_conf

    t0 = time.perf_counter()
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    prep = (time.perf_counter() - t0) * 1000.0
    if _try(gray_bgr, "gray", prep):
        return best_text, best_conf

    if ENABLE_ENHANCED:
        try:
            t0 = time.perf_counter()
            enhanced = cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)
            prep = (time.perf_counter() - t0) * 1000.0
            if _try(enhanced, "enhanced", prep):
                return best_text, best_conf
        except Exception:
            pass

    t0 = time.perf_counter()
    band = _band(crop)
    prep = (time.perf_counter() - t0) * 1000.0
    if band is not None:
        _try(band, "band", prep)

    return best_text, best_conf

def square_ocr(crop):
    h, w = crop.shape[:2]
    py, px = max(2, int(h*.05)), max(2, int(w*.03))
    s = crop[py:h-py, px:w-px]
    sh = s.shape[0]
    split, gap = int(sh*.48), max(1, int(sh*.04))
    top = s[:split]
    bottom = s[min(sh, split+gap):]
    tt, tc = run_ocr(top)
    bt, bc = run_ocr(bottom)
    return tt, tc, bt, bc

while True:
    msg = conn.recv()
    if msg["type"] == "ocr":
        t_dec = time.perf_counter()
        arr = np.frombuffer(msg["jpeg"], np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        decode_ms = (time.perf_counter() - t_dec) * 1000.0
        _VARIANT_LOG.clear()
        t = time.perf_counter()
        try:
            if msg["mode"] == "normal":
                txt, cf = run_ocr(im)
                payload = {"mode":"normal", "text":txt, "conf":cf}
            else:
                tt, tc, bt, bc = square_ocr(im)
                payload = {"mode":"square", "top_text":tt, "top_conf":tc,
                           "bottom_text":bt, "bottom_conf":bc}
            total_ms = (time.perf_counter()-t)*1000.0
            variants = list(_VARIANT_LOG)
            payload.update({"jid":msg["jid"], "fid":msg["fid"],
                            "ms":total_ms,
                            "decode_ms":round(decode_ms, 2),
                            "variants":variants,
                            "variant_count":len(variants),
                            "prep_total_ms":round(sum(v["prep_ms"] for v in variants), 2),
                            "infer_total_ms":round(sum(v["infer_ms"] for v in variants), 2)})
            conn.send({"type":"result", "payload":payload})
        except Exception as e:
            conn.send({"type":"error", "jid":msg["jid"], "fid":msg["fid"],
                       "error":repr(e), "traceback":traceback.format_exc()})
    elif msg["type"] == "stop":
        break
conn.close()
