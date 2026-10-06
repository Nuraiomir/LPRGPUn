#!/usr/bin/env python3
"""
Real-video A/B benchmark:
YOLO11n ONNX (existing workers/yolo_gpu_worker.py)
        -> same crop
        -> PaddleOCR and internal DeepSeek OCR-2

Important:
- Does NOT use ultralytics.
- Uses the same YOLO ONNX worker protocol as the existing production pipeline.
- Both OCR engines receive the same detected crop.
- Saves frame-level JSON for later analysis.
"""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing.connection import Listener

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "videos" / "20260909_171120.mp4"
DEFAULT_MODEL = ROOT / "model" / "best11n_512.onnx"
YOLO_WORKER = ROOT / "workers" / "yolo_gpu_worker.py"
OUT_DIR = ROOT / "runs"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_yolo_worker(model_path):
    host = "127.0.0.1"
    port = free_port()
    authkey = os.urandom(32)

    listener = Listener((host, port), authkey=authkey)

    cmd = [
        sys.executable,
        str(YOLO_WORKER),
        host,
        str(port),
        authkey.hex(),
        str(model_path),
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    conn = listener.accept()
    ready = conn.recv()

    if ready.get("type") != "ready":
        proc.kill()
        raise RuntimeError(f"YOLO worker did not become ready: {ready}")

    return proc, listener, conn, ready


def stop_yolo_worker(proc, listener, conn):
    try:
        conn.send({"type": "stop"})
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass
    try:
        listener.close()
    except Exception:
        pass

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def jpeg_bytes(img, quality=90):
    ok, enc = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def crop_from_det(frame, det):
    if det is None:
        return None

    x1, y1, x2, y2, conf = det
    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def load_paddle():
    from paddlex.inference import create_predictor

    model_name = os.environ.get(
        "LPR_OCR_MODEL",
        "en_PP-OCRv5_mobile_rec",
    )

    predictor = create_predictor(
        model_name,
        device="gpu:0",
    )

    return predictor, model_name


def paddle_text(result):
    """
    Extract text from PaddleX result without assuming one exact output
    object shape. The production worker uses PaddleX predictor output.
    """
    if result is None:
        return "", None

    if isinstance(result, dict):
        text = result.get("rec_text") or result.get("text") or ""
        score = result.get("rec_score") or result.get("score")
        return str(text), score

    # Common PaddleX result object patterns.
    for attr in ("rec_text", "text"):
        if hasattr(result, attr):
            text = getattr(result, attr)
            if callable(text):
                text = text()
            score = None
            for sattr in ("rec_score", "score"):
                if hasattr(result, sattr):
                    score = getattr(result, sattr)
                    if callable(score):
                        score = score()
                    break
            return str(text or ""), score

    return str(result), None


def run_paddle(predictor, crop):
    t0 = time.perf_counter()

    try:
        # Same basic image input type used by the existing PaddleX worker.
        outputs = list(predictor(crop))
        if not outputs:
            return {
                "text": "",
                "confidence": None,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "error": None,
            }

        text, score = paddle_text(outputs[0])

        return {
            "text": text,
            "confidence": score,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "error": None,
        }

    except Exception as e:
        return {
            "text": "",
            "confidence": None,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "error": repr(e),
        }


def image_data_url(img):
    return "data:image/jpeg;base64," + base64.b64encode(
        jpeg_bytes(img, quality=95)
    ).decode("ascii")


def run_deepseek(session, crop, gateway_url, api_key, model):
    t0 = time.perf_counter()

    prompt = (
        "Read ONLY the license plate characters visible in the image. "
        "Return ONLY the license plate number as plain text. "
        "Do not explain anything. "
        "Do not describe the image. "
        "Do not return Markdown, punctuation, or additional words. "
        "If the license plate cannot be read, return an empty string."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(crop)
                        },
                    },
                ],
            }
        ],
        "max_tokens": 32,
        "temperature": 0,
    }

    try:
        r = session.post(
            gateway_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return {
            "text": str(text or ""),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "error": None,
            "model": data.get("model"),
        }

    except Exception as e:
        return {
            "text": "",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "error": repr(e),
            "model": None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument(
        "--frame-step",
        type=int,
        default=6,
        help="Run YOLO every Nth frame, matching the current pipeline.",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 = full video.",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Optional output JSON path.",
    )
    args = ap.parse_args()

    video = Path(args.video)
    model = Path(args.model)

    if not video.exists():
        raise FileNotFoundError(video)
    if not model.exists():
        raise FileNotFoundError(model)
    if not YOLO_WORKER.exists():
        raise FileNotFoundError(YOLO_WORKER)

    gateway_key = os.environ.get("LPR_GATEWAY_KEY")
    if not gateway_key:
        raise RuntimeError("LPR_GATEWAY_KEY is not set")

    gateway_url = os.environ.get(
        "LPR_GATEWAY_URL",
        "https://test-mlops-ai-gateway.apps.prod.aida.bcc.kz/v1",
    )
    deepseek_model = os.environ.get(
        "LPR_DEEPSEEK_MODEL",
        "deepseek-ocr-2/deepseek-ocr-2",
    )

    print("=== Real-video YOLO11 + PaddleOCR vs DeepSeek OCR-2 ===")
    print("Video:", video)
    print("YOLO model:", model)
    print("YOLO worker:", YOLO_WORKER)
    print("Frame step:", args.frame_step)
    print("DeepSeek model:", deepseek_model)
    print()

    print("[1/4] Loading PaddleOCR...")
    paddle, paddle_model = load_paddle()
    print("Paddle model:", paddle_model)

    print("[2/4] Starting YOLO ONNX worker...")
    yolo_proc, listener, yolo_conn, yolo_ready = start_yolo_worker(model)
    print("YOLO providers:", yolo_ready.get("providers"))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        stop_yolo_worker(yolo_proc, listener, yolo_conn)
        raise RuntimeError(f"Cannot open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print("Video FPS:", fps)
    print("Video frames:", total_video_frames)
    print()
    print("[3/4] Running identical-crop A/B...")

    http = requests.Session()

    rows = []
    frame_id = 0
    processed = 0
    detected = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.max_frames and frame_id >= args.max_frames:
                break

            if frame_id % max(1, args.frame_step) != 0:
                frame_id += 1
                continue

            processed += 1

            # Same YOLO worker used by the existing project.
            t0 = time.perf_counter()
            yolo_conn.send({
                "type": "frame",
                "fid": frame_id,
                "jpeg": jpeg_bytes(frame, quality=90),
            })

            while True:
                msg = yolo_conn.recv()
                if msg.get("fid") != frame_id:
                    continue
                break

            yolo_wall_ms = (time.perf_counter() - t0) * 1000.0

            det = msg.get("det")
            if det is None:
                rows.append({
                    "frame_id": frame_id,
                    "timestamp_sec": (
                        frame_id / fps if fps > 0 else None
                    ),
                    "det": None,
                    "yolo_ms": msg.get("ms"),
                    "yolo_roundtrip_ms": yolo_wall_ms,
                    "paddle": None,
                    "deepseek": None,
                })
                frame_id += 1
                continue

            detected += 1
            crop = crop_from_det(frame, det)

            if crop is None:
                frame_id += 1
                continue

            # IMPORTANT: both OCR engines receive this exact same crop.
            paddle_result = run_paddle(paddle, crop)
            deepseek_result = run_deepseek(
                http,
                crop,
                gateway_url,
                gateway_key,
                deepseek_model,
            )

            rows.append({
                "frame_id": frame_id,
                "timestamp_sec": (
                    frame_id / fps if fps > 0 else None
                ),
                "det": {
                    "x1": float(det[0]),
                    "y1": float(det[1]),
                    "x2": float(det[2]),
                    "y2": float(det[3]),
                    "confidence": float(det[4]),
                },
                "crop_shape": [
                    int(crop.shape[1]),
                    int(crop.shape[0]),
                ],
                "yolo_ms": msg.get("ms"),
                "yolo_roundtrip_ms": yolo_wall_ms,
                "paddle": paddle_result,
                "deepseek": deepseek_result,
            })

            if processed % 20 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"processed={processed} detected={detected} "
                    f"elapsed={elapsed:.1f}s"
                )

            frame_id += 1

    finally:
        cap.release()
        http.close()
        stop_yolo_worker(yolo_proc, listener, yolo_conn)

    elapsed = time.perf_counter() - started

    if args.out:
        out_path = Path(args.out)
    else:
        tag = video.stem
        out_path = OUT_DIR / (
            f"ab_yolo11_paddle_vs_deepseek_{tag}.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark": "real_video_yolo11_paddle_vs_deepseek",
        "video": str(video),
        "video_name": video.name,
        "yolo_model": str(model),
        "yolo_worker": str(YOLO_WORKER),
        "frame_step": args.frame_step,
        "video_fps": fps,
        "video_total_frames": total_video_frames,
        "processed_frames": processed,
        "detected_frames": detected,
        "elapsed_wall_sec": elapsed,
        "ocr_backend": {
            "paddle": {
                "model": paddle_model,
                "device": "gpu:0",
            },
            "deepseek": {
                "gateway": gateway_url,
                "model": deepseek_model,
            },
        },
        "rows": rows,
    }

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("[4/4] DONE")
    print("Processed frames:", processed)
    print("Detected frames:", detected)
    print("Elapsed:", round(elapsed, 2), "sec")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
