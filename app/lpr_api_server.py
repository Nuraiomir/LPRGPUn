"""
HTTP server for license plate recognition.

    POST /frame?session_id=<id>      body: one JPEG frame (raw bytes)
    GET  /                           health check

Each session_id gets its own LPRRecognizer, so frames from different phones
or employees never share voting state. Requests of the same session are
processed one at a time; different sessions run in parallel. Sessions that
stay idle for SESSION_TTL_SEC are dropped.

Optional query parameters:
    profile=1   add a per-stage timing breakdown ("profile") to the response
    t=<sec>     use this timestamp for voting instead of the server clock.
                For offline A/B runs only: voting windows are measured in
                seconds, so a faster run would otherwise fit more frames into
                each window and look better just for being faster.

Response fields: ok, plate, confirmed, changed, bbox, confidence,
ocr_confidence, plate_type, raw_text, session_id, processing_time_ms,
frame_size, ocr_variant_mode.

Errors return {"ok": false, "error": ..., "error_code": ...}:
    400 bad_request         missing/invalid Content-Length, session_id or t
    400 decode_failed       body is not a decodable image
    404 not_found           unknown path
    413 payload_too_large   body larger than MAX_FRAME_BYTES
    503 worker_unavailable  a GPU worker timed out or crashed (it is restarted)
    500 internal_error      anything else

Not implemented yet: authentication, rate limiting, TLS.
"""

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_workers_client import WorkerError, WorkerStartupError, Workers  # noqa: E402
from lpr_recognizer import LPRRecognizer  # noqa: E402


HOST, PORT = "0.0.0.0", 8765

MAX_FRAME_BYTES = 10 * 1024 * 1024
SESSION_TTL_SEC = 30 * 60
MAX_SESSIONS = 1000
SOCKET_TIMEOUT_SEC = 30
SESSION_ID_RE = re.compile(r"[A-Za-z0-9._:@-]{1,128}")

_PROFILE_MS_FIELDS = ("yolo_ms", "ocr_total_ms", "ocr_square_ms", "ocr_normal_ms",
                      "ocr_fallback_ms", "crop_ms", "voting_ms", "total_ms")


class _Session:
    __slots__ = ("recognizer", "lock", "last_used")

    def __init__(self):
        self.recognizer = LPRRecognizer()
        self.lock = threading.Lock()
        self.last_used = time.monotonic()


class SessionStore:
    """Thread-safe map session_id -> recognizer, with idle expiry and a size cap."""

    def __init__(self, ttl_sec=SESSION_TTL_SEC, max_sessions=MAX_SESSIONS):
        self._ttl = ttl_sec
        self._max = max_sessions
        self._sessions = {}
        self._lock = threading.Lock()

    def get(self, session_id):
        now = time.monotonic()
        with self._lock:
            expired = [k for k, s in self._sessions.items() if now - s.last_used > self._ttl]
            for key in expired:
                del self._sessions[key]

            session = self._sessions.get(session_id)
            if session is None:
                if len(self._sessions) >= self._max:
                    oldest = min(self._sessions, key=lambda k: self._sessions[k].last_used)
                    del self._sessions[oldest]
                session = self._sessions[session_id] = _Session()
            session.last_used = now
            return session

    def count(self):
        with self._lock:
            return len(self._sessions)


WORKERS = None
SESSIONS = SessionStore()


class _ClientError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_TIMEOUT_SEC

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, code, message):
        self._send_json({"ok": False, "error": message, "error_code": code}, status)

    def do_GET(self):
        self._send_json({
            "ok": True,
            "service": "lpr-api-server",
            "port": PORT,
            "session_count": SESSIONS.count(),
            "ocr_variant_mode": getattr(WORKERS, "ocr_variant_mode", "unknown"),
            "worker_restarts": getattr(WORKERS, "restarts", None),
        })

    def _read_frame(self):
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise _ClientError(400, "bad_request", "Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError:
            raise _ClientError(400, "bad_request", "invalid Content-Length") from None
        if length <= 0:
            raise _ClientError(400, "bad_request", "empty request body")
        if length > MAX_FRAME_BYTES:
            raise _ClientError(413, "payload_too_large",
                               f"frame is {length} bytes, limit is {MAX_FRAME_BYTES}")

        data = self.rfile.read(length)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise _ClientError(400, "decode_failed", "body is not a decodable image")
        return frame, length

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/frame":
            self._send_error(404, "not_found", "Use POST /frame")
            return

        query = parse_qs(parsed.query)
        session_id = query.get("session_id", ["default"])[0]
        want_profile = query.get("profile", ["0"])[0] in ("1", "true", "yes")
        t_param = query.get("t", [None])[0]

        t_req = time.perf_counter()
        try:
            if not SESSION_ID_RE.fullmatch(session_id):
                raise _ClientError(400, "bad_request",
                                   "session_id must be 1-128 characters: letters, digits, . _ : @ -")
            if t_param is not None:
                try:
                    t_vote = float(t_param)
                except ValueError:
                    raise _ClientError(400, "bad_request", "t must be a number") from None

            t_read_start = time.perf_counter()
            frame, length = self._read_frame()
            t_decoded = time.perf_counter()

            session = SESSIONS.get(session_id)
            with session.lock:
                t = t_vote if t_param is not None else time.monotonic()
                proc_t0 = time.perf_counter()
                result = session.recognizer.process_frame(frame, WORKERS, t)
                proc_ms = (time.perf_counter() - proc_t0) * 1000.0
                profile = dict(session.recognizer.last_profile or {}) if want_profile else None

            result["session_id"] = session_id
            result["processing_time_ms"] = round(proc_ms, 1)
            result["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]
            result["ocr_variant_mode"] = getattr(WORKERS, "ocr_variant_mode", "unknown")

            if profile is not None:
                profile["body_read_ms"] = round((t_read_start - t_req) * 1000.0, 2)
                profile["jpeg_decode_ms"] = round((t_decoded - t_read_start) * 1000.0, 2)
                profile["request_bytes"] = length
                for key in _PROFILE_MS_FIELDS:
                    if key in profile:
                        profile[key] = round(profile[key], 2)
                profile["worker_detail"] = [
                    {k: round(v, 2) if isinstance(v, float) else v for k, v in d.items()}
                    for d in profile.get("worker_detail", [])
                ]
                result["profile"] = profile

            self._send_json({"ok": True, **result})

        except _ClientError as exc:
            self._send_error(exc.status, exc.code, str(exc))
        except (WorkerError, WorkerStartupError) as exc:
            print(f"[FRAME] worker unavailable: {exc}", flush=True)
            self._send_error(503, "worker_unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001
            print(f"[FRAME] internal error: {exc!r}", flush=True)
            self._send_error(500, "internal_error", repr(exc))

    def log_message(self, *args):
        print("[HTTP]", *args, flush=True)


def main(project_root, yolo_python, ocr_python, onnx_model,
         yolo_cuda_ld_path, ocr_cuda_ld_path, ocr_variant_mode="full", port=None):
    global WORKERS, PORT
    if port is not None:
        PORT = port

    print("=" * 70)
    print("LPR HTTP SERVER")
    print("=" * 70)
    print(f"OCR variant mode: {ocr_variant_mode}")
    print(f"Project root:     {project_root}")
    print(f"YOLO model:       {onnx_model}")
    print(f"Worker python:    {yolo_python}")
    print("\nStarting GPU workers...", flush=True)

    WORKERS = Workers(project_root, yolo_python, ocr_python, onnx_model,
                      yolo_cuda_ld_path, ocr_cuda_ld_path,
                      ocr_variant_mode=ocr_variant_mode)

    providers = WORKERS.yolo_info.get("providers", [])
    print(f"\nYOLO providers:   {providers}")
    print(f"YOLO on CUDA:     {'CUDAExecutionProvider' in providers}")
    print(f"OCR device:       {WORKERS.ocr_info.get('device')}")
    print(f"OCR backend:      {WORKERS.ocr_info.get('backend')}")
    print("=" * 70)
    print(f"READY: http://{HOST}:{PORT}/frame?session_id=<id>")
    print("=" * 70, flush=True)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        server.server_close()
        WORKERS.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    try:
        import config.gpu_env as gpu_env
    except ModuleNotFoundError:
        sys.exit("config/gpu_env.py not found. Create it with:\n"
                 "    cp config/gpu_env.example.py config/gpu_env.py")

    parser = argparse.ArgumentParser(description="LPR HTTP server")
    parser.add_argument("--ocr-variants", choices=["full", "no-enhanced"], default="full",
                        help="no-enhanced skips cv2.detailEnhance")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    main(project_root=gpu_env.PROJECT_ROOT,
         yolo_python=gpu_env.YOLO_PYTHON,
         ocr_python=gpu_env.OCR_PYTHON,
         onnx_model=gpu_env.ONNX_MODEL,
         yolo_cuda_ld_path=gpu_env.YOLO_CUDA_LD_PATH,
         ocr_cuda_ld_path=gpu_env.OCR_CUDA_LD_PATH,
         ocr_variant_mode=args.ocr_variants,
         port=args.port)
