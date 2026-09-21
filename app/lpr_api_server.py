"""
LPR API server -- Option C from the architecture discussion: a NEW, thin
HTTP adapter around the shared app/lpr_recognizer.py core, rather than
patching the old lpr_camera_server.py prototype's own copy of the
recognition logic (Option A) or writing a second, differently-behaved
implementation from scratch (Option B, which is what happened by accident
the first time -- see docs/LPR_API_CONTRACT_REVIEW.md).

lpr_camera_server.py is left in place, unmodified, per "do not delete old
working files" -- but it should be considered DEPRECATED once this server
has been verified end-to-end (see docs/architecture.md checklist), because
having two servers with two different recognition behaviors is exactly the
"behavioral drift" risk this refactor exists to remove.

What is NEW here relative to the prototype, and why:
  - `session_id` (query param, default "default"): each session_id gets its
    OWN LPRRecognizer instance and therefore its own temporal-voting state.
    This directly fixes the global-shared-state problem documented in
    LPR_API_CONTRACT_REVIEW.md ("if two cameras post to the same server,
    their frames get mixed into one vote history"). A client that never
    passes session_id gets the SAME single-shared-session behavior the
    prototype always had -- this is backward compatible by default.
  - `ocr_confidence`, `plate_type`, `raw_text` in the response: already
    computed internally by LPRRecognizer.process_frame(), simply not
    discarded. Additive only -- every field the prototype returned is still
    present with the same name and meaning.
  - `error_code` alongside `error` in error responses: a small, fixed set of
    machine-readable strings ("not_found", "decode_failed",
    "internal_error") instead of only a free-text repr(). This is NOT the
    full structured-error taxonomy proposed in LPR_API_CONTRACT.md section B
    -- just the smallest useful step, since "do not over-engineer" was an
    explicit instruction for this task.

What is UNCHANGED / NOT done here, on purpose:
  - No authentication (still open, like the prototype) -- proposed, not
    implemented, per LPR_API_CONTRACT.md section B.
  - No JSON-wrapped request body -- still raw JPEG bytes in, exactly like
    the prototype, so existing test tooling keeps working unchanged.
  - No request timeout on the YOLO/OCR calls -- still absent, same
    limitation as the prototype, documented, not silently fixed here.
  - No multi-plate-per-frame support -- Workers.detect() still returns at
    most one box, exactly like both prior implementations.

NOT YET VERIFIED end-to-end against real GPU hardware (no GPU in this
environment). See docs/architecture.md "Verification status" for the exact
steps to run before treating this as equivalent to v19 in production.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lpr_recognizer import LPRRecognizer  # noqa: E402
from gpu_workers_client import Workers  # noqa: E402

import cv2
import numpy as np
import time


HOST, PORT = "0.0.0.0", 8765


class SessionStore:
    """One LPRRecognizer per session_id. Not persisted across process
    restarts -- a restart resets all sessions, exactly like the prototype's
    single global Recognizer resets on restart today."""

    def __init__(self):
        self._sessions = {}

    def get(self, session_id):
        if session_id not in self._sessions:
            self._sessions[session_id] = LPRRecognizer()
        return self._sessions[session_id]

    def ids(self):
        return list(self._sessions.keys())


WORKERS = None
SESSIONS = SessionStore()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send_json({
            "ok": True,
            "service": "lpr-api-server",
            "port": PORT,
            "active_sessions": SESSIONS.ids(),
        })

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/frame":
            self._send_json(
                {"ok": False, "error": "Use POST /frame", "error_code": "not_found"},
                404,
            )
            return

        session_id = parse_qs(parsed.query).get("session_id", ["default"])[0]
        want_profile = parse_qs(parsed.query).get("profile", ["0"])[0] in ("1", "true", "yes")

        t_req = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length)
            t_read = time.perf_counter()

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("JPEG decode failed")
            t_decode = time.perf_counter()

            recognizer = SESSIONS.get(session_id)

            # Voting clock. By default this is the server's wall clock,
            # exactly as in production.
            #
            # EXPERIMENT-ONLY: if the client sends ?t=<seconds>, that value
            # is used as the voting timestamp instead. This exists because
            # the temporal windows (WINDOW_SEC=3.0, SWITCH_WINDOW_SEC=1.5)
            # are measured in real seconds: a FASTER server packs more
            # frames into the same 3-second window and would therefore look
            # more accurate purely by being faster. Feeding video time makes
            # an A/B comparison of OCR variants apples-to-apples. Production
            # clients do not send ?t and get unchanged behaviour.
            t_param = parse_qs(parsed.query).get("t", [None])[0]
            if t_param is not None:
                try:
                    t = float(t_param)
                except ValueError:
                    t = time.monotonic()
            else:
                t = time.monotonic()

            proc_t0 = time.perf_counter()
            result = recognizer.process_frame(frame, WORKERS, t)
            proc_ms = (time.perf_counter() - proc_t0) * 1000.0

            result["session_id"] = session_id
            # Server-side processing time (YOLO + OCR + voting), excluding
            # HTTP transfer. Lets the client separate network overhead from
            # actual inference cost.
            result["processing_time_ms"] = round(proc_ms, 1)
            result["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]

            if want_profile:
                # Optional, non-breaking: only present when ?profile=1.
                prof = dict(getattr(recognizer, "last_profile", None) or {})
                prof["body_read_ms"] = round((t_read - t_req) * 1000.0, 2)
                prof["jpeg_decode_ms"] = round((t_decode - t_read) * 1000.0, 2)
                prof["request_bytes"] = length
                for k in ("yolo_ms", "ocr_total_ms", "ocr_square_ms",
                          "ocr_normal_ms", "ocr_fallback_ms", "crop_ms",
                          "voting_ms", "total_ms"):
                    if k in prof:
                        prof[k] = round(prof[k], 2)
                for d in prof.get("worker_detail", []):
                    for k, v in list(d.items()):
                        if isinstance(v, float):
                            d[k] = round(v, 2)
                result["profile"] = prof

            t_ser = time.perf_counter()
            self._send_json({"ok": True, **result})
            if want_profile:
                # Serialization time can't be inside the payload it measures;
                # log it instead.
                ser_ms = (time.perf_counter() - t_ser) * 1000.0
                if ser_ms > 5.0:
                    print(f"[PROFILE] медленная сериализация/отправка: {ser_ms:.1f} мс", flush=True)

        except ValueError as e:
            self._send_json(
                {"ok": False, "error": repr(e), "error_code": "decode_failed"}, 500
            )
        except Exception as e:
            print("FRAME ERROR:", repr(e), flush=True)
            self._send_json(
                {"ok": False, "error": repr(e), "error_code": "internal_error"}, 500
            )

    def log_message(self, *args):
        print("[HTTP]", *args, flush=True)


def main(project_root, yolo_python, ocr_python, onnx_model,
         yolo_cuda_ld_path, ocr_cuda_ld_path, ocr_variant_mode="full",
         port=None):
    global WORKERS, PORT
    if port is not None:
        PORT = port
    print("=" * 70)
    print("LPR HTTP SERVER")
    print("=" * 70)
    print(f"Режим вариантов OCR: {ocr_variant_mode}"
          + ("  (эксперимент: без detailEnhance)" if ocr_variant_mode == "no-enhanced" else ""))
    print(f"Корень проекта:  {project_root}")
    print(f"Модель YOLO:     {onnx_model}")
    print(f"Python YOLO:     {yolo_python}")
    print(f"Python OCR:      {ocr_python}")
    print()
    print("Запуск GPU-воркеров, это займёт несколько секунд...", flush=True)

    WORKERS = Workers(
        Path(project_root), Path(yolo_python), Path(ocr_python), Path(onnx_model),
        yolo_cuda_ld_path, ocr_cuda_ld_path, ocr_variant_mode=ocr_variant_mode,
    )

    providers = WORKERS.yolo_info.get("providers", [])
    print()
    print("YOLO:")
    print(f"  ONNX Runtime providers: {providers}")
    print(f"  CUDA доступна:          {'CUDAExecutionProvider' in providers}")
    print("OCR:")
    print(f"  Устройство:             {WORKERS.ocr_info.get('device')}")
    print(f"  Бэкенд:                 {WORKERS.ocr_info.get('backend')}")
    print(f"  Paddle:                 {WORKERS.ocr_info.get('paddle')}")
    print()
    print("=" * 70)
    print(f"ГОТОВ: http://{HOST}:{PORT}/frame?session_id=<id>")
    print(f"Health-check: http://{HOST}:{PORT}/")
    print("=" * 70, flush=True)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера...", flush=True)
    finally:
        server.server_close()
        if WORKERS:
            WORKERS.close()
        print("Сервер остановлен.", flush=True)


if __name__ == "__main__":
    # Make the project root importable regardless of the current working
    # directory, so `python app/lpr_api_server.py` works from anywhere.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import config.gpu_env as gpu_env
    except ModuleNotFoundError:
        print("ОШИБКА: не найден config/gpu_env.py")
        print("Скопируйте config/gpu_env.example.py в config/gpu_env.py и проверьте пути.")
        sys.exit(1)

    import argparse
    ap = argparse.ArgumentParser(description="LPR HTTP server")
    ap.add_argument("--ocr-variants", choices=["full", "no-enhanced"],
                    default="full",
                    help="full = original,upscaled,gray,enhanced (продакшн). "
                         "no-enhanced = без cv2.detailEnhance (эксперимент B)")
    ap.add_argument("--port", type=int, default=PORT, help="порт сервера")
    cli = ap.parse_args()

    main(
        project_root=gpu_env.PROJECT_ROOT,
        yolo_python=gpu_env.YOLO_PYTHON,
        ocr_python=gpu_env.OCR_PYTHON,
        onnx_model=gpu_env.ONNX_MODEL,
        yolo_cuda_ld_path=gpu_env.YOLO_CUDA_LD_PATH,
        ocr_cuda_ld_path=gpu_env.OCR_CUDA_LD_PATH,
        ocr_variant_mode=cli.ocr_variants,
        port=cli.port,
    )
