"""
Spawns and talks to the two GPU worker subprocesses (workers/yolo_gpu_worker.py,
workers/ocr_gpu_worker.py) over multiprocessing.connection, exactly the same
IPC mechanism lpr_v19_universal.py and the old lpr_camera_server.py prototype
each used with their own embedded copies of this worker code. This file
replaces both of those embedded copies with ONE spawner pointed at the real,
shared worker files, so there is exactly one YOLO worker implementation and
one OCR worker implementation in the whole project (see workers/ directory).

This class is intentionally synchronous (one in-flight request per worker at
a time, guarded by a lock) -- for a live HTTP camera server, each frame
already arrives as one request that naturally waits for its response before
the client sends the next one. The async, backpressure/drop-stale FrameRelay
machinery in lpr_v19_universal.py exists to solve a DIFFERENT problem
(reading a recorded video file faster than the GPU can keep up) that does
not apply to a live per-frame API. See docs/architecture.md for this
reasoning written out in full.

NOT YET RUN against real GPU hardware from this environment (no GPU here).
Structurally mirrors the Workers class in the current lpr_camera_server.py
prototype, which IS known to work against the real GPU server, but updated
to the fid/jid protocol that workers/yolo_gpu_worker.py and
workers/ocr_gpu_worker.py (extracted from v19) actually speak. Verify with
the checklist in docs/architecture.md before relying on this in production.
"""

import os
import subprocess
import threading
import time
from itertools import count
from multiprocessing.connection import Listener
from pathlib import Path

import cv2


class WorkerStartupError(RuntimeError):
    pass


class Workers:
    def __init__(self, project_root, yolo_python, ocr_python, onnx_model,
                 yolo_cuda_ld_path, ocr_cuda_ld_path, ocr_variant_mode="full"):
        """
        project_root: Path to the directory containing workers/.
        yolo_python / ocr_python: Path to each worker's own venv interpreter.
        onnx_model: Path to best_512.onnx.
        ocr_variant_mode: "full" (production, validated) or "no-enhanced"
            (experiment B: skip the cv2.detailEnhance variant). Passed
            straight through to the OCR worker; changes nothing else.
        *_cuda_ld_path: LD_LIBRARY_PATH string each worker's venv needs to
            find its CUDA libraries (copy from config/gpu_env.example.py or
            your own working lpr_v19_universal.py / lpr_camera_server.py --
            these paths are environment-specific and NOT re-derived here).
        """
        self.ocr_variant_mode = ocr_variant_mode
        self._yolo_fid = count(1)
        self._ocr_jid = count(1)
        self._lock = threading.Lock()

        yolo_script = project_root / "workers" / "yolo_gpu_worker.py"
        ocr_script = project_root / "workers" / "ocr_gpu_worker.py"

        for script in (yolo_script, ocr_script):
            if not script.exists():
                raise WorkerStartupError(f"Не найден файл воркера: {script}")
        if not Path(onnx_model).exists():
            raise WorkerStartupError(f"Не найдена модель: {onnx_model}")
        for py in (yolo_python, ocr_python):
            if not Path(py).exists():
                raise WorkerStartupError(f"Не найден интерпретатор: {py}")

        log_dir = project_root / "runs" / "worker_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_log = log_dir / "yolo_worker.log"
        self.ocr_log = log_dir / "ocr_worker.log"

        yl = Listener(("127.0.0.1", 0), authkey=os.urandom(32))
        ol = Listener(("127.0.0.1", 0), authkey=os.urandom(32))

        self._yolo_proc = self._spawn(
            yolo_script, yolo_python, yolo_cuda_ld_path, yl,
            extra_args=[str(onnx_model)], log_path=self.yolo_log,
        )
        self._ocr_proc = self._spawn(
            ocr_script, ocr_python, ocr_cuda_ld_path, ol,
            extra_args=[ocr_variant_mode], log_path=self.ocr_log,
        )

        self._yolo_conn = self._accept_with_timeout(
            yl, self._yolo_proc, "YOLO", self.yolo_log)
        yolo_ready = self._yolo_conn.recv()
        if yolo_ready.get("type") != "ready":
            raise WorkerStartupError(f"YOLO-воркер не готов: {yolo_ready}")

        self._ocr_conn = self._accept_with_timeout(
            ol, self._ocr_proc, "OCR", self.ocr_log)
        ocr_ready = None
        while True:
            msg = self._ocr_conn.recv()
            if msg.get("type") == "booting":
                print(f"  OCR: {msg.get('stage')}", flush=True)
                continue
            ocr_ready = msg
            break
        if ocr_ready.get("type") != "ready":
            raise WorkerStartupError(f"OCR-воркер не готов: {ocr_ready}")

        self.yolo_info = yolo_ready
        self.ocr_info = ocr_ready
        yl.close()
        ol.close()

    @staticmethod
    def _spawn(script, python_exe, ld_path, listener, extra_args, log_path):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ld_path + ":" + env.get("LD_LIBRARY_PATH", "")
        args = [
            str(python_exe), str(script),
            "127.0.0.1", str(listener.address[1]), listener._authkey.hex(),
            *extra_args,
        ]
        # Worker output goes to a log file instead of being discarded, so a
        # crash at startup (import error, missing CUDA lib, etc.) is
        # actually visible instead of just hanging the accept() below.
        log_file = open(log_path, "wb")
        return subprocess.Popen(args, env=env,
                                 stdout=log_file, stderr=subprocess.STDOUT)

    @staticmethod
    def _accept_with_timeout(listener, proc, name, log_path, timeout=120.0):
        """accept() blocks forever if the worker died on startup. Poll the
        process while waiting and fail loudly with its log instead."""
        result = {}

        def _accept():
            try:
                result["conn"] = listener.accept()
            except Exception as exc:
                result["error"] = exc

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        deadline = time.time() + timeout
        while t.is_alive() and time.time() < deadline:
            if proc.poll() is not None:
                t.join(timeout=1.0)
                if "conn" in result:
                    break
                log = ""
                try:
                    log = open(log_path, encoding="utf-8", errors="replace").read()
                except OSError:
                    pass
                raise WorkerStartupError(
                    f"{name}-воркер завершился с кодом {proc.returncode}, "
                    f"не успев подключиться.\n"
                    f"Лог воркера ({log_path}):\n{log.strip() or '(пусто)'}"
                )
            t.join(timeout=0.2)

        if "error" in result:
            raise WorkerStartupError(f"{name}: ошибка accept: {result['error']!r}")
        if "conn" not in result:
            raise WorkerStartupError(
                f"{name}-воркер не подключился за {timeout:.0f} секунд. "
                f"Проверьте лог: {log_path}"
            )
        return result["conn"]

    def detect(self, frame_bgr):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t_encode = time.perf_counter()
        fid = next(self._yolo_fid)
        with self._lock:
            t_lock = time.perf_counter()
            self._yolo_conn.send({"type": "frame", "fid": fid, "jpeg": buf.tobytes()})
            msg = self._yolo_conn.recv()
            t_ipc = time.perf_counter()
        if msg["type"] == "error":
            raise RuntimeError(f"YOLO worker error: {msg['error']}")

        self.last_timing = {
            "jpeg_encode_ms": (t_encode - t0) * 1000.0,
            "lock_wait_ms": (t_lock - t_encode) * 1000.0,
            "ipc_roundtrip_ms": (t_ipc - t_lock) * 1000.0,
            "worker_inference_ms": float(msg.get("ms", 0.0)),
            "jpeg_bytes": len(buf),
        }
        return msg["det"]

    def ocr(self, crop_bgr, mode):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        t_encode = time.perf_counter()
        jid = next(self._ocr_jid)
        with self._lock:
            t_lock = time.perf_counter()
            self._ocr_conn.send({
                "type": "ocr", "mode": mode, "jpeg": buf.tobytes(),
                "jid": jid, "fid": jid,
            })
            msg = self._ocr_conn.recv()
            t_ipc = time.perf_counter()
        if msg["type"] == "error":
            raise RuntimeError(f"OCR worker error: {msg['error']}")

        payload = msg["payload"]
        self.last_timing = {
            "mode": mode,
            "jpeg_encode_ms": (t_encode - t0) * 1000.0,
            "lock_wait_ms": (t_lock - t_encode) * 1000.0,
            "ipc_roundtrip_ms": (t_ipc - t_lock) * 1000.0,
            "worker_inference_ms": float(payload.get("ms", 0.0)),
            "jpeg_bytes": len(buf),
            "crop_size": [int(crop_bgr.shape[1]), int(crop_bgr.shape[0])],
            # Deeper breakdown reported by the OCR worker itself.
            "worker_decode_ms": payload.get("decode_ms"),
            "worker_prep_ms": payload.get("prep_total_ms"),
            "worker_infer_ms": payload.get("infer_total_ms"),
            "variant_count": payload.get("variant_count"),
            "variants": payload.get("variants", []),
        }
        return payload

    def close(self):
        for conn in (getattr(self, "_yolo_conn", None), getattr(self, "_ocr_conn", None)):
            try:
                conn.send({"type": "stop"})
            except Exception:
                pass
        for proc in (getattr(self, "_yolo_proc", None), getattr(self, "_ocr_proc", None)):
            try:
                proc.terminate()
            except Exception:
                pass
