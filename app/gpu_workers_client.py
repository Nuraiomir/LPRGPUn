"""
Starts the YOLO and OCR worker subprocesses and talks to them over
multiprocessing.connection.

Each worker handles one request at a time. The two workers have separate
locks, so while one request is in OCR another can already be in YOLO.

Every call waits for its reply for a bounded time. If a worker does not
answer in time, or its process has died, it is restarted and the call raises
WorkerError; the next call goes to the fresh process. Replies carry the
request id, so a late reply from an abandoned request is discarded instead of
being returned for a different frame.

Worker output goes to runs/worker_logs/*.log and is appended across restarts,
so the reason for a crash stays visible.
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
    """A worker process could not be started or did not report ready."""


class WorkerError(RuntimeError):
    """A request failed: the worker timed out, crashed, or reported an error."""


def _read_log(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _accept(listener, proc, name, log_path, timeout):
    """listener.accept() with a deadline that also notices a dead process."""
    result = {}

    def run():
        try:
            result["conn"] = listener.accept()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        if proc.poll() is not None:
            thread.join(timeout=1.0)
            if "conn" in result:
                break
            raise WorkerStartupError(
                f"{name} worker exited with code {proc.returncode} before connecting.\n"
                f"Log ({log_path}):\n{_read_log(log_path)[-4000:] or '(empty)'}"
            )
        thread.join(timeout=0.2)

    if "error" in result:
        raise WorkerStartupError(f"{name}: accept failed: {result['error']!r}")
    if "conn" not in result:
        raise WorkerStartupError(
            f"{name} worker did not connect within {timeout:.0f} s. See {log_path}"
        )
    return result["conn"]


class _WorkerProcess:
    """One worker subprocess, its connection, and restart handling."""

    def __init__(self, name, script, python, ld_path, extra_args, log_path,
                 id_key, reply_timeout, startup_timeout):
        self.name = name
        self._script = script
        self._python = python
        self._ld_path = ld_path
        self._extra_args = [str(a) for a in extra_args]
        self.log_path = log_path
        self._id_key = id_key
        self.reply_timeout = reply_timeout
        self._startup_timeout = startup_timeout
        self._ids = count(1)
        self._lock = threading.Lock()
        self.restarts = 0
        self.proc = None
        self.conn = None
        self.info = None
        self._start()

    def _start(self):
        authkey = os.urandom(32)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        try:
            env = os.environ.copy()
            if self._ld_path:
                env["LD_LIBRARY_PATH"] = self._ld_path + ":" + env.get("LD_LIBRARY_PATH", "")
            args = [str(self._python), str(self._script),
                    "127.0.0.1", str(listener.address[1]), authkey.hex(),
                    *self._extra_args]
            with open(self.log_path, "ab") as log:
                self.proc = subprocess.Popen(args, env=env, stdout=log,
                                             stderr=subprocess.STDOUT)
            self.conn = _accept(listener, self.proc, self.name, self.log_path,
                                self._startup_timeout)
        finally:
            listener.close()

        deadline = time.monotonic() + self._startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.conn.poll(remaining):
                raise WorkerStartupError(f"{self.name} worker did not report ready")
            msg = self.conn.recv()
            if msg.get("type") == "booting":
                print(f"  {self.name}: {msg.get('stage')}", flush=True)
                continue
            break
        if msg.get("type") != "ready":
            raise WorkerStartupError(f"{self.name} worker not ready: {msg}")
        self.info = msg

    def _stop(self):
        try:
            self.conn.send({"type": "stop"})
        except Exception:  # noqa: BLE001
            pass
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def _restart(self, reason):
        print(f"[WORKER] restarting {self.name}: {reason}", flush=True)
        self._stop()
        self.restarts += 1
        self._start()

    def _reply_id(self, msg):
        if self._id_key in msg:
            return msg[self._id_key]
        payload = msg.get("payload")
        if isinstance(payload, dict):
            return payload.get(self._id_key)
        return None

    def request(self, message):
        """Sends one request and returns (reply, lock_wait_s, roundtrip_s)."""
        t_wait = time.perf_counter()
        with self._lock:
            t_locked = time.perf_counter()
            if self.proc.poll() is not None:
                self._restart(f"process exited with code {self.proc.returncode}")

            req_id = next(self._ids)
            message = dict(message, **{self._id_key: req_id})
            try:
                self.conn.send(message)
                deadline = time.monotonic() + self.reply_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self.conn.poll(remaining):
                        self._restart(f"no reply within {self.reply_timeout:.0f} s")
                        raise WorkerError(f"{self.name} worker timed out")
                    msg = self.conn.recv()
                    if self._reply_id(msg) == req_id:
                        break
                    # A reply to an earlier request that was abandoned; drop it.
            except (EOFError, OSError) as exc:
                self._restart(f"connection lost ({exc!r})")
                raise WorkerError(f"{self.name} worker connection lost") from exc
            t_done = time.perf_counter()

        if msg.get("type") == "error":
            raise WorkerError(f"{self.name} worker error: {msg.get('error')}")
        return msg, t_locked - t_wait, t_done - t_locked

    def close(self):
        with self._lock:
            self._stop()


class Workers:
    """
    project_root:   directory that contains workers/
    yolo_python, ocr_python:  interpreters of the worker environments
    onnx_model:     path to best_512.onnx
    *_cuda_ld_path: LD_LIBRARY_PATH for each worker (see config/gpu_env.example.py)
    ocr_variant_mode: "full" or "no-enhanced" (skips cv2.detailEnhance)
    """

    def __init__(self, project_root, yolo_python, ocr_python, onnx_model,
                 yolo_cuda_ld_path, ocr_cuda_ld_path, ocr_variant_mode="full",
                 yolo_timeout=10.0, ocr_timeout=15.0, startup_timeout=120.0):
        project_root = Path(project_root)
        yolo_script = project_root / "workers" / "yolo_gpu_worker.py"
        ocr_script = project_root / "workers" / "ocr_gpu_worker.py"

        for path, what in ((yolo_script, "worker script"), (ocr_script, "worker script"),
                           (Path(onnx_model), "model"), (Path(yolo_python), "interpreter"),
                           (Path(ocr_python), "interpreter")):
            if not path.exists():
                raise WorkerStartupError(f"{what} not found: {path}")

        log_dir = project_root / "runs" / "worker_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_log = log_dir / "yolo_worker.log"
        self.ocr_log = log_dir / "ocr_worker.log"
        self.ocr_variant_mode = ocr_variant_mode
        self._timing = threading.local()

        self._yolo = _WorkerProcess(
            "YOLO", yolo_script, yolo_python, yolo_cuda_ld_path, [onnx_model],
            self.yolo_log, id_key="fid", reply_timeout=yolo_timeout,
            startup_timeout=startup_timeout)
        try:
            self._ocr = _WorkerProcess(
                "OCR", ocr_script, ocr_python, ocr_cuda_ld_path, [ocr_variant_mode],
                self.ocr_log, id_key="jid", reply_timeout=ocr_timeout,
                startup_timeout=startup_timeout)
        except Exception:
            self._yolo.close()
            raise

        self.yolo_info = self._yolo.info
        self.ocr_info = self._ocr.info

    @property
    def last_timing(self):
        """Timing of the last call made by the current thread."""
        return getattr(self._timing, "value", None)

    @property
    def restarts(self):
        return {"yolo": self._yolo.restarts, "ocr": self._ocr.restarts}

    def detect(self, frame_bgr):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise WorkerError("JPEG encode failed for YOLO input")
        t_encode = time.perf_counter()
        msg, lock_wait, roundtrip = self._yolo.request({"type": "frame", "jpeg": buf.tobytes()})
        self._timing.value = {
            "jpeg_encode_ms": (t_encode - t0) * 1000.0,
            "lock_wait_ms": lock_wait * 1000.0,
            "ipc_roundtrip_ms": roundtrip * 1000.0,
            "worker_inference_ms": float(msg.get("ms", 0.0)),
            "jpeg_bytes": len(buf),
        }
        return msg["det"]

    def ocr(self, crop_bgr, mode):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise WorkerError("JPEG encode failed for OCR input")
        t_encode = time.perf_counter()
        msg, lock_wait, roundtrip = self._ocr.request(
            {"type": "ocr", "mode": mode, "jpeg": buf.tobytes(), "fid": 0})
        payload = msg["payload"]
        self._timing.value = {
            "mode": mode,
            "jpeg_encode_ms": (t_encode - t0) * 1000.0,
            "lock_wait_ms": lock_wait * 1000.0,
            "ipc_roundtrip_ms": roundtrip * 1000.0,
            "worker_inference_ms": float(payload.get("ms", 0.0)),
            "jpeg_bytes": len(buf),
            "crop_size": [int(crop_bgr.shape[1]), int(crop_bgr.shape[0])],
            "worker_decode_ms": payload.get("decode_ms"),
            "worker_prep_ms": payload.get("prep_total_ms"),
            "worker_infer_ms": payload.get("infer_total_ms"),
            "variant_count": payload.get("variant_count"),
            "variants": payload.get("variants", []),
        }
        return payload

    def close(self):
        for worker in (getattr(self, "_yolo", None), getattr(self, "_ocr", None)):
            if worker is not None:
                worker.close()
