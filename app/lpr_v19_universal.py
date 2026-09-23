
import cv2
import json
import os
import sys
import time
import subprocess
import threading
import queue
from pathlib import Path
from multiprocessing.connection import Listener



ROOT = Path(__file__).resolve().parent

# Recognition logic (constants, normalization, voting, vehicle switching)
# lives in lpr_recognizer.py and is shared with the HTTP server, so both
# entry points run exactly the same rules.
sys.path.insert(0, str(ROOT))
from lpr_recognizer import (  # noqa: E402
    LPRRecognizer,
    OCR_EVERY_N_DETECTIONS, SQUARE_ASPECT_MAX, MIN_SQUARE_W, MIN_SQUARE_H,
    MIN_FINAL_WEIGHT,
    valid_kz_plate, clean_text, normalize_top, normalize_bottom,
    add_vote, aggregate_all, square_candidate,
)

# Video path from the command line, relative to the project root.
DEFAULT_VIDEO_NAME = "videos/20260909_171120.mp4"
VIDEO_ARG = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_NAME
VIDEO = (ROOT.parent / VIDEO_ARG) if not Path(VIDEO_ARG).is_absolute() else Path(VIDEO_ARG)
VIDEO_STEM = VIDEO.stem

ONNX_MODEL = ROOT.parent / "model" / "best_512.onnx"
# Results/video output are namespaced by the input video's filename, so
# running against a second video never overwrites the first run's results.
RUN_DIR = ROOT.parent / "runs" / f"real_video_v18_{VIDEO_STEM}"
OUT = RUN_DIR / "results_vehicle_switch_GPU.json"
VIDEO_OUT = RUN_DIR / "result_vehicle_switch_GPU.mp4"

GPU_VENV = ROOT.parent / ".venv_gpu"
YOLO_VENV = GPU_VENV
OCR_VENV = GPU_VENV
YOLO_PYTHON = GPU_VENV / "bin" / "python"
OCR_PYTHON = GPU_VENV / "bin" / "python"

CUDA_LIB = ":".join([
    str(GPU_VENV / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
    str(GPU_VENV / "lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"),
    str(GPU_VENV / "lib/python3.12/site-packages/nvidia/cublas/lib"),
    str(GPU_VENV / "lib/python3.12/site-packages/nvidia/cudnn/lib"),
    str(GPU_VENV / "lib/python3.12/site-packages/nvidia/curand/lib"),
    str(GPU_VENV / "lib/python3.12/site-packages/nvidia/cufft/lib"),
    str(GPU_VENV / "lib/python3.12/site-packages/nvidia/nvjitlink/lib"),
])


# LIVE_MODE=False (recorded file): the YOLO queue blocks until the worker is
# free, so every frame selected by FRAME_STEP reaches YOLO and runs are
# repeatable.
# LIVE_MODE=True (live camera): only the newest frame is kept and an older
# unprocessed one is dropped, because latency matters more than completeness.
LIVE_MODE = False

FRAME_STEP = 6



# Vehicle switching: keep the currently confirmed plate until a different
# complete KZ plate is independently read at least twice in a short window.
# A single very strong valid read can confirm a new vehicle.


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


print("=" * 72)
print("REAL VIDEO TEST — v9 SQUARE TEMPORAL ROW VOTING")
print("=" * 72)
print(f"Видео: {VIDEO}")
print("Цель: устойчиво подтвердить все считанные номера, любой формы")
print("Метод: независимое временное голосование верхней/нижней строки")
print()


YOLO_WORKER = ROOT.parent / "workers" / "yolo_gpu_worker.py"
OCR_WORKER = ROOT.parent / "workers" / "ocr_gpu_worker.py"


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

GPU_YOLO = None
YOLO_READY = None
OCR_READY = None

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

    yp = _spawn(YOLO_WORKER, YOLO_PYTHON, CUDA_LIB, (ONNX_MODEL,), yl)

    OCR_VARIANT_MODE = os.environ.get("OCR_VARIANT_MODE", "full")
    print("OCR VARIANT MODE:", OCR_VARIANT_MODE, flush=True)
    op = _spawn(OCR_WORKER, OCR_PYTHON, CUDA_LIB, (OCR_VARIANT_MODE,), ol)

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
                        payload = msg["payload"]
                        print(
                            "[OCR PAYLOAD KEYS]",
                            list(payload.keys()),
                            flush=True
                        )
                        ocr_result_q.put(
                            (mode, payload, t, det)
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

    # YOLO runs in its own thread, so reading and writing frames continue
    # while the GPU is busy. Results come back through yolo_result_q together
    # with the frame they belong to. See LIVE_MODE for the queue behaviour.
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

    # Vehicle switching, including the guard against out-of-order OCR
    # results, is decided by the shared LPRRecognizer. confirmed_plate mirrors
    # its state for the rest of this loop; switch_events is the same list.
    switcher = LPRRecognizer()
    confirmed_plate = ""
    switch_events = switcher.switch_events

    def consider_plate(candidate, conf, t, source):
        nonlocal confirmed_plate
        ignored_before = switcher.out_of_order_decisions
        changed = switcher.consider_plate(candidate, conf, t, source)
        if switcher.out_of_order_decisions > ignored_before:
            print(
                f"[OUT-OF-ORDER DECISION IGNORED] "
                f"t={t:.2f}s < last_decision_t={switcher.last_decision_t:.2f}s "
                f"candidate={candidate} source={source}",
                flush=True,
            )
        if changed:
            old = confirmed_plate
            confirmed_plate = switcher.confirmed_plate
            print(f"*** VEHICLE SWITCH: {old or '-'} -> {candidate} ({source}) ***", flush=True)
        return changed

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
                        "ms": round(float(payload.get("ms", 0.0)), 2),
                        "decode_ms": round(float(payload.get("decode_ms", 0.0)), 2),
                        "variant_count": int(payload.get("variant_count", 0)),
                        "prep_total_ms": round(float(payload.get("prep_total_ms", 0.0)), 2),
                        "infer_total_ms": round(float(payload.get("infer_total_ms", 0.0)), 2),
                        "variants": payload.get("variants", []),
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
                    "ms": round(float(payload.get("ms", 0.0)), 2),
                    "decode_ms": round(float(payload.get("decode_ms", 0.0)), 2),
                    "variant_count": int(payload.get("variant_count", 0)),
                    "prep_total_ms": round(float(payload.get("prep_total_ms", 0.0)), 2),
                    "infer_total_ms": round(float(payload.get("infer_total_ms", 0.0)), 2),
                    "variants": payload.get("variants", []),
                })

                add_vote(top_votes, top, top_conf, t)
                add_vote(bottom_votes, bottom, bottom_conf, t)

                # Same decision as the HTTP server: see square_candidate()
                # in lpr_recognizer.py.
                candidate, final_conf = square_candidate(
                    top_votes, bottom_votes, square_readings, t
                )

                if candidate:
                    add_vote(final_votes, candidate, final_conf, t)
                    # Both row weights above their minimums is enough to
                    # confirm. Waiting for the combined vote to repeat made
                    # short-lived square plates leave the frame first.
                    consider_plate(candidate, final_conf, t, "square")

                visual_status = "SQUARE / OCR"
                visual_text = candidate or (
                    f"TOP={top or '-'}  BOTTOM={bottom or '-'}"
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

                    # While a vehicle is confirmed, also read this crop as a
                    # single-row plate: when the camera moves to the next
                    # car, YOLO's box can stay square for a few frames.
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

            # Every frame is drawn and written, whether or not it went to
            # YOLO; the overlay shows the latest detection and OCR state.
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

    # End-of-run report over the whole vote history. Online decisions use
    # only the last WINDOW_SEC seconds (see lpr_recognizer.py).
    top_summary = aggregate_all(top_votes)
    bottom_summary = aggregate_all(bottom_votes)
    final_summary = aggregate_all(final_votes)

    confirmed_square = ""
    if final_summary and final_summary[0][1] >= MIN_FINAL_WEIGHT:
        confirmed_square = final_summary[0][0]

    print()
    print("=" * 72)
    print("SQUARE ROW TEMPORAL VOTING RESULT")
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
        "version": "lpr_v19_universal",
        "ocr_variant_mode": OCR_VARIANT_MODE,
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
        "out_of_order_decisions_ignored": switcher.out_of_order_decisions,
        "last_decision_t": switcher.last_decision_t,
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
    print(f"ИТОГ — lpr_v19_universal, режим OCR: {OCR_VARIANT_MODE}")
    print("=" * 72)
    print(f"FPS:                 {fps:.2f}")
    print(f"Длительность:        {duration:.2f} сек")
    print(f"Detection:           {detections}")
    print(f"OCR попыток:         {ocr_attempts}")
    print(f"Square OCR кадров:   {square_ocr_frames}")
    print(f"ПОДТВЕРЖДЁННЫЙ SQUARE: {confirmed_square or 'НЕТ'}")
    print(f"ПОДТВЕРЖДЁННЫЙ FINAL:  {confirmed_plate or 'НЕТ'}")
    print(f"OUT-OF-ORDER DECISIONS IGNORED: {switcher.out_of_order_decisions}")
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
