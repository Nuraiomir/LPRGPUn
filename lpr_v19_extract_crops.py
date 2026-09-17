"""
lpr_v19_extract_crops.py -- OCR A/B experiment: crop extraction stage.

Runs ONLY the validated YOLO detection stage of lpr_v19_universal.py
against test videos and saves representative OCR-input crops to disk,
with metadata, so a later script (ocr_ab_test.py) can compare OCR models
offline on real crops. This script never performs OCR, temporal voting,
or vehicle switching -- it stops right where lpr_v19_universal.py would
normally call submit_ocr().

lpr_v19_universal.py itself is NEVER modified. It is imported as a module
(see _load_baseline() below) so the exact same objects/constants are
reused at runtime instead of being retyped -- this removes any risk of
transcription drift in the detection code.

--------------------------------------------------------------------------
REUSED FROM lpr_v19_universal.py (imported directly, not copied):
  - best_512.onnx + ONNX Runtime CUDAExecutionProvider GPU inference
  - the YOLO_WORKER subprocess script text (512x512 letterbox, decode,
    YOLO_CONF=0.40 threshold) -- the exact same file baseline writes and
    spawns, never re-typed
  - YOLO_VENV / YOLO_PYTHON / CUDA12 environment wiring, _spawn()/_env()
  - GPUYOLO IPC wrapper class
  - FRAME_STEP, YOLO_CONF, SQUARE_ASPECT_MAX, MIN_SQUARE_W, MIN_SQUARE_H

REPRODUCED VERBATIM (baseline keeps this as inline code inside
process_yolo_results() / the OCR_WORKER's square_ocr(), not as an
importable function, so the exact expressions are copied unchanged):
  - crop = frame[y1:y2, x1:x2]                     (crop coordinates)
  - aspect = w / max(1, h); the SQUARE_ASPECT_MAX / MIN_SQUARE_W /
    MIN_SQUARE_H classification into normal vs. square
  - the square branch's outer 5%/3% padding trim (py/px), followed by
    the SAME trim applied a second time plus the 48%/4% top/bottom row
    split, exactly as baseline's OCR_WORKER.square_ocr() does -- this is
    reproduced as a double trim on purpose, because that is literally
    what pixels reach the production OCR worker today.

INTENTIONALLY NOT REUSED / NOT PERFORMED (by design, per task spec):
  - temporal voting (add_vote / aggregate / best_top / best_bottom)
  - vehicle switching (consider_plate / switch_events)
  - OCR itself -- en_PP-OCRv5_mobile_rec is never loaded here; this
    script only saves the images OCR would have received
  - OCR_EVERY_N_DETECTIONS throttling (1-in-3 square sampling). That
    throttle exists in baseline purely to cap live OCR GPU load. Reusing
    it here for dataset building risks silently dropping short-lived
    square plates (e.g. a plate only ever detected 2-3 times) if the
    modulus happens not to land on them. It is replaced with a
    perceptual-hash de-duplicator (see CropDeduper) that only skips a
    frame when it is near-identical AND recently saved, and always saves
    the first frame of any new "burst" of detections regardless of
    similarity -- this is a deliberate, more conservative substitute.
  - the "NORMAL FALLBACK SUBMIT" resubmission on square detections
    (existed only to help vehicle-switch detection, not relevant here)
  - the async FrameRelay/threaded pipelining -- detection is called
    synchronously per sampled frame. There is no OCR call or video
    writer to overlap with in this script, so the added concurrency
    would only add risk without any benefit.

Usage:
    python lpr_v19_extract_crops.py video1.mp4 [video2.mp4 ...]
        [--out runs/ocr_ab_crops] [--expected-plates expected.json]

Run with whatever Python environment normally runs lpr_v19_universal.py
itself (needs cv2/numpy + stdlib multiprocessing). GPU YOLO inference
happens in a subprocess under YOLO_PYTHON, exactly as in baseline.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "lpr_v19_universal.py"


def _load_baseline():
    """
    Import lpr_v19_universal.py as a module WITHOUT running its main().
    main() only runs under `if __name__ == "__main__"`, so importing it
    here just gives us its constants/classes/functions/worker-script
    files -- it does not spawn any subprocess or open any video.
    """
    spec = importlib.util.spec_from_file_location(
        "lpr_v19_universal_baseline", BASELINE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    # Baseline reads sys.argv[1] as a video filename at import time.
    # Hide our own argv from it so it falls back to its own default
    # harmlessly (we never use baseline.VIDEO for anything).
    saved_argv = sys.argv
    sys.argv = [str(BASELINE_PATH)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved_argv
    return mod


baseline = _load_baseline()

# --- de-duplication tuning (extraction-only, not part of baseline) -----
DEDUP_HAMMING_MAX = 4        # of 64 hash bits; near-identical frame guard
DEDUP_MIN_SAVE_GAP_SEC = 0.75  # don't re-save a near-dup faster than this
BURST_GAP_SEC = 1.0          # gap since last detection of this kind that
                              # marks the start of a new "appearance" --
                              # its first frame is always saved


def _ahash(img, size=8):
    if img is None or img.size == 0:
        return np.zeros(size * size, dtype=bool)
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return gray > gray.mean()


def _hamming(a, b):
    return int(np.count_nonzero(a != b))


class CropDeduper:
    """
    Keeps the saved dataset representative instead of saving every
    FRAME_STEP-sampled detection. Tracked independently per "kind"
    ("normal" / "square", the square decision covers both split
    outputs together since they come from one detection).

    A frame is skipped only when BOTH:
      - it is visually near-identical (average-hash Hamming distance)
        to the last SAVED frame of the same kind, AND
      - it arrived within DEDUP_MIN_SAVE_GAP_SEC of that save.
    Any gap since the last DETECTION (not just save) of this kind
    longer than BURST_GAP_SEC is treated as a new appearance and is
    always saved -- this protects short-lived plates from ever being
    hashed away, since they rarely resemble whatever was saved before
    them and are, by definition, a fresh burst.
    """

    def __init__(self):
        self._last_hash = {}
        self._last_save_t = {}
        self._last_detect_t = {}

    def should_save(self, kind, image, t):
        h = _ahash(image)
        last_detect_t = self._last_detect_t.get(kind)
        new_burst = last_detect_t is None or (t - last_detect_t) > BURST_GAP_SEC
        self._last_detect_t[kind] = t

        if new_burst:
            save = True
        else:
            prev_h = self._last_hash.get(kind)
            if prev_h is None:
                save = True
            else:
                prev_save_t = self._last_save_t.get(kind, -1e9)
                near_dup = _hamming(h, prev_h) <= DEDUP_HAMMING_MAX
                too_soon = (t - prev_save_t) < DEDUP_MIN_SAVE_GAP_SEC
                save = not (near_dup and too_soon)

        if save:
            self._last_hash[kind] = h
            self._last_save_t[kind] = t
        return save


def start_yolo_worker():
    """Spawns the SAME YOLO worker subprocess baseline's main() spawns."""
    yl = Listener(("127.0.0.1", 0), authkey=os.urandom(32))
    yp = baseline._spawn(
        baseline.YOLO_WORKER, baseline.YOLO_PYTHON, baseline.CUDA12,
        (baseline.ONNX_MODEL,), yl,
    )
    yc = yl.accept()
    ready = yc.recv()
    if ready.get("type") != "ready":
        raise RuntimeError(f"YOLO worker failed to start: {ready}")
    print(f"[YOLO] GPU worker ready: {ready}", flush=True)
    return yl, yp, yc, baseline.GPUYOLO(yc)


def classify_and_prepare(frame, det):
    """
    Reproduces, verbatim, the crop coordinates / aspect classification /
    square preprocessing from lpr_v19_universal.py's process_yolo_results()
    and the OCR_WORKER's square_ocr(). Returns None if there is nothing
    usable to save.
    """
    x1, y1, x2, y2, conf = det
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]
    aspect = w / max(1, h)

    if aspect > baseline.SQUARE_ASPECT_MAX or (
        w < baseline.MIN_SQUARE_W or h < baseline.MIN_SQUARE_H
    ):
        return {
            "kind": "normal",
            "images": {"normal": crop},
            "dedup_image": crop,
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "orig_w": w,
            "orig_h": h,
            "aspect": aspect,
        }

    # Square branch, outer trim -- verbatim from process_yolo_results()
    # right before submit_ocr("square", sq, t, det).
    py = max(2, int(h * .05))
    px = max(2, int(w * .03))
    sq = crop[py:max(py + 1, h - py), px:max(px + 1, w - px)]
    if sq.size == 0:
        return None

    # Square branch, inner trim + row split -- verbatim from
    # OCR_WORKER's square_ocr(crop). This is a SECOND 5%/3% trim on top
    # of the one above, followed by a 48%/4% top/bottom split. Kept as
    # a double trim on purpose: these are the exact pixels production
    # OCR receives today.
    sh0, sw0 = sq.shape[:2]
    py2, px2 = max(2, int(sh0 * .05)), max(2, int(sw0 * .03))
    s = sq[py2:sh0 - py2, px2:sw0 - px2]
    if s.size == 0:
        return None
    sh = s.shape[0]
    split, gap = int(sh * .48), max(1, int(sh * .04))
    top = s[:split]
    bottom = s[min(sh, split + gap):]
    if top.size == 0 or bottom.size == 0:
        return None

    return {
        "kind": "square",
        "images": {"square_top": top, "square_bottom": bottom},
        "dedup_image": sq,
        "bbox": (x1, y1, x2, y2),
        "conf": conf,
        "orig_w": w,
        "orig_h": h,
        "aspect": aspect,
    }


def extract_video(video_path, out_root, expected_plates, gpu_yolo):
    video_path = Path(video_path)
    stem = video_path.stem
    out_dir = out_root / stem
    dirs = {
        "normal": out_dir / "normal",
        "square_top": out_dir / "square_top",
        "square_bottom": out_dir / "square_bottom",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    dedup = CropDeduper()
    manifest = []
    frame_idx = 0
    scanned = 0
    detections = 0
    saved_counts = {"normal": 0, "square_top": 0, "square_bottom": 0}
    skipped_dup = 0

    expected = expected_plates.get(stem) if expected_plates else None

    print(f"\n[{stem}] extracting from {video_path} (fps={fps:.2f})"
          + (f" expected_plate={expected}" if expected else ""), flush=True)

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % baseline.FRAME_STEP != 0:
            continue
        scanned += 1
        t = frame_idx / fps

        det = gpu_yolo.detect(frame)
        if det is None:
            continue
        detections += 1

        prep = classify_and_prepare(frame, det)
        if prep is None:
            continue

        if not dedup.should_save(prep["kind"], prep["dedup_image"], t):
            skipped_dup += 1
            continue

        crop_id = f"{stem}_f{frame_idx:07d}"
        for crop_type, image in prep["images"].items():
            fname = f"{crop_id}_{crop_type}.jpg"
            fpath = dirs[crop_type] / fname
            cv2.imwrite(str(fpath), image)
            saved_counts[crop_type] += 1
            manifest.append({
                "crop_id": crop_id,
                "video": video_path.name,
                "video_stem": stem,
                "frame_number": frame_idx,
                "timestamp_sec": round(t, 3),
                "crop_type": crop_type,
                "bbox": [int(v) for v in prep["bbox"][:4]],
                "detection_confidence": round(float(prep["conf"]), 4),
                "orig_crop_width": int(prep["orig_w"]),
                "orig_crop_height": int(prep["orig_h"]),
                "aspect_ratio": round(float(prep["aspect"]), 3),
                "saved_width": int(image.shape[1]),
                "saved_height": int(image.shape[0]),
                # Absolute path -- kept independent of ROOT so --out may
                # point anywhere (different disk/mount), and so
                # ocr_ab_test.py can load crops without assuming they
                # live under the repo directory.
                "file": str(fpath.resolve()),
                "expected_plate": expected,
            })

        if scanned % 200 == 0:
            print(
                f"  ... scanned {scanned} sampled frames, "
                f"{detections} detections, "
                f"{sum(saved_counts.values())} crops saved "
                f"({skipped_dup} near-dup skipped)", flush=True
            )

    cap.release()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.time() - t0
    print(
        f"[{stem}] sampled={scanned} detections={detections} "
        f"saved: normal={saved_counts['normal']} "
        f"square_top={saved_counts['square_top']} "
        f"square_bottom={saved_counts['square_bottom']} "
        f"skipped_near_dup={skipped_dup} elapsed={elapsed:.1f}s", flush=True
    )
    print(f"  manifest: {manifest_path}", flush=True)
    return manifest_path


def load_expected_plates(path):
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "videos", nargs="*", default=[baseline.DEFAULT_VIDEO_NAME],
        help="Video file(s) to extract crops from",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "runs" / "ocr_ab_crops"),
        help="Output root directory (default: runs/ocr_ab_crops)",
    )
    parser.add_argument(
        "--expected-plates", default=None,
        help="Optional JSON file mapping video_stem -> expected plate "
             "string (e.g. {\"20260909_171120\": \"979CBB02\"}), used "
             "later by ocr_ab_test.py for exact-match accuracy",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    expected_plates = load_expected_plates(args.expected_plates)

    print("=" * 72)
    print("OCR A/B CROP EXTRACTION (detection-only reuse of lpr_v19_universal.py)")
    print("=" * 72)
    print(f"Videos: {args.videos}")
    print(f"Output: {out_root}")
    print(
        f"FRAME_STEP={baseline.FRAME_STEP} YOLO_CONF={baseline.YOLO_CONF} "
        f"SQUARE_ASPECT_MAX={baseline.SQUARE_ASPECT_MAX} "
        f"MIN_SQUARE_W={baseline.MIN_SQUARE_W} MIN_SQUARE_H={baseline.MIN_SQUARE_H}"
    )

    yl, yp, yc, gpu_yolo = start_yolo_worker()
    manifest_paths = []
    try:
        for v in args.videos:
            vpath = (ROOT / v) if not Path(v).is_absolute() else Path(v)
            manifest_paths.append(
                extract_video(vpath, out_root, expected_plates, gpu_yolo)
            )
    finally:
        try:
            yc.send({"type": "stop"})
        except Exception:
            pass
        try:
            yc.close()
            yl.close()
        except Exception:
            pass
        try:
            yp.terminate()
            yp.wait(timeout=10)
        except Exception:
            try:
                yp.kill()
            except Exception:
                pass

    print()
    print("Done. Manifests:")
    for p in manifest_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
