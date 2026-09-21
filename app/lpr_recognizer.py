"""
LPR recognition core — the ONE authoritative implementation of everything
downstream of "here is a detected plate crop": classification, OCR-mode
selection, Kazakhstan normalization, temporal voting, and vehicle-switching.

PROVENANCE: every constant, regex, and formula in this file is copied
verbatim from lpr_v19_universal.py (validated on three real videos: the
545BDR05 / 979CBB02 / 049BXS02 sequences, all confirmed correctly, including
the v19 fix for short-lived square plates). Nothing here has been re-derived
or "improved" -- only reorganized from v19's nested closures (which only
make sense inside one long-running video-processing loop) into a class with
instance state, so the exact same logic can be called:
  (a) once per frame from a live HTTP request (app/lpr_api_server.py), and
  (b) once per selected frame from an offline video loop, if a future
      refactor of lpr_v19_universal.py itself wants to reuse it too.

lpr_v19_universal.py has NOT been modified and remains the reference
implementation. This module is a faithful extraction, not a rewrite, but it
has only been exercised by this project's unit tests (tests/test_lpr_recognizer.py)
with synthetic inputs -- it has NOT yet been run against the real GPU
workers on the three reference videos. Before trusting it as equivalent to
v19 in production, run it against the same three videos via
client/camera_client.py pointed at app/lpr_api_server.py and diff the
resulting plate sequences against v19's own JSON output. See
docs/architecture.md "Verification status" for exactly this checklist.

One deliberate behavioral note carried over unchanged from v19, flagged here
because it was tuned for a 60fps *offline* video loop and has not been
re-examined for a live camera feed at a lower, client-controlled frame rate:
OCR_EVERY_N_DETECTIONS=3 means only every 3rd square-classified detection
actually gets OCR'd. If the camera client already paces itself to a low fps
(see client/camera_client.py --fps), stacking this 1-in-3 sub-sampling on
top may be too aggressive. This is flagged, not changed, per the explicit
instruction not to alter recognition/voting behavior without justification.
"""

import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants -- copied verbatim from lpr_v19_universal.py. Do not change a
# value here without re-validating against the three reference videos.
# ---------------------------------------------------------------------------

YOLO_CONF = 0.40
SQUARE_ASPECT_MAX = 1.80
MIN_SQUARE_W = 130
MIN_SQUARE_H = 85

OCR_EVERY_N_DETECTIONS = 3  # see module docstring note above

WINDOW_SEC = 3.0
MIN_TOP_WEIGHT = 1.60
MIN_BOTTOM_WEIGHT = 2.00
MIN_FINAL_WEIGHT = 2.50  # kept only for aggregate_all()/reporting; the
                          # online gate below no longer re-checks this
                          # separately -- see the v19 comment preserved in
                          # LPRRecognizer._handle_square_result().

SWITCH_WINDOW_SEC = 1.5
SWITCH_CONFIRM_READS = 2
SWITCH_STRONG_CONF = 0.95


# ---------------------------------------------------------------------------
# Pure functions -- copied verbatim from lpr_v19_universal.py.
# ---------------------------------------------------------------------------

def valid_kz_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return bool(re.fullmatch(r"\d{3}[A-Z]{3}\d{2}", text))


def clean_text(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def normalize_top(s):
    """
    Square top row should be exactly 3 digits.
    We deliberately do NOT aggressively convert arbitrary letters to digits.
    This avoids turning a stray word into a false numeric candidate.
    """
    s = clean_text(s)

    m = re.search(r"(\d{3})", s)
    if m:
        return m.group(1)

    if len(s) == 3:
        trans = str.maketrans({
            "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
            "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6",
            "T": "7", "B": "8", "P": "9",
        })
        mapped = s.translate(trans)
        if mapped.isdigit() and len(mapped) == 3:
            return mapped

    return ""


def normalize_bottom(s):
    """Returns bottom row as RRLLL, e.g. 02BBT."""
    s = clean_text(s).replace("KZ", "")

    m = re.fullmatch(r"(\d{2})([A-Z]{3})", s)
    if m:
        return m.group(1) + m.group(2)

    m = re.fullmatch(r"([A-Z]{3})(\d{2})", s)
    if m:
        return m.group(2) + m.group(1)

    m = re.search(r"(\d{2})([A-Z]{3})", s)
    if m:
        return m.group(1) + m.group(2)

    m = re.search(r"([A-Z]{3})(\d{2})", s)
    if m:
        return m.group(2) + m.group(1)

    return ""


def add_vote(votes, value, conf, t):
    if not value or conf < 0.50:
        return
    weight = float(conf)
    if conf >= 0.90:
        weight += 0.40
    votes.append({"time": float(t), "value": value, "weight": weight, "conf": float(conf)})


def aggregate(votes, now):
    agg = defaultdict(lambda: {"weight": 0.0, "count": 0, "best_conf": 0.0})
    for v in votes:
        if now - v["time"] <= WINDOW_SEC:
            a = agg[v["value"]]
            a["weight"] += v["weight"]
            a["count"] += 1
            a["best_conf"] = max(a["best_conf"], v["conf"])
    result = [(value, a["weight"], a["count"], a["best_conf"]) for value, a in agg.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def best_top(votes, now):
    agg = aggregate(votes, now)
    if agg and agg[0][1] >= MIN_TOP_WEIGHT:
        return agg[0][0], agg[0][1], agg[0][2], agg

    # Character-level temporal fallback.
    recent = [v for v in votes if now - v["time"] <= WINDOW_SEC
              and len(v["value"]) == 3 and v["value"].isdigit()]
    if not recent:
        return "", 0.0, 0, agg

    pos = [defaultdict(float) for _ in range(3)]
    for v in recent:
        for i, ch in enumerate(v["value"]):
            pos[i][ch] += v["weight"]
    if any(not p for p in pos):
        return "", 0.0, 0, agg

    candidate = "".join(max(p.items(), key=lambda kv: kv[1])[0] for p in pos)
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
    """End-of-session reporting only -- not used by the online gate."""
    groups = {}
    for v in votes:
        value = v["value"]
        groups.setdefault(value, {"weight": 0.0, "count": 0, "best": 0.0})
        groups[value]["weight"] += float(v["weight"])
        groups[value]["count"] += 1
        groups[value]["best"] = max(groups[value]["best"], float(v["conf"]))
    rows = [(k, d["weight"], d["count"], d["best"]) for k, d in groups.items()]
    rows.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return rows


class LPRRecognizer:
    """
    One recognizer = one independent temporal-voting session. Create one
    instance per camera/session_id; do NOT share one instance across
    unrelated video sources (that is exactly the bug documented for the
    current lpr_camera_server.py prototype).

    Usage per incoming frame:
        result = recognizer.process_frame(frame_bgr, workers, t)

    `workers` must provide:
        workers.detect(frame_bgr) -> (x1,y1,x2,y2,conf) or None
        workers.ocr(crop_bgr, mode) -> dict matching the OCR worker's
            "normal" or "square" payload shape (see workers/ocr_gpu_worker.py)
    `t` is a monotonically increasing clock in seconds (e.g. time.monotonic()
    for a live server, or frame_idx/fps for an offline video). It MUST be
    monotonic per recognizer instance -- this is what last_decision_t
    protects.
    """

    def __init__(self):
        self.confirmed_plate = ""
        self.confirmed_history = []
        self.last_decision_t = -1.0
        self.out_of_order_decisions = 0
        self.switch_events = []
        self.last_profile = None

        self.top_votes = []
        self.bottom_votes = []
        self.final_votes = []

        self.square_ocr_frames = 0
        self.detections = 0
        self.ocr_attempts = 0

        self.square_readings = []
        self.normal_readings = []

    # -- vehicle switching gate, verbatim from lpr_v19_universal.py --------
    def consider_plate(self, candidate, conf, t, source):
        if not candidate or not valid_kz_plate(candidate):
            return False

        if t < self.last_decision_t - 1e-6:
            # An OCR result for an older moment finished after a newer
            # decision was already made -- keep it out of the confirmed
            # state, do not let it roll anything back.
            self.out_of_order_decisions += 1
            return False

        self.last_decision_t = max(self.last_decision_t, t)

        self.confirmed_history = [
            x for x in self.confirmed_history
            if t - x["time"] <= SWITCH_WINDOW_SEC
        ]
        self.confirmed_history.append({
            "plate": candidate, "time": t, "source": source, "conf": conf,
        })

        if candidate == self.confirmed_plate:
            return False

        reads = [x for x in self.confirmed_history if x["plate"] == candidate]
        strong_single = float(conf) >= SWITCH_STRONG_CONF
        if len(reads) >= SWITCH_CONFIRM_READS or strong_single:
            old = self.confirmed_plate
            self.confirmed_plate = candidate
            self.confirmed_history = []
            self.switch_events.append({
                "time": round(t, 2), "from": old, "to": candidate,
                "source": source, "confidence": round(float(conf), 3),
            })
            return True
        return False

    # -- per-detection orchestration, verbatim from process_yolo_results /
    #    process_ocr_results, just made synchronous instead of queue-based --
    def process_frame(self, frame_bgr, workers, t):
        """
        Runs one full detect -> classify -> OCR -> normalize -> vote ->
        switch cycle for a single frame. Returns a dict with the fields the
        API server exposes (see docs/LPR_API_CONTRACT.md for the wire
        format built on top of this).

        Profiling note: this method records per-stage timings into
        self.last_profile. The timing code does not alter any decision,
        threshold, ordering, or OCR call -- it only reads the clock around
        calls that already existed.
        """
        import time as _time
        prof = {
            "yolo_ms": 0.0,
            "ocr_calls": 0,
            "ocr_total_ms": 0.0,
            "ocr_square_ms": 0.0,
            "ocr_normal_ms": 0.0,
            "ocr_fallback_ms": 0.0,
            "crop_ms": 0.0,
            "voting_ms": 0.0,
            "worker_detail": [],
        }
        t_start = _time.perf_counter()

        t0 = _time.perf_counter()
        det = workers.detect(frame_bgr)
        prof["yolo_ms"] = (_time.perf_counter() - t0) * 1000.0
        if getattr(workers, "last_timing", None):
            prof["worker_detail"].append({"stage": "yolo", **workers.last_timing})

        changed = False
        ocr_confidence = None
        plate_type = None
        raw_text = None

        if det is not None:
            self.detections += 1
            x1, y1, x2, y2, det_conf = det
            t0 = _time.perf_counter()
            crop = frame_bgr[y1:y2, x1:x2]
            prof["crop_ms"] = (_time.perf_counter() - t0) * 1000.0

            if crop.size != 0:
                h, w = crop.shape[:2]
                aspect = w / max(1, h)

                if aspect > SQUARE_ASPECT_MAX or w < MIN_SQUARE_W or h < MIN_SQUARE_H:
                    plate_type = "normal"
                    self.ocr_attempts += 1
                    t0 = _time.perf_counter()
                    payload = workers.ocr(crop, "normal")
                    dt = (_time.perf_counter() - t0) * 1000.0
                    prof["ocr_calls"] += 1
                    prof["ocr_total_ms"] += dt
                    prof["ocr_normal_ms"] += dt
                    if getattr(workers, "last_timing", None):
                        prof["worker_detail"].append({"stage": "ocr_normal", **workers.last_timing})

                    t0 = _time.perf_counter()
                    changed, ocr_confidence, raw_text = self._handle_normal_result(payload, t)
                    prof["voting_ms"] += (_time.perf_counter() - t0) * 1000.0
                else:
                    plate_type = "square"
                    self.square_ocr_frames += 1
                    if self.square_ocr_frames % OCR_EVERY_N_DETECTIONS == 1:
                        self.ocr_attempts += 1
                        py = max(2, int(h * .05))
                        px = max(2, int(w * .03))
                        sq = crop[py:max(py + 1, h - py), px:max(px + 1, w - px)]

                        t0 = _time.perf_counter()
                        payload = workers.ocr(sq, "square")
                        dt = (_time.perf_counter() - t0) * 1000.0
                        prof["ocr_calls"] += 1
                        prof["ocr_total_ms"] += dt
                        prof["ocr_square_ms"] += dt
                        if getattr(workers, "last_timing", None):
                            prof["worker_detail"].append({"stage": "ocr_square", **workers.last_timing})

                        t0 = _time.perf_counter()
                        sq_changed, ocr_confidence = self._handle_square_result(payload, t)
                        prof["voting_ms"] += (_time.perf_counter() - t0) * 1000.0
                        changed = changed or sq_changed

                        # Generalized (v19) fallback: whichever vehicle is
                        # currently confirmed, also sample normal OCR on
                        # this still-square-shaped bbox, to help catch the
                        # next transition. Doubles the OCR calls for these
                        # frames -- see module docstring re: OCR_EVERY_N_DETECTIONS.
                        if self.confirmed_plate:
                            t0 = _time.perf_counter()
                            payload2 = workers.ocr(crop, "normal")
                            dt = (_time.perf_counter() - t0) * 1000.0
                            prof["ocr_calls"] += 1
                            prof["ocr_total_ms"] += dt
                            prof["ocr_fallback_ms"] += dt
                            if getattr(workers, "last_timing", None):
                                prof["worker_detail"].append({"stage": "ocr_fallback", **workers.last_timing})

                            t0 = _time.perf_counter()
                            n_changed, n_conf, n_raw = self._handle_normal_result(payload2, t)
                            prof["voting_ms"] += (_time.perf_counter() - t0) * 1000.0
                            changed = changed or n_changed
                            if ocr_confidence is None:
                                ocr_confidence = n_conf
                                raw_text = n_raw

        prof["total_ms"] = (_time.perf_counter() - t_start) * 1000.0
        prof["plate_type"] = plate_type
        prof["had_detection"] = det is not None
        self.last_profile = prof

        return {
            "plate": self.confirmed_plate,
            "confirmed": bool(self.confirmed_plate),
            "bbox": list(det) if det is not None else None,
            "confidence": float(det[4]) if det is not None else 0.0,
            "changed": changed,
            # Additive fields not present in the old prototype's response --
            # already computed internally, simply not thrown away:
            "ocr_confidence": ocr_confidence,
            "plate_type": plate_type,
            "raw_text": raw_text,
        }

    def _handle_normal_result(self, payload, t):
        raw_text = str(payload.get("text", ""))
        raw_conf = float(payload.get("conf", 0.0))
        text = clean_text(raw_text)
        changed = False
        if valid_kz_plate(text):
            self.normal_readings.append({
                "time": round(t, 2), "plate": text, "confidence": round(raw_conf, 3),
            })
            changed = self.consider_plate(text, raw_conf, t, "normal")
        return changed, raw_conf, raw_text

    def _handle_square_result(self, payload, t):
        top_text = payload.get("top_text", "")
        top_conf = float(payload.get("top_conf", 0.0))
        bottom_text = payload.get("bottom_text", "")
        bottom_conf = float(payload.get("bottom_conf", 0.0))

        top = normalize_top(top_text)
        bottom = normalize_bottom(bottom_text)

        self.square_readings.append({
            "time": round(t, 2),
            "top_raw": top_text, "top": top, "top_conf": round(top_conf, 3),
            "bottom_raw": bottom_text, "bottom": bottom, "bottom_conf": round(bottom_conf, 3),
        })

        add_vote(self.top_votes, top, top_conf, t)
        add_vote(self.bottom_votes, bottom, bottom_conf, t)

        best_t, top_weight, _, _ = best_top(self.top_votes, t)
        best_b, bottom_weight, _, _ = best_bottom(self.bottom_votes, t)

        candidate = ""
        if best_t and best_b:
            candidate = best_t + best_b[2:] + best_b[:2]

        changed = False
        final_conf = max(top_conf, bottom_conf)
        if (candidate and valid_kz_plate(candidate)
                and top_weight >= MIN_TOP_WEIGHT and bottom_weight >= MIN_BOTTOM_WEIGHT):
            final_conf = min(0.99, (top_weight + bottom_weight) / 4.0)
            add_vote(self.final_votes, candidate, final_conf, t)
            # v19 fix, preserved verbatim: MIN_TOP_WEIGHT + MIN_BOTTOM_WEIGHT
            # already exceeds MIN_FINAL_WEIGHT by construction (1.60 + 2.00
            # > 2.50), so confirm as soon as both individual thresholds are
            # met rather than requiring a second, redundant recurrence in a
            # separate final_votes window. This is what fixed the
            # short-lived-square-plate miss (979CBB02) in v19.
            changed = self.consider_plate(candidate, final_conf, t, "square")

        return changed, final_conf
