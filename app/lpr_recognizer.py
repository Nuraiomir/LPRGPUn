"""
License plate recognition core for Kazakhstan plates.

Everything that happens after YOLO has found a plate lives here: choosing
between the single-row and two-row (square) OCR path, normalizing OCR text to
the Kazakhstan format, temporal voting across frames, and deciding when the
camera has moved on to a different vehicle.

Both entry points use this module, so they run the same rules:
  app/lpr_v19_universal.py  offline video evaluation (uses consider_plate)
  app/lpr_api_server.py     HTTP server, one LPRRecognizer per session

Plate format: three digits, three Latin letters, two digits (e.g. 545BDR05).
A square plate carries the digits on the top row and region + letters on the
bottom row (e.g. top "633", bottom "02BBT" -> 633BBT02).

The HTTP path (process_frame) drops a confirmed plate that has not been read
for PLATE_HOLD_SEC seconds. The offline pipeline only uses consider_plate and
is not affected.

Note on OCR_EVERY_N_DETECTIONS: only every third square-shaped detection is
sent to OCR. The value was tuned for a 60 fps offline loop. A live camera that
sends far fewer frames per second may need a smaller value.
"""

import re
import time
from collections import defaultdict
from contextlib import contextmanager


# Detection and plate shape
YOLO_CONF = 0.40
SQUARE_ASPECT_MAX = 1.80      # wider than this -> single-row plate
MIN_SQUARE_W = 130            # smaller crops are treated as single-row
MIN_SQUARE_H = 85
OCR_EVERY_N_DETECTIONS = 3    # square detections: OCR every Nth one

# Temporal voting for square plates
WINDOW_SEC = 3.0
MIN_TOP_WEIGHT = 1.60
MIN_BOTTOM_WEIGHT = 2.00
# Used only by the end-of-run report. The online decision needs both row
# weights above their minimums, whose sum (3.60) already exceeds this value.
MIN_FINAL_WEIGHT = 2.50

# Vehicle switching
SWITCH_WINDOW_SEC = 1.5
SWITCH_CONFIRM_READS = 2      # matching reads needed to switch plates
SWITCH_STRONG_CONF = 0.95     # ...or one read at least this confident

# A confirmed plate that has not been read again for this many seconds is
# dropped, so the previous car is not shown after the camera has moved away.
# On the reference videos the longest gap between reads of a plate that was
# still in view was 0.7 s. Only reads of the confirmed plate itself keep it
# alive; a detection of some other plate does not. 0 disables the timeout.
PLATE_HOLD_SEC = 2.0

# Memory bounds for long-running sessions. Voting only looks back WINDOW_SEC,
# so older votes can never influence a decision. They are kept for ten
# windows before being dropped, which leaves room for small clock steps
# backwards without changing any result.
VOTE_RETENTION_SEC = WINDOW_SEC * 10
MAX_READINGS = 500            # diagnostic history kept per session


# ---------------------------------------------------------------------------
# Text normalization
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

    # More than three digits is ambiguous: "KZ 049" read with the Z as a 2
    # gives "2049", and the extra digit can sit on either side ("16330" for
    # 633). Taking the first three produced wrong plates such as 204BXS02,
    # so such a read does not vote at all.
    if len(re.sub(r"\D", "", s)) > 3:
        return ""

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


# ---------------------------------------------------------------------------
# Temporal voting
# ---------------------------------------------------------------------------

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

    # No single top-row value is strong enough: vote per character position.
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
    # A candidate built digit by digit is only as well supported as its
    # weakest digit. Summing the three positions counted every read three
    # times, so a single read could pass MIN_TOP_WEIGHT on its own.
    weight = min(pos[i][candidate[i]] for i in range(3))
    if weight >= MIN_TOP_WEIGHT:
        return candidate, weight, len(recent), agg
    return "", 0.0, 0, agg


def best_bottom(votes, now):
    agg = aggregate(votes, now)
    if not agg:
        return "", 0.0, 0, agg
    return agg[0][0], agg[0][1], agg[0][2], agg


def square_candidate(top_votes, bottom_votes, square_readings, t):
    """
    Decides whether the square-plate votes support a plate at time t.
    Returns (plate, confidence), or ("", 0.0) if they do not.

    The top and bottom rows are voted on separately, so their winners can come
    from different frames, even from two different cars in view within
    WINDOW_SEC. The plate is accepted only if its top and bottom were read
    together in at least one frame of the last WINDOW_SEC. square_readings are
    the normalized per-frame readings, newest last, including the current one.
    """
    best_t, top_weight, _, _ = best_top(top_votes, t)
    best_b, bottom_weight, _, _ = best_bottom(bottom_votes, t)
    if not best_t or not best_b:
        return "", 0.0
    plate = best_t + best_b[2:] + best_b[:2]
    if not valid_kz_plate(plate):
        return "", 0.0
    if top_weight < MIN_TOP_WEIGHT or bottom_weight < MIN_BOTTOM_WEIGHT:
        return "", 0.0
    read_together = any(
        r["top"] == best_t and r["bottom"] == best_b and t - r["time"] <= WINDOW_SEC
        for r in square_readings
    )
    if not read_together:
        return "", 0.0
    return plate, min(0.99, (top_weight + bottom_weight) / 4.0)


def aggregate_all(votes):
    """Totals over the whole history. Used for end-of-run reports only."""
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


# ---------------------------------------------------------------------------
# Per-frame profiling
# ---------------------------------------------------------------------------

class _FrameProfile:
    """Collects stage timings for one frame. Measurement only."""

    def __init__(self):
        self._start = time.perf_counter()
        self.data = {
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

    @contextmanager
    def timed(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.data[key] += (time.perf_counter() - t0) * 1000.0

    def record_worker(self, stage, workers):
        timing = getattr(workers, "last_timing", None)
        if timing:
            self.data["worker_detail"].append({"stage": stage, **timing})

    def ocr(self, kind, workers, image, mode):
        """Run one OCR call; kind is 'normal', 'square' or 'fallback'."""
        t0 = time.perf_counter()
        payload = workers.ocr(image, mode)
        dt = (time.perf_counter() - t0) * 1000.0
        self.data["ocr_calls"] += 1
        self.data["ocr_total_ms"] += dt
        self.data[f"ocr_{kind}_ms"] += dt
        self.record_worker(f"ocr_{kind}", workers)
        return payload

    def finish(self, plate_type, had_detection):
        self.data["total_ms"] = (time.perf_counter() - self._start) * 1000.0
        self.data["plate_type"] = plate_type
        self.data["had_detection"] = had_detection
        return self.data


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class LPRRecognizer:
    """
    One recognizer is one independent recognition session. Create one per
    camera or session_id; sharing an instance mixes the vote history of
    unrelated video sources.

    Per incoming frame:
        result = recognizer.process_frame(frame_bgr, workers, t)

    `workers` must provide:
        workers.detect(frame_bgr) -> (x1, y1, x2, y2, conf) or None
        workers.ocr(crop_bgr, mode) -> the OCR worker's "normal" or "square"
            payload (see workers/ocr_gpu_worker.py)

    `t` is a clock in seconds: time.monotonic() for a live server, or
    frame_idx / fps for a recorded video. A result whose t is older than the
    last accepted decision is ignored by consider_plate, so a late
    asynchronous OCR result cannot roll the confirmed plate back.
    """

    def __init__(self, plate_hold_sec=PLATE_HOLD_SEC):
        self.plate_hold_sec = plate_hold_sec
        self.last_confirmed_read_t = float("-inf")
        self.plate_clears = 0

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

        self._latest_t = float("-inf")

    def consider_plate(self, candidate, conf, t, source, fresh=True):
        """Returns True if this read switched the confirmed plate.

        fresh=False means the candidate comes from accumulated votes rather
        than from what was read in this frame. It counts for switching exactly
        as before, but does not keep the confirmed plate on screen."""
        if not candidate or not valid_kz_plate(candidate):
            return False

        if t < self.last_decision_t - 1e-6:
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
            if fresh:
                self.last_confirmed_read_t = t
            return False

        reads = [x for x in self.confirmed_history if x["plate"] == candidate]
        strong_single = float(conf) >= SWITCH_STRONG_CONF
        if len(reads) >= SWITCH_CONFIRM_READS or strong_single:
            old = self.confirmed_plate
            self.confirmed_plate = candidate
            self.confirmed_history = []
            self.last_confirmed_read_t = t
            self.switch_events.append({
                "time": round(t, 2), "from": old, "to": candidate,
                "source": source, "confidence": round(float(conf), 3),
            })
            return True
        return False

    def process_frame(self, frame_bgr, workers, t):
        """
        Runs detect -> classify -> OCR -> normalize -> vote -> switch for one
        frame. Stage timings are stored in self.last_profile.
        """
        prof = _FrameProfile()

        with prof.timed("yolo_ms"):
            det = workers.detect(frame_bgr)
        prof.record_worker("yolo", workers)

        changed = False
        ocr_confidence = None
        plate_type = None
        raw_text = None

        if det is not None:
            self.detections += 1
            x1, y1, x2, y2, det_conf = det
            with prof.timed("crop_ms"):
                crop = frame_bgr[y1:y2, x1:x2]

            if crop.size != 0:
                h, w = crop.shape[:2]
                aspect = w / max(1, h)

                if aspect > SQUARE_ASPECT_MAX or w < MIN_SQUARE_W or h < MIN_SQUARE_H:
                    plate_type = "normal"
                    self.ocr_attempts += 1
                    payload = prof.ocr("normal", workers, crop, "normal")
                    with prof.timed("voting_ms"):
                        changed, ocr_confidence, raw_text = self._handle_normal_result(payload, t)
                else:
                    plate_type = "square"
                    self.square_ocr_frames += 1
                    if self.square_ocr_frames % OCR_EVERY_N_DETECTIONS == 1:
                        self.ocr_attempts += 1
                        py = max(2, int(h * .05))
                        px = max(2, int(w * .03))
                        sq = crop[py:max(py + 1, h - py), px:max(px + 1, w - px)]

                        payload = prof.ocr("square", workers, sq, "square")
                        with prof.timed("voting_ms"):
                            sq_changed, _ = self._handle_square_result(payload, t)
                        changed = changed or sq_changed

                        # What OCR read in this frame. A square plate is only
                        # as reliable as its weaker row. (The vote-based
                        # confidence used for switching is not reported: it
                        # stays high for seconds after the plate leaves view.)
                        top_text = str(payload.get("top_text", ""))
                        bottom_text = str(payload.get("bottom_text", ""))
                        ocr_confidence = min(float(payload.get("top_conf", 0.0)),
                                             float(payload.get("bottom_conf", 0.0)))
                        raw_text = f"{top_text} / {bottom_text}"

                        # While a vehicle is confirmed, also read this crop as
                        # a single-row plate. When the camera moves to the next
                        # car, YOLO's box can stay square for a few frames;
                        # this catches the new plate sooner.
                        if self.confirmed_plate:
                            payload2 = prof.ocr("fallback", workers, crop, "normal")
                            with prof.timed("voting_ms"):
                                n_changed, n_conf, n_raw = self._handle_normal_result(payload2, t)
                            changed = changed or n_changed
                            if n_changed:
                                # this single-row read is what confirmed the plate
                                ocr_confidence, raw_text = n_conf, n_raw

        self.drop_stale_plate(t)
        self.last_profile = prof.finish(plate_type, det is not None)

        return {
            "plate": self.confirmed_plate,
            "confirmed": bool(self.confirmed_plate),
            "bbox": list(det) if det is not None else None,
            "confidence": float(det[4]) if det is not None else 0.0,
            "changed": changed,
            "ocr_confidence": ocr_confidence,
            "plate_type": plate_type,
            "raw_text": raw_text,
        }

    def drop_stale_plate(self, t):
        """Clears the confirmed plate if it was not read for plate_hold_sec.
        Returns True if the plate was cleared."""
        if not self.confirmed_plate or not self.plate_hold_sec or self.plate_hold_sec <= 0:
            return False
        if t - self.last_confirmed_read_t <= self.plate_hold_sec:
            return False
        self.confirmed_plate = ""
        self.confirmed_history = []
        # The square votes behind the cleared plate are stale by definition;
        # left in place they would confirm it again from old evidence as soon
        # as the next square crop is read, even if that crop is another car.
        self.top_votes.clear()
        self.bottom_votes.clear()
        self.final_votes.clear()
        self.plate_clears += 1
        return True

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
        self._trim_history(t)
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

        candidate, vote_conf = square_candidate(
            self.top_votes, self.bottom_votes, self.square_readings, t)

        changed = False
        final_conf = max(top_conf, bottom_conf)
        if candidate:
            final_conf = vote_conf
            add_vote(self.final_votes, candidate, final_conf, t)
            # Both row weights above their minimums is enough to confirm;
            # waiting for the combined vote to repeat made short-lived square
            # plates (e.g. 979CBB02) leave the frame before confirmation.
            #
            # The candidate comes from votes over the last WINDOW_SEC seconds,
            # so it can still name a car that has already left while the
            # camera reads the next one. It keeps the plate on screen only if
            # this frame's own reading matches at least one of its rows.
            plate_bottom = candidate[6:8] + candidate[3:6]
            fresh = (top == candidate[:3]) or (bottom == plate_bottom)
            changed = self.consider_plate(candidate, final_conf, t, "square", fresh=fresh)

        self._trim_history(t)
        return changed, final_conf

    def _trim_history(self, t):
        """Keeps per-session memory bounded without affecting any decision."""
        self._latest_t = max(self._latest_t, t)
        cutoff = self._latest_t - VOTE_RETENTION_SEC
        for votes in (self.top_votes, self.bottom_votes, self.final_votes):
            if votes and votes[0]["time"] < cutoff:
                votes[:] = [v for v in votes if v["time"] >= cutoff]
        for readings in (self.square_readings, self.normal_readings):
            if len(readings) > MAX_READINGS:
                del readings[:len(readings) - MAX_READINGS]
