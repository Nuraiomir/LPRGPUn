"""
Regression test for app/lpr_recognizer.py. No GPU needed.

Feeds LPRRecognizer the OCR readings that the GPU pipeline produced on
videos/20260908_150904.mp4 and checks that it makes the same decisions as
that run: in particular that the short-lived square plate 979CBB02 is
confirmed at 2.4 s and the switch to 221ZVZ05 happens at 3.7 s.

The readings are recorded pipeline output, not verified ground truth. This
test proves the voting and switching logic, not OCR accuracy.

Run:
    python3 tests/test_lpr_recognizer.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from lpr_recognizer import LPRRecognizer  # noqa: E402


class FakeWorkers:
    """Replays a fixed, pre-recorded sequence of OCR payloads instead of
    calling real GPU workers. Detection is not exercised here (bbox/aspect
    classification is assumed already decided -- see the note below); only
    the OCR-and-onward path is under test, called directly.
    """
    pass


def test_video2_catches_short_lived_square_plate_979CBB02():
    rec = LPRRecognizer()

    # Real square_readings captured from the validated v19 run on
    # 20260908_150904.mp4 (see this project's earlier results JSON), t=0.1..3.1.
    square_frames = [
        (0.1, "KZ979", 0.862, "UZCBB", 0.829),
        (0.4, "Kz 979", 0.876, "UZCBB", 0.906),
        (0.7, "Kz 979", 0.866, "02CBB", 0.988),
        (1.0, "Kz 919", 0.802, "UZCBB", 0.86),
        (1.3, "Kz 979", 0.747, "UZCBB", 0.896),
        (1.6, "Kz 979", 0.871, "UZCBB", 0.875),
        (2.0, "Kz 9/9", 0.698, "UZCBB", 0.879),
        (2.4, "kz 979", 0.828, "02CBB", 0.98),
        (3.1, "5", 0.773, "9U", 0.19),
    ]
    for t, top_raw, top_conf, bottom_raw, bottom_conf in square_frames:
        payload = {
            "top_text": top_raw, "top_conf": top_conf,
            "bottom_text": bottom_raw, "bottom_conf": bottom_conf,
        }
        rec._handle_square_result(payload, t)

    assert rec.confirmed_plate == "979CBB02", (
        f"expected 979CBB02 to be confirmed by t=3.1, got {rec.confirmed_plate!r} "
        f"-- this is exactly the v19 fix this module must preserve"
    )
    assert rec.switch_events[0]["to"] == "979CBB02"
    assert rec.switch_events[0]["source"] == "square"
    assert rec.switch_events[0]["time"] == 2.4, (
        f"real v19 run confirmed at t=2.4s, got t={rec.switch_events[0]['time']}"
    )
    print("[OK] 979CBB02 confirmed at t=2.4s via 'square' source, matching the real v19 run")

    # Real normal_readings immediately following, from the same run --
    # the vehicle then switches to 221ZVZ05.
    for t, plate, conf in [(3.5, "221ZVZ05", 0.867), (3.7, "221ZVZ05", 0.777)]:
        payload = {"text": plate, "conf": conf}
        rec._handle_normal_result(payload, t)

    assert rec.confirmed_plate == "221ZVZ05"
    assert rec.switch_events[-1]["to"] == "221ZVZ05"
    assert rec.switch_events[-1]["source"] == "normal"
    assert rec.switch_events[-1]["time"] == 3.7, (
        f"real v19 run switched at t=3.7s, got t={rec.switch_events[-1]['time']}"
    )
    print("[OK] switch 979CBB02 -> 221ZVZ05 at t=3.7s, matching the real v19 run")


def test_single_strong_read_confirms_immediately():
    """SWITCH_STRONG_CONF path: one read at >=0.95 confirms without waiting
    for a second read -- this is the mechanism ABSENT from the current
    lpr_camera_server.py prototype (documented in LPR_API_CONTRACT_REVIEW.md)."""
    rec = LPRRecognizer()
    changed = rec.consider_plate("545BDR05", 0.99, 0.1, "normal")
    assert changed is True
    assert rec.confirmed_plate == "545BDR05"
    print("[OK] a single conf=0.99 read confirms immediately (SWITCH_STRONG_CONF path)")


def test_out_of_order_result_does_not_roll_back_confirmed_plate():
    """last_decision_t protection -- absent from lpr_camera_server.py."""
    rec = LPRRecognizer()
    rec.consider_plate("545BDR05", 0.99, 5.0, "normal")
    stale_changed = rec.consider_plate("633BBT02", 0.99, 2.0, "square")  # t=2.0 < last_decision_t=5.0
    assert stale_changed is False
    assert rec.confirmed_plate == "545BDR05"
    assert rec.out_of_order_decisions == 1
    print("[OK] a result timestamped before the last decision cannot roll back confirmed_plate")


def test_invalid_format_never_confirms():
    rec = LPRRecognizer()
    assert rec.consider_plate("NOTAPLATE", 0.99, 1.0, "normal") is False
    assert rec.confirmed_plate == ""



# ---------------------------------------------------------------------------
# Stale plate timeout (HTTP path, process_frame)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

FRAME = np.zeros((120, 420, 3), np.uint8)
NORMAL_BOX = (0, 0, 400, 100, 0.9)          # aspect 4.0 -> single-row plate


class ScriptedWorkers:
    """detect() and ocr() answer from a script: None = no plate in view,
    otherwise (plate_text, ocr_conf)."""

    def __init__(self):
        self.next = None

    def detect(self, frame):
        return None if self.next is None else NORMAL_BOX

    def ocr(self, crop, mode):
        text, conf = self.next
        return {"text": text, "conf": conf}


def run(rec, workers, t, reading):
    workers.next = reading
    return rec.process_frame(FRAME, workers, t)


def test_stale_plate_is_cleared_after_hold():
    rec, w = LPRRecognizer(plate_hold_sec=2.0), ScriptedWorkers()
    r = run(rec, w, 0.0, ("545BDR05", 0.99))
    assert r["plate"] == "545BDR05" and r["changed"] is True
    assert run(rec, w, 1.9, None)["plate"] == "545BDR05", "cleared too early"
    r = run(rec, w, 2.1, None)
    assert r["plate"] == "" and r["confirmed"] is False, r
    assert r["changed"] is False, "a clear must not ask OCRM to search"
    assert rec.plate_clears == 1
    print("[OK] plate not read for more than 2 s is cleared; clearing does not set changed")


def test_plate_is_kept_while_it_keeps_being_read():
    rec, w = LPRRecognizer(plate_hold_sec=2.0), ScriptedWorkers()
    t = 0.0
    run(rec, w, t, ("545BDR05", 0.99))
    for _ in range(30):                       # 0.7 s gaps: the longest seen on real video
        t += 0.35
        run(rec, w, t, None)
        t += 0.35
        r = run(rec, w, t, ("545BDR05", 0.93))
        assert r["plate"] == "545BDR05", f"cleared at t={t:.2f} while still being read"
    assert rec.plate_clears == 0
    print("[OK] plate read every 0.7 s for 21 s is never cleared")


def test_reads_of_another_plate_do_not_keep_the_old_one():
    rec, w = LPRRecognizer(plate_hold_sec=2.0), ScriptedWorkers()
    run(rec, w, 0.0, ("545BDR05", 0.99))
    # Camera moves to the next car: its plate is seen, but read too weakly
    # and too rarely to confirm (reads further apart than SWITCH_WINDOW_SEC).
    # The old plate must not stay on screen meanwhile.
    run(rec, w, 0.9, ("633BBT02", 0.80))
    r = run(rec, w, 2.6, ("633BBT02", 0.80))
    assert r["plate"] == "", f"old plate kept alive by another car: {r['plate']!r}"
    print("[OK] reads of a different plate do not keep the previous plate on screen")


def test_cleared_plate_is_confirmed_again_as_a_change():
    rec, w = LPRRecognizer(plate_hold_sec=2.0), ScriptedWorkers()
    run(rec, w, 0.0, ("545BDR05", 0.99))
    run(rec, w, 3.0, None)                    # cleared
    r = run(rec, w, 4.0, ("545BDR05", 0.99))
    assert r["plate"] == "545BDR05" and r["changed"] is True, r
    print("[OK] the same car coming back is confirmed again with changed=true")


def test_zero_hold_disables_the_timeout():
    rec, w = LPRRecognizer(plate_hold_sec=0), ScriptedWorkers()
    run(rec, w, 0.0, ("545BDR05", 0.99))
    assert run(rec, w, 60.0, None)["plate"] == "545BDR05"
    print("[OK] plate_hold_sec=0 keeps the old behaviour")


def test_votes_for_a_gone_square_plate_do_not_keep_it_on_screen():
    """Square plates are decided from votes over the last 3 s. While the camera
    already reads the next car, those votes can still name the old plate; that
    must not keep the old plate on screen. Readings below are from the real run
    on videos/20260909_171120.mp4: at 10.4 s and 10.8 s the square path is
    reading the next car (694BPT05), not 633BBT02."""
    rec = LPRRecognizer(plate_hold_sec=2.0)
    square = [(7.5, "633", 0.995, "02BBT", 0.899), (7.8, "Kz 633", 0.855, "02BBT", 0.97),
              (8.1, "633", 0.993, "02BBT", 0.919), (9.0, "", 0.0, "JO", 0.289),
              (10.4, "0/BPT05", 0.856, "KZ67", 0.828), (10.8, "/BPT05", 0.826, "KZ671", 0.817)]
    shown = {}
    for t, top, tc, bottom, bc in square:
        rec._handle_square_result({"top_text": top, "top_conf": tc,
                                   "bottom_text": bottom, "bottom_conf": bc}, t)
        rec.drop_stale_plate(t)
        shown[t] = rec.confirmed_plate
    assert shown[8.1] == "633BBT02", shown
    assert shown[9.0] == "633BBT02", "cleared before 2 s had passed"
    assert shown[10.4] == "", f"old square plate kept alive by the next car: {shown[10.4]!r}"
    print("[OK] votes for a square plate that has left do not keep it on screen")


def test_square_plate_read_by_one_row_stays():
    """One readable row keeps a square plate on screen while the other row's
    votes are still within WINDOW_SEC. After that no full plate can be formed,
    and the plate is cleared PLATE_HOLD_SEC later."""
    rec = LPRRecognizer(plate_hold_sec=2.0)
    for t in (5.0, 5.3, 5.6):                 # both rows read; last top read at 5.6
        rec._handle_square_result({"top_text": "633", "top_conf": 0.97,
                                   "bottom_text": "02BBT", "bottom_conf": 0.96}, t)
    assert rec.confirmed_plate == "633BBT02"

    t = 5.6
    shown = {}
    while t < 11.5:                           # from now on only the bottom row reads
        t = round(t + 0.6, 1)
        rec._handle_square_result({"top_text": "6?3", "top_conf": 0.5,
                                   "bottom_text": "02BBT", "bottom_conf": 0.95}, t)
        rec.drop_stale_plate(t)
        shown[t] = rec.confirmed_plate
    # top votes last until 5.6 + 3.0 = 8.6; the last keep-alive is at 8.6,
    # so the plate stays through 10.6 and is gone after that
    assert all(p == "633BBT02" for tt, p in shown.items() if tt <= 10.4), shown
    assert all(p == "" for tt, p in shown.items() if tt >= 11.0), shown
    print("[OK] square plate with one readable row stays ~5 s, then is cleared")


def test_cleared_square_plate_is_not_revived_by_its_old_votes():
    """Real GPU run, videos/20260909_171120.mp4 at 10 fps: 633BBT02 was cleared
    at 10.2 s, then confirmed again at 10.4 s from its own 3-second-old votes
    while the camera was already on the next car. That must not happen."""
    rec = LPRRecognizer(plate_hold_sec=2.0)
    reads = {5.6: ("633", 0.972, "02BBT", 0.902), 5.9: ("633", 0.945, "02BBT", 0.981),
             6.2: ("633", 0.936, "02BBT", 0.974), 6.6: ("633", 0.883, "02BBT", 0.962),
             6.9: ("633", 0.968, "02BBT", 0.962), 7.2: ("633", 0.98, "02BBT", 0.92),
             7.5: ("633", 0.995, "02BBT", 0.899), 7.8: ("Kz 633", 0.855, "02BBT", 0.97),
             8.1: ("633", 0.993, "02BBT", 0.919), 9.0: ("", 0.0, "JO", 0.289),
             10.4: ("0/BPT05", 0.856, "KZ67", 0.828), 10.8: ("/BPT05", 0.826, "KZ671", 0.817)}
    shown = {}
    for step in range(56, 120):               # 5.6 s .. 11.9 s at 10 fps
        t = round(step * 0.1, 1)
        if t in reads:
            top, tc, bottom, bc = reads[t]
            rec._handle_square_result({"top_text": top, "top_conf": tc,
                                       "bottom_text": bottom, "bottom_conf": bc}, t)
        rec.drop_stale_plate(t)
        shown[t] = rec.confirmed_plate
    assert shown[10.2] == "", shown[10.2]
    revived = [t for t in shown if t > 10.2 and shown[t]]
    assert not revived, f"633BBT02 came back at {revived[0]} s with no fresh reading"
    print("[OK] a cleared square plate is not confirmed again from its old votes")


class SquareWorkers:
    """Square-shaped detection; OCR answers from a (top, top_conf, bottom, bottom_conf) reading."""

    def __init__(self):
        self.reading = None

    def detect(self, frame):
        return (0, 0, 200, 160, 0.9)          # aspect 1.25 -> square plate

    def ocr(self, crop, mode):
        top, tc, bottom, bc = self.reading
        if mode == "square":
            return {"top_text": top, "top_conf": tc, "bottom_text": bottom, "bottom_conf": bc}
        return {"text": "", "conf": 0.0}


def test_square_ocr_confidence_is_the_weaker_row_of_this_frame():
    """Real readings from videos/20260909_171120.mp4. At 9.0 s OCR read nothing
    ('' / 'JO'); the old code reported the vote-based 0.99 there."""
    rec, w = LPRRecognizer(), SquareWorkers()
    frame = np.zeros((200, 220, 3), np.uint8)

    def square_frame(t, reading):
        w.reading = reading
        for _ in range(3):                    # square OCR runs on every 3rd detection
            r = rec.process_frame(frame, w, t)
            if r["ocr_confidence"] is not None:
                return r
        raise AssertionError("square OCR did not run")

    for t in (7.5, 7.8, 8.1):
        r = square_frame(t, ("633", 0.995, "02BBT", 0.899))
    assert r["plate"] == "633BBT02"
    assert abs(r["ocr_confidence"] - 0.899) < 1e-9, r["ocr_confidence"]
    assert r["raw_text"] == "633 / 02BBT", r["raw_text"]

    r = square_frame(9.0, ("", 0.0, "JO", 0.289))
    assert r["plate"] == "633BBT02", "the plate itself is still confirmed from votes"
    assert r["ocr_confidence"] == 0.0, f"reported {r['ocr_confidence']} for an empty read"
    assert r["raw_text"] == " / JO", r["raw_text"]
    print("[OK] square ocr_confidence is this frame's weaker row; raw_text shows both rows")

if __name__ == "__main__":
    test_video2_catches_short_lived_square_plate_979CBB02()
    test_single_strong_read_confirms_immediately()
    test_out_of_order_result_does_not_roll_back_confirmed_plate()
    test_invalid_format_never_confirms()
    test_stale_plate_is_cleared_after_hold()
    test_plate_is_kept_while_it_keeps_being_read()
    test_reads_of_another_plate_do_not_keep_the_old_one()
    test_cleared_plate_is_confirmed_again_as_a_change()
    test_zero_hold_disables_the_timeout()
    test_votes_for_a_gone_square_plate_do_not_keep_it_on_screen()
    test_square_plate_read_by_one_row_stays()
    test_cleared_square_plate_is_not_revived_by_its_old_votes()
    test_square_ocr_confidence_is_the_weaker_row_of_this_frame()
    print("\nAll tests passed.")
