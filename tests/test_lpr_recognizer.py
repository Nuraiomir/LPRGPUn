"""
Regression test for app/lpr_recognizer.py.

IMPORTANT ON SCOPE: this test does NOT run YOLO or PaddleOCR -- there is no
GPU in this environment. What it verifies is narrower but still meaningful:
that LPRRecognizer, fed the EXACT OCR outputs that the real GPU pipeline
produced during the already-validated v19 run on Video 2
(20260908_150904.mp4, results_vehicle_switch_GPU.json from that run), makes
the SAME vehicle-switch decisions the real run made -- specifically that it
still catches the short-lived square plate 979CBB02, which is the whole
point of the v19 fix this module is supposed to preserve.

This is a REPLAY test against real captured ground truth, not synthetic
data. It does NOT prove the extraction is GPU-behavior-identical to v19 --
that requires actually running client/camera_client.py against
app/lpr_api_server.py on the real GPU server and diffing outputs, per
docs/architecture.md's verification checklist. What it DOES prove: the
normalization/voting/switching arithmetic itself was transplanted correctly.
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


if __name__ == "__main__":
    test_video2_catches_short_lived_square_plate_979CBB02()
    test_single_strong_read_confirms_immediately()
    test_out_of_order_result_does_not_roll_back_confirmed_plate()
    test_invalid_format_never_confirms()
    print("\nAll tests passed.")
