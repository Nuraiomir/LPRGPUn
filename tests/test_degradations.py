"""
Tests client/degradations.py. No GPU needed.

The benchmark compares two OCR modes in separate runs, so both runs must see
exactly the same degraded frames. That is the main thing checked here.

Run:
    python3 tests/test_degradations.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
from degradations import Degrader, all_variants, parse  # noqa: E402

FRAMES = [np.random.default_rng(i).integers(0, 256, (540, 960, 3), dtype=np.uint8) for i in range(4)]


def test_every_variant_runs_and_keeps_type():
    for spec in all_variants():
        out = Degrader(spec)(FRAMES[0])
        assert out.dtype == np.uint8 and out.ndim == 3, spec
        if not spec.startswith("phone"):
            assert out.shape == FRAMES[0].shape, f"{spec} changed the frame size"
    print(f"[OK] all {len(all_variants())} variants run and return valid frames")


def test_same_frames_in_every_run():
    """Two runs (two OCR modes) must receive identical degraded frames."""
    for spec in all_variants():
        d1 = Degrader(spec)
        run1 = [d1(f) for f in FRAMES]
        d2 = Degrader(spec)
        run2 = [d2(f) for f in FRAMES]
        assert all(np.array_equal(a, b) for a, b in zip(run1, run2, strict=True)), spec
    print("[OK] every variant gives identical frames in separate runs")


def test_phone_frames_are_phone_sized():
    big = np.zeros((2160, 3840, 3), np.uint8)
    sizes = {s: Degrader(s)(big).shape[1] for s in ("phone:1", "phone:2", "phone:3")}
    assert sizes == {"phone:1": 1920, "phone:2": 1280, "phone:3": 854}, sizes
    print("[OK] phone variants send 1080p, 720p and 480p frames")


def test_levels_get_stronger():
    img = FRAMES[1]
    brightness = [float(Degrader(f"dark:{lv}")(img).mean()) for lv in (1, 2, 3)]
    assert brightness[0] > brightness[1] > brightness[2], brightness
    contrast = [float(Degrader(f"lowcon:{lv}")(img).std()) for lv in (1, 2, 3)]
    assert contrast[0] > contrast[1] > contrast[2], contrast
    print("[OK] each level is darker (dark) or flatter (lowcon) than the one before")


def test_invalid_specs_are_rejected():
    for bad in ("fog:1", "blur:9", "darkblur:1"):
        try:
            parse(bad)
            raise AssertionError(f"{bad} was accepted")
        except ValueError:
            pass
    assert parse("blur") == ("blur", 1)
    print("[OK] unknown names and levels are rejected")


if __name__ == "__main__":
    test_every_variant_runs_and_keeps_type()
    test_same_frames_in_every_run()
    test_phone_frames_are_phone_sized()
    test_levels_get_stronger()
    test_invalid_specs_are_rejected()
    print("\nAll tests passed.")
