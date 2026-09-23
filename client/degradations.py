#!/usr/bin/env python3
"""
Image degradations that imitate hard field conditions, for testing how
recognition holds up on footage whose plates are already known.

Used by camera_client.py (--degrade NAME:LEVEL) to damage each frame just
before it is sent, so no degraded video files have to be written.

    far     plate far away, or a phone sending small frames: resolution lost
    blur    hand shake or motion: horizontal motion blur
    dark    dusk or an underground car park: dim and noisy
    lowcon  backlight, haze, a dirty plate: washed out, low contrast
    phone   the frame a phone app would actually send: smaller and JPEG-
            compressed. Unlike the others, the frame stays small, so pixel
            thresholds in the pipeline see real phone-sized plates.
    darkblur  dark and blurred at once (single level)

Levels 1, 2, 3 are mild, medium and strong. Strengths are set for 4K frames
(3840 px wide) and scaled to the actual frame width, so the same level means
the same loss of detail on smaller frames.

These are approximations: synthetic blur, noise and contrast loss are cleaner
than real dirt, rain, glare or angle. They do not replace real footage.

Preview one frame at every level:
    python3 client/degradations.py videos/20260909_171120.mp4 --at 6.2 --out /tmp/preview
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

_REF_WIDTH = 3840.0

# name -> level -> parameters
LEVELS = {
    "far":    {1: {"factor": 3}, 2: {"factor": 6}, 3: {"factor": 10}},
    "blur":   {1: {"length": 40}, 2: {"length": 80}, 3: {"length": 140}},
    "dark":   {1: {"gain": 0.40, "gamma": 1.6, "noise": 8},
               2: {"gain": 0.22, "gamma": 1.9, "noise": 12},
               3: {"gain": 0.12, "gamma": 2.2, "noise": 16}},
    "lowcon": {1: {"contrast": 0.40, "haze": 40},
               2: {"contrast": 0.22, "haze": 60},
               3: {"contrast": 0.12, "haze": 75}},
    "phone":  {1: {"width": 1920, "quality": 60},     # 1080p
               2: {"width": 1280, "quality": 40},     # 720p
               3: {"width": 854, "quality": 25}},     # 480p
    "darkblur": {2: {}},
}


def _scale(img):
    return img.shape[1] / _REF_WIDTH


def _far(img, rng, factor):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _blur(img, rng, length):
    # Horizontal motion blur is a 1-D average, so a box filter k x 1 gives the
    # same result as a k x k kernel with one non-zero row, far faster.
    k = max(3, int(round(length * _scale(img))) | 1)
    return cv2.blur(img, (k, 1))


def _lut(fn):
    x = np.arange(256, dtype=np.float32)
    return np.clip(fn(x), 0, 255).astype(np.uint8)


def _add_noise(img, sigma):
    # cv2.randn draws from OpenCV's own generator, seeded once per Degrader,
    # so every run sees the same noise; much faster than numpy on 4K frames.
    noise = np.empty(img.shape, np.int16)
    cv2.randn(noise, 0, sigma)
    return cv2.add(img.astype(np.int16), noise, dtype=cv2.CV_16S).clip(0, 255).astype(np.uint8)


def _dark(img, rng, gain, gamma, noise):
    lut = _lut(lambda x: np.power(x / 255.0, gamma) * gain * 255.0)
    return _add_noise(cv2.LUT(img, lut), noise)


def _lowcon(img, rng, contrast, haze):
    return cv2.LUT(img, _lut(lambda x: (x - 128.0) * contrast + 128.0 + haze))


def _phone(img, rng, width, quality):
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _darkblur(img, rng):
    return _dark(_blur(img, rng, **LEVELS["blur"][2]), rng, **LEVELS["dark"][2])


_FUNCS = {"far": _far, "blur": _blur, "dark": _dark, "lowcon": _lowcon,
          "phone": _phone, "darkblur": _darkblur}


def all_variants():
    """Every valid 'name:level' string, in a stable order."""
    return [f"{name}:{level}" for name in LEVELS for level in LEVELS[name]]


def parse(spec):
    """'blur:2' -> ('blur', 2). A bare name uses its first defined level."""
    name, _, level = spec.partition(":")
    if name not in LEVELS:
        raise ValueError(f"unknown degradation {name!r}; choose from {', '.join(LEVELS)}")
    level = int(level) if level else min(LEVELS[name])
    if level not in LEVELS[name]:
        raise ValueError(f"{name} has levels {sorted(LEVELS[name])}, not {level}")
    return name, level


class Degrader:
    """Applies one degradation to a sequence of frames with repeatable noise."""

    def __init__(self, spec, seed=0):
        self.name, self.level = parse(spec)
        self._params = LEVELS[self.name][self.level]
        self._rng = np.random.default_rng(seed)
        cv2.setRNGSeed(seed)

    def __call__(self, frame):
        return _FUNCS[self.name](frame, self._rng, **self._params)


def _preview(video, at_sec, out_dir):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {video} at {at_sec} s")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = ["original"] + all_variants()
    for spec in specs:
        img = frame if spec == "original" else Degrader(spec)(frame)
        if img.shape[:2] != frame.shape[:2]:   # phone frames: enlarge for side-by-side viewing only
            img = cv2.resize(img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        path = out_dir / f"{Path(video).stem}_{at_sec:.1f}s_{spec.replace(':', '_')}.jpg"
        cv2.imwrite(str(path), img)
        print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--at", type=float, default=6.0, help="time of the frame, s")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _preview(args.video, args.at, args.out)
