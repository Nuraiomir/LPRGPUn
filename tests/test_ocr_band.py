#!/usr/bin/env python3
"""
The band variant: when it runs, and that it cannot win by being wrong.

A detector box much taller than a plate holds the plate in a band with
bodywork around it, and reading the whole box loses characters. The worker
answers that with one more variant, the middle band cut to a plate's shape.

Two things have to hold, and they are what this file pins down:

    a well-shaped box is untouched   nothing extra runs, nothing changes
    the band is tried last           and wins only by strictly higher
                                     confidence, so a band that slices
                                     through the characters cannot displace
                                     a good read of the whole crop

No GPU and no paddle: the worker cannot be imported, so the cascade is read
out of its source and run against a stand-in recogniser.
"""

import os
import sys
import time
from pathlib import Path

import cv2                                          # noqa: F401  (used by exec)
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "workers" / "ocr_gpu_worker.py"


def cascade(answers, band_enabled=True):
    """run_ocr from the worker's source, wired to a scripted recogniser.

    answers maps a variant's image height to (text, conf), which is enough to
    tell the band apart from the full crop without guessing at pixels.
    """
    src = WORKER.read_text(encoding="utf-8")
    start = src.index("OCR_EARLY_EXIT_CONF = ")
    end = src.index("def square_ocr")
    os.environ["LPR_OCR_BAND"] = "1" if band_enabled else "0"

    seen = []

    def fake_ocr_once(image):
        h = int(image.shape[0])
        seen.append(h)
        return answers.get(h, ("", 0.0))

    ns = {"os": os, "cv2": cv2, "np": np, "time": time,
          "_ocr_once": fake_ocr_once, "ENABLE_ENHANCED": True}
    exec(compile(src[start:end], str(WORKER), "exec"), ns)
    return ns, seen


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'СБОЙ'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def main():
    failures = 0

    # A well-shaped box: 1949x428, the shape a plate read correctly comes in.
    # The band must never appear, whatever the confidences are.
    wide = np.zeros((428, 1949, 3), np.uint8)
    ns, seen = cascade({428: ("432BTY02", 0.80), 856: ("432BTY02", 0.85)})
    text, conf = ns["run_ocr"](wide)
    variants = [v["variant"] for v in ns["_VARIANT_LOG"]]
    failures += not check("на хорошей рамке полосы нет вообще",
                          "band" not in variants, f"варианты: {variants}")
    failures += not check("на хорошей рамке ответ прежний",
                          (text, round(conf, 2)) == ("432BTY02", 0.85))

    # A box shaped like the ones that lost their cars: 1817x966. The band is
    # 1817x404, a height nothing else in the cascade produces.
    tall = np.zeros((966, 1817, 3), np.uint8)

    # The band reads the plate where the full crop only ever loses a letter.
    ns, seen = cascade({966: ("860AX02", 0.80), 1932: ("860AX02", 0.83),
                        404: ("860AXS02", 0.94)})
    text, conf = ns["run_ocr"](tall)
    variants = [v["variant"] for v in ns["_VARIANT_LOG"]]
    failures += not check("на высокой рамке полоса пробуется",
                          "band" in variants, f"варианты: {variants}")
    failures += not check("полоса пробуется ПОСЛЕДНЕЙ",
                          variants[-1] == "band", f"варианты: {variants}")
    failures += not check("уверенная полоса выигрывает",
                          text == "860AXS02", f"получилось {text!r}")

    # The band slices through the characters and comes back confident but
    # wrong. It must not displace the weaker full-crop read, which is the
    # whole safety property.
    ns, _ = cascade({966: ("860AXS02", 0.88), 1932: ("860AXS02", 0.88),
                     404: ("60AXS0", 0.87)})
    text, conf = ns["run_ocr"](tall)
    failures += not check("менее уверенная полоса НЕ вытесняет полный вырез",
                          text == "860AXS02", f"получилось {text!r}")

    # Equal confidence is a tie, and a tie goes to whoever was tried first.
    ns, _ = cascade({966: ("860AXS02", 0.90), 1932: ("860AXS02", 0.90),
                     404: ("WRONG", 0.90)})
    text, _ = ns["run_ocr"](tall)
    failures += not check("при равной уверенности полный вырез сохраняется",
                          text == "860AXS02", f"получилось {text!r}")

    # Early exit still comes first: a confident full crop ends the cascade
    # before the band costs anything.
    ns, _ = cascade({966: ("860AXS02", 0.96)})
    ns["run_ocr"](tall)
    variants = [v["variant"] for v in ns["_VARIANT_LOG"]]
    failures += not check("уверенный полный вырез обрывает каскад до полосы",
                          variants == ["original"], f"варианты: {variants}")

    # And the switch really switches it off.
    ns, _ = cascade({966: ("860AX02", 0.80), 1932: ("860AX02", 0.83),
                     404: ("860AXS02", 0.94)}, band_enabled=False)
    text, _ = ns["run_ocr"](tall)
    variants = [v["variant"] for v in ns["_VARIANT_LOG"]]
    failures += not check("LPR_OCR_BAND=0 убирает полосу",
                          "band" not in variants, f"варианты: {variants}")
    failures += not check("и без полосы номер снова теряется",
                          text == "860AX02", f"получилось {text!r}")

    print()
    if failures:
        print(f"НЕ ПРОШЛО: {failures}")
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
