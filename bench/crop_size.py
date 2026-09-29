#!/usr/bin/env python3
"""
At what plate width does reading start to work?

The scanning page tells the operator to come closer while the plate is smaller
than the aiming guide asks for, and the threshold behind that (AIM_MIN_FILL in
web/demo.html) was set by eye, because nobody had measured the width at which
a reading becomes reliable. This measures it on footage we already have.

v19 now prints the crop size with every OCR reading. A reading counts as a
success when the plate it yields is one the video actually contains, taken from
bench/labels.json, so a confident misread does not count as success. Grouping
those by crop width gives the curve the page needs.

Width is reported as a share of the frame width, because that is what the page
can compare against: the box it draws and the guide it draws are both in the
coordinates of the frame it sent.

Two honest limits. These are car park videos from one camera, so the number
transfers to a phone only as a starting point. And a wide crop that is blurred
still fails, so width explains part of the failures, not all of them.

Usage:
    .venv_gpu/bin/python bench/crop_size.py
    python3 bench/crop_size.py --videos 20260923_152319 --frame-width 2160
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from lpr_recognizer import clean_text, extract_plate  # noqa: E402

# v19 submits a crop to the one-row OCR from two places, and the log says
# which. The second is a fallback: while a plate is confirmed, a square crop
# is ALSO read as one row, to catch the moment the camera moves to the next
# car. A two-row plate read as one row almost never yields a valid string, so
# mixing those readings into the curve measures plate shape, not crop width.
SUBMIT_LINE = re.compile(
    r"\[NORMAL (OCR|FALLBACK) SUBMIT\]\s+t=\s*([\d.]+)s\s+bbox=(\d+)x(\d+)"
)

RAW_LINE = re.compile(
    r"\[NORMAL OCR RAW\]\s+t=\s*([\d.]+)s\s+RAW=(.+?)\s+conf=([\d.]+)"
    r"(?:\s+w=(\d+)\s+h=(\d+))?"
)

# Buckets by plate width as a share of the frame width.
EDGES = [0.0, 0.05, 0.08, 0.11, 0.15, 0.20, 0.27, 0.35, 0.50, 1.01]


def frame_width_of(stem, override):
    if override:
        return override
    try:
        import cv2
    except ImportError:
        return None
    path = ROOT / "videos" / f"{stem}.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap.release()
    return w or None


def submit_kinds(log_path):
    """{(time, w, h): "normal" | "fallback"} from the submit lines."""
    kinds = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUBMIT_LINE.search(line)
        if m:
            key = (round(float(m.group(2)), 2), int(m.group(3)), int(m.group(4)))
            kinds[key] = "normal" if m.group(1) == "OCR" else "fallback"
    return kinds


def readings_in(log_path):
    """[(time, text, conf, w, h)]; w is None for logs written before this change."""
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RAW_LINE.search(line)
        if not m:
            continue
        try:
            raw = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            raw = m.group(2)
        w = int(m.group(4)) if m.group(4) else None
        h = int(m.group(5)) if m.group(5) else None
        out.append((float(m.group(1)), str(raw), float(m.group(3)), w, h))
    return out


def bucket_of(share):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= share < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", help="video names; default is every runs/raw_*.log")
    ap.add_argument("--frame-width", type=int, help="override the frame width in pixels")
    ap.add_argument("--min-aspect", type=float, default=0.0,
                    help="keep only boxes at least this wide relative to their height; "
                         "3.0 drops the ones that are not one-row plates")
    ap.add_argument("--labels", type=Path, default=ROOT / "bench" / "labels.json")
    args = ap.parse_args()

    labels = {k: {p["plate"] for p in v}
              for k, v in json.loads(args.labels.read_text(encoding="utf-8")).items()
              if not k.startswith("_")}

    stems = args.videos or sorted(p.stem[4:] for p in (ROOT / "runs").glob("raw_*.log"))
    if not stems:
        sys.exit("нет ни одного runs/raw_*.log")

    total = Counter()
    good = Counter()
    shapes = {}          # bucket -> aspect ratios, to spot boxes that
    misses = {}          # are not plate-shaped at all
    fallback = [0, 0]    # readings from the fallback path, and correct ones
    too_square = 0       # dropped by --min-aspect
    widths_px = []
    no_size = 0
    skipped = []

    for stem in stems:
        log = ROOT / "runs" / f"raw_{stem}.log"
        if not log.exists():
            skipped.append(f"{stem}: нет лога")
            continue
        wanted = labels.get(stem)
        if wanted is None:
            skipped.append(f"{stem}: нет в labels.json")
            continue
        fw = frame_width_of(stem, args.frame_width)
        if not fw:
            skipped.append(f"{stem}: не удалось узнать ширину кадра")
            continue
        kinds = submit_kinds(log)

        for _t, raw, _conf, w, h in readings_in(log):
            if w is None:
                no_size += 1
                continue
            ok = extract_plate(clean_text(raw)) in wanted
            # Readings from the fallback path are two-row plates pushed
            # through one-row OCR. They belong in their own count.
            if kinds.get((round(_t, 2), w, h), "normal") == "fallback":
                fallback[0] += 1
                fallback[1] += int(ok)
                continue
            if w / max(h, 1) < args.min_aspect:
                too_square += 1
                continue
            share = w / fw
            b = bucket_of(share)
            total[b] += 1
            widths_px.append(w)
            if ok:
                good[b] += 1
            shapes.setdefault(b, []).append(w / max(h, 1))
            if not ok and len(misses.setdefault(b, [])) < 4:
                misses[b].append((w, h, raw))

    for line in skipped:
        print(f"пропущено: {line}")
    if no_size:
        print(f"\n{no_size} чтений без размера выреза: это старые логи, "
              "сделанные до правки в v19. Прогони видео заново.")
    if not total:
        sys.exit("нет ни одного чтения с размером выреза")

    if fallback[0]:
        share = 100 * fallback[1] / fallback[0]
        print(f"\nИз таблицы исключены {fallback[0]} чтений запасного пути "
              f"(квадратный вырез, прочитанный как однорядный).")
        print(f"Верных среди них: {fallback[1]} ({share:.1f}%). Это отдельный вопрос: "
              "если их почти нет,")
        print("запасной путь тратит вызовы OCR впустую.")

    if too_square:
        print(f"\nОтброшено по форме ({args.min_aspect:.1f}:1 и уже): {too_square} чтений.")

    print(f"\nЧтений с размером: {sum(total.values())}, "
          f"ширина выреза от {min(widths_px)} до {max(widths_px)} пикселей\n")
    print("  ширина номера     чтений   верных    доля   форма")
    for b in range(len(EDGES) - 1):
        if not total[b]:
            continue
        lo, hi = EDGES[b], EDGES[b + 1]
        share = 100 * good[b] / total[b]
        bar = "#" * int(share / 5)
        asp = sorted(shapes.get(b, []))
        mid = asp[len(asp) // 2] if asp else 0
        print(f"  {lo * 100:4.0f}-{hi * 100:3.0f}% кадра  {total[b]:7d} {good[b]:8d}  "
              f"{share:5.1f}%  {mid:4.1f}:1  {bar}")

    # A one-row Kazakh plate is about 4.6:1. A group whose typical box is much
    # squarer is not holding one-row plates: it is two-row plates, plates cut by
    # the frame edge, or boxes that are not plates at all. Without this the
    # curve looks like "too close is bad" when the cause may be different.
    print("\n  Колонка «форма» это ширина к высоте. У обычного однорядного")
    print("  номера около 4.6:1. Сильно меньше означает, что в группе лежат")
    print("  не однорядные номера, и сравнивать её с остальными нельзя.")

    odd = [b for b in sorted(misses) if shapes.get(b) and
           sorted(shapes[b])[len(shapes[b]) // 2] < 3.0]
    for b in odd:
        lo, hi = EDGES[b], EDGES[b + 1]
        print(f"\n  Что не прочиталось в группе {lo * 100:.0f}-{hi * 100:.0f}%:")
        for w, h, raw in misses[b]:
            print(f"    {w}x{h} ({w / max(h, 1):.1f}:1)  {raw!r}")

    print("\n  Доля верных чтений считается от всех запусков OCR в этой группе.")

    # The lowest bucket whose success rate clears a bar, with enough readings
    # behind it to mean something.
    def first_over(pct, min_n=15):
        for b in range(len(EDGES) - 1):
            if total[b] >= min_n and 100 * good[b] / total[b] >= pct:
                return EDGES[b]
        return None

    half, most = first_over(50), first_over(80)
    print()
    if half is None:
        print("  Ни одна группа не дала даже половины верных чтений. Либо данных мало,")
        print("  либо дело не в ширине: проверь резкость на сохранённых кадрах.")
    else:
        print(f"  Половина чтений верна начиная с ширины около {half * 100:.0f}% кадра.")
    if most is not None:
        print(f"  Восемь из десяти  начиная с {most * 100:.0f}%.")
        print("\n  Отсюда для web/demo.html: рамка занимает GUIDE_WIDTH_FRAC = 0.62 кадра,")
        print(f"  значит AIM_MIN_FILL = {most:.2f} / 0.62 = {most / 0.62:.2f},")
        print("  то есть номер должен заполнять столько ширины рамки.")
    elif half is not None:
        print("  До восьми из десяти не дотянула ни одна группа: либо номера в этих")
        print("  видео мельче, чем нужно, либо мешает смаз, а не размер.")


if __name__ == "__main__":
    main()
