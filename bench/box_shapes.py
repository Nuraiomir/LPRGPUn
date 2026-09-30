#!/usr/bin/env python3
"""
Which box shape sent which crop down which OCR path.

A single-row plate is about 4.5 times wider than it is tall. A square plate is
close to 1.3. The pipeline routes a crop to the two-row reader when its box is
no wider than SQUARE_ASPECT_MAX and is big enough, so the two kinds are told
apart by that one number.

A plate seen at an angle breaks that. Its upright bounding box grows taller
than the plate itself: a 4.5:1 plate turned 30 degrees has a box of about
1.4:1, which is squarer than a real square plate. It then goes to the two-row
reader, which splits a single row of text into two, and the plate cannot be
read whatever the crop's quality.

This prints the shapes actually recorded in a run, so that stays a measurement
rather than an argument. Needs a run made after box_w/box_h were added to the
diagnostics; older result files are reported as such.

Usage:
    python3 bench/box_shapes.py 20260923_152319
    python3 bench/box_shapes.py 20260923_152319 --from 40 --to 57
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from lpr_recognizer import (SQUARE_ASPECT_MAX, MIN_SQUARE_W,  # noqa: E402
                            MIN_SQUARE_H)

# Nothing wider than SQUARE_ASPECT_MAX ever reaches the two-row reader: the
# routing sends it to the single-row path first. So "is there a wide box on the
# square path" cannot be the question. The question is whether the square-path
# boxes for a plate we missed are shaped like the boxes of the square plates we
# got right, and that is a comparison, not a threshold.


def load(stem):
    path = ROOT / "runs" / f"real_video_v18_{stem}" / "results_vehicle_switch_GPU.json"
    if not path.is_file():
        sys.exit(f"нет файла {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rows(data, key, lo, hi):
    out = []
    for r in data.get(key, []):
        if lo is not None and r["time"] < lo:
            continue
        if hi is not None and r["time"] > hi:
            continue
        if "box_aspect" not in r:
            continue
        out.append(r)
    return out


def spread(values):
    if not values:
        return "нет данных"
    values = sorted(values)
    mid = values[len(values) // 2]
    return (f"от {values[0]:.2f} до {values[-1]:.2f}, середина {mid:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="video name without extension")
    ap.add_argument("--from", dest="lo", type=float, help="only after this second")
    ap.add_argument("--to", dest="hi", type=float, help="only before this second")
    args = ap.parse_args()

    data = load(args.stem)
    normal = rows(data, "normal_readings", args.lo, args.hi)
    square = rows(data, "square_readings", args.lo, args.hi)

    if not normal and not square:
        print("В этом прогоне размеров рамки нет.")
        print("Перепрогони видео уже с новой диагностикой:")
        print(f"  .venv_gpu/bin/python app/lpr_v19_universal.py videos/{args.stem}.mp4")
        return 0

    print(f"{args.stem}: порог формы {SQUARE_ASPECT_MAX}, "
          f"минимальный размер для двухрядного пути {MIN_SQUARE_W}x{MIN_SQUARE_H}")
    print(f"\n  однорядным путём прочитано {len(normal)}: "
          f"ширина/высота {spread([r['box_aspect'] for r in normal])}")
    print(f"  двухрядным путём прочитано {len(square)}: "
          f"ширина/высота {spread([r['box_aspect'] for r in square])}")

    # The comparison this script exists for: the shapes of the boxes we got
    # right against the shapes of the boxes we lost.
    print("\n  Каждое двухрядное чтение в окне:")
    for r in square:
        print(f"    {r['time']:6.2f}s  {r['box_w']}x{r['box_h']}  "
              f"ширина/высота={r['box_aspect']}  "
              f"верх={r['top_raw']!r} низ={r['bottom_raw']!r}")

    if args.lo is not None or args.hi is not None:
        others = [r for r in data.get("square_readings", []) if "box_aspect" in r
                  and r not in square]
        if others:
            print(f"\n  Для сравнения, двухрядные чтения ВНЕ окна ({len(others)}):")
            print(f"    ширина/высота {spread([r['box_aspect'] for r in others])}")
            print("    Это в основном подтверждённые квадратные номера.")
            print("    Если рамки в окне заметно площе этих, значит там был")
            print("    наклонённый однорядный номер, а не квадратный.")

    print("\nЧитается это так:")
    print("  рамки в окне площе, чем у подтверждённых квадратных  ->")
    print("    однорядный номер под углом, виновато правило формы;")
    print("  рамки такие же  ->  форма ни при чём, причина в чтении строк.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
