#!/usr/bin/env python3
"""
Why a plate in the video was never confirmed: the detector, or the crop?

Three cars on the parking video are never recognised at all. field_misses.py
shows that no failed OCR reading belongs to them, which leaves two very
different explanations, and they need different work:

    no OCR ran at all in that stretch  -> the detector never produced anything
                                          plate-shaped. Work on detection.
    OCR ran and came back as junk      -> the crop reached OCR unreadable.
                                          Work on distance and framing.

Cars appear in the order bench/labels.json lists them, so a missing car has to
be between the confirmation of the previous car that WAS found and the next
one. This counts the OCR readings inside that stretch, prints them, and writes
out frames so the car can be looked at.

The window is bounded by confirmation times, and a plate is confirmed a little
after it becomes visible, so the stretch is approximately right rather than
exact. For looking at frames that is good enough.

Needs cv2 for the frames (use .venv_gpu). Without it the analysis still runs.

Usage:
    .venv_gpu/bin/python bench/missed_plates.py 20260923_152319
    .venv_gpu/bin/python bench/missed_plates.py 20260923_152319 --frames 20
    python3 bench/missed_plates.py 20260923_152319 --no-frames
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from lpr_recognizer import (clean_text, extract_plate,  # noqa: E402
                            MIN_TOP_WEIGHT, WINDOW_SEC)

RAW_LINE = re.compile(r"\[NORMAL OCR RAW\]\s+t=\s*([\d.]+)s\s+RAW=(.+?)\s+conf=([\d.]+)")


def readings_in(log_path):
    """[(time, text)] from a v19 log, or [] when the log is missing."""
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RAW_LINE.search(line)
        if not m:
            continue
        try:
            raw = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            raw = m.group(2)
        out.append((float(m.group(1)), str(raw)))
    return out


def confirmations(stem):
    """[(time, plate)] in order, from the saved run."""
    path = ROOT / "runs" / f"real_video_v18_{stem}" / "results_vehicle_switch_GPU.json"
    if not path.exists():
        sys.exit(f"нет прогона {path}\nСначала прогони видео через app/lpr_v19_universal.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(float(e["time"]), e["to"]) for e in data.get("switch_events", []) if e.get("to")]


def windows_for_missing(labels, found_at):
    """{plate: (start, end)} for every labelled plate that was never confirmed.

    found_at maps a plate to the moment it was confirmed. A missing plate sits
    between its nearest found neighbours in the label order.
    """
    out = {}
    for i, plate in enumerate(labels):
        if plate in found_at:
            continue
        start = 0.0
        for earlier in reversed(labels[:i]):
            if earlier in found_at:
                start = found_at[earlier]
                break
        end = None
        for later in labels[i + 1:]:
            if later in found_at:
                end = found_at[later]
                break
        out[plate] = (start, end)
    return out


def dump_frames(video, plate, start, end, count, out_dir):
    """Writes `count` frames spread across the window. Returns how many landed."""
    try:
        import cv2
    except ImportError:
        print("    (кадры не сохранены: нет cv2, запусти через .venv_gpu)")
        return 0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"    (кадры не сохранены: не открывается {video})")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    stop = end if end is not None else (total / fps if total else start + 5)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for k in range(count):
        t = start + (stop - start) * (k + 0.5) / count
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        name = out_dir / f"{plate}_{t:06.2f}s.jpg"
        if cv2.imwrite(str(name), frame):
            written += 1
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="video name without extension, e.g. 20260923_152319")
    ap.add_argument("--frames", type=int, default=14, help="frames to save per missing plate")
    ap.add_argument("--no-frames", action="store_true", help="analysis only")
    ap.add_argument("--labels", type=Path, default=ROOT / "bench" / "labels.json")
    args = ap.parse_args()

    data = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.stem not in data:
        sys.exit(f"В {args.labels} нет записи про {args.stem}.\n"
                 "Разметка этого видео есть только у тебя локально и не закоммичена, "
                 "поэтому на другой машине этот разбор не повторить.")
    labels = [p["plate"] for p in data[args.stem]]

    events = confirmations(args.stem)
    found_at = {}
    for t, plate in events:
        found_at.setdefault(plate, t)      # first confirmation of each plate

    missing = windows_for_missing(labels, found_at)
    readings = readings_in(ROOT / "runs" / f"raw_{args.stem}.log")
    video = ROOT / "videos" / f"{args.stem}.mp4"
    out_dir = ROOT / "runs" / f"missed_{args.stem}"

    print(f"{args.stem}: в разметке {len(labels)}, подтверждено {len(found_at)}, "
          f"не найдено {len(missing)}")
    if not readings:
        print(f"ВНИМАНИЕ: нет лога runs/raw_{args.stem}.log, "
              "разбор чтений невозможен. Прогони видео с сохранением полного вывода.")
    if not missing:
        print("Все номера из разметки подтверждены, разбирать нечего.")
        return

    for plate, (start, end) in missing.items():
        stop_text = f"{end:.2f}s" if end is not None else "конца видео"
        print(f"\n--- {plate}: должен быть между {start:.2f}s и {stop_text}")

        # A car that was already confirmed can stay in shot for seconds, and
        # its readings fall inside this window without belonging to it.
        inside, neighbours = [], 0
        for t, txt in readings:
            if t < start or (end is not None and t > end):
                continue
            if extract_plate(clean_text(txt)) in found_at:
                neighbours += 1
                continue
            inside.append((t, txt))
        if not inside:
            if neighbours:
                print(f"    ({neighbours} чтений в промежутке относятся к соседней машине)")
            print("    Своих чтений OCR в этом промежутке нет ни одного.")
            print("    Значит детектор не дал ни одной рамки, похожей на номер:")
            print("    это задача детекции, а не распознавания.")
        else:
            note = f" (ещё {neighbours} от соседней машины)" if neighbours else ""
            print(f"    OCR отработал {len(inside)} раз{note}, вот что он видел:")
            for t, txt in inside:
                mark = " <-- это он" if extract_plate(clean_text(txt)) == plate else ""
                print(f"      {t:6.2f}s  {txt!r}{mark}")

            # The verdict has to come from the readings, not from the fact that
            # there were any. An earlier version printed "the crop was
            # unreadable" whenever OCR had run at all, and said it about a car
            # whose plate one reading had got exactly right.
            hits = [(t, txt) for t, txt in inside
                    if extract_plate(clean_text(txt)) == plate]
            if not hits:
                print("    Ни одно чтение не сложилось в этот номер.")
                print("    Рамка была, но вырез оказался нечитаемым:")
                print("    это задача расстояния и кадрирования, а не детекции.")
            else:
                share = f"{len(hits)} из {len(inside)}"
                print(f"    Номер прочитан ПРАВИЛЬНО {share}, впервые в "
                      f"{hits[0][0]:.2f}s.")
                gaps = [b[0] - a[0] for a, b in zip(hits, hits[1:])]
                near = [g for g in gaps if g <= WINDOW_SEC]
                if len(hits) == 1:
                    print(f"    Но подтверждение требует набрать вес "
                          f"{MIN_TOP_WEIGHT} за {WINDOW_SEC:.0f} с, а одно "
                          f"чтение даёт максимум 1.40.")
                    print("    Одного верного чтения не хватает НИКОГДА, это "
                          "защита от случайного кадра.")
                    print("    Нужно второе верное чтение в том же окне: "
                          "работа над качеством выреза, не над порогом.")
                elif not near:
                    print(f"    Верные чтения есть, но между ними больше "
                          f"{WINDOW_SEC:.0f} с, и голоса не складываются.")
                else:
                    print("    Верные чтения складывались в одно окно: если "
                          "номер всё равно не подтверждён,")
                    print("    смотри уверенность этих чтений, веса не хватило.")

        if not args.no_frames:
            n = dump_frames(video, plate, start, end, args.frames, out_dir)
            if n:
                print(f"    сохранено кадров: {n} в runs/missed_{args.stem}/")

    if not args.no_frames:
        print(f"\nПосмотри картинки в {out_dir} и скажи, видно ли на них номер.")


if __name__ == "__main__":
    main()
