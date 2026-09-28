#!/usr/bin/env python3
"""
Why plates are missed on our own footage, as opposed to on the public dataset.

On the labelled AUTO.RIA dataset, 35 of the 37 remaining failures on new-format
plates are one confusion: O, Q and D in a small crop all come back as 0. That
dataset is cropped plates from 2019, though, and our parking video is a
different camera, angle and distance. Before spending GPU days on fine-tuning
the recogniser to tell those three letters apart, this checks whether the
readings we actually fail on in the field belong to the same family.

It reads the raw OCR lines v19 already prints ("[NORMAL OCR RAW] ..."), so no
new inference is needed, and attributes each failed reading to a plate we know
is in the video: a reading is counted as this confusion only when allowing the
digits in its letter slots to be any of their candidate letters reproduces a
labelled plate. No guessing which letter, so the attribution is evidence, not
assumption.

Produce the logs first, keeping the full output rather than grepping it:

    for v in 20260923_152319 20260909_171120 20260908_150904 20260908_150800; do
        .venv_gpu/bin/python app/lpr_v19_universal.py videos/$v.mp4 > runs/raw_$v.log 2>&1
    done

Usage:
    python3 bench/field_misses.py
    python3 bench/field_misses.py --logs "runs/raw_*.log"
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "bench"))
from letter_repair import DIGIT_TO_LETTER, LETTER_SLOTS, repairs_to  # noqa: E402
from lpr_recognizer import clean_text, extract_plate, valid_kz_plate  # noqa: E402

RAW_LINE = re.compile(r"\[NORMAL OCR RAW\]\s+t=\s*([\d.]+)s\s+RAW=(.+?)\s+conf=([\d.]+)")

CLASSES = ["верно", "чужой номер", "буква как цифра", "обрывок или пусто", "другое"]


def readings_in(log_path):
    """[(time, raw text, confidence)] from one v19 log."""
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RAW_LINE.search(line)
        if not m:
            continue
        try:
            raw = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            raw = m.group(2)
        out.append((float(m.group(1)), str(raw), float(m.group(3))))
    return out


def confirmed_in(stem):
    """Plates the run confirmed, from the saved JSON. Empty if it is not there."""
    path = ROOT / "runs" / f"real_video_v18_{stem}" / "results_vehicle_switch_GPU.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return [e.get("to", "") for e in data.get("switch_events", []) if e.get("to")]


def classify(raw, wanted):
    """(class, plate it belongs to, note) for one raw reading."""
    clean = clean_text(raw)
    plate = extract_plate(clean)
    if plate and plate in wanted:
        return "верно", plate, ""
    if plate:
        return "чужой номер", plate, f"{clean} -> {plate}"
    target = repairs_to(clean, wanted)
    if target:
        chars = []
        for start in range(len(clean) - 8 + 1):
            window = clean[start:start + 8]
            if window[:3] == target[:3] and window[6:] == target[6:]:
                chars = [f"{target[i]} как {window[i]}"
                         for i in LETTER_SLOTS
                         if window[i] != target[i]
                         and target[i] in DIGIT_TO_LETTER.get(window[i], ())]
                break
        return "буква как цифра", target, ", ".join(chars)
    if len(clean) < 6:
        return "обрывок или пусто", "", clean or "(пусто)"
    return "другое", "", clean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="runs/raw_*.log",
                    help="glob for v19 logs; the video name is taken from raw_<name>.log")
    ap.add_argument("--labels", type=Path, default=ROOT / "bench" / "labels.json")
    args = ap.parse_args()

    labels = {k: [p["plate"] for p in v]
              for k, v in json.loads(args.labels.read_text(encoding="utf-8")).items()
              if not k.startswith("_")}

    paths = sorted(Path(p) for p in glob(str(ROOT / args.logs)))
    if not paths:
        sys.exit(f"не найдено ни одного лога по образцу {args.logs}\n"
                 "Команда для их получения есть в описании этого скрипта.")

    totals = Counter()
    kz_gained_all = 0
    per_plate_all = {}

    for path in paths:
        stem = path.stem[4:] if path.stem.startswith("raw_") else path.stem
        wanted = labels.get(stem)
        if wanted is None:
            print(f"{path.name}: видео {stem} нет в labels.json, пропускаю\n")
            continue

        readings = readings_in(path)
        counts = Counter()
        notes = {}
        touched = Counter()
        kz_gained = 0
        for _t, raw, _conf in readings:
            clean = clean_text(raw)
            if extract_plate(clean) and not valid_kz_plate(clean):
                kz_gained += 1
            kind, plate, note = classify(raw, wanted)
            counts[kind] += 1
            if plate:
                touched[(kind, plate)] += 1
            bucket = notes.setdefault(kind, [])
            if note and note not in bucket and len(bucket) < 3:
                bucket.append(note)

        confirmed = confirmed_in(stem)
        print(f"=== {stem}   чтений OCR: {len(readings)}")
        if confirmed is None:
            print("    прогон JSON не найден, подтверждённые номера неизвестны")
        else:
            missed = [p for p in wanted if p not in confirmed]
            print(f"    в видео {len(wanted)}, подтверждено {len(set(confirmed))}, "
                  f"не найдено {len(missed)}"
                  + (f": {', '.join(missed)}" if missed else ""))
        for kind in CLASSES:
            if not counts[kind]:
                continue
            share = 100 * counts[kind] / max(len(readings), 1)
            line = f"    {kind:20s} {counts[kind]:5d}  ({share:4.1f}%)"
            if notes.get(kind):
                line += "   " + "; ".join(notes[kind])
            print(line)
        if kz_gained:
            print(f"    из них приняты только благодаря правилу KZ: {kz_gained}")

        hits = [(p, n) for (k, p), n in touched.items() if k == "буква как цифра"]
        if hits:
            print("    номера, пострадавшие от путаницы букв:")
            for plate, n in sorted(hits, key=lambda x: -x[1]):
                seen = touched.get(("верно", plate), 0)
                state = "и так прочитан верно" if seen else "ТОЛЬКО так и читался"
                print(f"      {plate}: {n} чтений, {state}")
                # A plate counts as blocked only if no video ever read it cleanly.
                per_plate_all[plate] = per_plate_all.get(plate, True) and not seen
        print()
        totals.update(counts)
        kz_gained_all += kz_gained

    if not totals:
        return
    failed = sum(totals[k] for k in CLASSES if k != "верно")
    print("=" * 62)
    print(f"Всего чтений: {sum(totals.values())}, из них неудачных: {failed}")
    for kind in CLASSES:
        if kind == "верно" or not totals[kind]:
            continue
        print(f"  {kind:20s} {totals[kind]:5d}  "
              f"({100 * totals[kind] / max(failed, 1):4.1f}% неудачных)")
    if kz_gained_all:
        print(f"\nПравило KZ дало {kz_gained_all} чтений, которые раньше отбрасывались.")

    letters = totals["буква как цифра"]
    blocked = sum(1 for only_damaged in per_plate_all.values() if only_damaged)
    print()
    if blocked:
        word = "номер" if blocked == 1 else "номера"
        print(f"  ВЫВОД: {blocked} {word} в поле читались ТОЛЬКО с путаницей букв,")
        print("  то есть та же семья ошибок, что и в датасете. Дообучение бьёт в цель.")
    elif letters:
        print("  ВЫВОД: путаница букв в поле встречается, но каждый такой номер")
        print("  хотя бы раз был прочитан чисто, и голосование его вытянуло.")
        print("  Дообучение ускорит подтверждение, но новых номеров почти не добавит.")
    else:
        print("  ВЫВОД: в поле этой путаницы нет. Мы теряем номера по другой причине,")
        print("  и дообучение на O/Q/D лечит болезнь, которой у нас не наблюдается.")
    print("=" * 62)


if __name__ == "__main__":
    main()
