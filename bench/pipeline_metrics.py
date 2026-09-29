#!/usr/bin/env python3
"""
Precision, recall and F1 of the whole pipeline, counted in vehicles.

Every other benchmark here measures one stage. This one measures the answer the
service actually gives: for each car that was in front of the camera, did the
service name it, and did it ever name a car that was not there.

    TP  a plate that is in the video and was confirmed
    FP  a plate that was confirmed and is not in the video
    FN  a plate that is in the video and was never confirmed

    precision = TP / (TP + FP)   of the plates we announced, how many were real
    recall    = TP / (TP + FN)   of the cars present, how many we found
    F1        harmonic mean of the two

This is not the detector's precision and recall, which are measured per box
against a ground-truth box during training and live in the training run's
output. A plate can be detected perfectly and still never confirmed, because
confirmation needs several agreeing readings. Reporting one number as the other
would overstate what the service does.

A false positive here is the expensive error: OCRM would open a different
vehicle's card. So precision is reported per video as well as overall, and any
wrong plate is printed with its name rather than folded into a count.

Counted over plates, not frames, so the sample is small: state the count next to
the metric and do not read a difference of one plate as a difference in quality.

Usage:
    python3 bench/pipeline_metrics.py
    python3 bench/pipeline_metrics.py --videos 20260923_152319
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def confirmations(stem):
    """Plates the run confirmed, in order, or None when there is no run."""
    path = ROOT / "runs" / f"real_video_v18_{stem}" / "results_vehicle_switch_GPU.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return [e["to"] for e in data.get("switch_events", []) if e.get("to")]


def f1_of(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", help="video names; default is all in labels.json")
    ap.add_argument("--labels", type=Path, default=ROOT / "bench" / "labels.json")
    args = ap.parse_args()

    labels = {k: v for k, v in
              json.loads(args.labels.read_text(encoding="utf-8")).items()
              if not k.startswith("_")}
    stems = args.videos or sorted(labels)

    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    by_type = {"обычные": {"tp": 0, "fn": 0}, "квадратные": {"tp": 0, "fn": 0}}
    all_wrong, missing_all = [], []

    for stem in stems:
        if stem not in labels:
            print(f"пропущено: {stem} нет в labels.json")
            continue
        got = confirmations(stem)
        if got is None:
            print(f"пропущено: {stem} без сохранённого прогона")
            continue

        wanted = [p["plate"] for p in labels[stem]]
        square = {p["plate"] for p in labels[stem] if p.get("square")}
        found = set(got)

        tp = sum(1 for p in wanted if p in found)
        fn = sum(1 for p in wanted if p not in found)
        wrong = sorted(found - set(wanted))
        fp = len(wrong)

        for plate in wanted:
            kind = "квадратные" if plate in square else "обычные"
            by_type[kind]["tp" if plate in found else "fn"] += 1

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        all_wrong += [f"{stem}: {w}" for w in wrong]
        missing_all += [f"{stem}: {p}" for p in wanted if p not in found]
        rows.append((stem, len(wanted), tp, fp, fn, *f1_of(tp, fp, fn)))

    if not rows:
        sys.exit("нечего считать")

    print("Сквозные метрики: считается машина, а не кадр и не символ.\n")
    print("  видео                машин    TP   FP   FN   точность  полнота    F1")
    for stem, n, tp, fp, fn, pr, rc, f1 in rows:
        print(f"  {stem:20s} {n:5d} {tp:5d} {fp:4d} {fn:4d}    "
              f"{pr:6.3f}   {rc:6.3f}  {f1:6.3f}")

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    pr, rc, f1 = f1_of(tp, fp, fn)
    n = tp + fn
    print(f"\n  ВСЕГО                {n:5d} {tp:5d} {fp:4d} {fn:4d}    "
          f"{pr:6.3f}   {rc:6.3f}  {f1:6.3f}")

    print("\n  По типу номера (полнота):")
    for kind, d in by_type.items():
        total = d["tp"] + d["fn"]
        if not total:
            continue
        print(f"    {kind:12s} {d['tp']:3d} из {total:3d}   {100 * d['tp'] / total:5.1f}%")

    if all_wrong:
        print("\n  Подтверждены номера, которых в видео нет:")
        for w in all_wrong:
            print(f"    {w}")
        print("    Это дорогая ошибка: ОСРМ открыла бы карточку чужого залога.")
    else:
        print("\n  Ни одного подтверждённого номера, которого не было в кадре.")

    if missing_all:
        print("\n  Не найдены:")
        for m in missing_all:
            print(f"    {m}")

    print(f"\n  Выборка: {n} машин на {len(rows)} видео. Это мало: разница в одну")
    print("  машину двигает метрику на несколько процентов, так что рядом с цифрой")
    print("  всегда надо называть размер выборки.")
    print("\n  Это не precision и recall детектора: там метрика считается по рамкам")
    print("  против эталонных рамок и живёт в выводе обучения. Номер может быть")
    print("  прекрасно задетектирован и всё равно не подтверждён, потому что для")
    print("  подтверждения нужно несколько согласных чтений.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # piping into head closes the stream; not a failure
        pass
