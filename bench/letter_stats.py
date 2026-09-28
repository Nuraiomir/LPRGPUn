#!/usr/bin/env python3
"""
Counts how much material the dataset holds for the one confusion that is left.

After the KZ-block fix, 35 of the 37 remaining OCR failures on new-format
plates are the same thing: O, Q and D in a small or blurred crop all come back
as the digit 0. No post-processing rule can settle that -- the information is
not in the image -- so the only route besides frame voting is to teach the
recognizer these three letters apart.

Training can only teach what it can see. If the training split holds thousands
of O and a handful of Q, fine-tuning will simply harden the model's existing
habit of answering O, and the result will be worse than today. This script
answers that before any GPU time is spent: how often each letter appears in
the letter slots, per split, and how many plates carry the hard ones.

Reads the annotation JSON only -- no images, no OCR, no GPU.

Usage:
    python3 bench/letter_stats.py ~/datasets_ocr_kz
    python3 bench/letter_stats.py ~/datasets_ocr_kz --hard OQD
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

NEW_KZ = re.compile(r"\d{3}([A-Z]{3})\d{2}")   # 123ABC02; group 1 = letter slots
LETTER_SLOTS = 3


def labels_in(split_dir):
    """Every plate string in one split, skipping annotations that won't parse."""
    plates, broken = [], 0
    for path in sorted((split_dir / "ann").glob("*.json")):
        try:
            text = json.loads(path.read_text(encoding="utf-8")).get("description", "")
        except (json.JSONDecodeError, OSError):
            broken += 1
            continue
        if text:
            plates.append(text.strip().upper())
    return plates, broken


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset root (has train/, val/, test/)")
    ap.add_argument("--hard", default="OQD",
                    help="the letters that get confused with each other (default: OQD)")
    args = ap.parse_args()

    hard = [c.upper() for c in args.hard]
    splits = [d.name for d in sorted(args.root.iterdir())
              if d.is_dir() and (d / "ann").is_dir()]
    if not splits:
        sys.exit(f"нет ни одной части с папкой ann/ в {args.root}")

    per_split = {}
    for split in splits:
        plates, broken = labels_in(args.root / split)
        letters = Counter()
        new_format = 0
        with_hard = Counter()          # plates carrying at least one hard letter
        for plate in plates:
            m = NEW_KZ.fullmatch(plate)
            if not m:
                continue
            new_format += 1
            slots = m.group(1)
            letters.update(slots)
            for c in set(slots) & set(hard):
                with_hard[c] += 1
        per_split[split] = {"всего": len(plates), "новых": new_format,
                            "буквы": letters, "с буквой": with_hard, "битых": broken}

    print(f"Датасет: {args.root}\n")
    for split, d in per_split.items():
        print(f"{split}: {d['всего']} номеров, из них нового формата {d['новых']}"
              + (f", битых файлов {d['битых']}" if d["битых"] else ""))
    train = per_split.get("train")

    print("\nСколько раз буква стоит в буквенном разряде (всего разрядов = номеров x 3):")
    header = "  буква  " + "".join(f"{s:>12s}" for s in splits)
    print(header)
    every = sorted({c for d in per_split.values() for c in d["буквы"]})
    for c in every:
        mark = " <-" if c in hard else ""
        row = f"  {c:5s}  " + "".join(f"{per_split[s]['буквы'][c]:12d}" for s in splits)
        print(row + mark)

    print(f"\nНомеров, где встречается спорная буква ({', '.join(hard)}):")
    print(header)
    for c in hard:
        print(f"  {c:5s}  " + "".join(f"{per_split[s]['с буквой'][c]:12d}" for s in splits))

    if not train:
        print("\nЧасти train нет — судить о пригодности для обучения не по чему.")
        return

    print("\n" + "=" * 62)
    counts = {c: train["буквы"][c] for c in hard}
    biggest = max(counts.values()) if counts else 0
    smallest = min(counts.values()) if counts else 0
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = 100 * n / max(sum(counts.values()), 1)
        print(f"  {c}: {n} примеров в обучении ({share:.0f}% от спорных)")

    print()
    if smallest == 0:
        print("  ВЫВОД: одной из букв в обучении нет вообще. Дообучение на этом")
        print("  датасете различать их не научит — нужны кадры с недостающей буквой.")
    elif smallest < 100:
        print(f"  ВЫВОД: самой редкой буквы всего {smallest} примеров. Этого мало:")
        print("  модель закрепит частый вариант и станет отвечать им ещё увереннее.")
        print("  Прежде чем обучать, стоит добрать кадры именно с редкой буквой.")
    elif biggest > 4 * smallest:
        print(f"  ВЫВОД: перекос {biggest / max(smallest, 1):.0f}:1. Примеры есть, но обучать")
        print("  надо с выравниванием классов, иначе редкая буква утонет в частой.")
    else:
        print("  ВЫВОД: материала достаточно и перекос умеренный — дообучение осмысленно.")
    print("=" * 62)
    print("\n  Замер для сравнения уже есть: 95.4% на новом формате (val, 785 из 823).")
    print("  После обучения запускать тот же bench/ocr_dataset_eval.py на той же части.")


if __name__ == "__main__":
    main()
