#!/usr/bin/env python3
"""
Shows the votes behind every confirmation in a benchmark run, so a rule for
"too close to call" can be chosen from real numbers instead of guessed.

For a square plate it prints the competing row values and their weights, taken
from the diagnostics the server recorded at that moment (needs runs made with
--profile). For a single-row plate it prints the readings of the last
SWITCH_WINDOW_SEC seconds, which is what the switch rule looks at.

At the end it compares the margins of correct and wrong confirmations: if the
wrong ones are consistently closer calls than the correct ones, a threshold
exists; if the ranges overlap, no threshold on margin alone can separate them.

Usage:
    python3 bench/vote_margins.py runs/hard_votes
"""

import json
import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from lpr_recognizer import SWITCH_WINDOW_SEC, clean_text, valid_kz_plate  # noqa: E402


def load_labels(root):
    data = json.loads((Path(root) / "bench" / "labels.json").read_text(encoding="utf-8"))
    return {k: [p["plate"] for p in v] for k, v in data.items() if not k.startswith("_")}


def rows(votes):
    """'633 2.70 (2 чтения)' for the strongest values."""
    return " | ".join(f"{v['value']} вес {v['weight']:.2f} ({v['reads']} чт.)" for v in votes[:3])


def margin(votes):
    """Absolute and relative lead of the strongest value over the runner-up."""
    if not votes:
        return None, None
    if len(votes) == 1:
        return votes[0]["weight"], float("inf")
    return votes[0]["weight"] - votes[1]["weight"], votes[0]["weight"] / max(1e-9, votes[1]["weight"])


def normal_window(responses, t0):
    """Valid plates read in the SWITCH_WINDOW_SEC before t0, with their confidence."""
    out = {}
    for r in responses:
        t, x = r["video_time"], r["response"]
        if not (t0 - SWITCH_WINDOW_SEC <= t <= t0) or x.get("plate_type") != "normal":
            continue
        text = clean_text(x.get("raw_text") or "")
        if valid_kz_plate(text):
            out.setdefault(text, []).append(round(float(x.get("ocr_confidence") or 0.0), 2))
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/hard_votes")
    labels = load_labels(".")
    correct_margins, wrong_margins = [], []
    sections = {"ЧУЖИЕ ПОДТВЕРЖДЕНИЯ": [], "ПРАВИЛЬНЫЕ ПОДТВЕРЖДЕНИЯ": []}

    for f in sorted(glob(str(root / "*" / "*.json"))):
        video, variant = os.path.basename(f)[:-5].split("__")
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        responses = d.get("responses", [])
        for r in responses:
            t, x = r["video_time"], r["response"]
            if not x.get("changed") or not x.get("plate"):
                continue
            plate = x["plate"]
            good = plate in labels[video]
            lines = [f"  {variant:9s} {video} {t:6.2f} с  {plate}  ({x.get('plate_type')})"]
            votes = (x.get("profile") or {}).get("square_votes")
            if votes:
                m_top, ratio_top = margin(votes["top"])
                m_bot, _ = margin(votes["bottom"])
                lines.append(f"      верх: {rows(votes['top'])}")
                lines.append(f"      низ:  {rows(votes['bottom'])}")
                lines.append(f"      перевес верха: {m_top:.2f}" +
                             ("" if ratio_top == float("inf") else f" ({ratio_top:.1f}x)") +
                             ("   ЕДИНСТВЕННЫЙ КАНДИДАТ" if ratio_top == float("inf") else ""))
                (correct_margins if good else wrong_margins).append(
                    (m_top, ratio_top, plate, variant, video))
            else:
                w = normal_window(responses, t)
                lines.append("      чтения за последние 1.5 с: " +
                             (", ".join(f"{p} x{len(c)} {c}" for p, c in w.items()) or "нет"))
            sections["ПРАВИЛЬНЫЕ ПОДТВЕРЖДЕНИЯ" if good else "ЧУЖИЕ ПОДТВЕРЖДЕНИЯ"].extend(lines)

    for name, lines in sections.items():
        print(f"\n===== {name}")
        print("\n".join(lines) if lines else "  нет")

    print("\n===== ПЕРЕВЕС ВЕРХНЕЙ СТРОКИ: ЧУЖИЕ ПРОТИВ ПРАВИЛЬНЫХ")
    for name, data in (("чужие", wrong_margins), ("правильные", correct_margins)):
        finite = [m for m, ratio, *_ in data if ratio != float("inf")]
        alone = sum(1 for _, ratio, *_ in data if ratio == float("inf"))
        if finite:
            print(f"  {name:11s} {len(data):2d} шт.: перевес от {min(finite):.2f} до {max(finite):.2f}"
                  f"; без соперника: {alone}")
        else:
            print(f"  {name:11s} {len(data):2d} шт.: соперников не было ни разу")
    if wrong_margins and correct_margins:
        w = [m for m, ratio, *_ in wrong_margins if ratio != float("inf")]
        c = [m for m, ratio, *_ in correct_margins if ratio != float("inf")]
        if w and c:
            print(f"\n  максимальный перевес у чужого: {max(w):.2f}")
            print(f"  минимальный перевес у правильного (при наличии соперника): {min(c):.2f}")
            print("  порог разделяет случаи" if max(w) < min(c)
                  else "  диапазоны пересекаются: перевеса одного недостаточно")


if __name__ == "__main__":
    main()
