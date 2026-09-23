#!/usr/bin/env python3
"""
Would waiting a little before confirming a newly seen plate avoid the wrong
ones? This replays saved benchmark runs and answers that from the readings
that were actually recorded. It changes nothing and needs no GPU.

The rule being tried, on single-row plates only:

    a candidate that meets the current switch rule is not confirmed at once.
    It becomes "pending". It is confirmed at the first reading that arrives
    at least DELAY seconds later, and only if it still leads the readings of
    the last SWITCH_WINDOW_SEC seconds. If another candidate takes the lead,
    the wait starts again for that one.

Delay 0 must reproduce what the server really did; the script checks that
first and says so. Without that check the rest would mean nothing.

Square plates are left as they happened: the saved responses carry one
confidence for the whole plate, while the real voting uses the two rows
separately, so they cannot be replayed exactly.

Usage:
    python3 bench/confirm_delay_sim.py runs/hard_votes
    python3 bench/confirm_delay_sim.py runs/hard_votes --delays 0 0.3 0.5 1.0
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from lpr_recognizer import (  # noqa: E402
    SWITCH_CONFIRM_READS, SWITCH_STRONG_CONF, SWITCH_WINDOW_SEC,
    clean_text, valid_kz_plate,
)

DEFAULT_DELAYS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def weight(conf):
    return conf + 0.4 if conf >= 0.90 else conf


def load_run(path):
    """Readings and the events the server actually produced."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    reads, square_events, clears = [], [], []
    for r in d.get("responses", []):
        t, x = r["video_time"], r["response"]
        if x.get("plate_type") == "normal" and x.get("raw_text"):
            plate = clean_text(x["raw_text"])
            if valid_kz_plate(plate):
                reads.append((t, plate, float(x.get("ocr_confidence") or 0.0)))
        if x.get("changed") and x.get("plate") and x.get("plate_type") == "square":
            square_events.append((t, x["plate"]))
    prev = None
    for r in d.get("responses", []):
        cur = r["response"].get("plate") or ""
        if prev and not cur:
            clears.append(r["video_time"])
        prev = cur
    actual = [(e["video_time"], e["to"], e.get("plate_type")) for e in d.get("switch_events", [])]
    end = max((r["video_time"] for r in d.get("responses", [])), default=0.0)
    return reads, square_events, clears, actual, end


def simulate(reads, square_events, clears, delay):
    """
    Returns the confirmations this delay would have produced, as (time, plate).

    With delay 0 this is exactly LPRRecognizer.consider_plate for single-row
    plates: a reading confirms its own plate once that plate has been read
    SWITCH_CONFIRM_READS times inside SWITCH_WINDOW_SEC, or once on a reading
    at least SWITCH_STRONG_CONF confident, and a switch clears the history.

    With a delay the candidate has to be read again at least `delay` seconds
    later while still meeting that rule. If another plate meets the rule in
    between, it becomes the pending one and the wait starts over.
    """
    events, confirmed, history = [], "", []
    pending, pending_since = None, None
    others = sorted([(t, p, "square") for t, p in square_events] + [(t, "", "clear") for t in clears])
    oi = 0

    for t, plate, conf in reads:
        while oi < len(others) and others[oi][0] <= t:      # square confirmations and clears
            ot, op, kind = others[oi]
            confirmed = op if kind == "square" else ""
            if kind == "square":
                events.append((ot, op))
            history, pending = [], None
            oi += 1

        history = [x for x in history if t - x[0] <= SWITCH_WINDOW_SEC]
        history.append((t, plate, conf))
        if plate == confirmed:
            continue

        same = [x for x in history if x[1] == plate]
        if len(same) < SWITCH_CONFIRM_READS and conf < SWITCH_STRONG_CONF:
            continue

        if pending != plate:
            pending, pending_since = plate, t
        if t - pending_since >= delay:
            confirmed = plate
            events.append((t, plate))
            history, pending = [], None

    return events


def check_zero_delay(runs):
    """
    Delay 0 must reproduce the confirmations the server made. Where it does
    not, that run is left out of the numbers below.

    The usual reason is a square-shaped detection: the server also reads such
    a crop as a single-row plate, but the saved response only carries the two
    rows, so those readings cannot be replayed.
    """
    def same(got, want, tol=0.02):
        """Times may differ by a hundredth of a second through rounding."""
        return (len(got) == len(want)
                and all(a[1] == b[1] and abs(a[0] - b[0]) <= tol for a, b in zip(got, want, strict=True)))

    trusted, skipped = [], []
    for name, (reads, sq, clears, actual, _) in runs.items():
        got = [(round(t, 2), p) for t, p in simulate(reads, sq, clears, 0.0)]
        want = [(round(t, 2), p) for t, p, _ in actual if p]
        (trusted if same(got, want) else skipped).append((name, want, got))
    print(f"Проверка симулятора: при задержке 0 совпал с сервером в "
          f"{len(trusted)} прогонах из {len(runs)}")
    if skipped:
        print(f"  не совпал в {len(skipped)}, они исключены из подсчёта:")
        for name, want, got in skipped[:3]:
            print(f"    {name}")
            print(f"      сервер:    {want}")
            print(f"      симулятор: {got}")
        if len(skipped) > 3:
            print(f"    ... и ещё {len(skipped) - 3}")
    return {name for name, _, _ in trusted}


def score(events, expected, end):
    wrong_time, wrong_events, first_correct = 0.0, [], {}
    for i, (t, plate) in enumerate(events):
        t_end = events[i + 1][0] if i + 1 < len(events) else end
        if plate in expected:
            first_correct.setdefault(plate, t)
        else:
            wrong_events.append((t, plate))
            wrong_time += t_end - t
    return wrong_events, wrong_time, first_correct


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--delays", type=float, nargs="+", default=DEFAULT_DELAYS)
    ap.add_argument("--labels", default="bench/labels.json")
    args = ap.parse_args()

    labels = {k: [p["plate"] for p in v] for k, v in
              json.loads(Path(args.labels).read_text(encoding="utf-8")).items()
              if not k.startswith("_")}

    runs, videos = {}, {}
    for f in sorted(glob(str(args.run_dir / "*" / "*.json"))):
        video, variant = os.path.basename(f)[:-5].split("__")
        name = f"{variant} {video}"
        runs[name] = load_run(f)
        videos[name] = video
    if not runs:
        sys.exit(f"нет прогонов в {args.run_dir}")

    trusted = check_zero_delay(runs)
    runs = {k: v for k, v in runs.items() if k in trusted}
    if not runs:
        sys.exit("Ни один прогон не прошёл проверку, считать нечего.")
    print(f"\nЦифры ниже посчитаны по {len(runs)} прогонам.")

    baseline_first = {}
    print(f"  {'задержка':>9s} {'чужих':>6s} {'чужой на экране':>16s} {'правильных':>11s} "
          f"{'позже на, с':>12s}")
    rows = []
    for delay in args.delays:
        n_wrong = n_correct = 0
        wrong_time = 0.0
        later, lost, details = [], [], []
        for name, (reads, sq, clears, _actual, end) in runs.items():
            expected = labels[videos[name]]
            events = simulate(reads, sq, clears, delay)
            wrong_events, wt, first = score(events, expected, end)
            n_wrong += len(wrong_events)
            wrong_time += wt
            n_correct += len(first)
            if delay == args.delays[0]:
                baseline_first[name] = first
            else:
                for plate, t in first.items():
                    if plate in baseline_first.get(name, {}):
                        later.append(t - baseline_first[name][plate])
                for plate in baseline_first.get(name, {}):
                    if plate not in first:
                        lost.append(f"{name} {plate}")
            for t, p in wrong_events:
                details.append(f"{name} {t:.2f}с {p}")
        med = sorted(later)[len(later) // 2] if later else 0.0
        rows.append((delay, n_wrong, wrong_time, n_correct, med, details, lost))
        print(f"  {delay:8.1f}с {n_wrong:6d} {wrong_time:15.1f}с {n_correct:11d} {med:12.2f}")

    print("\nКакие чужие номера остаются:")
    for delay, _n_wrong, _, _, _, details, _lost in rows:
        print(f"  {delay:.1f}с: " + (", ".join(details) if details else "нет"))

    print("\nКакие правильные номера перестают подтверждаться:")
    for delay, _n_wrong, _, _, _, _details, lost in rows:
        if delay == args.delays[0]:
            continue
        print(f"  {delay:.1f}с: {len(lost)} шт." + (("  " + ", ".join(lost[:6])) if lost else ""))
        if len(lost) > 6:
            print(f"        ... и ещё {len(lost) - 6}")




if __name__ == "__main__":
    main()
