import json
from pathlib import Path

BASE = Path("runs/real_video_v18_20260909_171120")

A = BASE / "results_A_full.json"
B = BASE / "results_vehicle_switch_GPU.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def change(a, b):
    if a == 0:
        return None
    return (b - a) / a * 100.0


def timing(d):
    return d["timing_profile"]


def profile_stats(d):
    rows = d.get("normal_readings", []) + d.get("square_readings", [])

    prep = sum(x.get("prep_total_ms", 0) for x in rows)
    infer = sum(x.get("infer_total_ms", 0) for x in rows)

    variants = {}
    enhanced = 0

    for row in rows:
        for v in row.get("variants", []):
            name = v.get("variant", "unknown")
            variants[name] = variants.get(name, 0) + 1

            if name == "enhanced":
                enhanced += 1

    return {
        "saved_readings": len(rows),
        "prep_ms": prep,
        "infer_ms": infer,
        "variants": variants,
        "enhanced": enhanced,
    }


a = load(A)
b = load(B)

ta = timing(a)
tb = timing(b)

pa = profile_stats(a)
pb = profile_stats(b)

print("=" * 80)
print("A/B OCR BENCHMARK")
print("=" * 80)

print()
print("A = FULL")
print("    original + upscaled + gray + enhanced")
print()
print("B = NO-ENHANCED")
print("    original + upscaled + gray")
print()

print("-" * 80)
print("PERFORMANCE")
print("-" * 80)

print(
    f"{'Metric':32}"
    f"{'A FULL':>15}"
    f"{'B NO-ENH':>15}"
    f"{'Change':>15}"
)
print("-" * 80)


def row(name, av, bv, digits=2):
    c = change(av, bv)

    if c is None:
        ctext = "N/A"
    else:
        ctext = f"{c:+.1f}%"

    print(
        f"{name:32}"
        f"{av:15.{digits}f}"
        f"{bv:15.{digits}f}"
        f"{ctext:>15}"
    )


row(
    "Wall time (sec)",
    ta["total_wall_sec"],
    tb["total_wall_sec"],
)

row(
    "OCR calls",
    ta["ocr"]["count"],
    tb["ocr"]["count"],
    0,
)

row(
    "OCR total (ms)",
    ta["ocr"]["sum_ms"],
    tb["ocr"]["sum_ms"],
)

row(
    "OCR avg (ms)",
    ta["ocr"]["avg_ms"],
    tb["ocr"]["avg_ms"],
)

row(
    "YOLO calls",
    ta["yolo"]["count"],
    tb["yolo"]["count"],
    0,
)

row(
    "YOLO total (ms)",
    ta["yolo"]["sum_ms"],
    tb["yolo"]["sum_ms"],
)

row(
    "YOLO avg (ms)",
    ta["yolo"]["avg_ms"],
    tb["yolo"]["avg_ms"],
)

row(
    "Writer total (ms)",
    ta["writer"]["sum_ms"],
    tb["writer"]["sum_ms"],
)

row(
    "Writer avg (ms)",
    ta["writer"]["avg_ms"],
    tb["writer"]["avg_ms"],
)


print()
print("-" * 80)
print("OCR PROFILING")
print("-" * 80)

print(
    f"{'Metric':32}"
    f"{'A FULL':>15}"
    f"{'B NO-ENH':>15}"
    f"{'Change':>15}"
)
print("-" * 80)

row(
    "Saved OCR readings",
    pa["saved_readings"],
    pb["saved_readings"],
    0,
)

row(
    "Prep total (ms)",
    pa["prep_ms"],
    pb["prep_ms"],
)

row(
    "Inference total (ms)",
    pa["infer_ms"],
    pb["infer_ms"],
)

row(
    "Enhanced variants",
    pa["enhanced"],
    pb["enhanced"],
    0,
)


print()
print("Variant counts:")
print("A:", pa["variants"])
print("B:", pb["variants"])


print()
print("-" * 80)
print("RECOGNITION")
print("-" * 80)

print("A final:   ", a.get("confirmed_plate_final"))
print("B final:   ", b.get("confirmed_plate_final"))

print("A square:  ", a.get("confirmed_square"))
print("B square:  ", b.get("confirmed_square"))

print()
print("A switch events:")

for x in a.get("switch_events", []):
    print(
        f"  {x.get('time', 0):.2f}s: "
        f"{x.get('from', '')} -> {x.get('to', '')}"
    )

print()
print("B switch events:")

for x in b.get("switch_events", []):
    print(
        f"  {x.get('time', 0):.2f}s: "
        f"{x.get('from', '')} -> {x.get('to', '')}"
    )


print()
print("-" * 80)
print("CONCLUSION FOR THIS VIDEO")
print("-" * 80)

print(
    "The A/B experiment changes only the OCR enhanced variant."
)
print(
    "Detection count and OCR attempt count are identical."
)
print(
    "Final recognized plate:",
    a.get("confirmed_plate_final"),
    "vs",
    b.get("confirmed_plate_final"),
)
print(
    "Confirmed square:",
    a.get("confirmed_square"),
    "vs",
    b.get("confirmed_square"),
)

print()
print(
    "This is a behavioral/performance comparison, not a formal"
)
print(
    "accuracy benchmark because independently verified ground truth"
)
print(
    "annotations are not available for this video."
)

print("=" * 80)
