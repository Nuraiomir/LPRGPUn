"""
ocr_ab_test.py -- OCR A/B test: en_PP-OCRv5_mobile_rec vs PP-OCRv6_tiny_rec

Answers one narrow question: on the SAME real crops extracted by
lpr_v19_extract_crops.py, does PP-OCRv6_tiny_rec offer a useful
accuracy/latency improvement over the validated PP-OCRv5 baseline?

This script never touches YOLO, temporal voting, vehicle switching or
threshold logic from lpr_v19_universal.py -- it only loads previously
saved crop images from disk and calls PaddleX's create_predictor(), the
same call baseline's OCR_WORKER subprocess makes, once per model per
crop. lpr_v19_universal.py is imported (read-only) purely to reuse its
plate-normalization helpers (clean_text/normalize_top/normalize_bottom/
valid_kz_plate) so "looks like a valid KZ plate component" means exactly
what it means in production -- it is never modified.

Design choices worth being explicit about:
  - Each model gets exactly ONE inference call per crop (no multi-variant
    "try harder" search). Baseline's run_ocr()/OCR_WORKER ladder
    (original -> upscale -> gray -> enhance, with an early exit once
    confidence >= 0.92) is a *runtime strategy*, and how many variants it
    tries depends on the model's OWN confidence -- reusing it here would
    silently call one model more times than the other and bias the
    latency comparison. The one preprocessing step kept is the small
    upscale baseline applies unconditionally to any crop under 32px tall
    (a degenerate-input safety net, not a recognition strategy), applied
    identically for both models.
  - Both models run against a `.copy()` of the exact same in-memory
    decoded image per crop, so nothing about the input can differ or be
    mutated between the two calls.
  - Confidence, exact-match correctness and latency are reported and
    aggregated completely independently -- a higher average confidence
    from one model is never treated as evidence of higher accuracy.

Must run inside the paddlex GPU venv (it re-execs itself into it if the
CUDA lib path isn't already set):
    /home/jovyan/work/_shared/nurai/.venv_paddlex_gpu/bin/python ocr_ab_test.py

Usage:
    ocr_ab_test.py [--crops-dir runs/ocr_ab_crops] [--out-dir runs/ocr_ab_test]
                    [--limit N]
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OCR_VENV = ROOT / ".venv_paddlex_gpu"
_REEXEC_FLAG = "_LPR_OCR_AB_LD_SET"


def _ensure_cuda_ld_path():
    """
    Mirrors the `ocr_ld` construction in lpr_v19_universal.py's main()
    (same OCR_VENV, same python3.10 nvidia sub-paths, same order) so
    Paddle finds the venv's own bundled CUDA runtime/cuDNN/cuBLAS. Must
    happen before paddle is ever imported, so this re-execs the process
    once with the correct LD_LIBRARY_PATH already set at process start
    (same reason baseline sets it on Popen()'s env rather than after
    the interpreter is already running).
    """
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    ocr_ld = ":".join([
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cublas/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cudnn/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/curand/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cufft/lib"),
        str(OCR_VENV / "lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"),
        os.environ.get("LD_LIBRARY_PATH", ""),
    ])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ocr_ld
    env[_REEXEC_FLAG] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_ensure_cuda_ld_path()

import argparse
import csv
import importlib.util
import json
import re
import statistics
import time

import cv2
import numpy as np

BASELINE_PATH = ROOT / "lpr_v19_universal.py"


def _load_baseline():
    spec = importlib.util.spec_from_file_location(
        "lpr_v19_universal_baseline", BASELINE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = [str(BASELINE_PATH)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved_argv
    return mod


baseline = _load_baseline()

import paddle
from paddlex.inference import create_predictor

MODELS = ["en_PP-OCRv5_mobile_rec", "PP-OCRv6_tiny_rec"]

_PLATE_RE = re.compile(r"(\d{3})([A-Z]{3})(\d{2})")


def split_expected(expected):
    """
    Splits a full expected plate ("979CBB02") into the components each
    crop_type should be compared against: the full string for "normal",
    the 3-digit top row for "square_top", and the RRLLL bottom row
    (e.g. "02CBB") for "square_bottom" -- matching normalize_bottom()'s
    own RRLLL convention in lpr_v19_universal.py.
    """
    if not expected:
        return None, None, None
    cleaned = baseline.clean_text(expected)
    m = _PLATE_RE.fullmatch(cleaned)
    if not m:
        return None, None, None
    d3, l3, d2 = m.groups()
    return cleaned, d3, d2 + l3


def _extract_best(results):
    """Verbatim result-parsing loop from lpr_v19_universal.py's OCR_WORKER."""
    best_text, best_conf = "", 0.0
    for item in results:
        text = getattr(item, "rec_text", None)
        conf = getattr(item, "rec_score", None)
        if text is None and isinstance(item, dict):
            text = item.get("rec_text") or item.get("text")
            conf = item.get("rec_score") or item.get("score")
        if text is None:
            continue
        try:
            conf = float(conf or 0.0)
        except Exception:
            conf = 0.0
        text = str(text).strip()
        if text and conf > best_conf:
            best_text, best_conf = text, conf
    return best_text, best_conf


def run_ocr_single(predictor, image):
    if image is None or image.size == 0:
        return "", 0.0, 0.0
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, _w = image.shape[:2]
    if h < 32:
        scale = max(2.0, 64.0 / max(1, h))
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    t0 = time.perf_counter()
    try:
        results = list(predictor(image))
    except Exception:
        results = []
    ms = (time.perf_counter() - t0) * 1000.0

    text, conf = _extract_best(results)
    return text, conf, ms


def looks_valid_component(crop_type, text):
    if crop_type == "normal":
        return baseline.valid_kz_plate(text)
    if crop_type == "square_top":
        return bool(baseline.normalize_top(text))
    if crop_type == "square_bottom":
        return bool(baseline.normalize_bottom(text))
    return False


def normalize_for_type(crop_type, text):
    if crop_type == "normal":
        return baseline.clean_text(text)
    if crop_type == "square_top":
        return baseline.normalize_top(text)
    if crop_type == "square_bottom":
        return baseline.normalize_bottom(text)
    return baseline.clean_text(text)


def expected_for_type(crop_type, expected_full, expected_top, expected_bottom):
    if crop_type == "normal":
        return expected_full
    if crop_type == "square_top":
        return expected_top
    if crop_type == "square_bottom":
        return expected_bottom
    return None


def load_manifest_records(crops_dir):
    records = []
    for manifest_path in sorted(Path(crops_dir).glob("*/manifest.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        records.extend(data)
    return records


def resolve_crop_path(file_field):
    """
    manifest "file" entries are absolute paths written by
    lpr_v19_extract_crops.py. Also accepts a path relative to ROOT for
    manifests hand-edited or produced by an older version of that script.
    """
    p = Path(file_field)
    return p if p.is_absolute() else (ROOT / p)


def load_models():
    predictors = {}
    for name in MODELS:
        print(f"Loading {name} on gpu:0 ...", flush=True)
        predictors[name] = create_predictor(name, device="gpu:0")
    return predictors


def warm_up(predictors, sample_image):
    for predictor in predictors.values():
        for _ in range(2):
            try:
                list(predictor(sample_image.copy()))
            except Exception:
                pass


def run_ab_test(records, predictors):
    rows = []
    total = len(records)
    for i, rec in enumerate(records, 1):
        img_path = resolve_crop_path(rec["file"])
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"WARNING: could not read {img_path}, skipping", flush=True)
            continue

        crop_type = rec["crop_type"]
        expected_full, expected_top, expected_bottom = split_expected(
            rec.get("expected_plate")
        )
        expected_value = expected_for_type(
            crop_type, expected_full, expected_top, expected_bottom
        )

        for model_name, predictor in predictors.items():
            text, conf, ms = run_ocr_single(predictor, image.copy())
            normalized = normalize_for_type(crop_type, text)
            exact_match = bool(expected_value) and (normalized == expected_value)

            rows.append({
                "model": model_name,
                "crop_id": rec["crop_id"],
                "crop_type": crop_type,
                "video_stem": rec.get("video_stem"),
                "file": rec["file"],
                "raw_text": text,
                "normalized_text": normalized,
                "confidence": round(conf, 4),
                "latency_ms": round(ms, 3),
                "is_empty": (text == ""),
                "looks_valid_component": looks_valid_component(crop_type, text),
                "expected_value": expected_value,
                "has_ground_truth": expected_value is not None,
                "exact_match": exact_match,
            })

        if i % 100 == 0 or i == total:
            print(f"  ... processed {i}/{total} crops", flush=True)
    return rows


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def summarize(rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    summary = {}
    for model_name, items in by_model.items():
        latencies = sorted(x["latency_ms"] for x in items)
        confidences = [x["confidence"] for x in items]
        n = len(items)
        empties = sum(1 for x in items if x["is_empty"])
        valid = sum(1 for x in items if x["looks_valid_component"])
        gt_items = [x for x in items if x["has_ground_truth"]]
        exact = sum(1 for x in gt_items if x["exact_match"])

        summary[model_name] = {
            "count": n,
            "avg_latency_ms": round(statistics.mean(latencies), 3) if n else 0.0,
            "median_latency_ms": round(statistics.median(latencies), 3) if n else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 95), 3) if n else 0.0,
            "avg_confidence": round(statistics.mean(confidences), 4) if n else 0.0,
            "empty_count": empties,
            "empty_rate": round(empties / n, 4) if n else 0.0,
            "valid_component_count": valid,
            "valid_component_rate": round(valid / n, 4) if n else 0.0,
            "ground_truth_count": len(gt_items),
            "exact_match_count": exact,
            "exact_match_accuracy": round(exact / len(gt_items), 4) if gt_items else None,
        }
    return summary


def summarize_by_type(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["crop_type"]), []).append(r)

    out = {}
    for (model_name, crop_type), items in groups.items():
        n = len(items)
        valid = sum(1 for x in items if x["looks_valid_component"])
        gt = [x for x in items if x["has_ground_truth"]]
        matches = sum(1 for x in gt if x["exact_match"])
        out.setdefault(model_name, {})[crop_type] = {
            "count": n,
            "valid_component_rate": round(valid / n, 4) if n else 0.0,
            "ground_truth_count": len(gt),
            "exact_match_accuracy": round(matches / len(gt), 4) if gt else None,
        }
    return out


def print_summary(summary, by_type):
    print()
    print("=" * 78)
    print("OCR A/B TEST SUMMARY")
    print("=" * 78)
    for model_name, s in summary.items():
        print(f"\n[{model_name}]")
        print(f"  crops tested:          {s['count']}")
        print(f"  avg latency:           {s['avg_latency_ms']:.2f} ms")
        print(f"  median latency:        {s['median_latency_ms']:.2f} ms")
        print(f"  p95 latency:           {s['p95_latency_ms']:.2f} ms")
        print(f"  avg confidence:        {s['avg_confidence']:.3f}")
        print(f"  empty results:         {s['empty_count']} ({s['empty_rate'] * 100:.1f}%)")
        print(f"  valid-component rate:  {s['valid_component_rate'] * 100:.1f}%")
        if s["exact_match_accuracy"] is not None:
            print(
                f"  exact-match accuracy:  {s['exact_match_accuracy'] * 100:.1f}% "
                f"({s['exact_match_count']}/{s['ground_truth_count']} known samples)"
            )
        else:
            print("  exact-match accuracy:  n/a (no expected_plate ground truth in manifest)")

        if model_name in by_type:
            print("  by crop type:")
            for crop_type, t in sorted(by_type[model_name].items()):
                acc = (
                    f"{t['exact_match_accuracy'] * 100:.1f}%"
                    if t["exact_match_accuracy"] is not None else "n/a"
                )
                print(
                    f"    {crop_type:14s} n={t['count']:5d} "
                    f"valid_component={t['valid_component_rate'] * 100:5.1f}% "
                    f"exact_match={acc}"
                )

    print()
    print("-" * 78)
    print("NOTE: higher confidence does NOT imply higher correctness.")
    print("Judge exact_match_accuracy (ground-truth crops only) and")
    print("valid_component_rate independently from avg_confidence and latency.")
    print("-" * 78)


def write_outputs(rows, summary, by_type, args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "crops_dir": str(args.crops_dir),
        "models": MODELS,
        "paddle_version": paddle.__version__,
        "num_crops_tested": len(rows) // max(1, len(MODELS)),
        "total_ocr_calls": len(rows),
        "limit": args.limit,
    }

    json_path = out_dir / "ocr_ab_results.json"
    json_path.write_text(
        json.dumps(
            {
                "run": run_meta,
                "summary": summary,
                "summary_by_crop_type": by_type,
                "results": rows,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = out_dir / "ocr_ab_results.csv"
    fieldnames = [
        "model", "crop_id", "crop_type", "video_stem", "file",
        "raw_text", "normalized_text", "confidence", "latency_ms",
        "is_empty", "looks_valid_component", "expected_value",
        "has_ground_truth", "exact_match",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"\nJSON results: {json_path}")
    print(f"CSV results:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--crops-dir", default=str(ROOT / "runs" / "ocr_ab_crops"),
        help="Root directory produced by lpr_v19_extract_crops.py",
    )
    parser.add_argument(
        "--out-dir", default=str(ROOT / "runs" / "ocr_ab_test"),
        help="Where to write ocr_ab_results.json/.csv",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only test the first N crops (deterministic order), for a quick smoke test",
    )
    args = parser.parse_args()

    records = load_manifest_records(args.crops_dir)
    if not records:
        print(f"No crops found under {args.crops_dir}. Run lpr_v19_extract_crops.py first.")
        return
    # Deterministic order for reproducibility, independent of filesystem
    # directory-listing order.
    records.sort(key=lambda r: (r["crop_type"], r["file"]))
    if args.limit:
        records = records[: args.limit]

    print("=" * 78)
    print("OCR A/B TEST -- en_PP-OCRv5_mobile_rec vs PP-OCRv6_tiny_rec")
    print("=" * 78)
    print(f"Crops dir: {args.crops_dir}")
    print(f"Crops to test: {len(records)}")
    print(f"Paddle: {paddle.__version__}")
    print()

    predictors = load_models()

    first_image = cv2.imread(str(resolve_crop_path(records[0]["file"])), cv2.IMREAD_COLOR)
    if first_image is None:
        first_image = np.full((64, 128, 3), 128, np.uint8)
    warm_up(predictors, first_image)

    rows = run_ab_test(records, predictors)
    summary = summarize(rows)
    by_type = summarize_by_type(rows)

    print_summary(summary, by_type)
    write_outputs(rows, summary, by_type, args, Path(args.out_dir))


if __name__ == "__main__":
    main()
