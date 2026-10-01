#!/usr/bin/env python3
"""
PaddleOCR против Tesseract на одних и тех же вырезах, по одним и тем же правилам.

Два движка меряются в одном прогоне, на одной и той же выборке, и оба
проходят через одинаковый разбор: clean_text, потом extract_plate, потом
проверка казахстанского формата. Иначе сравнение было бы про правила, а не
про движки.

Что считается:

  прочитано правильно   номер после разбора совпал с разметкой. Это та же
                        величина, что 95.4% у нынешнего распознавателя
  дало формат номера    разбор вообще получил что-то формата номера; может
                        быть чужим номером, и это хуже, чем ничего
  совпало дословно      строка целиком равна разметке, без вытаскивания
  время на вырез        медиана, миллисекунды

Тессеракту даются лучшие условия, какие у него бывают, иначе сравнение
нечестное: белый список из латиницы и цифр, режимы «одна строка», «одно
слово» и «строка без сегментации», и те же четыре варианта картинки, что
перебирает наш распознаватель. Берётся самый уверенный ответ из всех.

Тессеракту нужны две вещи: пакет pytesseract и сама программа tesseract.

    .venv_gpu/bin/pip install pytesseract
    which tesseract || bash tools/install_tesseract_local.sh

Установщик кладёт программу в ~/.tess, без sudo, из тех же apt-источников,
которыми машина и так пользуется. В конце он печатает три строки export,
которые надо выполнить перед запуском этого сравнения.

Запуск:

    .venv_gpu/bin/python bench/ocr_engine_compare.py ~/datasets_ocr_kz/<папка> \\
        --split test --limit 300
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "bench"))

from lpr_recognizer import clean_text, extract_plate, valid_kz_plate  # noqa: E402
from ocr_dataset_eval import find_samples  # noqa: E402

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TESS_CONFIGS = (
    ("psm7", f"--oem 1 --psm 7 -c tessedit_char_whitelist={WHITELIST}"),
    ("psm8", f"--oem 1 --psm 8 -c tessedit_char_whitelist={WHITELIST}"),
    ("psm13", f"--oem 1 --psm 13 -c tessedit_char_whitelist={WHITELIST}"),
)


def variants(crop):
    """Те же четыре варианта картинки, что перебирает workers/ocr_gpu_worker.py."""
    up = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    out = [("original", crop), ("upscaled", up),
           ("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))]
    try:
        out.append(("enhanced", cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)))
    except Exception:
        pass
    return out


# То же число, что OCR_EARLY_EXIT_CONF в workers/ocr_gpu_worker.py. Скопировано,
# а не импортировано: тот модуль на первой строке тянет paddle. Тессеракт
# получает то же правило досрочной остановки, что и наш распознаватель, иначе
# сравнение по времени было бы не про движки, а про число попыток.
EARLY_EXIT_CONF = 0.92


def tesseract_read(crop, pytesseract):
    """Лучший из вариантов картинки и режимов: (текст, уверенность 0..1)."""
    best_text, best_conf = "", 0.0
    for _, image in variants(crop):
        if best_conf >= EARLY_EXIT_CONF:
            break
        for _, config in TESS_CONFIGS:
            try:
                data = pytesseract.image_to_data(
                    image, config=config, output_type=pytesseract.Output.DICT)
            except Exception:
                continue
            words, confs = [], []
            for text, conf in zip(data["text"], data["conf"]):
                text = str(text).strip()
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = -1.0
                if text and conf >= 0:
                    words.append(text)
                    confs.append(conf / 100.0)
            if not words:
                continue
            text = "".join(words)
            conf = sum(confs) / len(confs)
            if conf > best_conf:
                best_text, best_conf = text, conf
            if best_conf >= EARLY_EXIT_CONF:
                break
    return best_text, best_conf


def grade(raw, label):
    """Одинаковый разбор для обоих движков."""
    cleaned = clean_text(str(raw))
    plate = extract_plate(cleaned)
    return {
        "raw": str(raw)[:60],
        "plate": plate,
        "format_ok": bool(plate) and valid_kz_plate(plate),
        "correct": bool(plate) and plate == label,
        "exact": cleaned == label,
    }


def run_tesseract(samples):
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:
        print(f"  тессеракт недоступен: {e!r}")
        print("  как поставить без sudo, написано в шапке файла")
        return None
    rows, times = [], []
    for i, (path, label) in enumerate(samples, 1):
        crop = cv2.imread(str(path))
        if crop is None:
            continue
        t0 = time.perf_counter()
        raw, conf = tesseract_read(crop, pytesseract)
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        row = grade(raw, label)
        row.update({"file": path.name, "label": label,
                    "conf": round(conf, 3), "ms": round(ms, 1)})
        rows.append(row)
        if i % 25 == 0:
            print(f"    тессеракт {i}/{len(samples)}", flush=True)
    return rows, times


def run_paddle(samples):
    try:
        import config.gpu_env as gpu_env
    except ImportError:
        sys.path.insert(0, str(ROOT))
        import config.gpu_env as gpu_env
    from gpu_workers_client import Workers

    workers = Workers(gpu_env.PROJECT_ROOT, gpu_env.YOLO_PYTHON, gpu_env.OCR_PYTHON,
                      gpu_env.ONNX_MODEL, gpu_env.YOLO_CUDA_LD_PATH,
                      gpu_env.OCR_CUDA_LD_PATH)
    print(f"  {workers.ocr_info.get('backend')} на {workers.ocr_info.get('device')}")
    rows, times = [], []
    try:
        for i, (path, label) in enumerate(samples, 1):
            crop = cv2.imread(str(path))
            if crop is None:
                continue
            t0 = time.perf_counter()
            payload = workers.ocr(crop, "normal")
            ms = (time.perf_counter() - t0) * 1000.0
            times.append(ms)
            row = grade(payload.get("text", ""), label)
            row.update({"file": path.name, "label": label,
                        "conf": round(float(payload.get("conf", 0.0)), 3),
                        "ms": round(ms, 1)})
            rows.append(row)
            if i % 25 == 0:
                print(f"    paddle {i}/{len(samples)}", flush=True)
    finally:
        workers.close()
    return rows, times


def report(name, rows, times):
    n = len(rows) or 1
    correct = sum(r["correct"] for r in rows)
    fmt = sum(r["format_ok"] for r in rows)
    exact = sum(r["exact"] for r in rows)
    wrong_plate = sum(1 for r in rows if r["format_ok"] and not r["correct"])
    return {
        "engine": name, "samples": len(rows),
        "correct": correct, "correct_share": round(correct / n, 4),
        "format_ok": fmt, "wrong_plate": wrong_plate,
        "exact": exact, "exact_share": round(exact / n, 4),
        "median_ms": round(statistics.median(times), 1) if times else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="папка с train/ test/ val/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300,
                    help="0 = вся часть; тессеракт медленный, начни с 300")
    ap.add_argument("--engines", default="paddle,tesseract")
    # No fixed default: one shared name let a later run quietly overwrite an
    # earlier run's per-crop rows, and the two could then no longer be
    # compared. The name is built from what actually varies between runs, so
    # two different runs cannot land on the same file.
    ap.add_argument("--out", type=Path, default=None,
                    help="по умолчанию runs/ocr_<движки>_<модель>_<часть><N>.json")
    args = ap.parse_args()

    dataset = args.dataset.expanduser().resolve()
    samples = find_samples(dataset, args.split)
    if not samples:
        raise SystemExit(f"в {dataset / args.split} нет размеченных картинок")
    if args.limit:
        samples = samples[:args.limit]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    # Which models are actually in play. Both are chosen outside this script
    # (an environment variable for Paddle, a traineddata directory for
    # Tesseract), so the run's own name has to carry them: otherwise two runs
    # that differ only by model overwrite each other and cannot be compared,
    # which is exactly what happened once.
    paddle_model = os.environ.get("LPR_OCR_MODEL", "en_PP-OCRv5_mobile_rec")
    tess_data = os.environ.get("TESSDATA_PREFIX", "")
    tess_tag = Path(tess_data.rstrip("/")).name or "system"
    tags = []
    if "paddle" in engines:
        tags.append(paddle_model)
    if "tesseract" in engines:
        tags.append(f"tess-{tess_tag}")

    if args.out is None:
        part = f"{args.split}{len(samples)}"
        args.out = ROOT / "runs" / f"ocr_{'_'.join(tags)}_{part}.json"

    print(f"датасет: {dataset}")
    print(f"вырезов: {len(samples)} из части «{args.split}»")
    print(f"движки: {', '.join(engines)}")
    if "paddle" in engines:
        print(f"модель paddle: {paddle_model}")
    if "tesseract" in engines:
        print(f"данные tesseract: {tess_data or '(системные)'}")
    print(f"результат будет записан в: {args.out}\n")

    results, details = [], {}
    for engine in engines:
        print(f"{engine}:")
        got = run_paddle(samples) if engine == "paddle" else run_tesseract(samples)
        if not got:
            continue
        rows, times = got
        results.append(report(engine, rows, times))
        details[engine] = rows
        print()

    if not results:
        raise SystemExit("ни один движок не отработал")

    print("=" * 74)
    print("СРАВНЕНИЕ НА ОДНОЙ И ТОЙ ЖЕ ВЫБОРКЕ")
    print("=" * 74)
    head = f"{'движок':<12}{'правильно':>12}{'доля':>9}{'формат':>9}{'чужой':>8}{'мс':>8}"
    print(head)
    print("-" * 74)
    for r in results:
        print(f"{r['engine']:<12}{r['correct']:>7}/{r['samples']:<4}"
              f"{r['correct_share']:>9.3f}{r['format_ok']:>9}"
              f"{r['wrong_plate']:>8}{r['median_ms']:>8.0f}")

    print("\nКолонки:")
    print("  правильно  номер после разбора совпал с разметкой")
    print("  формат     разбор получил что-то формата номера, верного или нет")
    print("  чужой      формат верный, а номер другой. Это самая дорогая ошибка:")
    print("             сервис показал бы залог не той машины")
    print("  мс         медиана на один вырез")

    if len(results) == 2:
        a, b = results
        d = (a["correct_share"] - b["correct_share"]) * 100
        print(f"\nРазница в чтении: {abs(d):.1f} пункта в пользу "
              f"{a['engine'] if d > 0 else b['engine']}.")
        if b["median_ms"] and a["median_ms"]:
            slower = max(a["median_ms"], b["median_ms"]) / min(a["median_ms"], b["median_ms"])
            slow = a["engine"] if a["median_ms"] > b["median_ms"] else b["engine"]
            print(f"По скорости {slow} медленнее в {slower:.0f} раз.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dataset": str(dataset), "split": args.split, "samples": len(samples),
         "summary": results, "rows": details},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nподробности по каждому вырезу: {args.out}")


if __name__ == "__main__":
    main()
