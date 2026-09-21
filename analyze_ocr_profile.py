#!/usr/bin/env python3
"""
Разбор уже собранного профиля (runs/http_profile.json).

Отвечает на вопрос: откуда берутся всплески 500-800 мс -- из кодирования
JPEG, из IPC, из ожидания блокировки или из самого инференса PaddleX.

Ничего не запускает и не меняет. Только читает JSON, который уже сохранил
camera_client.py с флагом --profile.

Запуск:
    python3 tools/analyze_ocr_profile.py runs/http_profile.json
"""

import json
import sys
from collections import defaultdict


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100))]


def stats_line(name, vals, width=34):
    if not vals:
        print(f"  {name:{width}s} нет данных")
        return
    s = sorted(vals)
    avg = sum(s) / len(s)
    print(f"  {name:{width}s} n={len(s):4d}  сред={avg:7.1f}  "
          f"мед={pct(s,50):7.1f}  p95={pct(s,95):7.1f}  макс={s[-1]:7.1f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)

    profiles = data.get("profiles", [])
    if not profiles:
        print("В файле нет секции 'profiles'.")
        print("Прогон нужно было делать с флагом --profile.")
        sys.exit(1)

    # Собираем все вызовы воркеров из всех запросов
    calls = []
    for p in profiles:
        for d in p.get("worker_detail", []):
            calls.append({
                "video_time": p.get("video_time"),
                "stage": d.get("stage"),
                "jpeg_encode_ms": float(d.get("jpeg_encode_ms", 0.0)),
                "lock_wait_ms": float(d.get("lock_wait_ms", 0.0)),
                "ipc_roundtrip_ms": float(d.get("ipc_roundtrip_ms", 0.0)),
                "worker_inference_ms": float(d.get("worker_inference_ms", 0.0)),
                "jpeg_bytes": d.get("jpeg_bytes"),
                "crop_size": d.get("crop_size"),
                "variants": d.get("variants", []),
                "variant_count": d.get("variant_count"),
                "worker_prep_ms": d.get("worker_prep_ms"),
                "worker_infer_ms": d.get("worker_infer_ms"),
                "worker_decode_ms": d.get("worker_decode_ms"),
                "request_server_ms": p.get("server_ms"),
                "request_ocr_calls": p.get("ocr_calls", 0),
            })

    if not calls:
        print("В профиле нет worker_detail. Нужен прогон с обновлённым сервером.")
        sys.exit(1)

    ocr_calls = [c for c in calls if c["stage"] != "yolo"]
    yolo_calls = [c for c in calls if c["stage"] == "yolo"]

    print("=" * 78)
    print("РАЗБОР ПРОФИЛЯ: КУДА УХОДИТ ВРЕМЯ В ВЫЗОВАХ OCR")
    print("=" * 78)
    print(f"Файл:              {sys.argv[1]}")
    print(f"Запросов:          {len(profiles)}")
    print(f"Вызовов YOLO:      {len(yolo_calls)}")
    print(f"Вызовов OCR:       {len(ocr_calls)}")

    # --- Пункт 9: доходит ли сам инференс до 500-700 мс ---
    print()
    print("-" * 78)
    print("ГЛАВНЫЙ ВОПРОС: всплеск внутри инференса или в обвязке?")
    print("-" * 78)
    for label, key in [
        ("кодирование JPEG", "jpeg_encode_ms"),
        ("ожидание блокировки", "lock_wait_ms"),
        ("IPC туда-обратно", "ipc_roundtrip_ms"),
        ("инференс внутри воркера", "worker_inference_ms"),
    ]:
        stats_line(label, [c[key] for c in ocr_calls])

    inf = [c["worker_inference_ms"] for c in ocr_calls]
    ipc = [c["ipc_roundtrip_ms"] for c in ocr_calls]
    over_500 = [c for c in ocr_calls if c["worker_inference_ms"] > 500]
    print()
    print(f"  Вызовов, где инференс сам по себе > 500 мс: {len(over_500)}")
    if inf and ipc:
        # IPC включает в себя инференс, разница это чистая передача данных
        transport = [c["ipc_roundtrip_ms"] - c["worker_inference_ms"] for c in ocr_calls]
        stats_line("чистая передача (IPC минус инференс)", transport)

    # --- Пункт 5: корзины по времени ---
    print()
    print("-" * 78)
    print("РАСПРЕДЕЛЕНИЕ ВЫЗОВОВ OCR ПО ВРЕМЕНИ (по инференсу воркера)")
    print("-" * 78)
    buckets = [(0, 50), (50, 100), (100, 200), (200, 300),
               (300, 500), (500, 800), (800, 10**9)]
    for lo, hi in buckets:
        sel = [c for c in ocr_calls if lo <= c["worker_inference_ms"] < hi]
        label = f"{lo}-{hi} мс" if hi < 10**9 else f">{lo} мс"
        share = 100.0 * len(sel) / len(ocr_calls) if ocr_calls else 0
        bar = "#" * int(share / 2)
        print(f"  {label:>12s}  {len(sel):4d}  ({share:5.1f}%)  {bar}")

    # --- Пункты 7 и 8: normal vs square, первый вызов vs fallback ---
    print()
    print("-" * 78)
    print("СРАВНЕНИЕ ТИПОВ ВЫЗОВОВ OCR")
    print("-" * 78)
    by_stage = defaultdict(list)
    for c in ocr_calls:
        by_stage[c["stage"]].append(c["worker_inference_ms"])
    names = {
        "ocr_normal": "обычный номер (первый вызов)",
        "ocr_square": "квадратный номер (верх+низ)",
        "ocr_fallback": "повторный OCR (fallback)",
    }
    for stage in ("ocr_normal", "ocr_square", "ocr_fallback"):
        stats_line(names.get(stage, stage), by_stage.get(stage, []))

    print()
    print("  Суммарное время инференса по типам:")
    total_inf = sum(inf) or 1.0
    for stage, vals in sorted(by_stage.items(), key=lambda kv: -sum(kv[1])):
        s = sum(vals)
        print(f"    {names.get(stage, stage):34s} {s/1000.0:7.1f} сек  ({100.0*s/total_inf:5.1f}%)")

    # --- Размер кропа против времени ---
    sized = [c for c in ocr_calls if c.get("crop_size")]
    if sized:
        print()
        print("-" * 78)
        print("ЗАВИСИМОСТЬ ОТ РАЗМЕРА КРОПА")
        print("-" * 78)
        groups = defaultdict(list)
        for c in sized:
            w, h = c["crop_size"]
            area = w * h
            if area < 20000:
                key = "мелкий (<20k пикс)"
            elif area < 100000:
                key = "средний (20k-100k)"
            elif area < 400000:
                key = "крупный (100k-400k)"
            else:
                key = "очень крупный (>400k)"
            groups[key].append(c["worker_inference_ms"])
        for key in ["мелкий (<20k пикс)", "средний (20k-100k)",
                    "крупный (100k-400k)", "очень крупный (>400k)"]:
            if key in groups:
                stats_line(key, groups[key])

    # --- Разбор по вариантам препроцессинга (если сервер их прислал) ---
    variant_rows = []
    for c in ocr_calls:
        for v in c.get("variants", []) or []:
            variant_rows.append({**v, "stage": c["stage"], "video_time": c["video_time"]})

    if variant_rows:
        print()
        print("-" * 78)
        print("РАЗБОР ПО ВАРИАНТАМ ПРЕПРОЦЕССИНГА ВНУТРИ run_ocr")
        print("-" * 78)
        by_var = defaultdict(list)
        prep_by_var = defaultdict(list)
        for v in variant_rows:
            by_var[v["variant"]].append(float(v.get("infer_ms", 0.0)))
            prep_by_var[v["variant"]].append(float(v.get("prep_ms", 0.0)))

        order = ["original", "upscaled", "gray", "enhanced"]
        print("  Инференс PaddleX по вариантам:")
        for name in order:
            if name in by_var:
                stats_line(f"  {name}", by_var[name])
        print()
        print("  Подготовка изображения (CPU) по вариантам:")
        for name in order:
            if name in prep_by_var and sum(prep_by_var[name]) > 0:
                stats_line(f"  {name}", prep_by_var[name])

        print()
        print("  Суммарный вклад каждого варианта:")
        tot = sum(sum(v) for v in by_var.values()) + sum(sum(v) for v in prep_by_var.values())
        tot = tot or 1.0
        for name in order:
            if name in by_var:
                s = sum(by_var[name]) + sum(prep_by_var.get(name, []))
                print(f"    {name:12s} вызовов={len(by_var[name]):5d}  "
                      f"{s/1000.0:7.1f} сек  ({100.0*s/tot:5.1f}%)")

        # Сколько вызовов дошло до всех четырёх вариантов
        counts = defaultdict(int)
        for c in ocr_calls:
            n = c.get("variant_count")
            if n is not None:
                counts[n] += 1
        if counts:
            print()
            print("  Сколько вариантов успевало отработать до early-exit:")
            total_c = sum(counts.values()) or 1
            for n in sorted(counts):
                share = 100.0 * counts[n] / total_c
                print(f"    {n} вариант(ов): {counts[n]:5d}  ({share:5.1f}%)  {'#' * int(share/2)}")
            worst = counts.get(max(counts), 0)
            print()
            print(f"  Вызовов, где early-exit НЕ сработал и прогнались все варианты: {worst}")

    # --- Пункт 6: топ-20 самых медленных вызовов OCR ---
    print()
    print("-" * 78)
    print("20 САМЫХ МЕДЛЕННЫХ ВЫЗОВОВ OCR, ПОЛНАЯ РАЗБИВКА")
    print("-" * 78)
    print(f"  {'видео':>7s} {'тип':>13s} {'JPEG':>7s} {'блок':>7s} "
          f"{'IPC':>8s} {'инференс':>9s} {'передача':>9s} {'кроп':>12s}")
    for c in sorted(ocr_calls, key=lambda x: -x["worker_inference_ms"])[:20]:
        transport = c["ipc_roundtrip_ms"] - c["worker_inference_ms"]
        crop = c.get("crop_size")
        crop_s = f"{crop[0]}x{crop[1]}" if crop else "-"
        stage_short = c["stage"].replace("ocr_", "")
        print(f"  {c['video_time']:6.2f}с {stage_short:>13s} "
              f"{c['jpeg_encode_ms']:6.1f} {c['lock_wait_ms']:6.1f} "
              f"{c['ipc_roundtrip_ms']:7.1f} {c['worker_inference_ms']:8.1f} "
              f"{transport:8.1f} {crop_s:>12s}")

    # --- Вывод ---
    print()
    print("=" * 78)
    print("ВЫВОД")
    print("=" * 78)
    if not ocr_calls:
        return
    avg_inf = sum(inf) / len(inf)
    avg_enc = sum(c["jpeg_encode_ms"] for c in ocr_calls) / len(ocr_calls)
    avg_lock = sum(c["lock_wait_ms"] for c in ocr_calls) / len(ocr_calls)
    avg_tr = sum(c["ipc_roundtrip_ms"] - c["worker_inference_ms"] for c in ocr_calls) / len(ocr_calls)

    parts = [("инференс PaddleX", avg_inf), ("кодирование JPEG", avg_enc),
             ("ожидание блокировки", avg_lock), ("передача данных", avg_tr)]
    parts.sort(key=lambda x: -x[1])
    total = sum(v for _, v in parts) or 1.0
    print("Средний вызов OCR складывается так:")
    for name, val in parts:
        print(f"  {name:24s} {val:7.1f} мс  ({100.0*val/total:5.1f}%)")
    print()
    dominant = parts[0][0]
    print(f"Доминирует: {dominant}")
    if dominant == "инференс PaddleX":
        print("Значит проблема внутри модели или её препроцессинга, а не в")
        print("архитектуре воркеров. Следующий шаг: замер по вариантам")
        print("препроцессинга внутри run_ocr (original/upscaled/gray/enhanced).")
    else:
        print("Значит проблема в обвязке, а не в самой модели.")


if __name__ == "__main__":
    main()
