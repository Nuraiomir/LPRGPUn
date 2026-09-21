#!/usr/bin/env python3
"""
Сравнение двух прогонов офлайн-пайплайна app/lpr_v19_universal.py.

Читает два JSON вида results_vehicle_switch_GPU.json и печатает
сравнительную таблицу A против B.

Это поведенческий и производительный тест, НЕ измерение точности:
для видео нет проверенной разметки, поэтому скрипт нигде не называет
один режим лучше другого. Он только показывает, что изменилось.

Запуск:
    python3 compare_ab_offline.py A.json B.json
"""

import json
import sys
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_variants(d):
    """Собирает статистику по вариантам препроцессинга из сохранённых чтений.

    ВАЖНО: профиль сохраняется только для тех OCR-вызовов, что попали в
    square_readings / normal_readings. Это НЕ все вызовы OCR (см. пункт
    о качестве данных в выводе)."""
    counts = Counter()
    prep_ms = Counter()
    infer_ms = Counter()
    readings_with_profile = 0

    for key in ("square_readings", "normal_readings"):
        for r in d.get(key, []) or []:
            variants = r.get("variants")
            if not variants:
                continue
            readings_with_profile += 1
            for v in variants:
                name = v.get("variant", "?")
                counts[name] += 1
                prep_ms[name] += float(v.get("prep_ms", 0.0) or 0.0)
                infer_ms[name] += float(v.get("infer_ms", 0.0) or 0.0)

    return counts, prep_ms, infer_ms, readings_with_profile


def sum_field(d, field):
    total = 0.0
    found = 0
    for key in ("square_readings", "normal_readings"):
        for r in d.get(key, []) or []:
            if field in r and r[field] is not None:
                total += float(r[field])
                found += 1
    return total, found


def fmt(v, spec="{:.2f}"):
    if v is None:
        return "нет"
    if isinstance(v, (int, float)):
        try:
            return spec.format(v)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def row(label, a, b, spec="{:.2f}", show_delta=True):
    sa, sb = fmt(a, spec), fmt(b, spec)
    delta = ""
    if (show_delta and isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and a):
        d = (b - a) / a * 100.0
        if abs(d) >= 0.5:
            delta = f"{d:+7.1f}%"
        else:
            delta = "      ~0%"
    print(f"  {label:34s} {sa:>14s} {sb:>14s}   {delta}")


def section(title):
    print()
    print(f"  {title}")
    print("  " + "-" * 74)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    a = load(sys.argv[1])
    b = load(sys.argv[2])

    va = collect_variants(a)
    vb = collect_variants(b)
    counts_a, prep_a, infer_a, prof_a = va
    counts_b, prep_b, infer_b, prof_b = vb

    enh_a = counts_a.get("enhanced", 0)
    enh_b = counts_b.get("enhanced", 0)

    # --- Главная проверка целостности: не падали ли вызовы OCR ---
    # В исправном прогоне каждая попытка OCR даёт результат, а fallback-вызовы
    # добавляются сверху. Поэтому завершённых вызовов должно быть НЕ МЕНЬШЕ,
    # чем попыток. Если меньше, часть вызовов упала с ошибкой и прогон
    # недействителен. Проверка по одному enhanced это не ловит: упавший вызов
    # тоже даёт ноль enhanced.
    def integrity(d):
        attempts = d.get("ocr_attempts")
        done = get(d, "timing_profile", "ocr", "count")
        if not isinstance(attempts, int) or not isinstance(done, int):
            return None, attempts, done
        return done >= attempts, attempts, done

    broken = []
    for label, d in (("A", a), ("B", b)):
        ok, att, done = integrity(d)
        if ok is False:
            broken.append((label, att, done))

    if broken:
        print("!" * 80)
        print("СРАВНЕНИЕ НЕВОЗМОЖНО: В ПРОГОНЕ ПАДАЛИ ВЫЗОВЫ OCR")
        print("!" * 80)
        for label, att, done in broken:
            print(f"  {label}: попыток OCR {att}, завершено только {done}, "
                  f"потеряно не меньше {att - done}")
        print()
        print("  Упавший вызов теряет результат целиком, включая хорошие варианты.")
        print("  Такой прогон быстрее не потому, что убран detailEnhance, а потому,")
        print("  что часть работы просто не выполнилась.")
        print()
        print("  Проверьте вывод прогона на строки 'OCR worker error'.")
        sys.exit(1)

    print("=" * 80)
    print("СРАВНЕНИЕ A / B: ОФЛАЙН-ПАЙПЛАЙН")
    print("=" * 80)
    print(f"  A = {sys.argv[1]}")
    print(f"  B = {sys.argv[2]}")
    print()
    va_ = a.get("video", "?")
    vb_ = b.get("video", "?")
    print(f"  видео A: {va_}")
    print(f"  видео B: {vb_}")
    if va_ != vb_:
        print()
        print("  ВНИМАНИЕ: прогоны сделаны на РАЗНЫХ видео, сравнение некорректно.")

    # --- Проверка, что режимы действительно разные ---
    print()
    print(f"  вызовов enhanced в A: {enh_a}")
    print(f"  вызовов enhanced в B: {enh_b}")
    if enh_a > 0 and enh_b > 0:
        print()
        print("  ВНИМАНИЕ: enhanced запускался в обоих прогонах.")
        print("  Похоже, режим no-enhanced не применился. Сравнивать нельзя.")
    elif enh_a == 0 and enh_b == 0:
        print()
        print("  ВНИМАНИЕ: enhanced не запускался ни в одном прогоне.")
        print("  Либо оба в режиме no-enhanced, либо ни одному кадру он не понадобился.")

    print()
    print(f"  {'':34s} {'A':>14s} {'B':>14s}   {'изменение':>9s}")

    # ---------------- Распознавание ----------------
    section("РАСПОЗНАВАНИЕ")
    row("итоговый номер", a.get("confirmed_plate_final") or "-",
        b.get("confirmed_plate_final") or "-", "{}", show_delta=False)
    row("подтверждённый квадратный", a.get("confirmed_square") or "-",
        b.get("confirmed_square") or "-", "{}", show_delta=False)

    sw_a = a.get("switch_events", []) or []
    sw_b = b.get("switch_events", []) or []
    row("переключений автомобиля", len(sw_a), len(sw_b), "{}")

    seq_a = [e.get("to") for e in sw_a]
    seq_b = [e.get("to") for e in sw_b]

    print()
    print("    Последовательность A: " + (" -> ".join(seq_a) or "(пусто)"))
    print("    Последовательность B: " + (" -> ".join(seq_b) or "(пусто)"))
    print()
    if seq_a == seq_b:
        print("    Последовательности СОВПАДАЮТ.")
    else:
        print("    Последовательности РАЗЛИЧАЮТСЯ.")
        only_a = [p for p in seq_a if p not in seq_b]
        only_b = [p for p in seq_b if p not in seq_a]
        if only_a:
            print(f"      только в A: {', '.join(only_a)}")
        if only_b:
            print(f"      только в B: {', '.join(only_b)}")

    # Тайминги переключений
    if seq_a == seq_b and sw_a:
        print()
        print("    Время переключений:")
        print(f"      {'номер':>12s} {'A, сек':>9s} {'B, сек':>9s} {'разница':>10s}")
        for ea, eb in zip(sw_a, sw_b):
            ta, tb = ea.get("time"), eb.get("time")
            if isinstance(ta, (int, float)) and isinstance(tb, (int, float)):
                print(f"      {str(ea.get('to')):>12s} {ta:9.2f} {tb:9.2f} {tb-ta:+10.2f}")

    # ---------------- Производительность ----------------
    section("ВРЕМЯ ВЫПОЛНЕНИЯ")
    row("общее время, сек", get(a, "timing_profile", "total_wall_sec"),
        get(b, "timing_profile", "total_wall_sec"))
    row("старт OCR, сек", get(a, "timing_profile", "ocr_ready_sec"),
        get(b, "timing_profile", "ocr_ready_sec"))

    section("YOLO")
    row("вызовов", get(a, "timing_profile", "yolo", "count"),
        get(b, "timing_profile", "yolo", "count"), "{}")
    row("суммарно, сек", (get(a, "timing_profile", "yolo", "sum_ms") or 0) / 1000.0,
        (get(b, "timing_profile", "yolo", "sum_ms") or 0) / 1000.0)
    row("среднее, мс", get(a, "timing_profile", "yolo", "avg_ms"),
        get(b, "timing_profile", "yolo", "avg_ms"))

    section("OCR (время на стороне воркера: декод + препроцессинг + PaddleX)")
    row("вызовов", get(a, "timing_profile", "ocr", "count"),
        get(b, "timing_profile", "ocr", "count"), "{}")
    row("суммарно, сек", (get(a, "timing_profile", "ocr", "sum_ms") or 0) / 1000.0,
        (get(b, "timing_profile", "ocr", "sum_ms") or 0) / 1000.0)
    row("среднее, мс", get(a, "timing_profile", "ocr", "avg_ms"),
        get(b, "timing_profile", "ocr", "avg_ms"))

    w_a, w_b = get(a, "timing_profile", "writer"), get(b, "timing_profile", "writer")
    if w_a or w_b:
        section("ЗАПИСЬ ВИДЕО (в боевом режиме не нужна)")
        row("вызовов", get(a, "timing_profile", "writer", "count"),
            get(b, "timing_profile", "writer", "count"), "{}")
        row("суммарно, сек", (get(a, "timing_profile", "writer", "sum_ms") or 0) / 1000.0,
            (get(b, "timing_profile", "writer", "sum_ms") or 0) / 1000.0)

    # ---------------- Счётчики пайплайна ----------------
    section("СЧЁТЧИКИ ПАЙПЛАЙНА")
    for label, key in [
        ("кадров всего", "frames_total"),
        ("кадров проверено", "checked_frames"),
        ("детекций", "detections"),
        ("попыток OCR", "ocr_attempts"),
        ("кадров square OCR", "square_ocr_frames"),
    ]:
        row(label, a.get(key), b.get(key), "{}")

    # ---------------- Варианты ----------------
    section("ВАРИАНТЫ ПРЕПРОЦЕССИНГА (только по сохранённым чтениям)")
    all_names = ["original", "upscaled", "gray", "enhanced"]
    extra = sorted(set(counts_a) | set(counts_b) - set(all_names))
    for name in all_names + [n for n in extra if n not in all_names]:
        if counts_a.get(name) or counts_b.get(name):
            row(f"{name}: запусков", counts_a.get(name, 0), counts_b.get(name, 0), "{}")

    print()
    for name in all_names:
        if prep_a.get(name) or prep_b.get(name):
            row(f"{name}: препроцессинг, мс", prep_a.get(name, 0.0), prep_b.get(name, 0.0))
    print()
    for name in all_names:
        if infer_a.get(name) or infer_b.get(name):
            row(f"{name}: инференс PaddleX, мс", infer_a.get(name, 0.0), infer_b.get(name, 0.0))

    section("СУММАРНО ПО СОХРАНЁННЫМ ЧТЕНИЯМ")
    pa, na = sum_field(a, "prep_total_ms")
    pb, nb = sum_field(b, "prep_total_ms")
    ia, _ = sum_field(a, "infer_total_ms")
    ib, _ = sum_field(b, "infer_total_ms")
    row("препроцессинг всего, мс", pa, pb)
    row("инференс PaddleX всего, мс", ia, ib)
    row("чтений с профилем", prof_a, prof_b, "{}")

    # ---------------- Чтения ----------------
    section("ЧТЕНИЯ НОМЕРОВ")
    row("normal_readings", len(a.get("normal_readings", []) or []),
        len(b.get("normal_readings", []) or []), "{}")
    row("square_readings", len(a.get("square_readings", []) or []),
        len(b.get("square_readings", []) or []), "{}")

    # Совпадение чтений по времени
    def readings_map(d, key, field):
        out = {}
        for r in d.get(key, []) or []:
            t = r.get("time")
            if t is not None:
                out[round(float(t), 2)] = r.get(field)
        return out

    for key, field, label in [("normal_readings", "plate", "обычные"),
                              ("square_readings", "top", "квадратные, верх"),
                              ("square_readings", "bottom", "квадратные, низ")]:
        ma, mb = readings_map(a, key, field), readings_map(b, key, field)
        common = set(ma) & set(mb)
        if not common:
            continue
        diff = [(t, ma[t], mb[t]) for t in sorted(common) if ma[t] != mb[t]]
        print()
        print(f"    {label}: общих моментов {len(common)}, расхождений {len(diff)}")
        for t, x, y in diff[:10]:
            print(f"      {t:6.2f} сек   A={str(x or '-'):10s}  B={str(y or '-'):10s}")
        if len(diff) > 10:
            print(f"      ... ещё {len(diff) - 10}")
        only_a = sorted(set(ma) - set(mb))
        only_b = sorted(set(mb) - set(ma))
        if only_a:
            print(f"      моментов только в A: {len(only_a)}  {[round(x,2) for x in only_a[:8]]}")
        if only_b:
            print(f"      моментов только в B: {len(only_b)}  {[round(x,2) for x in only_b[:8]]}")

    # ---------------- Оговорки ----------------
    print()
    print("=" * 80)
    print("ЧТО ЭТИ ЦИФРЫ ЗНАЧАТ И ЧЕГО ОНИ НЕ ЗНАЧАТ")
    print("=" * 80)
    print("  1. Это сравнение поведения и скорости, а не измерение точности.")
    print("     Для видео нет проверенной разметки, поэтому нельзя сказать,")
    print("     какой режим распознаёт правильнее. Совпадение последовательностей")
    print("     означает лишь, что оба режима дали одинаковый результат, а не то,")
    print("     что этот результат верен.")
    print()
    print("  2. Профиль по вариантам собран только по сохранённым чтениям.")
    att_a, att_b = a.get("ocr_attempts"), b.get("ocr_attempts")
    print(f"     В A: попыток OCR {att_a}, чтений с профилем {prof_a}.")
    print(f"     В B: попыток OCR {att_b}, чтений с профилем {prof_b}.")
    print("     Вызовы, не давшие валидного номера, в этот профиль не попали,")
    print("     поэтому суммы по вариантам занижены относительно реальных.")
    print()
    print("  3. Время OCR это время на стороне воркера целиком: декодирование")
    print("     JPEG, препроцессинг на CPU и один или несколько вызовов PaddleX.")
    print("     Это не чистое время инференса на видеокарте.")
    print()
    print("  4. Результат получен на одном видео. Для вывода о том, можно ли")
    print("     убрать enhanced насовсем, нужны прогоны на нескольких видео,")
    print("     включая трудные условия съёмки.")


if __name__ == "__main__":
    main()
