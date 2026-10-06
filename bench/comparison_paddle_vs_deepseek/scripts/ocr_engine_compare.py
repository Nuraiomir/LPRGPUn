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


# Распознаватель, развёрнутый в кластере банка и отданный через AI Gateway.
# Говорит на языке OpenAI chat/completions: картинка уходит внутри сообщения
# как data URL, ответ приходит обычным текстом.
#
# Две вещи, которые отличают его от локальных движков и которые надо помнить:
#
#   уверенности нет. chat/completions её не возвращает. На ней у нас держатся
#   голосование, ранний выход 0.92 и смена машины 0.95, поэтому в колонке
#   "conf" здесь всегда 0, и сравнивать по ней нельзя
#
#   время включает сеть. Это не время модели, а время запроса: сеть, очередь
#   на шлюзе, инференс. Для решения "годится ли как живой распознаватель"
#   важно именно оно, потому что столько и будет ждать камера
#
# Запрос "Free OCR." выбран не наугад: на пробе он прочитал номер, а
# инструкция на русском вернула пустую строку. Но проба это один вырез, и на
# сотне картина может быть другой, поэтому формулировку можно выбрать флагом.
GATEWAY_PROMPTS = {
    "free": "Free OCR.",
    "plate": "Read the license plate. Answer with the plate characters only.",
    "plate_ru": ("Прочитай номерной знак на изображении. "
                 "В ответе только символы номера, без пояснений."),
}

# ЧЕСТНОСТЬ СРАВНЕНИЯ. Первый прогон дал шлюзу то же, что PaddleOCR: плотный
# вырез номера, примерно 300 на 60 точек. Для PaddleOCR это ровно его вход, он
# на таких строках и обучался. Для модели документного OCR это вход далеко за
# пределами того, что она видела: такие модели обучают на страницах, и
# крошечная картинка из восьми символов для них нетипична.
#
# То есть низкие 44 из 100 могут быть не свойством модели, а свойством того,
# что мы ей дали. Проверяется это двумя дешёвыми способами, и оба здесь
# флагами: увеличить вырез до привычного модели размера и поменять
# формулировку запроса. Пока это не проверено, вывод "шлюз читает хуже"
# держится на одном варианте входа, а не на модели.
def _prepare_for_gateway(path, upscale_to):
    """(байты картинки, mime). При upscale_to вырез увеличивается по ширине."""
    raw = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    if not upscale_to:
        return raw, mime

    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return raw, mime
    h, w = image.shape[:2]
    if w >= upscale_to:
        return raw, mime
    scale = upscale_to / w
    # INTER_CUBIC, а не NEAREST: увеличение должно выглядеть как фотография
    # побольше, а не как пиксельная лестница.
    bigger = cv2.resize(image, (upscale_to, max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", bigger, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return raw, mime
    return buf.tobytes(), "image/jpeg"


def run_gateway(samples, prompt_key="free", upscale_to=0):
    key = os.environ.get("LPR_GATEWAY_KEY", "").strip()
    base = os.environ.get("LPR_GATEWAY_URL", "").strip()
    model = os.environ.get("LPR_GATEWAY_MODEL", "deepseek-ocr-2/deepseek-ocr-2")
    if not key or not base:
        print("  нужны LPR_GATEWAY_KEY и LPR_GATEWAY_URL")
        print("  ключ задавать так, чтобы он не попал в историю команд:")
        print('    read -s -p "ключ: " LPR_GATEWAY_KEY; export LPR_GATEWAY_KEY; echo')
        return None

    import base64
    import urllib.error
    import urllib.request

    prompt = GATEWAY_PROMPTS[prompt_key]
    url = base.rstrip("/") + "/chat/completions"
    print(f"  {model} через {url}")
    print(f"  запрос: {prompt_key} ({prompt[:50]!r})")
    print(f"  вход: {'вырез как есть' if not upscale_to else f'увеличен до {upscale_to} точек по ширине'}")
    rows, times, failed = [], [], 0
    for i, (path, label) in enumerate(samples, 1):
        raw_bytes, mime = _prepare_for_gateway(path, upscale_to)
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(raw_bytes).decode("ascii")}},
            ]}],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/json")

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                answer = json.loads(resp.read().decode("utf-8", "replace"))
            text = answer["choices"][0]["message"]["content"]
        except Exception as e:
            # Один упавший запрос не должен обрушить весь прогон: сеть моргает,
            # а повторять час измерений из-за одного таймаута незачем. Падения
            # считаются и печатаются в конце.
            text = ""
            failed += 1
            if failed <= 3:
                print(f"    запрос {i} не прошёл: {type(e).__name__}")
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)

        row = grade(text, label)
        row.update({"file": path.name, "label": label, "conf": 0.0, "ms": round(ms, 1)})
        rows.append(row)
        if i % 25 == 0:
            print(f"    шлюз {i}/{len(samples)}", flush=True)
    if failed:
        print(f"  запросов не прошло: {failed} из {len(samples)}")
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


def report(name, model, rows, times):
    """name  ключ запуска (paddle, tesseract, gateway)
    model  что на самом деле читало номер, это и попадает в таблицу.

    Две разные вещи, и раньше в таблице стояла первая. «gateway» это способ
    доступа, а не распознаватель: через шлюз может стоять любая модель, и
    строка «gateway 44/100» через месяц ничего не говорит о том, что мерили.
    В колонке теперь имя модели, а ключ остаётся в JSON для сверки запуска.
    """
    n = len(rows) or 1
    correct = sum(r["correct"] for r in rows)
    fmt = sum(r["format_ok"] for r in rows)
    exact = sum(r["exact"] for r in rows)
    wrong_plate = sum(1 for r in rows if r["format_ok"] and not r["correct"])
    return {
        "engine": name, "model": model, "samples": len(rows),
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
    ap.add_argument("--gateway-prompt", default="free",
                    choices=sorted(GATEWAY_PROMPTS),
                    help="формулировка запроса к шлюзу")
    ap.add_argument("--gateway-upscale", type=int, default=0,
                    help="увеличить вырез до N точек по ширине перед отправкой "
                         "на шлюз. 0 = как есть. Документный OCR обучался на "
                         "страницах, и плотный вырез номера для него нетипичен")
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
    gw_model = os.environ.get("LPR_GATEWAY_MODEL",
                              "deepseek-ocr-2/deepseek-ocr-2").split("/")[-1]
    # Ключ запуска → что этим ключом на самом деле запускается.
    gw_label = gw_model
    if args.gateway_prompt != "free":
        gw_label += f"/{args.gateway_prompt}"
    if args.gateway_upscale:
        gw_label += f"/x{args.gateway_upscale}"
    model_of = {
        "paddle": paddle_model,
        "tesseract": f"tesseract/{tess_tag}",
        "gateway": gw_label,
    }
    tags = []
    if "paddle" in engines:
        tags.append(paddle_model)
    if "tesseract" in engines:
        tags.append(f"tess-{tess_tag}")
    if "gateway" in engines:
        tag = "gw-" + gw_model
        if args.gateway_prompt != "free":
            tag += f"-{args.gateway_prompt}"
        if args.gateway_upscale:
            tag += f"-up{args.gateway_upscale}"
        tags.append(tag)

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
    if "gateway" in engines:
        print(f"модель через шлюз: {gw_model}")
        print(f"шлюз: {os.environ.get('LPR_GATEWAY_URL', '(не задан)')}")
    print(f"результат будет записан в: {args.out}\n")

    results, details = [], {}
    for engine in engines:
        print(f"{engine}:")
        runner = {"paddle": run_paddle,
                  "tesseract": run_tesseract,
                  "gateway": run_gateway}.get(engine)
        if runner is None:
            print(f"  неизвестный движок: {engine}")
            continue
        if engine == "gateway":
            got = runner(samples, prompt_key=args.gateway_prompt,
                         upscale_to=args.gateway_upscale)
        else:
            got = runner(samples)
        if not got:
            continue
        rows, times = got
        results.append(report(engine, model_of.get(engine, engine), rows, times))
        details[engine] = rows
        print()

    if not results:
        raise SystemExit("ни один движок не отработал")

    print("=" * 88)
    print("СРАВНЕНИЕ НА ОДНОЙ И ТОЙ ЖЕ ВЫБОРКЕ")
    print("=" * 88)
    head = (f"{'модель':<26}{'правильно':>12}{'доля':>9}"
            f"{'формат':>9}{'чужой':>8}{'мс':>8}")
    print(head)
    print("-" * 88)
    for r in results:
        print(f"{r['model']:<26}{r['correct']:>7}/{r['samples']:<4}"
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
              f"{a['model'] if d > 0 else b['model']}.")
        if b["median_ms"] and a["median_ms"]:
            slower = max(a["median_ms"], b["median_ms"]) / min(a["median_ms"], b["median_ms"])
            slow = a["model"] if a["median_ms"] > b["median_ms"] else b["model"]
            print(f"По скорости {slow} медленнее в {slower:.0f} раз.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dataset": str(dataset), "split": args.split, "samples": len(samples),
         "summary": results, "rows": details},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nподробности по каждому вырезу: {args.out}")


if __name__ == "__main__":
    main()
