#!/usr/bin/env python3
"""
Проверяет OCR, развёрнутый в кластере и отданный через AI Gateway.

Шлюз говорит на языке OpenAI chat/completions. В примере из документации
показан только текстовый запрос, а нам нужно отправить картинку, и способ
передачи картинки в таком API бывает разный. Поэтому первый шаг не
"встроить", а "выяснить": принимает ли шлюз изображение, в каком виде, что
возвращает и сколько это занимает времени.

Скрипт ничего не встраивает в сервис. Он отправляет несколько пробных
запросов и печатает ответы как есть, чтобы по ним написать рабочий клиент.

КЛЮЧ. Берётся только из переменной окружения LPR_GATEWAY_KEY. В файлы не
пишется, в вывод не печатается, в репозиторий не попадает. Задавать так:

    read -s -p "ключ: " LPR_GATEWAY_KEY; export LPR_GATEWAY_KEY; echo

read -s не показывает ввод на экране и не оставляет ключ в истории команд.

Запуск:

    python3 tools/probe_gateway_ocr.py --crop путь/к/вырезу.jpg

    python3 tools/probe_gateway_ocr.py --crop crop.jpg \\
        --base-url https://<шлюз>/v1 --model deepseek-ocr-2/deepseek-ocr-2
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "deepseek-ocr-2/deepseek-ocr-2"

# Варианты запроса, которые имеет смысл попробовать. Разные шлюзы принимают
# картинку по-разному, и какой из них работает, выясняется только опытом.
PROMPTS = [
    ("free_ocr", "Free OCR."),
    ("plate", "Прочитай номерной знак на изображении. "
              "В ответе только символы номера, без пояснений."),
]


def post(url, key, payload, timeout):
    """(код ответа, тело, миллисекунды). Исключений не бросает."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, body, (time.perf_counter() - t0) * 1000.0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, body, (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000.0


def short(text, limit=600):
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"... (ещё {len(text) - limit} символов)"


def answer_of(body):
    """Текст ответа модели, если тело разобралось как ожидается."""
    try:
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="адрес шлюза до /v1, без завершающего слэша")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--crop", type=Path,
                    help="вырез с номером; без него проверяется только доступ")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    key = os.environ.get("LPR_GATEWAY_KEY", "").strip()
    if not key:
        raise SystemExit(
            "нет ключа. Задай его, не оставляя в истории команд:\n"
            "    read -s -p \"ключ: \" LPR_GATEWAY_KEY; export LPR_GATEWAY_KEY; echo")

    url = args.base_url.rstrip("/") + "/chat/completions"
    print(f"шлюз:  {url}")
    print(f"модель: {args.model}")
    print(f"ключ:   задан, {len(key)} символов (не печатается)\n")

    # 1. Доступ и авторизация. Текстовый запрос из документации: если он не
    #    проходит, с картинками разбираться рано.
    print("=" * 70)
    print("1. ДОСТУП: текстовый запрос")
    print("=" * 70)
    code, body, ms = post(url, key, {
        "model": args.model,
        "messages": [{"role": "user", "content": "Hello"}],
    }, args.timeout)
    print(f"  код {code}, {ms:.0f} мс")
    print(f"  ответ: {short(body)}\n")
    if code == 0:
        print("  Сеть до шлюза не прошла. Проверь, что сервер его видит:")
        print(f"    curl -sS -o /dev/null -w '%{{http_code}}\\n' {args.base_url}")
        return
    if code in (401, 403):
        print("  Ключ не принят. Проверь, что скопирован целиком и не истёк.")
        return
    if code == 404:
        print("  Адрес или имя модели не найдены. Сверь model id со шлюзом.")
        return

    if not args.crop:
        print("Картинку не передавали: --crop не задан. Для проверки чтения")
        print("номера запусти ещё раз с --crop путь/к/вырезу.jpg")
        return

    crop = args.crop.expanduser()
    if not crop.is_file():
        raise SystemExit(f"нет файла: {crop}")
    b64 = base64.b64encode(crop.read_bytes()).decode("ascii")
    suffix = crop.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else "image/jpeg"
    print(f"вырез: {crop.name}, {crop.stat().st_size / 1024:.0f} КБ, {mime}\n")

    # 2. Картинка в виде data URL внутри content. Это обычный для
    #    OpenAI-совместимых шлюзов способ, и начинать надо с него.
    for name, prompt in PROMPTS:
        print("=" * 70)
        print(f"2. КАРТИНКА, запрос «{name}»")
        print("=" * 70)
        code, body, ms = post(url, key, {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        }, args.timeout)
        print(f"  код {code}, {ms:.0f} мс")
        text = answer_of(body)
        if text is not None:
            print(f"  ПРОЧИТАНО: {text!r}")
        else:
            print(f"  ответ целиком: {short(body)}")
        print()

    print("=" * 70)
    print("ЧТО СМОТРЕТЬ В ЭТОМ ВЫВОДЕ")
    print("=" * 70)
    print("  Код 200 и осмысленный текст: шлюз принимает картинки, можно")
    print("  писать клиент и мерить на выборке.")
    print()
    print("  Код 400: формат запроса не тот. В теле ответа обычно написано,")
    print("  чего шлюз ждёт; пришли его целиком, подберём.")
    print()
    print("  Время ответа важно не меньше текста. У нас камера шлёт около")
    print("  трёх кадров в секунду, и бюджет на один вызов примерно 160 мс.")
    print("  Если шлюз отвечает за секунды, живым распознавателем он быть не")
    print("  сможет, и это надо выяснить до встраивания, а не после.")
    print()
    print("  И отдельно: вернулась ли вместе с текстом уверенность. В ответе")
    print("  chat/completions её обычно нет. На ней у нас держатся")
    print("  голосование, ранний выход и смена машины.")


if __name__ == "__main__":
    main()
