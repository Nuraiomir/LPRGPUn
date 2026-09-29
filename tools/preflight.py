#!/usr/bin/env python3
"""
One command that answers "is everything working" before a demo.

It starts the real service on a spare port with the real keys and certificate,
puts every part of the scenario through it, and prints one line per check. What
it covers: the tests, the service starting at all, the scanning page being
served with the aiming guide and the OCRM card in it, the access key being
required, a frame being recognised, a plate found in the test database, a plate
NOT found answering 200 rather than an error, a field visit being created, and
the Postman collection still being valid JSON that carries a key.

The offline pipeline is checked with --videos, which runs the four videos and
compares the metrics against the last recorded result. That takes minutes and
needs the GPU, so it is off by default.

The certificate is self-signed, so the checks skip verification when talking to
localhost. That is the one thing here that a real client must not do, and it is
written out rather than hidden.

Usage:
    .venv_gpu/bin/python tools/preflight.py
    .venv_gpu/bin/python tools/preflight.py --videos
"""

import argparse
import json
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ROOT / "config" / "api_keys.txt"
CERT = ROOT / "config" / "server.pem"
KEY = ROOT / "config" / "server.key"
FRAME = ROOT / "tools" / "sample_frame.jpg"
COLLECTION = ROOT / "tools" / "LPR_API.postman_collection.json"

# Plates that must behave differently, and the reason the pair is here: the one
# the database holds and the one it does not. Confusing the second with a
# recognition failure is the mistake the whole screen is built to avoid.
PLATE_FOUND = "545BDR05"
PLATE_MISSING = "231BED02"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  {'OK  ' if ok else 'СБОЙ'}  {name}" + (f"   {detail}" if detail else ""),
          flush=True)
    return ok


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url, data=None, key=None, method=None, ctx=None):
    """(status, parsed json or raw text). Never raises for an HTTP status."""
    req = urllib.request.Request(url, data=data, method=method)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if data:
        req.add_header("Content-Type",
                       "application/json" if data[:1] == b"{" else "image/jpeg")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
            body, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    except Exception as e:                       # noqa: BLE001 - reported, not raised
        return 0, str(e)
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body.decode("utf-8", "replace")


def run_tests():
    print("\nТесты")
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if not files:
        check("файлы тестов найдены", False, "папка tests пуста")
        return
    for path in files:
        proc = subprocess.run([sys.executable, str(path)], cwd=ROOT,
                              capture_output=True, text=True, timeout=600)
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        check(path.name, proc.returncode == 0,
              "" if proc.returncode == 0 else (tail[-1][:90] if tail else ""))


def check_files():
    print("\nФайлы на месте")
    check("ключ доступа config/api_keys.txt", KEYS.is_file())
    check("сертификат config/server.pem", CERT.is_file())
    check("ключ TLS config/server.key", KEY.is_file())
    check("кадр-образец tools/sample_frame.jpg", FRAME.is_file())

    if COLLECTION.is_file():
        try:
            coll = json.loads(COLLECTION.read_text(encoding="utf-8"))
            names = [i.get("name", "") for i in coll.get("item", [])]
            has_key = (coll.get("auth", {}).get("type") == "bearer")
            check("коллекция Postman читается", True, f"{len(names)} запросов")
            check("коллекция шлёт ключ доступа", has_key,
                  "" if has_key else "без ключа все запросы к сервису дадут 401")
        except json.JSONDecodeError as e:
            check("коллекция Postman читается", False, str(e)[:80])
    else:
        check("коллекция Postman есть", False)


def start_server(port, key_arg):
    args = [sys.executable, str(ROOT / "app" / "lpr_api_server.py"),
            "--port", str(port), "--api-keys-file", str(KEYS)]
    if CERT.is_file() and KEY.is_file():
        args += ["--cert", str(CERT), "--key", str(KEY)]
    proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    banner = []
    start = time.monotonic()
    while time.monotonic() - start < 180:
        line = proc.stdout.readline()
        if not line:
            break
        banner.append(line.rstrip())
        if "READY:" in line:
            return proc, banner
        if proc.poll() is not None:
            break
    return proc, banner


def check_service(base, api_key, ctx):
    """Every HTTP check, against any running instance.

    Separate from starting the server so it can be run against a stand-in
    that has no GPU, which is how these checks are themselves tested.
    """
    status, body = call(f"{base}/", ctx=ctx)
    check("health отвечает", status == 200 and isinstance(body, dict)
          and body.get("ok") is True, f"HTTP {status}")

    status, page = call(f"{base}/demo", ctx=ctx)
    page = page if isinstance(page, str) else ""
    check("страница сканирования отдаётся", status == 200 and "<video" in page,
          f"HTTP {status}")
    check("на странице есть рамка наведения", "AIM_MIN_FILL" in page)
    check("на странице есть карточка ОСРМ", "ocrmCard" in page)
    check("на странице есть текст про отсутствие в базе",
          "не найден в базе" in page)

    frame = FRAME.read_bytes() if FRAME.is_file() else b""
    if frame:
        status, body = call(f"{base}/frame?session_id=preflight", data=frame, ctx=ctx)
        check("кадр без ключа отвергается", status == 401,
              f"HTTP {status}")
        status, body = call(f"{base}/frame?session_id=preflight", data=frame,
                            key=api_key, ctx=ctx)
        check("кадр с ключом принимается", status == 200
              and isinstance(body, dict) and body.get("ok") is True,
              f"HTTP {status}")

    status, body = call(f"{base}/vehicle?plate={PLATE_FOUND}", key=api_key, ctx=ctx)
    found = isinstance(body, dict) and body.get("found") is True
    check(f"{PLATE_FOUND} находится в справочнике", status == 200 and found,
          f"HTTP {status}")
    if found:
        check("в карточке есть заёмщик, кредит и залог",
              all(k in body for k in ("borrower", "credit", "collateral")))

    status, body = call(f"{base}/vehicle?plate={PLATE_MISSING}", key=api_key, ctx=ctx)
    missing_ok = (status == 200 and isinstance(body, dict)
                  and body.get("found") is False)
    check(f"{PLATE_MISSING} отвечает «нет в базе», а не ошибкой",
          missing_ok, f"HTTP {status} (должно быть 200, не 404)")

    payload = json.dumps({"plate": PLATE_FOUND, "note": "preflight"}).encode()
    status, body = call(f"{base}/visit", data=payload, key=api_key,
                        method="POST", ctx=ctx)
    check("выезд создаётся", status == 201 and isinstance(body, dict)
          and str(body.get("visit_id", "")).startswith("TEST-VISIT"),
          f"HTTP {status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", action="store_true",
                    help="also run the four videos and compare the metrics (minutes, GPU)")
    args = ap.parse_args()

    print("=" * 62)
    print("ПРОВЕРКА ПЕРЕД ПОКАЗОМ")
    print("=" * 62)

    check_files()
    run_tests()

    if not KEYS.is_file():
        print("\nБез ключа доступа сервис не проверить. Останавливаюсь.")
        return summarize()
    api_key = next((ln.strip() for ln in KEYS.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")), "")

    print("\nСервис")
    port = free_port()
    proc, banner = start_server(port, api_key)
    started = any("READY:" in ln for ln in banner)
    if not check("сервис запустился", started,
                 "" if started else (banner[-1][:90] if banner else "нет вывода")):
        proc.kill()
        return summarize()

    ocrm_line = next((ln for ln in banner if "Test OCRM" in ln), "")
    check("тестовый справочник загружен", "vehicles" in ocrm_line,
          ocrm_line.split(":", 1)[-1].strip() if ocrm_line else "строки нет")

    scheme = "https" if CERT.is_file() else "http"
    base = f"{scheme}://127.0.0.1:{port}"
    # Self-signed certificate on localhost: a real client must verify.
    ctx = ssl._create_unverified_context() if scheme == "https" else None

    try:
        check_service(base, api_key, ctx)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    if args.videos:
        check_videos()

    return summarize()


def check_videos():
    print("\nОфлайн-прогон по видео")
    names = ["20260923_152319", "20260909_171120", "20260908_150904", "20260908_150800"]
    missing = [n for n in names if not (ROOT / "videos" / f"{n}.mp4").is_file()]
    if missing:
        check("видео на месте", False, f"нет: {', '.join(missing)}")
        return
    for name in names:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "app" / "lpr_v19_universal.py"),
             f"videos/{name}.mp4"],
            cwd=ROOT, capture_output=True, text=True, timeout=1800)
        check(f"прогон {name}", proc.returncode == 0,
              "" if proc.returncode == 0 else proc.stderr.strip().splitlines()[-1][:80])

    proc = subprocess.run([sys.executable, str(ROOT / "bench" / "pipeline_metrics.py")],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    line = next((ln for ln in proc.stdout.splitlines() if "ВСЕГО" in ln), "")
    nums = re.findall(r"\d+\.\d+|\d+", line)
    # машин, TP, FP, FN, точность, полнота, F1
    ok = len(nums) >= 7 and nums[1] == "25" and nums[2] == "0"
    check("метрики те же: 25 из 28, чужих 0", ok, line.strip() or "строки ВСЕГО нет")


def summarize():
    bad = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 62)
    if bad:
        print(f"НЕ ПРОШЛО: {len(bad)} из {len(results)}")
        for name in bad:
            print(f"   {name}")
        print("\nПоказывать в таком виде нельзя, пока это не починено.")
    else:
        print(f"ВСЁ ПРОШЛО: {len(results)} проверок")
        print("\nСервис поднимается, страница отдаётся, ключ требуется,")
        print("кадр распознаётся, обе ветки поиска в базе работают, выезд создаётся.")
    print("=" * 62)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
