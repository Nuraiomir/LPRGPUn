#!/usr/bin/env python3
"""
Simulated-camera client for the LPR API server.

This does NOT replace physical camera testing (see docs/architecture.md,
"NOT YET VERIFIED"). It is a transport/integration test: it turns a
recorded video into a sequence of HTTP POSTs exactly the way a future real
camera client would, so the API layer (server, sessions, JSON contract) can
be exercised end-to-end before physical camera access exists.

Usage:
    python client/camera_client.py \\
        --video 20260909_171120.mp4 \\
        --server http://127.0.0.1:8765 \\
        --fps 10 \\
        --jpeg-quality 90 \\
        --session-id cam1

If the server currently running is the OLD lpr_camera_server.py prototype
rather than the new app/lpr_api_server.py, the JSON shape is still
compatible for the fields both share (ok/plate/confirmed/bbox/confidence/
changed) -- but its recognition behavior is the OLDER, documented-as-buggy
logic (see docs/LPR_API_CONTRACT_REVIEW.md), not v19's. This script does not
know or care which server it's talking to; it only speaks the shared HTTP
contract. Check the server's own startup banner to know which one you're
running against.
"""

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Path to a video file to simulate a camera feed")
    p.add_argument("--server", default="http://127.0.0.1:8765", help="Base URL of the LPR API server")
    p.add_argument("--fps", type=float, default=10.0, help="Simulated camera frame rate (frames actually sent per second)")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG encode quality (1-100)")
    p.add_argument("--session-id", default=None, help="session_id для всего видео (один и тот же на все кадры)")
    p.add_argument("--max-frames", type=int, default=None, help="Остановиться после N кадров (по умолчанию: всё видео)")
    p.add_argument("--timeout", type=float, default=30.0, help="Таймаут одного запроса, секунд")
    p.add_argument("--save-responses", default=None, help="Путь для сохранения всех ответов сервера в JSON")
    p.add_argument("--profile", action="store_true", help="Запросить у сервера разбивку по стадиям (?profile=1)")
    p.add_argument("--video-time-voting", action="store_true",
                   help="Передавать серверу время кадра в видео как метку для голосования. "
                        "Нужно для честного A/B: иначе более быстрый режим получает "
                        "преимущество просто потому, что кадры укладываются плотнее в окно голосования.")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video {args.video!r}", file=sys.stderr)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(video_fps / args.fps))

    base_url = args.server.rstrip("/") + "/frame"
    base_params = []
    if args.session_id:
        base_params.append(f"session_id={args.session_id}")
    if args.profile:
        base_params.append("profile=1")

    def build_url(video_time):
        params = list(base_params)
        if args.video_time_voting:
            params.append(f"t={video_time:.4f}")
        return base_url + ("?" + "&".join(params) if params else "")

    url = build_url(0.0)

    print(f"Simulating a {args.fps:.1f} fps camera from {args.video!r} "
          f"(source video is {video_fps:.1f} fps -> sending every {step}th frame)")
    print(f"POSTing to {url}")
    print()

    frame_idx = 0
    sent = 0
    failed = 0
    timeouts = 0
    http_errors = 0
    latencies = []
    server_times = []
    profiles = []
    responses = []
    switch_log = []
    last_plate = ""
    t_start = time.monotonic()
    next_send_at = t_start
    send_interval = 1.0 / args.fps if args.fps > 0 else 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        if args.max_frames is not None and sent >= args.max_frames:
            break

        # Pace ACTUAL wall-clock sending to --fps, not just video-frame
        # skipping. This matters: the server's temporal voting windows
        # (WINDOW_SEC, SWITCH_WINDOW_SEC) are keyed to real wall-clock
        # seconds between requests (time.monotonic() on the server side),
        # not to any timestamp this client sends. Sending frames as fast as
        # possible would compress everything into a fraction of a second of
        # server-side time and would NOT reproduce the timing the offline
        # video evaluator (frame_idx/video_fps) used to validate v19.
        now = time.monotonic()
        if now < next_send_at:
            time.sleep(next_send_at - now)
        next_send_at = max(next_send_at + send_interval, time.monotonic())

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            continue

        req_t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                build_url(frame_idx / video_fps), data=buf.tobytes(), method="POST",
                headers={"Content-Type": "image/jpeg"},
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                status = r.status
                raw = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            status = e.code
            raw = e.read().decode("utf-8", errors="replace")
            http_errors += 1
        except socket.timeout:
            print(f"[t={frame_idx/video_fps:6.2f}s] ТАЙМАУТ ({args.timeout} сек)")
            timeouts += 1
            failed += 1
            continue
        except Exception as e:
            print(f"[t={frame_idx/video_fps:6.2f}s] ЗАПРОС НЕ ПРОШЁЛ: {e!r}")
            failed += 1
            continue
        latency_ms = (time.monotonic() - req_t0) * 1000.0
        latencies.append(latency_ms)
        sent += 1

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[t={frame_idx/video_fps:6.2f}s] НЕВЕРНЫЙ ОТВЕТ (код {status}): {raw[:200]!r}")
            failed += 1
            continue

        if not result.get("ok", False):
            print(f"[t={frame_idx/video_fps:6.2f}s] ОШИБКА СЕРВЕРА: {result.get('error')}")
            failed += 1
            continue

        if args.save_responses:
            responses.append({
                "video_time": round(frame_idx / video_fps, 3),
                "latency_ms": round(latency_ms, 1),
                "response": result,
            })

        srv_ms = result.get("processing_time_ms")
        if isinstance(srv_ms, (int, float)):
            server_times.append(float(srv_ms))

        prof = result.get("profile")
        if prof:
            profiles.append({
                "video_time": round(frame_idx / video_fps, 2),
                "total_ms": prof.get("total_ms", 0.0),
                "server_ms": srv_ms,
                "client_ms": round(latency_ms, 1),
                "yolo_ms": prof.get("yolo_ms", 0.0),
                "ocr_total_ms": prof.get("ocr_total_ms", 0.0),
                "ocr_square_ms": prof.get("ocr_square_ms", 0.0),
                "ocr_normal_ms": prof.get("ocr_normal_ms", 0.0),
                "ocr_fallback_ms": prof.get("ocr_fallback_ms", 0.0),
                "ocr_calls": prof.get("ocr_calls", 0),
                "jpeg_decode_ms": prof.get("jpeg_decode_ms", 0.0),
                "body_read_ms": prof.get("body_read_ms", 0.0),
                "voting_ms": prof.get("voting_ms", 0.0),
                "plate_type": prof.get("plate_type"),
                "had_detection": prof.get("had_detection"),
                "worker_detail": prof.get("worker_detail", []),
            })

        plate = result.get("plate", "")
        changed = result.get("changed", False)
        marker = ""
        if changed:
            marker = "   <<< СМЕНА НОМЕРА"
            switch_log.append({
                "video_time": round(frame_idx / video_fps, 2),
                "from": last_plate,
                "to": plate,
                "plate_type": result.get("plate_type"),
            })
        last_plate = plate

        ocr_conf = result.get("ocr_confidence")
        ocr_str = f"{ocr_conf:.2f}" if isinstance(ocr_conf, (int, float)) else "-"
        srv_str = f"{srv_ms:5.0f}" if isinstance(srv_ms, (int, float)) else "    -"

        print(
            f"[t={frame_idx/video_fps:6.2f}s] номер={plate or '-':10s} "
            f"det={result.get('confidence', 0.0):.2f} "
            f"ocr={ocr_str:>4s} "
            f"тип={str(result.get('plate_type') or '-'):6s} "
            f"сервер={srv_str}мс "
            f"всего={latency_ms:6.1f}мс{marker}"
        )

    elapsed = time.monotonic() - t_start
    cap.release()

    print()
    print("=" * 64)
    print("РЕЗУЛЬТАТЫ HTTP-ТЕСТА")
    print("=" * 64)
    print(f"Видео:                    {args.video}")
    print(f"session_id:               {args.session_id or '(не передавался)'}")
    print(f"Отправлено кадров:        {sent}")
    print(f"Успешных ответов:         {sent - failed}")
    print(f"Ошибок всего:             {failed}")
    print(f"  в т.ч. HTTP-ошибок:     {http_errors}")
    print(f"  в т.ч. таймаутов:       {timeouts}")
    print(f"Общее время:              {elapsed:.1f} сек")
    print(f"Запрошенная частота:      {args.fps:.1f} кадр/сек")
    if elapsed > 0:
        print(f"Фактическая частота:      {sent/elapsed:.1f} кадр/сек")

    def stats(values, title, unit="мс"):
        if not values:
            return
        s = sorted(values)
        def pct(p):
            return s[min(len(s) - 1, int(len(s) * p / 100))]
        avg = sum(s) / len(s)
        print()
        print(title)
        print(f"  минимум:                {s[0]:.0f} {unit}")
        print(f"  среднее:                {avg:.0f} {unit}")
        print(f"  медиана:                {pct(50):.0f} {unit}")
        print(f"  95-й процентиль:        {pct(95):.0f} {unit}")
        print(f"  максимум:               {s[-1]:.0f} {unit}")
        return avg

    avg_total = stats(latencies, "Полная задержка клиента (сеть + обработка):")
    avg_server = stats(server_times, "Обработка на сервере (YOLO + OCR + голосование):")

    if avg_total and avg_server:
        print()
        print(f"Накладные расходы HTTP:   {avg_total - avg_server:.0f} мс "
              f"({100*(avg_total-avg_server)/avg_total:.0f}% от общей задержки)")

    if avg_total:
        print()
        print("=" * 64)
        print(f"РЕАЛЬНАЯ ПРОПУСКНАЯ СПОСОБНОСТЬ: {1000.0/avg_total:.1f} кадр/сек")
        print("=" * 64)
        print("Это последовательный режим: кадр отправлен, ждём ответ,")
        print("только потом следующий кадр. Без параллельных запросов.")

    if switch_log:
        print()
        print("Распознанные номера по порядку:")
        for ev in switch_log:
            frm = ev["from"] or "-"
            print(f"  {ev['video_time']:6.2f} сек   {frm:10s} -> {ev['to']:10s} ({ev['plate_type']})")

    if profiles:
        print()
        print("=" * 64)
        print("РАЗБОР ЗАДЕРЖЕК ПО СТАДИЯМ")
        print("=" * 64)

        for thr in (100, 200, 300, 500):
            n = sum(1 for p in profiles if (p["server_ms"] or 0) > thr)
            print(f"Запросов дольше {thr:4d} мс:   {n:4d}  ({100.0*n/len(profiles):5.1f}%)")

        groups = {}
        for p in profiles:
            if not p["had_detection"]:
                key = "без детекции (только YOLO)"
            elif p["ocr_calls"] == 0:
                key = "детекция есть, OCR пропущен"
            elif p["ocr_calls"] == 1 and p["plate_type"] == "normal":
                key = "обычный номер (1 вызов OCR)"
            elif p["ocr_calls"] == 1 and p["plate_type"] == "square":
                key = "квадратный номер (1 вызов OCR)"
            else:
                key = f"квадратный + fallback ({p['ocr_calls']} вызова OCR)"
            groups.setdefault(key, []).append(p["server_ms"] or 0.0)

        print()
        print("Время обработки по типу запроса:")
        print(f"  {'тип запроса':38s} {'кол-во':>7s} {'среднее':>9s} {'медиана':>9s} {'макс':>8s}")
        for key in sorted(groups, key=lambda k: -sum(groups[k]) / max(1, len(groups[k]))):
            vals = sorted(groups[key])
            avg = sum(vals) / len(vals)
            med = vals[len(vals) // 2]
            print(f"  {key:38s} {len(vals):7d} {avg:8.0f}мс {med:8.0f}мс {vals[-1]:7.0f}мс")

        tot_yolo = sum(p["yolo_ms"] for p in profiles)
        tot_ocr = sum(p["ocr_total_ms"] for p in profiles)
        tot_sq = sum(p["ocr_square_ms"] for p in profiles)
        tot_nm = sum(p["ocr_normal_ms"] for p in profiles)
        tot_fb = sum(p["ocr_fallback_ms"] for p in profiles)
        tot_dec = sum(p["jpeg_decode_ms"] for p in profiles)
        tot_vote = sum(p["voting_ms"] for p in profiles)
        tot_all = tot_yolo + tot_ocr + tot_dec + tot_vote

        print()
        print("Куда ушло всё время сервера:")
        if tot_all > 0:
            for name, val in [
                ("YOLO (детекция)", tot_yolo),
                ("OCR всего", tot_ocr),
                ("   в т.ч. квадратные номера", tot_sq),
                ("   в т.ч. обычные номера", tot_nm),
                ("   в т.ч. повторный OCR (fallback)", tot_fb),
                ("Декодирование JPEG", tot_dec),
                ("Голосование и состояние", tot_vote),
            ]:
                print(f"  {name:36s} {val/1000.0:7.1f} сек  ({100.0*val/tot_all:5.1f}%)")

        print()
        print("20 самых медленных запросов:")
        print(f"  {'время видео':>11s} {'сервер':>8s} {'YOLO':>7s} {'OCR':>8s} "
              f"{'вызовов':>8s} {'тип':>8s}")
        for p in sorted(profiles, key=lambda x: -(x["server_ms"] or 0))[:20]:
            print(f"  {p['video_time']:10.2f}с {p['server_ms'] or 0:7.0f}мс "
                  f"{p['yolo_ms']:6.0f}мс {p['ocr_total_ms']:7.0f}мс "
                  f"{p['ocr_calls']:8d} {str(p['plate_type'] or '-'):>8s}")

    print()
    print(f"Итоговый номер: {last_plate or '(не подтверждён)'}")

    if args.save_responses:
        out = {
            "video": args.video,
            "session_id": args.session_id,
            "requested_fps": args.fps,
            "jpeg_quality": args.jpeg_quality,
            "frames_sent": sent,
            "failed": failed,
            "http_errors": http_errors,
            "timeouts": timeouts,
            "wall_time_sec": round(elapsed, 2),
            "effective_fps": round(sent / elapsed, 2) if elapsed else 0,
            "latency_ms": {
                "all": [round(x, 1) for x in latencies],
                "avg": round(avg_total, 1) if avg_total else None,
            },
            "server_processing_ms": {
                "all": [round(x, 1) for x in server_times],
                "avg": round(avg_server, 1) if avg_server else None,
            },
            "switch_events": switch_log,
            "final_plate": last_plate,
            "profiles": profiles,
            "responses": responses,
        }
        with open(args.save_responses, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены: {args.save_responses}")

    print("=" * 64)


if __name__ == "__main__":
    main()
