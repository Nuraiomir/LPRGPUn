#!/usr/bin/env python3
"""
Hard-conditions benchmark: does OCR mode "full" (with cv2.detailEnhance)
recognize plates that "no-enhanced" misses, and what does it cost in speed?

For each OCR mode the script starts the HTTP server, sends every video once
per degradation through client/camera_client.py at camera-like 10 fps, and
saves each run. Then it scores the runs against bench/labels.json, which lists
the plates actually present in each video.

Scores per run:
    found    plates from the labels that were confirmed
    missed   plates from the labels that were never confirmed
    wrong    confirmed plates that are not in the video. Each one would make
             OCRM look up a different car, so this matters more than a miss.
    server   average and 95th-percentile processing time per frame on the
             server. Measured by the server itself, so the time the client
             spends degrading frames does not distort it.

Run everything (about 40 minutes on the RTX 4090):
    .venv_gpu/bin/python bench/hard_conditions.py

Subsets, and scoring existing runs without running again:
    .venv_gpu/bin/python bench/hard_conditions.py --videos 20260909_171120 --variants none blur:2 phone:2
    .venv_gpu/bin/python bench/hard_conditions.py --score-only

Finished runs are skipped when the script is started again, so an interrupted
benchmark can be resumed. Use --force to run everything again.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))
from degradations import all_variants  # noqa: E402

MODES = ("full", "no-enhanced")
OUT = ROOT / "runs" / "hard_conditions"


def load_labels(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def run_path(mode, video, variant):
    return OUT / mode / f"{video}__{variant.replace(':', '-')}.json"


def start_server(mode, port):
    log = OUT / f"server_{mode}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "app" / "lpr_api_server.py"),
         "--ocr-variants", mode, "--port", str(port)],
        cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            handle.close()
            sys.exit(f"server ({mode}) exited with code {proc.returncode}, see {log}")
        if "READY" in log.read_text(encoding="utf-8", errors="replace"):
            return proc, handle
        time.sleep(1)
    proc.terminate()
    sys.exit(f"server ({mode}) did not become ready in 300 s, see {log}")


def stop_server(proc, handle):
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    handle.close()


def run_all(labels, videos, variants, modes, port, fps, force, profile=False):
    todo = {m: [(v, d) for v in videos for d in variants
                if force or not run_path(m, v, d).exists()] for m in modes}
    total = sum(len(t) for t in todo.values())
    if total == 0:
        print("All runs already exist. Use --force to run them again.")
        return
    done = 0
    t_start = time.monotonic()
    for mode in modes:
        if not todo[mode]:
            continue
        print(f"\n=== starting server, OCR mode {mode} ===", flush=True)
        proc, handle = start_server(mode, port)
        try:
            for video, variant in todo[mode]:
                out = run_path(mode, video, variant)
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(ROOT / "client" / "camera_client.py"),
                       "--video", str(ROOT / "videos" / f"{video}.mp4"),
                       "--server", f"http://127.0.0.1:{port}",
                       "--session-id", f"{mode}.{video}.{variant.replace(':', '-')}",
                       "--fps", str(fps), "--video-time-voting",
                       "--save-responses", str(out)]
                if variant != "none":
                    cmd += ["--degrade", variant]
                if profile:
                    cmd += ["--profile"]
                t0 = time.monotonic()
                res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                done += 1
                status = "ok" if res.returncode == 0 and out.exists() else f"FAILED ({res.returncode})"
                if status != "ok":
                    (out.with_suffix(".log")).write_text(res.stdout + res.stderr, encoding="utf-8")
                    out.unlink(missing_ok=True)
                elapsed = time.monotonic() - t_start
                eta = elapsed / done * (total - done)
                print(f"[{done:3d}/{total}] {mode:11s} {video} {variant:10s} "
                      f"{time.monotonic() - t0:5.1f} s  {status}   осталось ~{eta / 60:.0f} мин", flush=True)
        finally:
            stop_server(proc, handle)


def _p95(values):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * 0.95))]


def score_run(path, expected):
    d = json.loads(path.read_text(encoding="utf-8"))
    confirmed = [e["to"] for e in d.get("switch_events", []) if e.get("to")]
    exp = [p["plate"] for p in expected]
    square = {p["plate"] for p in expected if p.get("square")}
    found = [p for p in exp if p in confirmed]
    return {
        "found": len(found), "total": len(exp),
        "square_found": len([p for p in found if p in square]), "square_total": len(square),
        "missed": [p for p in exp if p not in confirmed],
        "wrong": sorted({p for p in confirmed if p not in exp}),
        "srv_avg": (d.get("server_processing_ms") or {}).get("avg") or 0.0,
        "srv_p95": _p95((d.get("server_processing_ms") or {}).get("all") or []),
        "errors": d.get("failed", 0),
    }


def score(labels, videos, variants, modes):
    table = {}
    for mode in modes:
        for video in videos:
            for variant in variants:
                p = run_path(mode, video, variant)
                if p.exists():
                    table[(mode, video, variant)] = score_run(p, labels[video])
    if not table:
        sys.exit("No runs found. Run the benchmark first.")

    def cell(r):
        if r is None:
            return f"{'нет прогона':>22s}"
        wrong = f" ЧУЖИХ {len(r['wrong'])}" if r["wrong"] else ""
        err = f" ОШИБОК {r['errors']}" if r["errors"] else ""
        return f"{r['found']}/{r['total']} {r['srv_avg']:4.0f}/{r['srv_p95']:4.0f}мс{wrong}{err}".rjust(22)

    for video in videos:
        print(f"\n== {video}")
        print(f"   {'условия':10s}  {'full: найдено ср/p95':>22s}  {'no-enhanced':>22s}   разница")
        for variant in variants:
            a = table.get(("full", video, variant))
            b = table.get(("no-enhanced", video, variant))
            note = ""
            if a and b:
                if a["found"] > b["found"]:
                    note = "full нашёл больше: " + ", ".join(sorted(set(b["missed"]) - set(a["missed"])))
                elif b["found"] > a["found"]:
                    note = "no-enhanced нашёл больше: " + ", ".join(sorted(set(a["missed"]) - set(b["missed"])))
                if a["wrong"] != b["wrong"]:
                    note += f"  чужие: full {a['wrong']}, no-enh {b['wrong']}"
            print(f"   {variant:10s}  {cell(a)}  {cell(b)}   {note}")

    print("\n== Итого по условиям (все видео вместе)")
    print(f"   {'условия':10s}  {'full: найдено / квадр. / чужих / сервер':>40s}   {'no-enhanced':>40s}")
    for variant in variants:
        row = []
        for mode in modes:
            rs = [table[(mode, v, variant)] for v in videos if (mode, v, variant) in table]
            if not rs:
                row.append(f"{'нет прогонов':>40s}")
                continue
            f = sum(r["found"] for r in rs); t = sum(r["total"] for r in rs)
            sf = sum(r["square_found"] for r in rs); st = sum(r["square_total"] for r in rs)
            w = sum(len(r["wrong"]) for r in rs)
            ms = sum(r["srv_avg"] for r in rs) / len(rs)
            row.append(f"{f:2d}/{t:<2d}  {sf}/{st} кв.  {w} чужих  {ms:4.0f} мс/кадр".rjust(40))
        print(f"   {variant:10s}  {row[0]}   {row[1] if len(row) > 1 else ''}")

    print("\nНайдено = номер из видео был подтверждён. Чужие = подтверждён номер,")
    print("которого на видео нет: ОСРМ стала бы искать другую машину.")
    print("Сервер = среднее и 95-й процентиль времени обработки кадра на сервере.")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(ROOT / "bench" / "labels.json"))
    ap.add_argument("--videos", nargs="+", help="video names without .mp4 (default: all in labels)")
    ap.add_argument("--variants", nargs="+", help="'none' and/or NAME:LEVEL (default: none + all)")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write runs (use a new directory to keep earlier results)")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--profile", action="store_true",
                    help="also save the per-stage timing and the row votes behind each decision")
    ap.add_argument("--force", action="store_true", help="run again even if results exist")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    if args.out is not None:
        OUT = args.out
    labels = load_labels(args.labels)
    videos = args.videos or list(labels)
    unknown = [v for v in videos if v not in labels]
    if unknown:
        sys.exit(f"no labels for: {', '.join(unknown)} (add them to {args.labels})")
    variants = args.variants or (["none"] + all_variants())
    bad = [v for v in variants if v != "none" and v not in all_variants()]
    if bad:
        sys.exit(f"unknown variants: {bad}; choose from none, {', '.join(all_variants())}")

    if not args.score_only:
        run_all(labels, videos, variants, args.modes, args.port, args.fps, args.force,
                profile=args.profile)
    score(labels, videos, variants, args.modes)


if __name__ == "__main__":
    main()
