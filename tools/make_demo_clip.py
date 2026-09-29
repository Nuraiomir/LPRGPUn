#!/usr/bin/env python3
"""
Prepares a video for showing, without ffmpeg.

Two things are needed before a demo and both normally reach for ffmpeg, which
is not installed on the GPU box and needs an administrator:

    a clip to play          the run's own output is written with the mp4v codec,
                            which browsers do not play, and a minute of 720x1280
                            at 60 fps is hundreds of megabytes
    a file to feed Chromium instead of a camera, so the whole page can be
                            demonstrated when the room's light defeats the lens

OpenCV carries its own encoders, so both are made here. The clip is re-encoded,
optionally shortened and narrowed. H.264 is tried first, then VP9 in WebM,
which browsers play just as readily and which this box does have; the codec
actually used is reported rather than assumed, and the extension follows it, so
asking for --out clip.mp4 can correctly produce clip.webm. The camera file is
Y4M, which is uncompressed and therefore large, so it is cut short by default.

Usage:
    .venv_gpu/bin/python tools/make_demo_clip.py \\
        runs/real_video_v18_20260923_152319/result_vehicle_switch_GPU.mp4 \\
        --out ~/demo_parking.mp4 --seconds 60 --width 720

    .venv_gpu/bin/python tools/make_demo_clip.py videos/20260923_152319.mp4 \\
        --y4m /tmp/fakecam.y4m --seconds 12 --width 720
"""

import argparse
import os
import sys
from pathlib import Path

import cv2


def open_source(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"не открывается: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, w, h, total


def target_size(w, h, width):
    """Keeps the proportions. Even numbers: odd ones break some encoders."""
    if not width or width >= w:
        return w // 2 * 2, h // 2 * 2
    k = width / w
    return max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2)


# Tried in order. The first three play in a browser; mp4v is the last resort
# and does not. Which of them a given OpenCV can write depends on how it was
# built: on the GPU box H.264 is absent but VP9 is there, so the codec is
# probed rather than assumed, and the container has to match the codec, hence
# the extension travelling with it.
CODECS = (("avc1", ".mp4"), ("H264", ".mp4"),
          ("VP90", ".webm"), ("VP80", ".webm"),
          ("mp4v", ".mp4"))


def open_writer(out, fps, size):
    """(writer, fourcc, path). Probes codecs; the path may change extension."""
    # A codec that will not open makes OpenCV and ffmpeg print several alarming
    # lines from inside the C library. They are expected here and mean only
    # "try the next one", so they are hidden for the length of the probe.
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        for fourcc, ext in CODECS:
            path = Path(out).with_suffix(ext)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                     fps, size)
            if writer.isOpened():
                return writer, fourcc, path
            writer.release()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    return None, None, None


def write_clip(src, out, seconds, width):
    cap, fps, w, h, total = open_source(src)
    ow, oh = target_size(w, h, width)
    limit = int(fps * seconds) if seconds else total or 10**9

    writer, used, path = open_writer(out, fps, (ow, oh))
    if writer is None:
        cap.release()
        sys.exit("ни один кодек не открылся")

    n = 0
    while n < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if (ow, oh) != (w, h):
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()

    size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
    print(f"  {path}")
    print(f"    {ow}x{oh}, {n} кадров, {n / fps:.0f} с, кодек {used}, {size_mb:.0f} МБ")
    if used == "mp4v":
        print("    ВНИМАНИЕ: ни H.264, ни VP9 этот OpenCV писать не умеет.")
        print("    Браузер такой файл не откроет, нужен обычный плеер.")
    else:
        print("    Открывается в браузере: перетащи файл в окно Chromium.")
    return used


def write_y4m(src, out, seconds, width):
    """Chromium plays a Y4M file in place of a camera. Uncompressed, so short."""
    cap, fps, w, h, total = open_source(src)
    ow, oh = target_size(w, h, width)
    limit = int(fps * seconds) if seconds else total or 10**9
    rate = f"{int(round(fps))}:1"

    n = 0
    with open(out, "wb") as f:
        f.write(f"YUV4MPEG2 W{ow} H{oh} F{rate} Ip A1:1 C420mpeg2\n".encode("ascii"))
        while n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if (ow, oh) != (w, h):
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            f.write(b"FRAME\n")
            f.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes())
            n += 1
    cap.release()

    size_mb = Path(out).stat().st_size / 1024 / 1024
    print(f"  {out}")
    print(f"    {ow}x{oh}, {n} кадров, {n / fps:.0f} с, {size_mb:.0f} МБ (без сжатия)")
    # All three flags are needed: the first grants the camera permission
    # without a dialog, the second registers a fake camera at all, the third
    # points it at this file. Without the second Chromium answers
    # getUserMedia with "Requested device not found".
    print("    Запуск с ним вместо камеры:")
    print(f"      chromium --use-fake-ui-for-media-stream \\\n"
          f"               --use-fake-device-for-media-stream \\\n"
          f"               --use-file-for-fake-video-capture={out} \\\n"
          f"               --ignore-certificate-errors https://localhost:8765/demo")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video to read")
    ap.add_argument("--out", help="write a playable clip here "
                                  "(extension follows the codec that works)")
    ap.add_argument("--y4m", help="write a Chromium camera file here")
    ap.add_argument("--seconds", type=float, default=0,
                    help="keep only the first N seconds (0 = all)")
    ap.add_argument("--width", type=int, default=0,
                    help="scale down to this width, keeping the proportions")
    args = ap.parse_args()

    if not args.out and not args.y4m:
        sys.exit("нечего делать: укажи --out и/или --y4m")
    if not Path(args.source).is_file():
        sys.exit(f"нет файла: {args.source}")

    print(f"Источник: {args.source}")
    if args.out:
        write_clip(args.source, args.out, args.seconds, args.width)
    if args.y4m:
        if not args.seconds:
            print("  (для Y4M без --seconds файл будет очень большим)")
        write_y4m(args.source, args.y4m, args.seconds, args.width)


if __name__ == "__main__":
    main()
