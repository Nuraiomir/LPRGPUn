#!/usr/bin/env python3
"""
Prepares frames from a video for labelling: samples them evenly, runs the
current detector over each one and writes its boxes as a YOLO label file.

The labels are a starting point, not the truth. On real footage the detector
misses plates and fires on badges and lettering, which is exactly why the
model is being retrained. Open the frames in a label editor, delete the wrong
boxes and draw the missing ones.

Frames are sampled across the whole video, including ones where the detector
found nothing: those are the most useful examples to train on.

Output (the layout labelImg and Ultralytics both understand):

    <out>/images/<video>_000123.jpg
    <out>/labels/<video>_000123.txt      one line per box: 0 cx cy w h (0..1)
    <out>/classes.txt

Usage:
    python3 tools/prepare_frames.py videos/20260923_152319.mp4 \
        --model ~/best.pt --out ~/labelling/parking --count 150
"""

import argparse
import sys
from pathlib import Path

import cv2


def sample_positions(total, count):
    """Evenly spaced frame numbers, first and last included."""
    if count >= total:
        return list(range(total))
    step = (total - 1) / (count - 1) if count > 1 else 0
    return [int(round(i * step)) for i in range(count)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, help="detector weights (.pt); omit to write empty labels")
    ap.add_argument("--count", type=int, default=150, help="frames to take (default 150)")
    ap.add_argument("--conf", type=float, default=0.40, help="detection threshold, as in the pipeline")
    ap.add_argument("--imgsz", type=int, default=512, help="detector input size, as in the pipeline")
    ap.add_argument("--max-side", type=int, default=1920,
                    help="downscale frames so the long side is at most this (0 keeps 4K)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wanted = sample_positions(total, args.count)

    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(str(args.model))

    img_dir, lbl_dir = args.out / "images", args.out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("license_plate\n", encoding="utf-8")

    stem = args.video.stem
    saved = boxes_total = empty = 0
    for n in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        if not ok:
            continue
        if args.max_side:
            h, w = frame.shape[:2]
            if max(h, w) > args.max_side:
                k = args.max_side / max(h, w)
                frame = cv2.resize(frame, (int(round(w * k)), int(round(h * k))),
                                   interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

        lines = []
        if model is not None:
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                lines.append(f"0 {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

        name = f"{stem}_{n:06d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        saved += 1
        boxes_total += len(lines)
        empty += not lines

    cap.release()
    print(f"видео: {args.video.name}, {total} кадров, {fps:.1f} кадр/с")
    print(f"сохранено кадров: {saved} в {img_dir}")
    print(f"черновых рамок:   {boxes_total}")
    print(f"кадров без рамок: {empty}  (модель ничего не нашла; если номер там есть, нарисуйте его)")
    print(f"\nРазметка: {lbl_dir}\nКлассы:   {args.out / 'classes.txt'}")


if __name__ == "__main__":
    main()
