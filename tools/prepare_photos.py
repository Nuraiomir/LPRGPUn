#!/usr/bin/env python3
"""
Prepares a folder of photos for labelling: copies them under tidy names, scales
them down and writes a YOLO label file per photo using the current detector.

Same idea as tools/prepare_frames.py, but the input is photos rather than a
video. Photos with no plate in them are kept with an empty label file: those
teach the detector not to fire on badges, lights and lettering.

The labels are a starting point. Open the folder in a label editor, delete
wrong boxes and draw missing ones.

Output:
    <out>/images/<prefix>_0001.jpg
    <out>/labels/<prefix>_0001.txt      one line per box: 0 cx cy w h (0..1)
    <out>/classes.txt

Usage:
    python3 tools/prepare_photos.py ~/photos_raw \
        --model ~/training/runs/real_v2/weights/best.pt \
        --out ~/labelling/photos_v1 --prefix photo
"""

import argparse
import sys
from pathlib import Path

import cv2

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder with the photos (searched recursively)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, help="detector weights (.pt); omit to write empty labels")
    ap.add_argument("--prefix", default="photo", help="name prefix for the copies")
    ap.add_argument("--conf", type=float, default=0.40, help="detection threshold, as in the pipeline")
    ap.add_argument("--imgsz", type=int, default=512, help="detector input size, as in the pipeline")
    ap.add_argument("--max-side", type=int, default=1920,
                    help="downscale so the long side is at most this (0 keeps the original)")
    args = ap.parse_args()

    photos = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not photos:
        sys.exit(f"no photos found in {args.folder}")

    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(str(args.model))

    img_dir, lbl_dir = args.out / "images", args.out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("license_plate\n", encoding="utf-8")

    saved = boxes_total = empty = unreadable = 0
    for path in photos:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  не прочитан: {path.name}")
            unreadable += 1
            continue
        if args.max_side:
            h, w = img.shape[:2]
            if max(h, w) > args.max_side:
                k = args.max_side / max(h, w)
                img = cv2.resize(img, (int(round(w * k)), int(round(h * k))),
                                 interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

        lines = []
        if model is not None:
            res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                lines.append(f"0 {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

        name = f"{args.prefix}_{saved + 1:04d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                             encoding="utf-8")
        saved += 1
        boxes_total += len(lines)
        empty += not lines

    print(f"найдено файлов: {len(photos)}, сохранено: {saved}"
          + (f", не прочитано: {unreadable}" if unreadable else ""))
    print(f"черновых рамок: {boxes_total}")
    print(f"без рамок:      {empty}  (проверьте: если номер там есть, нарисуйте его)")
    print(f"\nКартинки: {img_dir}\nРазметка: {lbl_dir}")


if __name__ == "__main__":
    main()
