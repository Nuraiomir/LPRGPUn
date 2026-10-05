#!/usr/bin/env python3
"""
Собирает копию обучающего набора без кадров из тестовых видео.

Зачем. В training/lpr_real_v1 нашлись 300 кадров, вырезанных из
videos/20260923_152319.mp4, то есть из видео, на котором считается сквозная
метрика. Модель обучалась на тех самых кадрах, на которых её потом проверяют,
и число "25 машин из 28" из-за этого не является честным измерением.

Что делает скрипт. Берёт исходный набор, выбрасывает все картинки, чьё
содержимое совпадает с кадрами перечисленных видео, и раскладывает остальное в
новую папку с тем же делением на train и val. Исходный набор не трогается.

Совпадение ищется по хешу содержимого, потому что при сборке датасета картинки
переименовали: по именам кадр из видео в обучающем наборе не найти.

    python3 tools/make_clean_dataset.py \\
        --source ~/training/lpr_real_v1 \\
        --frames ~/nurai_gpu/real_all \\
        --videos 20260923_152319 \\
        --out ~/training/lpr_real_clean

Потом обучение на новом наборе и замер заново:

    ~/.venv_train/bin/python tools/train_yolo11n.py \\
        --data ~/training/lpr_real_clean/data.yaml --epochs 100 \\
        --weights yolov8n.pt --name lpr_clean_8n --out-name best8n_clean_512.onnx
"""
import argparse
import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frames_of(folder, videos):
    """Хеши кадров, вырезанных из перечисленных видео."""
    pattern = re.compile("^(" + "|".join(re.escape(v) for v in videos) + r")_\d+\.",
                         re.IGNORECASE)
    out = set()
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUFFIXES and pattern.match(p.name):
            out.add(digest(p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path,
                    help="исходный набор: images/train, images/val, labels/...")
    ap.add_argument("--frames", required=True, type=Path,
                    help="папка, где лежат кадры видео (real_all)")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="имена видео без расширения, например 20260923_152319")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    source = args.source.expanduser()
    frames_dir = args.frames.expanduser()
    out = args.out.expanduser()
    for p in (source, frames_dir):
        if not p.is_dir():
            raise SystemExit(f"нет папки: {p}")
    if out.exists():
        raise SystemExit(f"{out} уже есть. Удали её сам или задай другой --out: "
                         f"молча перезаписывать датасет опасно.")

    print(f"ищу кадры видео {', '.join(args.videos)} в {frames_dir}...")
    banned = frames_of(frames_dir, args.videos)
    print(f"  найдено кадров: {len(banned)}\n")
    if not banned:
        raise SystemExit("кадров не нашлось: проверь --frames и --videos")

    kept = Counter()
    dropped = Counter()
    for split in ("train", "val"):
        img_dir = source / "images" / split
        lbl_dir = source / "labels" / split
        if not img_dir.is_dir():
            print(f"  нет {img_dir}, пропускаю")
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted(img_dir.iterdir()):
            if not (img.is_file() and img.suffix.lower() in SUFFIXES):
                continue
            if digest(img) in banned:
                dropped[split] += 1
                continue
            shutil.copy2(img, out / "images" / split / img.name)
            label = lbl_dir / (img.stem + ".txt")
            if label.is_file():
                shutil.copy2(label, out / "labels" / split / label.name)
            kept[split] += 1
        print(f"  {split}: оставлено {kept[split]}, выброшено {dropped[split]}")

    # data.yaml пишется заново, с абсолютным путём: относительный указывал бы
    # на исходный набор, и обучение пошло бы по старым данным незаметно.
    (out / "data.yaml").write_text(
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8")

    total_kept = sum(kept.values())
    total_dropped = sum(dropped.values())
    print("\n" + "=" * 64)
    print("ЧИСТЫЙ НАБОР СОБРАН")
    print("=" * 64)
    print(f"  было:      {total_kept + total_dropped}")
    print(f"  выброшено: {total_dropped}  (кадры из тестовых видео)")
    print(f"  осталось:  {total_kept}")
    print(f"  папка:     {out}")
    print(f"  описание:  {out / 'data.yaml'}")
    if total_dropped == 0:
        print("\n  Ничего не выброшено: либо утечки нет, либо --frames не та папка.")
    print("\nДальше обучение на нём и замер заново. Старые числа на видео")
    print(f"{', '.join(args.videos)} до этого момента считать нельзя.")


if __name__ == "__main__":
    main()
