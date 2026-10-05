#!/usr/bin/env python3
"""
Собирает набор для проверки детектора из размеченных картинок, которых нет в
обучении.

Зачем. Сравнивать детекторы на выборке, которую они видели при обучении,
бессмысленно. В training/lpr_real_v1 проверочная часть всего 134 картинки, и
разница между двумя моделями там измеряется одной рамкой. При этом рядом
лежат размеченные картинки, которые в обучение не попали вовсе, и их никто не
использует.

Что делает. Берёт папку с images/ и labels/, выбрасывает всё, что по
содержимому совпадает с обучающим набором, и складывает остаток в новый набор
целиком как val. Такой набор годится только для проверки, обучать на нём
нельзя: он для того и собран.

    python3 tools/make_holdout.py \\
        --source ~/nurai_gpu/real_all \\
        --exclude ~/training/lpr_real_v1 \\
        --out ~/training/holdout_real

Проверка обеих моделей на нём, OCR не участвует:

    ~/.venv_train/bin/yolo val model=runs/detect/lpr_yolov8n_fair/weights/best.pt \\
        data=~/training/holdout_real/data.yaml imgsz=512
"""
import argparse
import hashlib
import shutil
from pathlib import Path

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashes(root):
    return {digest(p) for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUFFIXES}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path,
                    help="папка с images/ и labels/")
    ap.add_argument("--exclude", required=True, type=Path,
                    help="обучающий набор: всё, что есть в нём, выбрасывается")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    source = args.source.expanduser()
    exclude = args.exclude.expanduser()
    out = args.out.expanduser()
    for p in (source, exclude):
        if not p.is_dir():
            raise SystemExit(f"нет папки: {p}")
    if out.exists():
        raise SystemExit(f"{out} уже есть, задай другой --out или удали её сам")

    img_dir = source / "images"
    lbl_dir = source / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"нет {img_dir}: нужна папка с images/ и labels/")

    print(f"читаю обучающий набор {exclude}...")
    seen = hashes(exclude)
    print(f"  в нём {len(seen)} уникальных картинок\n")

    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    kept = skipped = unlabelled = 0
    for img in sorted(img_dir.rglob("*")):
        if not (img.is_file() and img.suffix.lower() in SUFFIXES):
            continue
        if digest(img) in seen:
            skipped += 1
            continue
        label = lbl_dir / (img.stem + ".txt")
        if not label.is_file():
            unlabelled += 1
            continue
        shutil.copy2(img, out / "images" / "val" / img.name)
        shutil.copy2(label, out / "labels" / "val" / label.name)
        kept += 1

    if kept == 0:
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit("не осталось ни одной картинки: всё уже в обучении")

    # train указывает на ту же папку, что val: Ultralytics требует оба поля,
    # а обучать на этом наборе нельзя и не нужно. Если кто-то попробует, он
    # обучится на проверочных данных, и это будет видно сразу по идеальным
    # метрикам.
    (out / "data.yaml").write_text(
        f"path: {out}\n"
        "train: images/val\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8")

    print("=" * 60)
    print("НАБОР ДЛЯ ПРОВЕРКИ СОБРАН")
    print("=" * 60)
    print(f"  взято:            {kept}   (ни одна модель их не видела)")
    print(f"  пропущено:        {skipped}   (уже в обучении)")
    if unlabelled:
        print(f"  без разметки:     {unlabelled}   (пропущены, размечать нечем)")
    print(f"  папка:            {out}")
    print(f"  описание:         {out / 'data.yaml'}")
    print("\nЭто проверка ДЕТЕКТОРА: нашлась ли рамка и насколько точно.")
    print("OCR в ней не участвует, что внутри рамки написано, не проверяется.")


if __name__ == "__main__":
    main()
