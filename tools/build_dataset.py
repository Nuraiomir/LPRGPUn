#!/usr/bin/env python3
"""
Builds one dataset out of the old set the model was trained on and the newly
labelled real frames.

Frames from one video look alike, so splitting them at random would put
near-identical frames into both halves and make the validation score look
better than it is. Here the frames are ordered by time and cut into blocks of
consecutive frames (roughly one car each); every Nth block goes to validation
as a whole. Validation then contains cars the model never trained on.

The old validation set is kept as well, so the training run also shows whether
the model is losing what it already did well.

Usage:
    python3 tools/build_dataset.py \
        --old ~/mixed_lpr --new ~/labelling/parking_labeled \
        --out ~/training/lpr_real_v1
"""

import argparse
import shutil
import sys
from pathlib import Path


def copy_pairs(items, img_dir, lbl_dir, prefix):
    """items: list of (image path, label path). Returns how many were copied."""
    n = 0
    for img, lbl in items:
        shutil.copy2(img, img_dir / f"{prefix}{img.name}")
        if lbl.exists():
            shutil.copy2(lbl, lbl_dir / f"{prefix}{lbl.name}")
        else:
            (lbl_dir / f"{prefix}{img.stem}.txt").write_text("", encoding="utf-8")
        n += 1
    return n


def old_pairs(root, split):
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    return [(i, root / "labels" / split / f"{i.stem}.txt") for i in imgs]


def new_pairs(root):
    imgs = sorted((root / "images").glob("*.jpg"))
    return [(i, root / "labels" / f"{i.stem}.txt") for i in imgs]


def split_by_blocks(pairs, blocks, val_every):
    """Cut the time-ordered frames into blocks; every val_every-th goes to val."""
    if not pairs:
        return [], []
    size = max(1, len(pairs) // blocks)
    train, val = [], []
    for i, pair in enumerate(pairs):
        (val if (i // size) % val_every == val_every - 1 else train).append(pair)
    return train, val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True, help="existing dataset (images/train, ...)")
    ap.add_argument("--new", type=Path, required=True, help="labelled frames (images/, labels/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=15, help="blocks to cut the new frames into")
    ap.add_argument("--val-every", type=int, default=4, help="every Nth block goes to validation")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} already exists and is not empty")

    dirs = {}
    for split in ("train", "val"):
        dirs[split] = (args.out / "images" / split, args.out / "labels" / split)
        for d in dirs[split]:
            d.mkdir(parents=True, exist_ok=True)

    counts = {}
    counts["old train"] = copy_pairs(old_pairs(args.old, "train"), *dirs["train"], "old_")
    counts["old val"] = copy_pairs(old_pairs(args.old, "val"), *dirs["val"], "old_")

    new_train, new_val = split_by_blocks(new_pairs(args.new), args.blocks, args.val_every)
    counts["new train"] = copy_pairs(new_train, *dirs["train"], "real_")
    counts["new val"] = copy_pairs(new_val, *dirs["val"], "real_")

    (args.out / "data.yaml").write_text(
        f"path: {args.out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8")

    def boxes_in(split):
        return sum(len([x for x in f.read_text().splitlines() if x.strip()])
                   for f in (args.out / "labels" / split).glob("*.txt"))

    train_names = {f.name for f in (args.out / "images" / "train").glob("*.jpg")}
    val_names = {f.name for f in (args.out / "images" / "val").glob("*.jpg")}
    shared = train_names & val_names

    print("собрано:")
    for k, v in counts.items():
        print(f"  {k:10s} {v:5d} кадров")
    for split in ("train", "val"):
        n = len(list((args.out / "images" / split).glob("*.jpg")))
        real = len(list((args.out / "images" / split).glob("real_*.jpg")))
        print(f"\n{split}: {n} кадров ({real} реальных), рамок {boxes_in(split)}")
    if shared:
        print(f"\nВНИМАНИЕ: {len(shared)} кадров лежат и в обучении, и в проверке, "
              f"например {sorted(shared)[:3]}.")
        print("Метрики проверки будут завышены. Стоит разобраться до обучения.")
    else:
        print("\nОбучение и проверка не пересекаются.")
    print(f"\ndata.yaml: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
