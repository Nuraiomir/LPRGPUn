#!/usr/bin/env python3
"""
Measures how well the current OCR reads a labelled dataset of plate crops.

Until now every benchmark in this project compared one configuration against
another, because no verified labels existed. A labelled dataset finally allows
the number that matters: how many plates are read exactly right.

Two things are counted separately, and conflating them would make the result
meaningless:

    OCR correct      the characters OCR returned equal the label
    pipeline accepts the reading also passes valid_kz_plate(), i.e. the
                     three-letter format the service currently confirms

A plate in the older Kazakh layout (3 digits, 2 letters, 2 digits) can be read
perfectly and still be rejected by the pipeline. That is a product decision,
not an OCR failure, so it is reported on its own line.

Dataset layout (AUTO.RIA / nomeroff-net style):

    <root>/<split>/ann/<name>.json     {"description": "249AS16", ...}
    <root>/<split>/img/<name>.png

Usage:
    python3 bench/ocr_dataset_eval.py ~/datasets/ocr_kz --scan-only
    python3 bench/ocr_dataset_eval.py ~/datasets/ocr_kz --split test --limit 500
    python3 bench/ocr_dataset_eval.py ~/datasets/ocr_kz --split test --report runs/ocr_eval
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "bench"))
# extract_plate is imported, not copied: the benchmark has to measure the rule
# the service actually runs. A second copy here would keep reporting the old
# number after the pipeline changed, which is exactly the mistake this file is
# meant to catch.
from lpr_recognizer import clean_text, extract_plate, valid_kz_plate  # noqa: E402
from letter_repair import repair_letter_slots  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
NEW_KZ = re.compile(r"\d{3}[A-Z]{3}\d{2}")     # 123ABC02, what the service confirms
OLD_KZ = re.compile(r"\d{3}[A-Z]{2}\d{2}")     # 249AS16, the pre-2012 layout

def plate_format(text):
    if NEW_KZ.fullmatch(text):
        return "new (3 цифры + 3 буквы + 2 цифры)"
    if OLD_KZ.fullmatch(text):
        return "old (3 цифры + 2 буквы + 2 цифры)"
    return "other"


def find_samples(root, split):
    """Returns [(image path, label)] for one split."""
    ann_dir, img_dir = root / split / "ann", root / split / "img"
    if not ann_dir.is_dir():
        return []
    samples, missing = [], 0
    for ann in sorted(ann_dir.glob("*.json")):
        try:
            label = json.loads(ann.read_text(encoding="utf-8")).get("description", "")
        except (OSError, ValueError):
            continue
        image = None
        for suffix in IMAGE_SUFFIXES:
            candidate = img_dir / (ann.stem + suffix)
            if candidate.exists():
                image = candidate
                break
        if image is None:
            missing += 1
            continue
        samples.append((image, clean_text(label)))
    if missing:
        print(f"  {split}: {missing} аннотаций без картинки, пропущены")
    return samples


def splits_of(root):
    return [d.name for d in sorted(root.iterdir()) if (d / "ann").is_dir()]


def scan(root):
    """Prints what the dataset contains, without touching the GPU."""
    total = Counter()
    formats = Counter()
    examples = {}
    lengths = Counter()
    for split in splits_of(root):
        samples = find_samples(root, split)
        total[split] = len(samples)
        for _, label in samples:
            kind = plate_format(label)
            formats[kind] += 1
            lengths[len(label)] += 1
            examples.setdefault(kind, []).append(label)

    print("\nКартинок по частям датасета:")
    for split, n in total.items():
        print(f"  {split:6s} {n:6d}")
    print(f"  {'всего':6s} {sum(total.values()):6d}")

    print("\nФорматы номеров:")
    n = sum(formats.values()) or 1
    for kind, count in formats.most_common():
        sample = ", ".join(examples[kind][:4])
        print(f"  {kind:36s} {count:6d}  ({100 * count / n:4.1f}%)   {sample}")

    print("\nДлина номера:", ", ".join(f"{k}: {v}" for k, v in sorted(lengths.items())))

    accepted = formats["new (3 цифры + 3 буквы + 2 цифры)"]
    print(f"\nПроверку формата сервиса прошло бы {accepted} из {n} "
          f"({100 * accepted / n:.1f}%) — остальные отвергаются по формату, "
          f"как бы хорошо OCR их ни прочитал.")


def levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(root, split, limit, report_dir, ocr_mode):
    samples = find_samples(root, split)
    if not samples:
        sys.exit(f"в {root / split} нет размеченных картинок")
    if limit:
        samples = samples[:limit]

    sys.path.insert(0, str(ROOT))
    try:
        import config.gpu_env as gpu_env
    except ModuleNotFoundError:
        sys.exit("config/gpu_env.py не найден: cp config/gpu_env.example.py config/gpu_env.py")
    from gpu_workers_client import Workers

    print(f"\nЗапуск OCR ({ocr_mode})...", flush=True)
    workers = Workers(gpu_env.PROJECT_ROOT, gpu_env.YOLO_PYTHON, gpu_env.OCR_PYTHON,
                      gpu_env.ONNX_MODEL, gpu_env.YOLO_CUDA_LD_PATH, gpu_env.OCR_CUDA_LD_PATH,
                      ocr_variant_mode=ocr_mode)
    print(f"OCR: {workers.ocr_info.get('device')} / {workers.ocr_info.get('backend')}\n", flush=True)

    stats = Counter()
    per_format = {}
    confusions = Counter()
    shapes = Counter()
    shape_examples = {}
    wrong_pulls = []
    wrong_repairs = []
    failures = []
    char_errors = char_total = 0
    t_start = time.monotonic()

    try:
        for i, (path, label) in enumerate(samples, 1):
            image = cv2.imread(str(path))
            if image is None:
                stats["не прочитана картинка"] += 1
                continue

            payload = workers.ocr(image, "normal")
            got = clean_text(payload.get("text", ""))
            conf = float(payload.get("conf", 0.0))

            kind = plate_format(label)
            bucket = per_format.setdefault(kind, Counter())
            bucket["всего"] += 1
            stats["всего"] += 1

            correct = got == label
            accepted = valid_kz_plate(got)

            # What a rule that pulls the plate out of a longer reading would do.
            pulled = extract_plate(got)
            if pulled == label:
                stats["извлечение: верно"] += 1
            elif pulled:
                stats["извлечение: НЕВЕРНЫЙ номер"] += 1
                if len(wrong_pulls) < 12:
                    wrong_pulls.append(f"{label} -> {got} -> {pulled}")
            elif kind == "new (3 цифры + 3 буквы + 2 цифры)":
                stats["извлечение: ничего"] += 1

            # Only readings the service rejects outright are offered to the
            # repair rule, so it can never change a plate confirmed today.
            if not pulled:
                repaired, certainty = repair_letter_slots(got)
                if repaired:
                    hit = "верно" if repaired == label else "НЕВЕРНО"
                    stats[f"буквы {certainty}: {hit}"] += 1
                    if hit == "НЕВЕРНО" and len(wrong_repairs) < 12:
                        wrong_repairs.append(f"{label} -> {got} -> {repaired} ({certainty})")

            bucket["OCR верно" if correct else "OCR неверно"] += 1
            if correct:
                stats["OCR верно"] += 1
            if correct and accepted:
                stats["принято сервисом"] += 1
                bucket["принято сервисом"] += 1

            char_errors += levenshtein(got, label)
            char_total += len(label)

            if not correct:
                # A wrong reading of the right length is a character mix-up.
                # A reading of the wrong length means OCR returned a cut-off or
                # empty string, which is a different failure with a different fix.
                if not got:
                    shape = "ничего не вернул"
                elif len(got) < len(label):
                    shape = "короче: часть номера потеряна"
                elif len(got) > len(label):
                    shape = "длиннее: прихватил лишнее"
                else:
                    shape = "та же длина, перепутаны символы"
                    for a, b in zip(label, got, strict=True):
                        if a != b:
                            confusions[f"{a} -> {b}"] += 1
                shapes[shape] += 1
                shape_examples.setdefault(shape, []).append(f"{label} -> {got or '(пусто)'}")
                failures.append({"image": str(path), "expected": label, "got": got,
                                 "conf": round(conf, 3), "format": kind})

            if i % 200 == 0:
                rate = i / (time.monotonic() - t_start)
                print(f"  {i}/{len(samples)}   верно {stats['OCR верно']}   "
                      f"{rate:.0f} шт/с", flush=True)
    finally:
        workers.close()

    n = stats["всего"] or 1
    elapsed = time.monotonic() - t_start
    print("\n" + "=" * 62)
    print(f"РЕЗУЛЬТАТ на части «{split}»: {n} номеров за {elapsed:.0f} с "
          f"({n / max(elapsed, 1e-9):.0f} шт/с)")
    print("=" * 62)
    print(f"  OCR прочитал точно:       {stats['OCR верно']:5d}  ({100 * stats['OCR верно'] / n:5.1f}%)")
    print(f"  из них принял бы сервис:  {stats['принято сервисом']:5d}  "
          f"({100 * stats['принято сервисом'] / n:5.1f}%)")
    print(f"  ошибок в символах:        {100 * char_errors / max(char_total, 1):5.1f}%")

    print("\n  Правило сервиса: извлечь номер из строки (снимает KZ и прочий мусор):")
    ok_pull = stats["извлечение: верно"]
    bad_pull = stats["извлечение: НЕВЕРНЫЙ номер"]
    print(f"    верно:                  {ok_pull:5d}  ({100 * ok_pull / n:5.1f}%)"
          f"   было {stats['принято сервисом']} ({100 * stats['принято сервисом'] / n:.1f}%)")
    print(f"    НЕВЕРНЫЙ номер:         {bad_pull:5d}  ({100 * bad_pull / n:5.1f}%)"
          "   <- это опаснее пропуска: ОСРМ искала бы другую машину")
    for example in wrong_pulls:
        print(f"      {example}")

    print("\n  Ещё не в сервисе: починить букву, прочитанную как цифру (106B0A11 -> 106BOA11):")
    for certainty in ("однозначно", "догадка"):
        ok = stats[f"буквы {certainty}: верно"]
        bad = stats[f"буквы {certainty}: НЕВЕРНО"]
        if not (ok or bad):
            continue
        share = 100 * bad / max(ok + bad, 1)
        print(f"    {certainty:11s} +{ok:4d} верно, {bad:4d} НЕВЕРНО "
              f"({share:.0f}% из добавленных были бы чужой машиной)")
    if not any(stats[f"буквы {c}: {h}"] for c in ("однозначно", "догадка")
               for h in ("верно", "НЕВЕРНО")):
        print("    ни одного случая — правило ничего не изменило бы")
    for example in wrong_repairs:
        print(f"      {example}")

    print("\n  По форматам номера:")
    for kind, bucket in sorted(per_format.items()):
        total = bucket["всего"]
        ok = bucket["OCR верно"]
        taken = bucket["принято сервисом"]
        print(f"    {kind:36s} {total:5d}   OCR верно {ok:5d} ({100 * ok / max(total, 1):5.1f}%)"
              f"   принято {taken:5d}")

    wrong = sum(shapes.values()) or 1
    if shapes:
        print("\n  Вид ошибок:")
        for shape, count in shapes.most_common():
            examples = ", ".join(shape_examples[shape][:3])
            print(f"    {shape:34s} {count:5d}  ({100 * count / wrong:4.1f}%)   {examples}")

    if confusions:
        print("\n  Чаще всего путает символы (ожидалось -> прочитано):")
        for pair, count in confusions.most_common(12):
            print(f"    {pair}   {count}")

    if report_dir:
        write_report(Path(report_dir), failures, stats, per_format, confusions, split, n)


def write_report(out, failures, stats, per_format, confusions, split, total):
    """An HTML page with every misread crop, to look at them with your own eyes."""
    out.mkdir(parents=True, exist_ok=True)
    crops = out / "crops"
    crops.mkdir(exist_ok=True)

    rows = []
    for i, f in enumerate(failures):
        image = cv2.imread(f["image"])
        if image is None:
            continue
        name = f"{i:05d}.png"
        cv2.imwrite(str(crops / name), image)
        h, w = image.shape[:2]
        rows.append(
            f'<tr><td><img src="crops/{name}" alt=""></td>'
            f'<td class="exp">{f["expected"]}</td>'
            f'<td class="got">{f["got"] or "(пусто)"}</td>'
            f'<td>{f["conf"]}</td><td>{w}x{h}</td><td>{f["format"]}</td></tr>')

    page = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Ошибки OCR — {split}</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 24px; background:#fafafa; color:#18181b }}
 table {{ border-collapse: collapse; background:#fff }}
 td, th {{ border:1px solid #d4d4d8; padding:8px 10px; vertical-align:middle }}
 img {{ max-height:64px; display:block }}
 .exp {{ font-weight:700; letter-spacing:1px }}
 .got {{ color:#b00020; letter-spacing:1px }}
 .sum {{ margin-bottom:18px; line-height:1.6 }}
</style></head><body>
<h1>Неверно прочитанные номера — часть «{split}»</h1>
<div class="sum">
Всего проверено: {total}<br>
Прочитано точно: {stats['OCR верно']} ({100 * stats['OCR верно'] / max(total,1):.1f}%)<br>
Ошибок в таблице: {len(rows)}
</div>
<table><tr><th>кроп</th><th>ожидалось</th><th>прочитано</th><th>уверенность</th>
<th>размер</th><th>формат</th></tr>
{chr(10).join(rows)}
</table></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"\n  Отчёт с картинками: {out / 'index.html'}  ({len(rows)} шт.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="folder holding train/ test/ val/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--report", default=None, help="folder for the HTML gallery of misreads")
    ap.add_argument("--ocr-mode", choices=["full", "no-enhanced"], default="full")
    ap.add_argument("--scan-only", action="store_true", help="just describe the dataset")
    args = ap.parse_args()

    root = args.dataset.expanduser()
    if not root.is_dir():
        sys.exit(f"нет такой папки: {root}")
    # The archive often unpacks into a folder of the same name.
    if not splits_of(root):
        nested = [d for d in root.iterdir() if d.is_dir() and splits_of(d)]
        if len(nested) == 1:
            root = nested[0]
            print(f"использую вложенную папку: {root}")
    if not splits_of(root):
        sys.exit(f"в {root} не найдено частей с ann/ и img/")

    scan(root)
    if not args.scan_only:
        evaluate(root, args.split, args.limit, args.report, args.ocr_mode)


if __name__ == "__main__":
    main()
