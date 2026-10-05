#!/usr/bin/env python3
"""
Порог уверенности детектора: как он меняет находки и ложные рамки.

ЗАЧЕМ ЭТОТ ЗАМЕР. Руководитель говорит, что YOLO11n справляется лучше, чем
YOLOv8n. На отложенной выборке она действительно находит больше: полнота
0.915 против 0.870. Но она же выдаёт около восьми процентов ложных рамок
(точность 0.924 против 1.000 у восьмой), и именно из-за них сквозной прогон
дал два неверных номера. Вывод «оставляем восьмую» опирается целиком на эти
ложные рамки.

А ложные рамки это ровно то, что убирает порог уверенности. Все предыдущие
замеры сделаны на одном пороге 0.40, и пока не проверено, что будет на 0.55 и
0.70, вывод держится на одной точке. Если на более высоком пороге YOLO11n
теряет ложные рамки и сохраняет находки, руководитель прав и модель надо
менять. Если вместе с ложными уходят и находки, вывод подтверждён числами, а
не мнением.

ДВЕ МЕТРИКИ, И ВТОРАЯ ВАЖНЕЕ.

  точность и полнота по всем рамкам
      Обычная метрика качества детектора: все рамки выше порога, после
      подавления пересечений, сравниваются с разметкой по IoU. Так считают
      все, и так сравнимо с чужими результатами.

  попадание первой рамки (top-1)
      Наш сервис берёт ОДНУ рамку, самую уверенную, и отдаёт её
      распознавателю. Значит ложная рамка вредит только тогда, когда она
      обогнала настоящую. Для нашего конвейера решает именно это число, и
      именно оно объясняет те два неверных номера.

Полнота по всем рамкам может расти, а попадание первой рамки падать: это
происходит, когда модель находит номер И ещё что-то более уверенное.

ЧТО НУЖНО. Отложенная выборка в формате YOLO: images/ и labels/ рядом, либо
val/images и val/labels внутри. Собирается tools/make_holdout.py. Обучающие
картинки в неё попадать не должны, иначе замер меряет запоминание.

ЗАПУСК.

    .venv_gpu/bin/python bench/detector_conf_sweep.py \\
        model/best_512.onnx ~/yolo11n_512.onnx \\
        --dataset ~/training/holdout_real \\
        --conf 0.25 0.40 0.55 0.70

Декодирование выхода модели здесь своё, а не из воркера. Это не дублирование
по невнимательности: воркеру нужна одна рамка на одном пороге, а здесь нужны
все рамки на четырёх порогах, с подавлением пересечений. Разные задачи, и
смешивать их в одной функции значило бы усложнить ту, что работает в сервисе.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Порог IoU, при котором рамка считается попавшей в разметку. 0.5 это
# общепринятое значение, и менять его тут нельзя: иначе числа перестанут быть
# сравнимыми с чем бы то ни было.
IOU_MATCH = 0.5

# Порог подавления пересечений. Модель выдаёт тысячи рамок, и без подавления
# «все рамки выше порога» это не находки, а один и тот же номер много раз.
IOU_NMS = 0.45

INPUT_SIZE = 512
PAD_VALUE = 114


def letterbox(image, size=INPUT_SIZE):
    """Вписывает кадр в квадрат модели, не искажая пропорции."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes):
    """Оставляет по одной рамке на каждое найденное место.

    boxes: список (x1, y1, x2, y2, conf), любого порядка. Возвращает
    отсортированный по уверенности список без сильно пересекающихся рамок.
    """
    kept = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(box[:4], k[:4]) < IOU_NMS for k in kept):
            kept.append(box)
    return kept


class RawDetector:
    """Все рамки модели, без порога: порог применяется потом, на готовом выходе.

    Так одна прогонка модели по картинке обслуживает все четыре порога сразу.
    Иначе картинка считалась бы четыре раза, и сравнение порогов стоило бы
    вчетверо дороже без всякой пользы.
    """

    def __init__(self, model_path):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        wanted = ([p for p in ("CUDAExecutionProvider",) if p in providers]
                  + ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(model_path), providers=wanted)
        self.input_name = self.session.get_inputs()[0].name
        self.providers = self.session.get_providers()

    def boxes(self, image):
        """Список (x1, y1, x2, y2, conf) в пикселях картинки, без порога."""
        canvas, scale, dx, dy = letterbox(image)
        arr = canvas[:, :, ::-1].astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None]
        pred = self.session.run(None, {self.input_name: arr})[0]
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1] and pred.shape[0] <= 10:
            pred = pred.T

        fh, fw = image.shape[:2]
        out = []
        for row in pred:
            if len(row) < 5:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            conf = float(row[4]) if len(row) == 5 else float(np.max(row[4:]))
            if conf <= 0.01:
                # Отсекаем совсем нулевые: их десятки тысяч, и они не станут
                # находкой ни при каком разумном пороге, а время на них уходит.
                continue
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2:
                cx *= INPUT_SIZE
                cy *= INPUT_SIZE
                bw *= INPUT_SIZE
                bh *= INPUT_SIZE
            x1 = max(0.0, min(fw - 1.0, (cx - bw / 2 - dx) / scale))
            y1 = max(0.0, min(fh - 1.0, (cy - bh / 2 - dy) / scale))
            x2 = max(1.0, min(float(fw), (cx + bw / 2 - dx) / scale))
            y2 = max(1.0, min(float(fh), (cy + bh / 2 - dy) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((x1, y1, x2, y2, conf))
        return out


def find_pairs(dataset):
    """Пары (картинка, файл разметки). Понимает оба обычных расположения."""
    dataset = Path(dataset).expanduser().resolve()
    roots = []
    for candidate in (dataset, dataset / "val", dataset / "test", dataset / "train"):
        if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            roots.append(candidate)
    if not roots:
        raise SystemExit(f"в {dataset} не нашла пары images/ и labels/")

    pairs = []
    for root in roots:
        for image in sorted((root / "images").iterdir()):
            if image.suffix.lower() not in SUFFIXES:
                continue
            label = root / "labels" / (image.stem + ".txt")
            if label.is_file():
                pairs.append((image, label))
    return roots, pairs


def read_truth(label_path, width, height):
    """Рамки разметки в пикселях. Формат YOLO: класс cx cy w h, доли."""
    truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        truth.append((
            (cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height,
        ))
    return truth


def score(all_boxes, truths, threshold):
    """Считает обе метрики на одном пороге.

    all_boxes: список (на картинку) всех рамок модели без порога
    truths:    список (на картинку) рамок разметки
    """
    tp = fp = fn = 0
    top1_hit = top1_miss = top1_none = 0
    # Картинки без размеченного номера считаются отдельно. На них «ничего не
    # нашла» это ПРАВИЛЬНЫЙ ответ, а не промах, и смешивать их с картинками,
    # где номер есть, нельзя: иначе модель, которая просто меньше срабатывает,
    # выглядит хуже на пустых картинках и лучше на непустых одновременно.
    # В первой версии этого скрипта они были свалены в одну колонку, и числа
    # можно было прочитать неправильно.
    quiet_ok = false_alarm = 0

    for boxes, truth in zip(all_boxes, truths):
        above = nms([b for b in boxes if b[4] >= threshold])

        # Обычная метрика: каждая рамка разметки может быть найдена один раз.
        taken = set()
        for box in above:
            best_i, best_iou = -1, 0.0
            for i, gt in enumerate(truth):
                if i in taken:
                    continue
                value = iou(box[:4], gt)
                if value > best_iou:
                    best_i, best_iou = i, value
            if best_i >= 0 and best_iou >= IOU_MATCH:
                taken.add(best_i)
                tp += 1
            else:
                fp += 1
        fn += len(truth) - len(taken)

        # Что увидит сервис: одна рамка, самая уверенная.
        if not truth:
            # Номера на картинке нет. Молчание это успех, любая рамка это
            # ложная тревога.
            if above:
                false_alarm += 1
            else:
                quiet_ok += 1
            continue

        if not above:
            top1_none += 1
        elif any(iou(above[0][:4], gt) >= IOU_MATCH for gt in truth):
            top1_hit += 1
        else:
            top1_miss += 1

    with_plate = sum(1 for t in truths if t) or 1
    return {
        "conf": threshold,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "top1_hit": top1_hit,
        "top1_wrong": top1_miss,
        "top1_empty": top1_none,
        "quiet_ok": quiet_ok,
        "false_alarm": false_alarm,
        # Доля считается только по картинкам, где номер есть.
        "top1_share": round(top1_hit / with_plate, 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path, help="файлы .onnx")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="отложенная выборка: images/ и labels/")
    ap.add_argument("--conf", nargs="+", type=float,
                    default=[0.25, 0.40, 0.55, 0.70])
    ap.add_argument("--limit", type=int, default=0, help="0 = вся выборка")
    ap.add_argument("--out", type=Path, default=None,
                    help="по умолчанию runs/detector_conf_sweep.json")
    args = ap.parse_args()

    for model in args.models:
        if not model.expanduser().is_file():
            raise SystemExit(f"нет файла модели: {model}")

    roots, pairs = find_pairs(args.dataset)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit("в выборке нет размеченных картинок")

    if args.out is None:
        args.out = ROOT / "runs" / "detector_conf_sweep.json"

    print(f"выборка: {args.dataset}")
    for root in roots:
        print(f"  часть: {root.name}")
    print(f"картинок: {len(pairs)}")
    print(f"пороги: {', '.join(f'{c:g}' for c in args.conf)}")
    print(f"IoU попадания: {IOU_MATCH}, IoU подавления: {IOU_NMS}\n")

    # Разметка читается один раз: она одна для всех моделей и всех порогов.
    truths, sizes = [], []
    for image_path, label_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"не читается картинка: {image_path}")
        h, w = image.shape[:2]
        sizes.append((w, h))
        truths.append(read_truth(label_path, w, h))

    plates = sum(len(t) for t in truths)
    with_plate = sum(1 for t in truths if t)
    print(f"рамок в разметке: {plates}")
    print(f"картинок с номером: {with_plate}, без номера: {len(pairs) - with_plate}\n")

    report = {}
    for model in args.models:
        model = model.expanduser()
        name = model.stem
        print(f"{name}:")
        detector = RawDetector(model)
        print(f"  провайдеры: {detector.providers}")

        all_boxes, times = [], []
        for i, (image_path, _label) in enumerate(pairs, 1):
            image = cv2.imread(str(image_path))
            t0 = time.perf_counter()
            boxes = detector.boxes(image)
            times.append((time.perf_counter() - t0) * 1000.0)
            all_boxes.append(boxes)
            if i % 50 == 0:
                print(f"    {i}/{len(pairs)}", flush=True)

        rows = [score(all_boxes, truths, c) for c in args.conf]
        median_ms = float(np.median(times))
        report[name] = {"model": str(model), "providers": detector.providers,
                        "median_ms": round(median_ms, 2), "rows": rows}

        print(f"  {'порог':>7}{'точность':>11}{'полнота':>10}{'лишних':>9}"
              f"{'топ-1':>8}{'не та':>8}{'не нашла':>10}{'ложн.тревог':>13}")
        for row in rows:
            print(f"  {row['conf']:>7.2f}{row['precision']:>11.3f}"
                  f"{row['recall']:>10.3f}{row['fp']:>9}"
                  f"{row['top1_hit']:>8}{row['top1_wrong']:>8}"
                  f"{row['top1_empty']:>10}{row['false_alarm']:>13}")
        print(f"  медиана на картинку: {median_ms:.1f} мс\n")

    print("=" * 74)
    print("ЧТО ЭТО ЗНАЧИТ")
    print("=" * 74)
    print("  точность  доля верных среди выданных рамок. Низкая означает")
    print("            ложные рамки: на фарах, наклейках, надписях")
    print("  полнота   доля найденных номеров из размеченных")
    print("  лишних    сколько рамок не попало ни в один размеченный номер")
    print("  Дальше только по картинкам, ГДЕ НОМЕР ЕСТЬ:")
    print("  топ-1     на скольких САМАЯ УВЕРЕННАЯ рамка попала в номер. Для")
    print("            нашего сервиса решает это число: он берёт одну рамку и")
    print("            отдаёт её распознавателю")
    print("  не та     самая уверенная рамка попала не в номер. Это те самые")
    print("            кадры, на которых сквозной прогон даёт чужой номер")
    print("  не нашла  ни одной рамки выше порога, хотя номер на картинке есть")
    print()
    print("  И отдельно по картинкам, ГДЕ НОМЕРА НЕТ:")
    print("  ложн.тревог  модель выдала рамку там, где номера нет. На таких")
    print("               картинках правильный ответ это молчание")
    print()

    # Сравнение по тому числу, которое решает: лучший порог у каждой модели.
    if len(report) >= 2:
        print("Лучший порог по топ-1 у каждой модели:")
        best = {}
        for name, data in report.items():
            row = max(data["rows"], key=lambda r: r["top1_hit"])
            best[name] = row
            with_plate = sum(1 for t in truths if t)
            print(f"  {name:<28} порог {row['conf']:.2f}: "
                  f"{row['top1_hit']} из {with_plate} "
                  f"({row['top1_share']:.3f}), не та рамка {row['top1_wrong']}"
                  f", ложных тревог {row['false_alarm']}")
        names = list(best)
        a, b = best[names[0]], best[names[1]]
        diff = (a["top1_share"] - b["top1_share"]) * 100
        if abs(diff) < 0.5:
            print(f"\n  Разница {abs(diff):.1f} пункта: на этой выборке модели"
                  f" неразличимы.")
        else:
            winner = names[0] if diff > 0 else names[1]
            print(f"\n  Разница {abs(diff):.1f} пункта в пользу {winner}"
                  f" на её лучшем пороге.")
        print("\n  Прежде чем менять модель в сервисе: выигрыш должен держаться")
        print("  и на сквозном прогоне по видео, а не только здесь. Рамка это")
        print("  ещё не номер.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dataset": str(args.dataset), "images": len(pairs), "plates": plates,
        "iou_match": IOU_MATCH, "iou_nms": IOU_NMS, "models": report,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nподробности: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
