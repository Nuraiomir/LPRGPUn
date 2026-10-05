#!/usr/bin/env python3
"""
Достаёт кадры из видео, чтобы их разметить и добавить в обучение.

ЗАЧЕМ. Детектор обучен почти целиком на ФОТОГРАФИЯХ, а работает по КАДРАМ с
камеры. Это разные картинки: на кадре смазанность от движения, другой угол,
другая выдержка, чаще блики. Видеокадров в обучающем наборе было 150, и все
150 оказались кадрами видео, по которому мы же считаем сквозную метрику, то
есть утечкой. После их удаления видеокадров в обучении не осталось вовсе.

Отсюда и разница между моделями: YOLO11n находит больше, но выдаёт ложные
рамки, и ни одна из них не видела при обучении того, с чем работает. Самое
полезное, что можно сделать для обоих детекторов, это добавить в обучение
размеченные видеокадры.

ГЛАВНАЯ ОПАСНОСТЬ, И ПОЧЕМУ ЭТОТ СКРИПТ ОТКАЗЫВАЕТСЯ РАБОТАТЬ.

Кадры из видео, по которым считается сквозная метрика, в обучение попадать
НЕ ДОЛЖНЫ. Если попадут, метрика начнёт измерять запоминание, а не
распознавание, и будет расти, пока продукт не станет хуже. Это уже
случилось один раз: 150 кадров видео 20260923_152319 лежали в обучении, и
сквозные 25 из 28 были завышены.

Поэтому скрипт знает список видео, по которым считается метрика, и
отказывается резать их на кадры. Снять для обучения надо НОВОЕ видео, не то,
на котором проверяемся.

ЗАПУСК.

    python3 tools/extract_frames.py ~/новое_видео.mp4 \\
        --out ~/training/frames_new --every 15

    --every 15   брать каждый 15-й кадр. При 30 кадрах в секунду это два
                 кадра в секунду: соседние кадры почти одинаковы, и
                 размечать их все значит тратить время впустую, а обучение
                 от почти одинаковых картинок не улучшается.

Дальше кадры надо РАЗМЕТИТЬ. Скрипт этого не делает и делать не может:
разметка это рамки, нарисованные человеком, и именно она определяет, чему
модель научится.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

# Видео, по которым считается сквозная метрика. Читается из того же файла,
# что и сама метрика, чтобы список не разошёлся с действительностью: если
# добавят новое проверочное видео, запрет появится сам.
LABELS = ROOT / "bench" / "labels.json"


def scoring_videos():
    """Имена видео (без расширения), по которым считается сквозная метрика."""
    try:
        data = json.loads(LABELS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Нет файла метрики: запрещать нечего, но и молчать нельзя.
        print(f"ВНИМАНИЕ: не прочитала {LABELS}, проверку на утечку сделать "
              f"не могу. Убедись сама, что это видео не используется для "
              f"проверки.", file=sys.stderr)
        return set()
    return {k for k in data if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="папка для кадров. Создаётся, если нет")
    ap.add_argument("--every", type=int, default=15,
                    help="брать каждый N-й кадр (по умолчанию 15)")
    ap.add_argument("--limit", type=int, default=0,
                    help="сколько кадров максимум. 0 = без предела")
    ap.add_argument("--allow-scoring-video", action="store_true",
                    help="НЕ ИСПОЛЬЗОВАТЬ для обучения. Только если кадры "
                         "нужны для разбора ошибок, а не для обучения")
    args = ap.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"нет файла: {video}")

    if video.stem in scoring_videos() and not args.allow_scoring_video:
        raise SystemExit(
            f"\n{video.name} это проверочное видео: по нему считается сквозная\n"
            f"метрика. Кадры из него в обучение попадать не должны, иначе\n"
            f"метрика начнёт измерять запоминание, а не распознавание.\n"
            f"Это уже случалось: 150 кадров такого видео лежали в обучении.\n\n"
            f"Для обучения нужно НОВОЕ видео. Если кадры нужны не для\n"
            f"обучения, а чтобы посмотреть на ошибки, добавь\n"
            f"--allow-scoring-video и не кладите результат в обучающий набор.")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV не открыл видео: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"видео:  {video.name}")
    print(f"кадров: {total}, {fps:.1f} в секунду")
    print(f"беру:   каждый {args.every}-й"
          + (f", примерно {fps / args.every:.1f} в секунду" if fps else ""))
    print(f"в папку: {out}\n")

    saved = index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % args.every == 0:
            # Имя кадра содержит имя видео и номер кадра. По имени потом
            # видно, откуда кадр, и это единственный способ проверить набор
            # на утечку глазами, а не только по содержимому файлов.
            name = f"{video.stem}_f{index:06d}.jpg"
            cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if args.limit and saved >= args.limit:
                break
        index += 1
    capture.release()

    print(f"сохранено кадров: {saved}")
    print("\nЧТО ДАЛЬШЕ")
    print("  1. Разметить кадры: на каждом обвести номерной знак. Без разметки")
    print("     кадры для обучения бесполезны.")
    print("  2. Проверить набор на утечку перед обучением:")
    print("       python3 tools/dataset_overlap.py <обучающий> <проверочный>")
    print("  3. Обучить:")
    print("       python3 tools/train_yolo11n.py --weights yolo11n.pt \\")
    print("           --name lpr_yolo11n_video --out-name yolo11n_video_512.onnx")


if __name__ == "__main__":
    sys.exit(main())
