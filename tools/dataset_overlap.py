#!/usr/bin/env python3
"""
Что из размеченных папок уже попало в обучение, а что лежит без дела.

Сравнение по именам файлов ничего не доказывает: при сборке датасета картинки
переименовывают, и одна и та же фотография оказывается под двумя именами.
Поэтому здесь считается хеш содержимого.

Отдельно проверяется утечка. В папке real_all лежат кадры с именами вида
20260923_152319_000023.jpg, то есть вырезанные из видео, на котором мы меряем
сквозное качество. Если такие кадры попали в обучение, цифра "25 машин из 28"
завышена: модель видела эти самые кадры. Это надо знать до того, как число
попадёт в отчёт, а не после.

    python3 tools/dataset_overlap.py \\
        ~/training/lpr_real_v1 ~/mixed_lpr ~/nurai_gpu/real_all ~/photos_raw/Grnz

Первая папка считается обучающим набором, остальные сравниваются с ней.
"""
import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
# Имена кадров, вырезанных из видео: 20260923_152319_000023.jpg
FRAME = re.compile(r"(\d{8}_\d{6})_\d+\.", re.IGNORECASE)


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root):
    """({хеш: [пути]}, сколько файлов посмотрели, что помешало).

    Обход намеренно терпимый к ошибкам: папка может оказаться нечитаемой, а
    битая ссылка выглядит как файл, пока его не откроешь. Но МОЛЧА пропускать
    такое нельзя. Инструмент, который на нечитаемой папке говорит «картинок
    ноль» и дальше печатает вывод, хуже, чем инструмент, который падает: по
    его выводу делают вывод об утечке, а он ничего не смотрел.
    """
    out = {}
    looked = 0
    problems = []
    try:
        entries = sorted(root.rglob("*"))
    except OSError as exc:
        return out, 0, [f"не обошла {root}: {exc}"]

    for p in entries:
        try:
            if not p.is_file():
                continue
        except OSError as exc:
            problems.append(f"{p}: {exc}")
            continue
        looked += 1
        if p.suffix.lower() not in SUFFIXES:
            continue
        try:
            out.setdefault(digest(p), []).append(p)
        except OSError as exc:
            problems.append(f"{p}: {exc}")
    return out, looked, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    args = ap.parse_args()

    roots = [f.expanduser() for f in args.folders]
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        raise SystemExit("нет таких папок: " + ", ".join(str(m) for m in missing))

    print("считаю хеши, это минуту-две...\n")
    maps = {}
    empty = []
    for r in roots:
        maps[r], looked, problems = scan(r)
        kinds = Counter(p.suffix.lower() for paths in maps[r].values()
                        for p in paths)
        print(f"  {r}: {len(maps[r])} уникальных картинок "
              f"(файлов просмотрено {looked})")
        if kinds:
            print("      " + ", ".join(f"{suf or 'без расширения'}: {n}"
                                       for suf, n in kinds.most_common()))
        for problem in problems[:5]:
            print(f"      не прочиталось: {problem}")
        if not maps[r]:
            empty.append((r, looked))

    if empty:
        # Это и есть тот случай, ради которого всё выше. Пустая папка
        # означает, что сравнивать не с чем, и любой вывод об утечке будет
        # выводом ни о чём.
        print("\n" + "=" * 72)
        print("НЕ МОГУ СРАВНИВАТЬ")
        print("=" * 72)
        for r, looked in empty:
            print(f"  {r}: картинок не нашла, а файлов просмотрела {looked}.")
            if looked:
                print("      Файлы есть, но ни один не картинка нужного типа.")
                print(f"      Ищу только: {', '.join(sorted(SUFFIXES))}")
            else:
                print("      Внутри вообще ничего не видно. Проверь путь и")
                print("      права на чтение:")
                print(f"        ls -la {r}")
                print(f"        find {r} -type f | head")
        raise SystemExit("\nПроверка не выполнена. Вывода об утечке не будет: "
                         "по пустой папке его делать нельзя.")

    train = roots[0]
    train_hashes = set(maps[train])

    print("\n" + "=" * 72)
    print(f"ЧТО УЖЕ ЕСТЬ В {train.name}")
    print("=" * 72)
    for r in roots[1:]:
        hs = set(maps[r])
        common = hs & train_hashes
        new = hs - train_hashes
        share = 100 * len(common) / max(len(hs), 1)
        print(f"\n{r}")
        print(f"  всего уникальных   {len(hs)}")
        print(f"  уже в обучении     {len(common)}  ({share:.0f}%)")
        print(f"  НОВЫХ              {len(new)}")
        if new:
            example = sorted(maps[r][h][0].name for h in list(new)[:3])
            print(f"  например: {', '.join(example)}")

    print("\n" + "=" * 72)
    print("УТЕЧКА: кадры из тестовых видео внутри обучающего набора")
    print("=" * 72)
    # Имя внутри обучающего набора могло быть изменено, поэтому ищем по
    # содержимому: берём кадры видео из любой папки и смотрим, есть ли их
    # хеши среди обучающих.
    # Считать надо КАРТИНКИ, а не вхождения. Один и тот же кадр лежит и в
    # real_all под своим именем, и в обучающем наборе под новым, и проход по
    # всем папкам подряд засчитывал его дважды: 150 кадров превращались в 300.
    # Поэтому хеши сначала собираются в множество, и только потом считаются.
    leaked = Counter()
    seen = {}
    for r in roots:
        for h, paths in maps[r].items():
            if h not in train_hashes or h in seen:
                continue
            for p in paths:
                m = FRAME.search(p.name)
                if m:
                    seen[h] = m.group(1)
                    break
    for video in seen.values():
        leaked[video] += 1
    if leaked:
        print("  НАЙДЕНА. Эти кадры есть и в тестовых видео, и в обучении:")
        for video, count in leaked.most_common():
            print(f"    {video}: {count} кадров")
        print("\n  Это значит, что сквозная метрика на этих видео завышена:")
        print("  модель обучалась на тех самых кадрах, на которых её меряют.")
        print("  Такие кадры надо убрать из обучения и переобучить, либо")
        print("  мерить на видео, кадров из которого в обучении нет.")
    else:
        print(f"  Кадров из тестовых видео в {train.name} не нашла.")
        print()
        print("  Это НЕ значит автоматически, что сквозная метрика честная.")
        print("  Значит ровно одно: среди картинок первой папки нет файлов,")
        print("  совпадающих по содержимому с кадрами тестовых видео, чьи")
        print(f"  имена похожи на {FRAME.pattern}.")
        print()
        print("  Проверь сама, что выполнены два условия:")
        print(f"    1. {train.name} это действительно обучающий набор,")
        print("       а не что-то другое: первая папка в команде это обучение.")
        print("    2. Кадры из тестовых видео, если они есть, названы так,")
        print("       чтобы их узнал шаблон выше. Переименованный кадр по")
        print("       содержимому найдётся, а по имени нет, и тогда видео")
        print("       не будет названо.")


if __name__ == "__main__":
    main()
