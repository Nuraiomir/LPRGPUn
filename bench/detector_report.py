#!/usr/bin/env python3
"""
Один отчёт по нескольким детекторам: скорость, метрики обучения, машины.

Собирает в одну таблицу три разные вещи, которые обычно живут порознь и
поэтому их путают:

  скорость модели     чистый session.run, отдельно на CPU и на GPU. Это то
                      число, которое меняется при смене модели
  метрики детектора   точность, полнота и mAP на отложенной выборке, из
                      вывода обучения. Считаются по рамкам
  машины              сколько машин из 28 система назвала на четырёх видео и
                      сколько назвала чужих. Считается по машинам

Третье не выводится из второго. Рамка может быть найдена идеально, а номер не
подтверждён, потому что для подтверждения нужно несколько согласных чтений.
Ровно поэтому строки в таблице не сходятся друг с другом, и это не ошибка.

Модель опознаётся по имени файла, по той же схеме, что в
app/lpr_v19_universal.py: model/best_512.onnx это прогоны без суффикса,
любая другая модель это прогоны с суффиксом __<имя без .onnx>.

    .venv_gpu/bin/python bench/detector_report.py \\
        model/best_512.onnx model/best8n_fair_512.onnx model/best11n_512.onnx

Перед запуском по каждой модели должны быть прогнаны видео:

    for v in videos/*.mp4; do
        LPR_ONNX_MODEL=model/<имя>.onnx .venv_gpu/bin/python \\
            app/lpr_v19_universal.py "$v"
    done
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# CUDA в этом проекте приезжает пакетами pip, и путь к её библиотекам
# собирает config/gpu_env.py. Воркеры получают его через окружение при
# запуске, а этот скрипт запускают руками, и без него onnxruntime не находит
# libcudnn.so.9: провайдер CUDA просто не создаётся, а колонка GPU выходит
# пустой. Динамический загрузчик читает LD_LIBRARY_PATH только при старте
# процесса, поэтому выставить её на ходу нельзя -- надо перезапуститься.
def _reexec_with_cuda_libs():
    import os
    if os.environ.get("LPR_BENCH_REEXEC"):
        return
    try:
        sys.path.insert(0, str(ROOT))
        import config.gpu_env as gpu_env
        ld = gpu_env.CUDA_LD_PATH
    except Exception:
        return
    if not ld or ld in os.environ.get("LD_LIBRARY_PATH", ""):
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LPR_BENCH_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_reexec_with_cuda_libs()


import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

sys.path.insert(0, str(ROOT / "bench"))

from pipeline_metrics import confirmations, f1_of  # noqa: E402

DEFAULT_MODEL = "best_512.onnx"


def params_of(path):
    try:
        import onnx
        m = onnx.load(str(path))
        return sum(int(np.prod(i.dims)) for i in m.graph.initializer)
    except Exception:
        return None


def speed(path, provider, runs, warmup):
    """Медиана чистого session.run в миллисекундах, или None."""
    try:
        sess = ort.InferenceSession(str(path), providers=[provider])
    except Exception:
        return None
    if provider not in sess.get_providers():
        return None
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    data = np.random.default_rng(0).random(shape, dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: data})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: data})
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def training_metrics(out_name):
    """Метрики из прогона обучения, который сделал именно эту модель."""
    for path in sorted((ROOT / "runs").glob("train_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("out_name") == out_name:
            return d
    return None


def vehicles(tag, labels):
    """TP, FP, FN по машинам на всех видео из разметки, или None."""
    tp = fp = fn = 0
    wrong, missing = [], []
    seen_any = False
    for stem, cars in labels.items():
        got = confirmations(stem, tag)
        if got is None:
            continue
        seen_any = True
        wanted = [c["plate"] for c in cars]
        found = set(got)
        tp += sum(1 for p in wanted if p in found)
        for p in wanted:
            if p not in found:
                fn += 1
                missing.append(f"{stem}: {p}")
        for p in sorted(found - set(wanted)):
            fp += 1
            wrong.append(f"{stem}: {p}")
    if not seen_any:
        return None
    return {"tp": tp, "fp": fp, "fn": fn, "wrong": wrong, "missing": missing}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--labels", type=Path, default=ROOT / "bench" / "labels.json")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "detector_report.json")
    args = ap.parse_args()

    labels = {k: v for k, v in
              json.loads(args.labels.read_text(encoding="utf-8")).items()
              if not k.startswith("_")}
    total_cars = sum(len(v) for v in labels.values())

    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in available]
    print(f"onnxruntime {ort.__version__}")
    print(f"замер скорости: {', '.join(providers)}, прогонов {args.runs}")
    print(f"разметка: {total_cars} машин на {len(labels)} видео\n")

    rows = []
    for path in args.models:
        if not path.is_file():
            print(f"нет файла, пропускаю: {path}")
            continue
        print(f"меряю {path.name}...", flush=True)
        tag = "" if path.name == DEFAULT_MODEL else f"__{path.stem}"
        row = {
            "model": path.name,
            "tag": tag,
            "params_m": (lambda n: round(n / 1e6, 2) if n else None)(params_of(path)),
            "size_mb": round(path.stat().st_size / 2**20, 1),
            "train": training_metrics(path.name),
            "vehicles": vehicles(tag, labels),
        }
        for provider in providers:
            short = "gpu" if "CUDA" in provider else "cpu"
            ms = speed(path, provider, args.runs, args.warmup)
            row[short] = round(ms, 2) if ms else None
        rows.append(row)

    if not rows:
        raise SystemExit("ни одной модели не нашлось")

    print("\n" + "=" * 92)
    print("СРАВНЕНИЕ ДЕТЕКТОРОВ")
    print("=" * 92)
    print(f"{'модель':<26}{'млн':>6}{'GPU мс':>9}{'CPU мс':>9}"
          f"{'полнота*':>10}{'mAP50':>8}{'машин':>9}{'чужих':>8}{'F1':>8}")
    print("-" * 92)
    for r in rows:
        t = r["train"] or {}
        v = r["vehicles"]
        if v:
            _, _, f1 = f1_of(v["tp"], v["fp"], v["fn"])
            cars = f"{v['tp']}/{v['tp'] + v['fn']}"
            wrong = str(v["fp"])
            f1s = f"{f1:.3f}"
        else:
            cars, wrong, f1s = "нет", "нет", "нет"
        # Собрано заранее: подстановка с кавычками внутри f-строки работает
        # не во всех версиях Python, а этот отчёт должен считаться везде.
        gpu = "%.2f" % r["gpu"] if r.get("gpu") else "нет"
        cpu = "%.2f" % r["cpu"] if r.get("cpu") else "нет"
        rec = "%.3f" % t["recall"] if t.get("recall") else "-"
        m50 = "%.3f" % t["map50"] if t.get("map50") else "-"
        par = "%.2f" % r["params_m"] if r["params_m"] else "?"
        print(f"{r['model']:<26}{par:>6}{gpu:>9}{cpu:>9}"
              f"{rec:>10}{m50:>8}{cars:>9}{wrong:>8}{f1s:>8}")

    print("\n* полнота и mAP50 относятся к РАМКАМ на отложенной выборке обучения")
    print("  и берутся из вывода обучения. Прочерк значит, что эта модель")
    print("  обучена не этим скриптом и его сводки нет.")
    print("  «машин» и «чужих» считаются по машинам на четырёх видео, и это")
    print("  другая величина: номер может быть найден и не подтверждён.")

    for r in rows:
        v = r["vehicles"]
        if v and v["wrong"]:
            print(f"\n{r['model']}: подтверждены номера, которых в видео нет")
            for w in v["wrong"]:
                print(f"    {w}")
            print("    Это дорогая ошибка: ОСРМ открыла бы чужой залог.")

    for r in rows:
        t = r["train"]
        if t and t.get("starting_weights"):
            print(f"\n{r['model']}: обучена с {t['starting_weights']}, "
                  f"{t.get('epochs')} эпох")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nподробности: {args.out}")
    print(f"\nВыборка {total_cars} машин. Это мало: разница в одну машину двигает")
    print("метрику на несколько процентов. Называй размер выборки рядом с числом.")


if __name__ == "__main__":
    main()
