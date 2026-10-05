#!/usr/bin/env python3
"""
Чистое время одного прогона детектора: сколько стоит сама модель.

Зачем отдельный скрипт. В выводе конвейера есть строка вида
"YOLO: 570 calls, 54.2 ms avg", и её легко принять за скорость модели. Это не
она. Туда входит доставка кадра воркеру, разбор JPEG, letterbox, обратное
преобразование координат и ожидание в очереди. Сама модель занимает оттуда
единицы процентов, и сравнивать модели по этому числу бессмысленно: оно почти
не меняется, какую модель ни поставь.

Здесь меряется только `session.run`, на одном и том же подготовленном тензоре,
после прогрева. Это то число, которое меняется при смене модели, и то, к
которому относятся опубликованные сравнения.

CPU и GPU считаются отдельно, потому что ответ у них разный, и это важная
часть вопроса: модель может быть заметно быстрее на процессоре и при этом
одинаковой на видеокарте.

    .venv_gpu/bin/python bench/detector_speed.py model/best_512.onnx model/best11n_512.onnx
    .venv_gpu/bin/python bench/detector_speed.py model/*.onnx --runs 300
"""
import argparse
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

IMGSZ = 512


def params_of(path):
    """Параметров в модели, по весам внутри файла."""
    try:
        import onnx
        m = onnx.load(str(path))
        return sum(int(np.prod(i.dims)) for i in m.graph.initializer)
    except Exception:
        return None


def bench(path, provider, runs, warmup):
    try:
        sess = ort.InferenceSession(str(path), providers=[provider])
    except Exception as e:
        return None, f"{provider} недоступен: {str(e)[:120]}"
    if provider not in sess.get_providers():
        return None, f"{provider} не подхватился, фактически {sess.get_providers()[0]}"

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    # Один и тот же тензор на все прогоны и на обе модели: разница во времени
    # тогда относится к модели, а не к данным.
    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float32)

    for _ in range(warmup):
        sess.run(None, {inp.name: data})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: data})
        times.append((time.perf_counter() - t0) * 1000.0)
    return times, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in available]
    print(f"onnxruntime {ort.__version__}, провайдеры: {', '.join(available)}")
    print(f"прогонов: {args.runs}, прогрев: {args.warmup}\n")

    rows = []
    for path in args.models:
        if not path.is_file():
            print(f"нет файла: {path}")
            continue
        n = params_of(path)
        size_mb = path.stat().st_size / 2**20
        row = {"model": path.name,
               "params_m": round(n / 1e6, 2) if n else None,
               "size_mb": round(size_mb, 1)}
        for provider in providers:
            times, err = bench(path, provider, args.runs, args.warmup)
            short = "gpu" if "CUDA" in provider else "cpu"
            if times is None:
                row[short] = None
                print(f"{path.name:<28} {short}: {err}")
                continue
            times.sort()
            row[short] = round(statistics.median(times), 2)
            row[f"{short}_p95"] = round(times[int(len(times) * 0.95)], 2)
        rows.append(row)

    if not rows:
        return

    print("=" * 76)
    print("ЧИСТОЕ ВРЕМЯ ОДНОГО ПРОГОНА МОДЕЛИ")
    print("=" * 76)
    head = f"{'модель':<30}{'млн параметров':>16}{'МБ':>7}{'GPU, мс':>11}{'CPU, мс':>11}"
    print(head)
    print("-" * 76)
    for r in rows:
        gpu = f"{r['gpu']:.2f}" if r.get("gpu") else "нет"
        cpu = f"{r['cpu']:.2f}" if r.get("cpu") else "нет"
        pm = f"{r['params_m']}" if r["params_m"] else "?"
        print(f"{r['model']:<30}{pm:>16}{r['size_mb']:>7.1f}{gpu:>11}{cpu:>11}")

    if len(rows) == 2 and all(r.get("cpu") for r in rows):
        a, b = rows
        for key, name in (("cpu", "процессоре"), ("gpu", "видеокарте")):
            if not (a.get(key) and b.get(key)):
                continue
            faster, slower = sorted(rows, key=lambda r: r[key])
            ratio = slower[key] / faster[key]
            if ratio < 1.05:
                print(f"\nНа {name} разницы нет: {a[key]:.2f} против {b[key]:.2f} мс.")
            else:
                print(f"\nНа {name} {faster['model']} быстрее в {ratio:.2f} раза "
                      f"({faster[key]:.2f} против {slower[key]:.2f} мс).")

    print("\nЭто время только модели. В конвейере к нему добавляется подготовка")
    print("кадра и передача между процессами, и там разница между моделями")
    print("тонет: смотри строку YOLO в выводе app/lpr_v19_universal.py.")


if __name__ == "__main__":
    main()
