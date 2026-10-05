"""
Trains a detector on the project's dataset and exports it to ONNX at 512, so
two detectors can be compared on the same benchmark.

Read this before comparing anything. The deployed detector was not trained the
way this script trains by default. Its run (`training/runs/real_v2/args.yaml`)
says `model: /home/omirzakn/best.pt, epochs: 12`: twelve epochs of fine-tuning
from weights that had already seen plates. Starting instead from `yolo11n.pt`
means starting from COCO, which knows about dogs and traffic lights and
nothing about a Kazakh plate. Same data, same image size, same batch, but a
different starting point, and on 1442 training images the starting point may
matter more than the architecture.

So a run of this script against the deployed model answers "is the model we
have better than this new one" and NOT "is YOLO11n better than YOLOv8n". For
the second question both have to start from the same place: train yolov8n.pt
the same way and compare those two.

The current detector is YOLOv8n: 3.03M parameters, 64 convolutions, no
attention block. YOLO11n is a different architecture (2.6M parameters, C3k2
and C2PSA blocks) that Ultralytics reports as both smaller and slightly more
accurate on COCO. Whether that holds on Kazakh plates is a question for this
script plus bench/, not for a table in someone's slides.

Two rules keep the comparison honest:

  Export at 512. workers/yolo_gpu_worker.py letterboxes every frame to 512
  and multiplies the model's normalized output by 512. An export at another
  size would still run and still return boxes -- wrong ones. This script
  refuses any other value.

  Same data.yaml, same split. Anything else measures the dataset, not the
  architecture.

This needs Ultralytics, which is AGPL-3.0 and lives in requirements-tools.txt,
not in the environment the service runs in. Training and export happen here;
the service only ever loads the exported .onnx. On this machine that
environment is ~/.venv_train, the one that already holds torch; .venv_gpu is
the service's and has no Ultralytics in it.

    ~/.venv_train/bin/python tools/train_yolo11n.py \\
        --data ~/training/lpr_real_v1/data.yaml --epochs 100

Afterwards the detector is compared end to end, on the four reference videos,
against the 25-of-28 baseline:

    for v in videos/*.mp4; do
        LPR_ONNX_MODEL=model/best11n_512.onnx .venv_gpu/bin/python \\
            app/lpr_v19_universal.py "$v"
    done
    .venv_gpu/bin/python bench/pipeline_metrics.py
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMGSZ = 512  # not a default: the worker hardcodes this. See the docstring.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="data.yaml, the same one the current detector was trained on")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="starting weights; downloaded on first use")
    ap.add_argument("--name", default="lpr_yolo11n",
                    help="run name under runs/detect/")
    ap.add_argument("--imgsz", type=int, default=IMGSZ,
                    help=f"kept for visibility; only {IMGSZ} is accepted")
    ap.add_argument("--device", default="0")
    # Each run gets its own .onnx. One shared name let a second run overwrite
    # the first, and the two could then no longer be compared -- the same
    # mistake the benchmark output files already made once.
    ap.add_argument("--out-name", default="best11n_512.onnx",
                    help="file under model/ to write the export to")
    args = ap.parse_args()
    if not args.out_name.endswith(".onnx"):
        raise SystemExit(f"--out-name должен заканчиваться на .onnx: {args.out_name}")

    if args.imgsz != IMGSZ:
        raise SystemExit(
            f"--imgsz {args.imgsz} нельзя: воркер letterbox-ит кадр в {IMGSZ} и "
            f"умножает выход модели на {IMGSZ}. Экспорт другого размера не упадёт, "
            f"а вернёт неправильные рамки, и это будет незаметно.")

    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"нет файла с описанием датасета: {data}")

    from ultralytics import YOLO  # imported late: the error above is cheaper

    print(f"датасет: {data}")
    print(f"старт с весов: {args.weights}")
    print(f"эпох: {args.epochs}, размер: {IMGSZ}, батч: {args.batch}\n")

    t0 = time.perf_counter()
    model = YOLO(args.weights)
    model.train(data=str(data), epochs=args.epochs, imgsz=IMGSZ,
                batch=args.batch, device=args.device, name=args.name,
                project=str(ROOT / "runs" / "detect"))
    train_min = (time.perf_counter() - t0) / 60.0

    # Validation on the split named in data.yaml, with the trained weights
    # reloaded from disk: the same file that is about to be exported.
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(data), imgsz=IMGSZ, device=args.device)

    onnx_tmp = Path(YOLO(str(best)).export(
        format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False))
    target = ROOT / "model" / args.out_name
    if target.exists():
        raise SystemExit(
            f"{target} уже есть. Перезаписать значит потерять модель, с которой "
            f"сравнивают. Задай другое --out-name или убери файл сам.")
    shutil.copy2(onnx_tmp, target)

    summary = {
        "weights": str(best),
        "onnx": str(target),
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": IMGSZ,
        "train_minutes": round(train_min, 1),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }
    summary["starting_weights"] = args.weights
    # The report script matches a training run to its exported model by this
    # field: the run is named by --name, the export by --out-name, and nothing
    # else ties the two together.
    summary["out_name"] = target.name
    out = ROOT / "runs" / f"train_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("ОБУЧЕНИЕ ЗАКОНЧЕНО")
    print("=" * 62)
    print(f"  время обучения   {summary['train_minutes']} мин")
    print(f"  точность         {summary['precision']:.3f}")
    print(f"  полнота          {summary['recall']:.3f}")
    print(f"  mAP50            {summary['map50']:.3f}")
    print(f"  mAP50-95         {summary['map50_95']:.3f}")
    print(f"\n  модель: {target}")
    print(f"  сводка: {out}")
    print("\nЭто метрики ДЕТЕКТОРА на отложенной выборке. Они не говорят,")
    print("сколько машин система назовёт: рамка может найтись, а номер не")
    print("прочитаться. Сквозную цифру даёт следующий шаг:\n")
    print('  for v in videos/*.mp4; do')
    print(f'      LPR_ONNX_MODEL=model/{target.name} .venv_gpu/bin/python \\')
    print('          app/lpr_v19_universal.py "$v"')
    print('  done')
    print(f'  .venv_gpu/bin/python bench/pipeline_metrics.py --detector {target.stem}')
    print("\nПрогоны нового детектора пишутся в отдельные папки, старые числа")
    print("остаются на месте.")


if __name__ == "__main__":
    main()
