"""
GPU environment configuration for app/lpr_api_server.py.

Copy this file to config/gpu_env.py and adjust if your paths differ.

Paths match the current repository layout on the server:

    nurai/
      app/        lpr_v19_universal.py, lpr_api_server.py, ...
      model/      best_512.onnx
      videos/     test videos
      runs/       offline run outputs
      .venv_kz_gpu/, .venv_paddlex_gpu/

The CUDA library paths below are copied verbatim from the working
configuration on the project's JupyterHub GPU container.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../nurai/

YOLO_PYTHON = PROJECT_ROOT / ".venv_kz_gpu" / "bin" / "python"
OCR_PYTHON = PROJECT_ROOT / ".venv_paddlex_gpu" / "bin" / "python"
ONNX_MODEL = PROJECT_ROOT / "model" / "best_512.onnx"

# Copied from lpr_camera_server.py's CUDA12 / OCR_CUDA constants.
YOLO_CUDA_LD_PATH = ":".join([
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib",
])

OCR_CUDA_LD_PATH = ":".join([
    str(OCR_PYTHON.parent.parent / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib"),
    str(OCR_PYTHON.parent.parent / "lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"),
    str(OCR_PYTHON.parent.parent / "lib/python3.10/site-packages/nvidia/cublas/lib"),
    str(OCR_PYTHON.parent.parent / "lib/python3.10/site-packages/nvidia/cudnn/lib"),
    str(OCR_PYTHON.parent.parent / "lib/python3.10/site-packages/nvidia/curand/lib"),
    str(OCR_PYTHON.parent.parent / "lib/python3.10/site-packages/nvidia/cufft/lib"),
])
