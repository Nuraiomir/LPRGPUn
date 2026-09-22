"""
GPU environment for app/lpr_api_server.py.

Copy to config/gpu_env.py (ignored by git) and adjust if your layout differs:

    cp config/gpu_env.example.py config/gpu_env.py

Expected layout:

    nurai_gpu/
      app/          lpr_v19_universal.py, lpr_api_server.py, ...
      workers/      yolo_gpu_worker.py, ocr_gpu_worker.py
      model/        best_512.onnx
      .venv_gpu/    one environment for both workers (onnxruntime-gpu, paddlepaddle-gpu, paddlex)

These values match the settings app/lpr_v19_universal.py uses on the host.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GPU_VENV = PROJECT_ROOT / ".venv_gpu"
YOLO_PYTHON = GPU_VENV / "bin" / "python"
OCR_PYTHON = GPU_VENV / "bin" / "python"
ONNX_MODEL = PROJECT_ROOT / "model" / "best_512.onnx"

# CUDA libraries shipped as pip packages (nvidia-*). The python3.X directory
# is discovered instead of hardcoded, so upgrading Python does not silently
# break the library path.
_NVIDIA_LIBS = ("cuda_runtime", "cuda_nvrtc", "cublas", "cudnn",
                "curand", "cufft", "nvjitlink")


def _nvidia_lib_path(venv: Path) -> str:
    site_packages = sorted(venv.glob("lib/python3.*/site-packages"))
    if not site_packages:
        return ""
    base = site_packages[-1] / "nvidia"
    return ":".join(str(base / name / "lib") for name in _NVIDIA_LIBS)


CUDA_LD_PATH = _nvidia_lib_path(GPU_VENV)

# Both workers run in the same environment, so they share one library path.
YOLO_CUDA_LD_PATH = CUDA_LD_PATH
OCR_CUDA_LD_PATH = CUDA_LD_PATH
