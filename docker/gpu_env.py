"""
GPU environment inside the container. Copied to /app/config/gpu_env.py by the
Dockerfile, in place of the machine-specific file the host uses.

Three differences from config/gpu_env.example.py, and all three are because a
container has no virtual environment:

  the interpreter    there is one Python, and both workers run under it, so
                     sys.executable is the answer rather than a venv path
  the library path   the CUDA libraries still arrive as pip packages
                     (nvidia-*), exactly as they do in .venv_gpu on the host,
                     so the path is discovered the same way, just under the
                     system site-packages
  the project root   always /app

The host's file is not reused because it names .venv_gpu, which does not exist
here, and the failure would appear only when a worker tried to start.
"""
import sys
import sysconfig
from pathlib import Path

PROJECT_ROOT = Path("/app")

# One interpreter for the server and both worker subprocesses.
YOLO_PYTHON = Path(sys.executable)
OCR_PYTHON = Path(sys.executable)

ONNX_MODEL = PROJECT_ROOT / "model" / "best_512.onnx"

# Same list as the host's file: the CUDA pieces ONNX Runtime and Paddle load
# at run time. They ship as pip packages, so the path is built rather than
# hardcoded, and a Python upgrade cannot silently break it.
_NVIDIA_LIBS = ("cuda_runtime", "cuda_nvrtc", "cublas", "cudnn",
                "curand", "cufft", "nvjitlink")


def _nvidia_lib_path() -> str:
    base = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    return ":".join(str(base / name / "lib") for name in _NVIDIA_LIBS)


CUDA_LD_PATH = _nvidia_lib_path()
YOLO_CUDA_LD_PATH = CUDA_LD_PATH
OCR_CUDA_LD_PATH = CUDA_LD_PATH
