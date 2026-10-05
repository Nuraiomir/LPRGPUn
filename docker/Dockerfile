# The service in one image: HTTP server, both GPU workers, the scanning page,
# the detector, and the invented test vehicles.
#
# Base image: plain Debian Python, not an nvidia/cuda one. The reason is in
# config/gpu_env.example.py: on the GPU host the CUDA libraries already come
# from pip packages (nvidia-cublas, nvidia-cudnn and the rest), not from the
# system, and the venv's library path is built out of them. Reproducing that
# needs no CUDA base image, only the host's NVIDIA driver, which the container
# toolkit passes in at run time. The result is an image that matches what is
# known to work rather than a second, untested way of getting CUDA.
#
# Build:
#     docker build -f docker/Dockerfile -t lpr-service:latest .
#
# Run (the host needs nvidia-container-toolkit):
#     docker run --gpus all -p 8765:8765 -v "$PWD/certs:/certs" lpr-service:latest
#
# This image is the prototype, not a production deployment. In particular it
# runs one process family in one container and keeps no state beyond /certs.
FROM python:3.12-slim

# libgomp1: OpenMP, needed by both ONNX Runtime and Paddle.
# libgl1 and libglib2.0-0: OpenCV links against them even in the headless
#   build, which drops the GUI but not the imaging libraries underneath.
# openssl: the entrypoint makes the self-signed certificate with it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        openssl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first and on their own layer: they are the slow part, and they
# change far less often than the code, so a code edit does not re-download
# several gigabytes of CUDA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The recogniser's weights are fetched by PaddleX on first use. Fetching them
# here means a container can start without reaching the network, which is the
# normal state of a bank's network. If this step fails at build time, the
# image still works: the download simply happens at first start instead, and
# then the container does need the network once.
RUN python -c "\
from paddlex.inference import create_predictor; \
create_predictor('en_PP-OCRv5_mobile_rec', device='cpu'); \
print('модель распознавателя скачана в образ')" \
    || echo "ВНИМАНИЕ: модель не скачалась, она загрузится при первом запуске (нужна сеть)"

# Only what the service runs. The benchmarks, the tests, the videos and the
# offline pipeline stay out: they belong to the machine that measures, not to
# the machine that serves.
COPY app/ app/
COPY workers/ workers/
COPY web/ web/
COPY model/ model/
COPY config/test_ocrm.json config/test_ocrm.json

# In place of the host's machine-specific file. See docker/gpu_env.py.
COPY docker/gpu_env.py config/gpu_env.py

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Where the certificate and the access key live. Declared so that a run
# without -v still keeps them for the life of the container rather than
# making a new certificate on every restart.
VOLUME ["/certs"]

EXPOSE 8765

# GET / is open on purpose, for exactly this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "\
import ssl, urllib.request; \
ctx = ssl._create_unverified_context(); \
urllib.request.urlopen('https://127.0.0.1:8765/', context=ctx, timeout=4)" \
    || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
