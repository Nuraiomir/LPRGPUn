# LPR GPU Pipeline

License plate recognition for Kazakhstan plates, running on an NVIDIA GPU.
Built as the backend for a mobile OCRM scenario: an employee points a phone
camera at a car, the plate is recognized and confirmed, and OCRM looks it up
in the bank database. This service only recognizes plates; it never sees or
returns customer data.

- Detection: YOLO (`model/best_512.onnx`) on ONNX Runtime with CUDA
- Text recognition: PaddleX PP-OCRv5 (`en_PP-OCRv5_mobile_rec`) on the GPU
- Single-row and two-row (square) plates
- Temporal voting across frames and automatic switching between vehicles

## Project structure

```text
app/
  lpr_recognizer.py      recognition rules: normalization, voting, vehicle switching
  lpr_v19_universal.py   offline pipeline: processes a video file
  lpr_api_server.py      HTTP server: POST /frame, one recognizer per session
  gpu_workers_client.py  starts and talks to the GPU worker processes
workers/
  yolo_gpu_worker.py     plate detection subprocess
  ocr_gpu_worker.py      text recognition subprocess
client/
  camera_client.py       sends a video to the server frame by frame, like a camera
config/
  gpu_env.example.py     paths for the HTTP server (copy to gpu_env.py)
model/best_512.onnx
videos/                  test videos
runs/                    results and benchmark history
tests/
compare_ab_offline.py    compares two offline runs (OCR mode A/B)
```

Both entry points use the same rules from `app/lpr_recognizer.py`.

## Environment

One virtual environment in the project root, used by both workers:

```text
.venv_gpu/   Python 3.12, onnxruntime-gpu 1.26.0, paddlepaddle-gpu 3.3.1, paddlex 3.3.13
```

Tested on an RTX 4090, driver 580.95.05. The environment is machine-specific
and not stored in Git.

## Offline pipeline

```bash
.venv_gpu/bin/python app/lpr_v19_universal.py videos/20260909_171120.mp4
```

Paths are relative to the project root. Without an argument the reference
video above is used. Results go to `runs/real_video_v18_<video>/`:
`results_vehicle_switch_GPU.json` and an annotated `result_vehicle_switch_GPU.mp4`.

### OCR mode

```bash
export OCR_VARIANT_MODE=full          # default
export OCR_VARIANT_MODE=no-enhanced   # skips cv2.detailEnhance
```

The OCR worker tries each crop as original, 2x upscaled, grayscale, and
finally `cv2.detailEnhance`, stopping as soon as confidence reaches 0.92.
`detailEnhance` runs on the CPU and is by far the slowest step. The mode is
recorded in the result JSON (`ocr_variant_mode`).

Compare two runs:

```bash
python3 compare_ab_offline.py <A.json> <B.json>
```

The script refuses to compare runs made in the same mode or runs in which OCR
calls were lost.

## HTTP server

```bash
cp config/gpu_env.example.py config/gpu_env.py     # once
.venv_gpu/bin/python app/lpr_api_server.py         # options: --ocr-variants no-enhanced, --port 8765
```

Send a video through it as if it were a camera:

```bash
.venv_gpu/bin/python client/camera_client.py \
    --video videos/20260909_171120.mp4 --server http://127.0.0.1:8765 \
    --session-id cam1 --fps 10
```

### API

```text
POST /frame?session_id=<id>
Content-Type: image/jpeg
<body: the JPEG frame as raw bytes>
```

`session_id`: 1 to 128 characters, letters, digits and `. _ : @ -`. Each
session has its own recognition state. Frames of one session are processed one
at a time; different sessions run in parallel. Sessions idle for 30 minutes are
removed.

Response:

```json
{
  "ok": true,
  "plate": "502ARV02",
  "confirmed": true,
  "changed": true,
  "bbox": [120, 250, 480, 330, 0.98],
  "confidence": 0.98,
  "ocr_confidence": 0.99,
  "plate_type": "normal",
  "raw_text": "502ARV02",
  "session_id": "cam1",
  "processing_time_ms": 42.1,
  "frame_size": [1920, 1080],
  "ocr_variant_mode": "full"
}
```

`plate` is the currently confirmed plate. `changed` is true only on the frame
where the confirmed plate changed; that is when OCRM should search the bank
database. `confidence` is the detector's, `ocr_confidence` the text reader's.

Errors return `{"ok": false, "error": "...", "error_code": "..."}`:

| Status | error_code | When |
|---|---|---|
| 400 | bad_request | missing or invalid Content-Length, session_id or `t` |
| 400 | decode_failed | body is not a decodable image |
| 404 | not_found | any path other than /frame |
| 413 | payload_too_large | body over 10 MB |
| 503 | worker_unavailable | a GPU worker timed out or crashed; it is restarted automatically |
| 500 | internal_error | anything else |

`GET /` is a health check: session count, OCR mode, worker restarts.

Optional query parameters: `profile=1` adds a per-stage timing breakdown,
`t=<seconds>` sets the voting timestamp (for offline A/B runs only).

## Tests

No GPU needed; the workers are replaced by stand-ins.

```bash
python3 tests/test_lpr_recognizer.py          # voting and switching on recorded OCR output
python3 tests/test_ocr_worker_static.py       # OCR worker code is well-formed
python3 tests/test_gpu_workers_client.py      # worker timeouts, crashes, restarts
python3 tests/test_api_server_limits.py       # size limit, validation, session locking
python3 tests/test_api_server_integration.py  # full HTTP path with camera_client
```

## Measured results

Offline pipeline on the host GPU, three daytime videos. The plate sequences are
pipeline output, not verified ground truth.

| Video | Plates | full | no-enhanced |
|---|---|---|---|
| 20260909_171120 | 545BDR05, 633BBT02, 694BPT05 | 26.4 s | 16.5 s |
| 20260908_150800 | 822AKH02, 049BXS02, 205BBG05, 502ARV02 | 16.0 s | 13.6 s |
| 20260908_150904 | 979CBB02, 221ZVZ05, 202NYM02, 669BKH02, 633BBT02, 820BAQ02 | 21.4 s | 18.9 s |

In both modes all 13 plates and every switch time were identical. OCR time
per call dropped by 85-91% without `detailEnhance`. The default is still
`full`: these videos contain almost no hard conditions (dirt, strong angle,
dusk), which is exactly where `detailEnhance` is meant to help.

With `no-enhanced`, YOLO becomes the largest cost: about 44 ms per frame in
the pipeline against about 1.3 ms of pure model inference. Most of that time
is spent outside the GPU.

## Known limitations

- Not yet tested with a real phone camera over the network.
- No authentication, rate limiting or TLS on the HTTP server.
- One plate per frame: the detector returns only the most confident box.
- The confirmed plate stays in the response until a new one is confirmed, so
  right after the camera moves away the previous car's plate is still shown.
- `OCR_EVERY_N_DETECTIONS = 3` was tuned for 60 fps video; a live camera
  sending fewer frames may need a lower value.
