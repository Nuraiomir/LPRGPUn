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
  degradations.py        damages frames to imitate hard conditions (--degrade)
bench/
  hard_conditions.py     OCR mode benchmark on degraded video, scored against labels
  labels.json            plates present in each test video
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
.venv_gpu/bin/python app/lpr_api_server.py         # options: --ocr-variants, --port, --plate-hold-sec
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
where a new plate was confirmed; that is when OCRM should search the bank
database.

`confidence` is the detector's confidence that there is a plate in `bbox`.
`ocr_confidence` and `raw_text` describe what OCR read in this frame, before
normalization, and are `null` when OCR did not run (no detection, or a square
detection that was not sampled):

- single-row plate: the read's confidence and text, e.g. `"545BDR05"`
- square plate: the lower of the two row confidences, and both rows as read,
  e.g. `"633 / 02BBT"`

`ocr_confidence` says how sure OCR is about the text it read, not whether the
text is a plate: `"0/BPT05 / KZ67"` can come with 0.80. Whether a plate was
recognized is given by `plate` and `confirmed`.

A confirmed plate that is not read again for 2 seconds is cleared: `plate`
becomes `""` and `confirmed` becomes `false`, with `changed` staying `false`.
OCRM should show vehicle data only while `confirmed` is `true`. Only reads of
the confirmed plate itself keep it on screen, so pointing the camera at the
next car does not extend the previous one. On the reference videos the longest
gap between reads of a plate still in view was 0.7 s. If the same car comes
back after a clear, it is confirmed again with `changed: true`. Change the
timeout with `--plate-hold-sec` (0 disables it). When sending a recorded video
through the server, use `camera_client.py --video-time-voting` so the timeout
is measured in video time rather than in processing time.

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
python3 tests/test_degradations.py            # degraded frames are identical across runs
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

### OCR A/B on `20260908_150904.mp4`

Offline replay of `videos/20260908_150904.mp4` (18.19 s, 59.98 FPS, 1,091
frames) through the same YOLO + OCR + voting pipeline, with `LIVE_MODE=False`
so the frames selected for processing are never dropped.

```text
Video
→ OpenCV
→ every 6th frame
→ YOLO on RTX 4090
→ license plate crop
→ PaddleX PP-OCRv5 on GPU
→ voting and vehicle switching
→ confirmed plate
```

The two modes differ only in the versions of each crop that OCR tries:

| Mode | Crop versions |
|---|---|
| full | original → upscaled → grayscale → detailEnhance |
| no-enhanced | original → upscaled → grayscale |

`detailEnhance` is CPU preprocessing; the PP-OCRv5 inference itself runs on
the GPU in both modes.

| Metric | Full | No-enhanced |
|---|---:|---:|
| Processing time | 21.44 s | 18.91 s |
| YOLO calls | 181 | 181 |
| Detections | 148 | 148 |
| OCR attempts | 118 | 118 |
| OCR calls | 126 | 126 |
| OCR time | 9.83 s | 1.46 s |
| Final plate | 820BAQ02 | 820BAQ02 |
| Square plate | 633BBT02 | 633BBT02 |
| Vehicle switches | 6 | 6 |

Both modes produced the same vehicle sequence, with all six switch timestamps
identical:

```text
979CBB02 → 221ZVZ05 → 202NYM02 → 669BKH02 → 633BBT02 → 820BAQ02
```

Removing `detailEnhance` cut total processing time by 11.8% and OCR time by
85.1%. The full run spent about 3.95 s in `detailEnhance`. That figure is
summed over the OCR calls whose profiling was saved, not all of them, so the
real total is at least that.

What this does and does not show:

- It compares processing time and pipeline behaviour. It is not an accuracy
  benchmark: there is no verified ground truth for this video.
- One video is not enough for a general conclusion about accuracy. Harder
  conditions still need to be tested: low light, glare, motion blur, strong
  perspective, dirt and greater distance.
- The source video is 59.98 FPS, but YOLO processes about every sixth frame.
  This is not 60 FPS recognition.
- With `no-enhanced` the 18.19 s video was processed in 18.91 s, roughly its
  own duration. Processing time is measured from the start of the run, so it
  also includes starting the GPU workers and writing the annotated output
  video.

## Hard-conditions benchmark

The three test videos are daytime footage with large, clear plates, which is
where `detailEnhance` is not expected to help. `bench/hard_conditions.py`
damages each frame before sending it, sends every video through the HTTP
server at 10 fps in both OCR modes, and scores each run against
`bench/labels.json`, the plates actually present in each video.

```bash
.venv_gpu/bin/python bench/hard_conditions.py                  # everything, about 40 min
.venv_gpu/bin/python bench/hard_conditions.py --score-only     # re-print scores of saved runs
```

Degradations, each at levels 1 (mild) to 3 (strong), set for 4K frames:

| Name | Imitates |
|---|---|
| `far` | plate far away: resolution lost, frame keeps its size |
| `blur` | hand shake or motion |
| `dark` | dusk or an underground car park |
| `lowcon` | backlight, haze, a dirty plate |
| `phone` | the frame a phone would send: 1080p, 720p or 480p, JPEG-compressed |
| `darkblur` | dark and blurred at once (level 2 only) |

`phone` sends the smaller frame as is. This matters because the pipeline's
square-plate thresholds are in pixels (`MIN_SQUARE_W = 130`,
`MIN_SQUARE_H = 85`): at 720p some square plates from the test videos fall
below them and would be read as single-row plates.

Per run the script reports plates found, plates missed, wrong plates (a
confirmed plate that is not in the video; OCRM would look up another car) and
server-side processing time per frame. Runs are saved in `runs/hard_conditions/`
and skipped when the script is started again. Inspect how a level looks with
`python3 client/degradations.py <video> --at <seconds> --out <dir>`.

Synthetic degradations are cleaner than real dirt, rain, glare or angle, so
this complements real hard-condition footage rather than replacing it. To add
such a video, put it in `videos/` and list its plates in `bench/labels.json`.

## Mock server for integration

`tools/mock_lpr_server.py` speaks the same HTTP contract but recognizes
nothing: it replays a fixed sequence of responses. It lets the client side be
written and tested before the real service is reachable. One standard-library
file, no GPU and no dependencies:

```bash
python3 tools/mock_lpr_server.py          # listens on 0.0.0.0:8765
```

Each request for a session moves one step along the scenario: nothing
recognized, a plate confirmed (`changed: true`), the same plate held
(`changed: false`), the plate cleared (`confirmed: false`), then a second,
two-row plate. Add `&step=<n>` to ask for one specific step instead, which
makes a request repeatable. `GET /scenario` returns the whole sequence.

Its responses carry the same fields as the real service, including the error
shapes, so client code written against it works against the real one. The
plates it returns are made up; no real plate is ever sent out with it.

`tools/LPR_API.postman_collection.json` imports into Postman and covers every
request and every error code; `tools/sample_frame.jpg` is a frame to attach
(it shows an invented plate). Point the `baseUrl` variable at the real service
when the network access is in place.

## Known limitations

- Not yet tested with a real phone camera over the network.
- No authentication, rate limiting or TLS on the HTTP server.
- One plate per frame: the detector returns only the most confident box.
- `OCR_EVERY_N_DETECTIONS = 3` was tuned for 60 fps video; a live camera
  sending fewer frames may need a lower value.
