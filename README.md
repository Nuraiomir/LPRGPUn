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

## Quick start

Everything below is run from the repository root on the GPU host. Each section
further down explains the same steps in more detail.

```bash
cd ~/nurai_gpu

# 1. Once: tell the server where the environment and the model are
cp config/gpu_env.example.py config/gpu_env.py

# 2. Once: an access key and a certificate. The key is compared in constant
#    time and never leaves this file; the certificate is what lets a browser
#    give the page access to the camera. Neither is committed (.gitignore).
mkdir -p config
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > config/api_keys.txt
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -subj "/CN=10.26.13.12" -keyout config/server.key -out config/server.pem

# 3. Start the service (loading the GPU workers takes a few seconds)
.venv_gpu/bin/python app/lpr_api_server.py \
    --cert config/server.pem --key config/server.key \
    --api-keys-file config/api_keys.txt
```

Then, with the key from `config/api_keys.txt`:

| What | How |
|---|---|
| Scan with a phone or laptop camera | open `https://<host>:8765/demo`, paste the key, press "Начать сканирование" |
| Check the service is alive | `curl -sk https://127.0.0.1:8765/` |
| Send one frame by hand | `curl -sk -X POST --data-binary @tools/sample_frame.jpg -H "Content-Type: image/jpeg" -H "Authorization: Bearer <key>" "https://127.0.0.1:8765/frame?session_id=test"` |
| Replay a recorded video as a camera | `.venv_gpu/bin/python client/camera_client.py --video videos/<file>.mp4 --server https://127.0.0.1:8765 --api-key <key> --insecure --session-id cam1 --fps 10` |
| Process a video offline, with an annotated output | `.venv_gpu/bin/python app/lpr_v19_universal.py videos/<file>.mp4` |
| Run the tests (no GPU needed) | `for t in tests/*.py; do .venv_gpu/bin/python $t \| tail -1; done` |
| Give integrators something to build against | `python3 tools/mock_lpr_server.py` (no GPU, no dependencies) |

If the server exits with `Address already in use`, an older instance is still
running: `ss -ltnp | grep 8765`.

## Project structure

Every file in the repository, and what it is for.

**The pipeline itself.**

```text
app/
  lpr_recognizer.py      the rules that decide anything: what counts as a
                         plate, normalization, voting, vehicle switching.
                         Both entry points import this one, so they cannot
                         drift apart
  lpr_api_server.py      HTTP server: POST /frame, one recognizer per
                         session, and it serves the scanning page
  lpr_v19_universal.py   offline pipeline: one video file in, a JSON of
                         everything that happened out. The measuring
                         instrument behind every number below
  gpu_workers_client.py  starts the worker subprocesses, talks to them, and
                         restarts one that dies
  ocrm_stub.py           the invented vehicle lookup standing where OCRM
                         will go
workers/
  yolo_gpu_worker.py     plate detection, its own process
  ocr_gpu_worker.py      text recognition, its own process
web/
  demo.html              the scanning page: camera, aiming guide, plate,
                         vehicle card, field-visit button
config/
  gpu_env.example.py     paths for the server (copy to gpu_env.py)
  test_ocrm.json         invented vehicles for the lookup
model/best_512.onnx      the detector
videos/                  test footage
runs/                    results and benchmark history
```

**Sending something to the service.**

```text
client/
  camera_client.py       replays a video into the server frame by frame, as
                         a camera would
  degradations.py        damages frames on the way to imitate dusk, blur,
                         distance and phone compression
tools/
  mock_lpr_server.py     answers the same API with no GPU and no
                         dependencies, so integrators can build against it
  preflight.py           one command that puts the whole scenario through a
                         real instance and prints a line per check
  make_demo_clip.py      prepares a clip to play or a file to feed Chromium
                         instead of a camera, without ffmpeg
  LPR_API.postman_collection.json
```

**Measuring.** Every number in this README was produced by one of these.

```text
bench/
  labels.json            the plates actually present in each test video.
                         Ground truth for everything else here
  pipeline_metrics.py    the headline score: vehicles found, wrong plates,
                         precision, recall, F1
  hard_conditions.py     the same score over deliberately damaged frames
  missed_plates.py       why one plate was never confirmed: the readings in
                         its window, and which of them were it
  box_shapes.py          the shape of the box each reading came from, which
                         is what "Why a plate is lost" below is built on
  crop_size.py           read rate against the share of the frame the plate
                         fills; the aiming guide is sized from this
  field_misses.py        sorts failed readings by what went wrong
  ocr_dataset_eval.py    the recogniser alone, over a labelled dataset
  letter_stats.py        which characters the recogniser confuses
  letter_repair.py       the repair rule those statistics suggested, kept
                         because the measurement said it does not pay
  vote_margins.py        how close a confirmation came to not happening
  confirm_delay_sim.py   how much sooner a lower threshold would confirm,
                         and what it would cost
  wrong_plate_time.py    how long a wrong plate stays on screen before it
                         is corrected
```

**Preparing training data.** Not part of running the service, and the only
thing here that needs Ultralytics.

```text
tools/
  build_dataset.py       assembles a training set from collected frames
  prepare_frames.py      pulls frames out of video and pre-labels them
  prepare_photos.py      the same for still photographs
```

**Tests.** The suite needs no GPU: the workers are replaced by stand-ins, so
it runs anywhere, and `tools/preflight.py` and GitHub Actions both run it.

```text
tests/                   the suite, run by preflight.py and by CI
bench/compare_ab_offline.py   compares two offline runs (OCR mode A/B)
tools/
  smoke_ocr_worker.py    starts the real OCR worker and times it. Needs the
  smoke_gpu_pipeline.py  GPU, run by hand, deliberately not named test_* so
                         the suite does not try to collect it
```

**Nothing lives in the repository root** except `README.md`, `.gitignore` and
the two requirements files. A tool belongs to `tools/`, a measurement to
`bench/`, a test to `tests/`.

## Environment

One virtual environment in the project root, used by both workers:

```bash
python3.12 -m venv .venv_gpu
.venv_gpu/bin/pip install -r requirements.txt          # to run the service
.venv_gpu/bin/pip install -r requirements-tools.txt    # to run the tests too
```

Tested on an RTX 4090, driver 580.95.05, Python 3.12. The GPU builds expect
CUDA on the machine. The environment itself is machine-specific and not stored
in Git.

The split between the two files is not tidiness. **Ultralytics is AGPL-3.0**,
and it is in the second one: it trained the detector and it exports the model,
but the service loads the exported `.onnx` through ONNX Runtime and never
imports it. So a developer installing the tools and a bank deploying the
service are two different licensing questions. `model/best_512.onnx` carries
`license: AGPL-3.0` in its own metadata, which is the fact to take to a lawyer
rather than an argument about whether a trained model is derivative.

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
python3 bench/compare_ab_offline.py <A.json> <B.json>
```

The script refuses to compare runs made in the same mode or runs in which OCR
calls were lost.

### Trying another recogniser

```bash
# Another recogniser, measured on the same benchmark. PaddleX has to know the
# name; an unknown one fails at worker startup, where the message is plain.
LPR_OCR_MODEL=PP-OCRv6_tiny_rec .venv_gpu/bin/python bench/ocr_dataset_eval.py \
    ~/datasets_ocr_kz/<dataset> --split val --ocr-mode full
```

`LPR_OCR_MODEL` replaces `en_PP-OCRv5_mobile_rec`. It exists so a different
recogniser can be put through `bench/ocr_dataset_eval.py` without editing code,
and so a failed experiment is undone by unsetting a variable. The name is
reported in the worker's `backend` field and in the server's startup banner.

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
| 401 | unauthorized | missing or wrong access key |
| 404 | not_found | any path other than /frame |
| 413 | payload_too_large | body over 10 MB |
| 503 | worker_unavailable | a GPU worker timed out or crashed; it is restarted automatically |
| 500 | internal_error | anything else |

### Vehicle lookup and field visit

Two endpoints complete the scenario after a plate is confirmed. Both need the
same access key.

```text
GET  /vehicle?plate=502ARV02
POST /visit          {"plate": "502ARV02", "note": "..."}
```

`GET /vehicle` answers `200` either way:

```json
{"ok": true, "found": true,  "plate": "545BDR05", "borrower": {...}, "credit": {...},
 "vehicle": {...}, "collateral": {...}, "organization": {...}, "database": "test"}
{"ok": true, "found": false, "plate": "231BED02", "database": "test"}
```

A plate that is not in the database is **not** an error and not a recognition
failure, so it is `found: false` with status 200 rather than a 404. The screen
must say the plate was recognised and simply is not in OCRM. `database` is
`not_configured` when no test file was loaded.

`POST /visit` answers `201` with a `visit_id`. Visits are kept in memory for the
life of the process.

**The vehicle data is invented.** `config/test_ocrm.json` holds made-up
borrowers, credits and vehicles so the whole scenario can be shown end to end;
it is not the bank's OCRM and must never be presented as an integration with
it. Plates are taken from our own test footage, and several plates that appear
in that footage are left out on purpose so the not-found path can be
demonstrated with a real car. Point `--test-ocrm` at another file to change it.

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
(it shows an invented plate). Two variables: point `baseUrl` at the real
service when the network access is in place, and put one line from
`config/api_keys.txt` in `apiKey`. The collection sends that key as a bearer
token on every request, which the mock server ignores and the real service
requires.

## Scanning page for a phone

`web/demo.html` is served by the server itself at `GET /demo`. It opens the
phone's back camera and scans continuously: a frame is taken, sent to
`POST /frame`, and the next one is only taken after the answer arrives, so
requests never queue up. In practice that settles at about 3 frames a second,
which is what the pipeline sustains.

The page shows what an OCRM screen would show: the confirmed plate over the
live picture, the detector's box around what it found, and a marker on the
frame where `changed` is `true` — the moment OCRM would search the bank
database. When the plate is cleared (`confirmed` back to `false`) the page goes
back to "scanning", which is what hiding the vehicle card looks like.

### Aiming guide

The page draws plate-shaped brackets in the middle of the picture and tells the
operator what to change: "point the camera at the plate" when nothing is
detected, "come closer" while the plate is smaller than the brackets ask for,
"hold steady" once it fills them. The brackets follow the detector's box, white
to amber to green.

This exists because of a measurement, not a hunch. `bench/field_misses.py` over
our four videos found 218 failed readings, of which 94 were fragments and 111
had characters lost or doubled, against 6 from a letter the recogniser
confuses. So what we lose, we lose to the crop rather than to the recogniser,
and distance is the one thing about a crop an operator can change.

That is as far as those numbers go. An earlier version of this section read
"almost everything we lose is a crop that arrived too small", which they do not
support: "characters lost" is equally the signature of a crop that is the wrong
shape, and shape had not been measured. It has been since, under "Why a plate
is lost" below, and for the two cars we lose on `20260923_152319` the answer is
the shape. The guide still earns its place for genuinely distant plates; it
would not have saved those two.

The guide is only a hint. The whole frame is still sent, so a plate outside the
brackets is recognised exactly as before. Cropping to the guide on the phone
would cut both traffic and server work, and it can also cut a plate in half, so
it stays out until it is measured.

The two numbers behind it are measured, not guessed. `bench/crop_size.py` over
468 readings on those videos, grouped by the share of the frame width the plate
took: almost nothing reads up to 15%, 41% at 15-20%, 60% at 20-27%, 75% at
27-35%, 81% at 35-50%, 68% above that. The guide is therefore sized for a plate
filling about 42% of the frame, and the hint asks for more below 27%.

The drop above 50% is not a reason to tell the operator to step back. Counting
only genuine one-row plates it shrinks to 90% against 84.5%, about one standard
error.

One limit worth stating before this is quoted: counting only one-row plates,
the success rate barely moves with width at all. Part of what the raw curve
calls "too small" is really "the detector's box came out too tall". That was
written here as a caveat and has since been measured; see below.

### Why a plate is lost

Every reading now records the box it came from (`box_w`, `box_h`,
`box_aspect`), and `bench/box_shapes.py` prints them. Over the 163 single-row
readings of `20260923_152319`, grouped by the shape of that box:

| box width / height | readings | gave a plate |
|---|---|---|
| under 2.0 | 9 | 0 |
| 2.0 to 2.5 | 31 | 14 |
| 2.5 to 3.0 | 34 | 25 |
| 3.0 to 3.5 | 20 | 14 |
| 3.5 to 4.5 | 51 | 47 |
| over 4.5 | 18 | 16 |

A single-row plate is about 4.5 times wider than it is tall, and readings from
a box that shape succeed about nine times in ten. The two cars this video never
confirms, `646BCD02` and `708NIA02`, sat between 1.8 and 2.1 the whole time
they were visible, in boxes around 1800 pixels wide. Nothing was too small. The
detector had taken in bodywork above and below the plate, so the plate filled
under half the crop's height and characters dropped out: `860AXS02` came back
as `860AX02`, `860XS02`, `1860AX02`. Below 1.8 the crop is routed to the
two-row reader as well, which splits one row of text in two: `646BCD02` arrives
there as `646P0` over `BCD/02`.

**Reading the middle band does not fix it, and the attempt is worth recording.**
The obvious answer is to cut a flat box down to 4.5:1 and read that as one more
OCR variant, taking whichever variant is most confident. Measured over the four
videos, that traded away exactly what the pipeline exists to protect:

| | plates found | wrong plates | precision | recall | F1 |
|---|---|---|---|---|---|
| without the band | 25 of 28 | 0 | 1.000 | 0.893 | 0.943 |
| with the band | 26 of 28 | 3 | 0.897 | 0.929 | 0.912 |

One more car found, three cars misidentified: `694BPT03` for `694BPT05`,
`116AEG19` for `776AEG19`, `136JDB04` for `136JDB02`. Each is one or two
characters off a real plate in the same video. The band slices through a
character, the clipped glyph reads as a different digit, and the result still
has a plate's shape, so the pattern accepts it. Worse, the same slice repeats
frame after frame, so the error is consistent and the voting confirms it:
voting protects against occasional errors, not systematic ones. Selecting a
variant by confidence alone cannot help here, because confidence says how sure
OCR is of the characters it read, not whether they are the plate's.

The measurement stands and the fix does not. A tight box is the detector's job,
so this belongs with detector training rather than with a crop heuristic.

A browser only grants a page access to the camera over HTTPS, so the server has
to run with `--cert` and `--key` (see above) for this to work from a phone.
`http://localhost` is the one exception, useful for testing on the server
itself. With a self-signed certificate the browser warns once and the warning
has to be accepted before the camera can start.

```bash
.venv_gpu/bin/python app/lpr_api_server.py \
    --cert config/server.pem --key config/server.key \
    --api-keys-file config/api_keys.txt
# then open https://<server address>:8765/demo on the phone
```

## Before a demo

```bash
.venv_gpu/bin/python tools/preflight.py            # minutes, no video
.venv_gpu/bin/python tools/preflight.py --videos   # adds the four-video run
```

It starts the service on a spare port with the real keys and certificate and
puts the whole scenario through it: the tests, the service starting, the page
being served with the aiming guide and the OCRM card in it, the key being
required, a frame recognised, a plate found, a plate NOT found answering 200
rather than an error, a visit created, and the Postman collection still valid
and still carrying a key. One line per check, and a list of what failed.

### What to show, in order

1. **Start the service.** The banner says it plainly: TLS on, access keys
   loaded, YOLO on CUDA, OCR device GPU, and how many test vehicles. It is the
   shortest proof that the parts are real.
2. **Open `https://localhost:8765/demo` and scan a plate.** A plate on a phone
   screen held up to the camera works. The brackets guide the aim, the hint
   asks for closer, the box turns green, the plate appears, the card follows.
3. **Scan a plate that is not in the database** (`231BED02`, `502ARV02`). The
   screen says the plate was recognised and is not in OCRM. Say out loud that
   this is not a recognition failure: it is the answer, and the two are kept
   apart everywhere, down to the 200 rather than a 404.
4. **Create the field visit.** It comes back with an id.
5. **Show the numbers**, not a claim: 25 vehicles of 28 on our own footage with
   no false plate, precision 1.000, recall 0.893, F1 0.943, and 95.4% on the
   labelled dataset. Say the sample is 28 vehicles.

Two things to say before being asked: the vehicle data is invented, and this
has not been tested from a phone over the network because the port is closed.

### If the camera will not start

A browser only grants camera access over HTTPS, and a self-signed certificate
has to be accepted once in that browser. Failing that, Chromium can play a
video file in place of a camera, which demonstrates the whole flow with real
plates. Chromium wants that file as Y4M, and `ffmpeg` is not installed on this
box, so `tools/make_demo_clip.py` writes it with OpenCV instead:

```bash
.venv_gpu/bin/python tools/make_demo_clip.py videos/20260923_152319.mp4 \
    --y4m /tmp/fakecam.y4m --seconds 12 --width 720

chromium --use-fake-ui-for-media-stream \
         --use-fake-device-for-media-stream \
         --use-file-for-fake-video-capture=/tmp/fakecam.y4m \
         --ignore-certificate-errors "https://localhost:8765/demo"
```

All three flags are needed. Without `--use-fake-device-for-media-stream` the
page gets "Requested device not found" and the picture stays black. Y4M is
uncompressed — about 10 MB per second at 720 wide — so keep `--seconds` small.

### Watching a run's result video

A run writes `result_*.mp4` with the mp4v codec, which no browser plays, and a
minute of it is hundreds of megabytes. The same tool re-encodes a piece of it
into something the browser opens:

```bash
.venv_gpu/bin/python tools/make_demo_clip.py \
    runs/real_video_v18_20260923_152319/result_vehicle_switch_GPU.mp4 \
    --out ~/demo_parking.mp4 --seconds 60 --width 720
```

H.264 is tried first; this box has no H.264 encoder but does have VP8, so the
file usually comes out as `.webm` — the tool prints the name it actually wrote.
Drag that file into a Chromium window to watch it.

Encoding runs at roughly real time: a minute of 60 fps footage takes about a
minute, and the tool prints how far it has got every few seconds. A 300 MB
`result_*.mp4` comes out as a handful of megabytes.

## Known limitations

- Tested with a computer's own camera (362 frames, 2.8/s, 30 ms per frame
  on the server). Not yet tested from a phone over the network.
- The vehicle lookup is a JSON file of invented records, not the bank's OCRM.
- No rate limiting on the HTTP server. Access keys and TLS are in place;
  see "Access keys and TLS".
- One plate per frame: the detector returns only the most confident box.
- `OCR_EVERY_N_DETECTIONS = 3` was tuned for 60 fps video; a live camera
  sending fewer frames may need a lower value.
