# LPR GPU Pipeline

GPU-accelerated License Plate Recognition (LPR) pipeline for detecting vehicles and recognizing license plate numbers from video.

The project uses:
- YOLO-based detection via ONNX Runtime
- NVIDIA CUDA GPU acceleration
- PaddleX / PP-OCRv5 for license plate OCR
- Temporal voting and vehicle switching logic
- JSON result generation
- Annotated output video generation

## Project Structure

```text
LPRGPUn/
├── app/
│   └── lpr_v19_universal.py
│
├── model/
│   └── best_512.onnx
│
├── videos/
│   ├── 20260908_150800.mp4
│   ├── 20260908_150904.mp4
│   └── 20260909_171120.mp4
│
├── runs/
│   ├── real_video_v18_20260908_150800/
│   │   └── results_vehicle_switch_GPU.json
│   ├── real_video_v18_20260908_150904/
│   │   └── results_vehicle_switch_GPU.json
│   └── real_video_v18_20260909_171120/
│       └── results_vehicle_switch_GPU.json
│
└── .gitignore
```

## Requirements

### Hardware

- NVIDIA GPU
- CUDA-compatible environment
- Linux

### Python environments

The pipeline uses two Python virtual environments:

```text
.venv_kz_gpu
.venv_paddlex_gpu
```

They are expected to be located in the project root:

```text
LPRGPUn/
├── .venv_kz_gpu/
├── .venv_paddlex_gpu/
├── app/
├── model/
├── videos/
└── runs/
```

The virtual environments are machine-specific and are not stored in Git.

## Model

The ONNX detection model is:

```text
model/best_512.onnx
```

The application resolves this model from the project root automatically.

## Running the Pipeline

From the project root:

```bash
cd ~/work/_shared/nurai
```

Run the pipeline with a video:

```bash
.venv_kz_gpu/bin/python \
app/lpr_v19_universal.py \
/home/jovyan/work/_shared/nurai/videos/20260909_171120.mp4
```

You can replace the video path with another video:

```bash
.venv_kz_gpu/bin/python \
app/lpr_v19_universal.py \
/home/jovyan/work/_shared/nurai/videos/20260908_150800.mp4
```

or:

```bash
.venv_kz_gpu/bin/python \
app/lpr_v19_universal.py \
/home/jovyan/work/_shared/nurai/videos/20260908_150904.mp4
```

For videos outside the repository:

```bash
.venv_kz_gpu/bin/python \
app/lpr_v19_universal.py \
/path/to/your/video.mp4
```

Absolute video paths are recommended with the current project configuration.

## Processing Pipeline

```text
Input Video
     │
     ▼
Frame Processing
     │
     ▼
YOLO / ONNX Runtime
     │
     ▼
Vehicle / License Plate Detection
     │
     ▼
OCR Candidate Selection
     │
     ▼
PaddleX / PP-OCRv5
     │
     ▼
Temporal Voting
     │
     ▼
Vehicle Switching Logic
     │
     ▼
Confirmed License Plate
     │
     ├──────────────► JSON Results
     │
     └──────────────► Output Video
```

## GPU Backends

The YOLO stage uses ONNX Runtime with:

```text
CUDAExecutionProvider
CPUExecutionProvider
```

The OCR stage uses PaddleX with GPU acceleration.

At startup, the application reports the detected backend/provider information in the console.

## Output

Results are stored in the project-level:

```text
runs/
```

A typical result directory:

```text
runs/
└── real_video_v18_<video_name>/
    ├── results_vehicle_switch_GPU.json
    └── result_vehicle_switch_GPU.mp4
```

### JSON results

The JSON output contains information about:
- detected license plates
- confirmed readings
- vehicle switching events
- OCR readings
- detection information
- processing/timing profile
- YOLO backend
- OCR backend

### Output video

The pipeline generates an annotated output video for the processed input.

## Test Run

A successful GPU test run was performed on:

```text
Video: 20260909_171120.mp4
FPS: 60
Frames: 931
Duration: ~15.5 seconds
YOLO backend: CUDAExecutionProvider
OCR backend: PaddleX / PP-OCRv5
```

The run confirmed license plate readings and vehicle switching events.

## Performance Example

One GPU test run produced approximately:

```text
YOLO calls: 154
YOLO average time: ~41.9 ms

OCR calls: 108
OCR average time: ~138.8 ms

Writer average time: ~4.1 ms

Total wall time: ~22 seconds
```

Actual performance depends on GPU, video resolution, FPS and processing configuration.

## Input Video Compatibility

The pipeline is not limited to the example videos in the repository.

A different video can be passed as an argument:

```bash
.venv_kz_gpu/bin/python \
app/lpr_v19_universal.py \
/path/to/your/video.mp4
```

Recognition quality depends on:
- license plate visibility
- image resolution
- lighting
- viewing angle
- motion blur
- plate size in the frame
- detector performance

## Main Entry Point

The primary application is:

```text
app/lpr_v19_universal.py
```

Other experimental and server-side files are intentionally not part of the clean repository.

## Repository

GitHub:

https://github.com/Nuraiomir/LPRGPUn
