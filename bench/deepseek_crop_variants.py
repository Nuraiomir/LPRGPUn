import base64
import os
import re
import time
from pathlib import Path

import cv2
import requests


BASE_URL = os.environ["LPR_GATEWAY_URL"].rstrip("/")
API_KEY = os.environ["LPR_GATEWAY_KEY"]
MODEL = "deepseek-ocr-2/deepseek-ocr-2"

SRC = Path(
    "/home/omirzakn/datasets_ocr_kz/"
    "autoriaNumberplateOcrKz-2019-04-26/test/img"
)

SAMPLES = [
    ("12190610.jpg-0.png", "249AS16"),
    ("12209128.jpg-1.png", "392AD07"),
    ("12476702.jpg-0.png", "553KKA01"),
]


def data_url(img):
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")

    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return "data:image/png;base64," + b64


def ask(session, img):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read the license plate in this image. "
                            "Return ONLY the plate characters. "
                            "Do not describe the image. "
                            "Do not return any other text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url(img)
                        },
                    },
                ],
            }
        ],
    }

    t0 = time.perf_counter()

    r = session.post(
        BASE_URL + "/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    latency = (time.perf_counter() - t0) * 1000

    r.raise_for_status()

    data = r.json()

    text = data["choices"][0]["message"]["content"]

    return str(text).strip(), latency


def normalize(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def variants(img):
    h, w = img.shape[:2]

    scale = max(1.0, 800 / w)

    up = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # мягкий контраст без агрессивного threshold
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)

    return {
        "original": img,
        "upscale": up,
        "upscale_clahe": enhanced,
    }


session = requests.Session()

for filename, expected in SAMPLES:
    path = SRC / filename
    img = cv2.imread(str(path))

    print()
    print("=" * 80)
    print(f"IMAGE: {filename}")
    print(f"GT:    {expected}")
    print(f"SIZE:  {img.shape[1]}x{img.shape[0]}")
    print("=" * 80)

    for name, variant in variants(img).items():
        try:
            raw, latency = ask(session, variant)

            print(
                f"{name:16s} "
                f"-> {raw!r:70s} "
                f"{latency:6.1f} ms "
                f"normalized={normalize(raw)!r}"
            )

        except Exception as e:
            print(f"{name:16s} ERROR: {e}")
