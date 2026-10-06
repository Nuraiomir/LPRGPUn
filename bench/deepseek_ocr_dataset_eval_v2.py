#!/usr/bin/env python3

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import requests


def clean_text(text: str) -> str:
    text = str(text or "").upper().strip()

    # убрать типичный префикс страны
    text = re.sub(r"\bKZ\b", "", text)

    # оставить только латиницу и цифры
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


def find_samples(root: Path, split: str):
    ann_dir = root / split / "ann"
    img_dir = root / split / "img"

    samples = []

    for ann_path in sorted(ann_dir.glob("*.json")):
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        label = clean_text(data.get("description", ""))

        if not label:
            continue

        stem = ann_path.stem

        candidates = [
            img_dir / stem,
            img_dir / f"{stem}.jpg",
            img_dir / f"{stem}.jpeg",
            img_dir / f"{stem}.png",
            img_dir / f"{stem}.webp",
        ]

        image_path = next((p for p in candidates if p.exists()), None)

        if image_path is None:
            matches = list(img_dir.glob(stem + ".*"))
            if matches:
                image_path = matches[0]

        if image_path is not None:
            samples.append((image_path, label))

    return samples


def image_to_data_url(path: Path) -> str:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")

    return f"data:{mime};base64,{encoded}"


def call_deepseek(session, base_url, api_key, model, image_path, prompt):
    url = base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path)
                        },
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()

    response = session.post(
        url,
        headers=headers,
        json=payload,
        timeout=120,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    response.raise_for_status()

    data = response.json()

    text = ""

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        pass

    return str(text or ""), latency_ms


def is_valid_kz_plate(text: str) -> bool:
    # Новый формат: 3 цифры + 3 буквы + 2 цифры
    if re.fullmatch(r"\d{3}[A-Z]{3}\d{2}", text):
        return True

    # Старый формат: 3 цифры + 2 буквы + 2 цифры
    if re.fullmatch(r"\d{3}[A-Z]{2}\d{2}", text):
        return True

    return False


def extract_plate(text: str):
    text = clean_text(text)

    # Сначала новый формат
    matches = re.findall(r"\d{3}[A-Z]{3}\d{2}", text)
    if len(matches) == 1:
        return matches[0]

    # Затем старый формат
    matches = re.findall(r"\d{3}[A-Z]{2}\d{2}", text)
    if len(matches) == 1:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LPR_GATEWAY_URL", ""),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LPR_GATEWAY_KEY", ""),
    )
    parser.add_argument(
        "--model",
        default="deepseek-ocr-2/deepseek-ocr-2",
    )
    parser.add_argument(
        "--prompt",
        default="free_ocr",
        choices=["free_ocr", "plate"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output",
        default="runs/deepseek_ocr_test_results.json",
    )

    args = parser.parse_args()

    if not args.base_url:
        raise SystemExit("ERROR: LPR_GATEWAY_URL is not set")

    if not args.api_key:
        raise SystemExit("ERROR: LPR_GATEWAY_KEY is not set")

    root = Path(args.dataset)

    samples = find_samples(root, args.split)

    if args.limit:
        samples = samples[:args.limit]

    print(f"Dataset: {root}")
    print(f"Split:   {args.split}")
    print(f"Samples: {len(samples)}")
    print(f"Model:   {args.model}")
    print(f"Prompt:  {args.prompt}")
    print()

    prompts = {
        "free_ocr": (
            "Read the text on this license plate image. "
            "Return only the license plate text. "
            "Do not add explanations."
        ),
        "plate": (
            "Read the license plate. "
            "Return only the license plate number. "
            "Do not add any explanation or other text."
        ),
    }

    session = requests.Session()

    results = []

    exact_raw = 0
    exact_normalized = 0
    exact_extracted = 0

    valid_gt = 0
    valid_gt_correct = 0

    total_latency = 0.0

    for i, (image_path, expected) in enumerate(samples, 1):
        print(
            f"[{i:03d}/{len(samples):03d}] "
            f"{image_path.name} "
            f"GT={expected}",
            end=" ",
            flush=True,
        )

        try:
            raw, latency_ms = call_deepseek(
                session=session,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                image_path=image_path,
                prompt=prompts[args.prompt],
            )

            normalized = clean_text(raw)
            extracted = extract_plate(raw)

            if raw.strip() == expected:
                exact_raw += 1

            if normalized == expected:
                exact_normalized += 1

            if extracted == expected:
                exact_extracted += 1

            if is_valid_kz_plate(expected):
                valid_gt += 1

                if extracted == expected:
                    valid_gt_correct += 1

            total_latency += latency_ms

            correct = extracted == expected

            print(
                f"-> {extracted or normalized or '<empty>'} "
                f"{'OK' if correct else 'FAIL'} "
                f"{latency_ms:.0f} ms"
            )

            results.append(
                {
                    "image": str(image_path),
                    "expected": expected,
                    "raw": raw,
                    "normalized": normalized,
                    "extracted": extracted,
                    "correct": correct,
                    "valid_gt_format": is_valid_kz_plate(expected),
                    "latency_ms": round(latency_ms, 1),
                }
            )

        except Exception as e:
            print(f"ERROR: {e}")

            results.append(
                {
                    "image": str(image_path),
                    "expected": expected,
                    "raw": "",
                    "normalized": "",
                    "extracted": None,
                    "correct": False,
                    "valid_gt_format": is_valid_kz_plate(expected),
                    "latency_ms": None,
                    "error": repr(e),
                }
            )

    n = len(samples)

    avg_latency = total_latency / n if n else 0

    print()
    print("=" * 70)
    print("DEEPSEEK OCR DATASET RESULTS")
    print("=" * 70)

    print(f"Samples:              {n}")
    print(f"Raw exact:            {exact_raw}/{n} "
          f"({exact_raw / n * 100:.1f}%)" if n else "Raw exact: 0")

    print(f"Normalized exact:     {exact_normalized}/{n} "
          f"({exact_normalized / n * 100:.1f}%)" if n else "Normalized exact: 0")

    print(f"Extracted exact:      {exact_extracted}/{n} "
          f"({exact_extracted / n * 100:.1f}%)" if n else "Extracted exact: 0")

    print()

    print(f"Valid KZ GT:          {valid_gt}/{n} "
          f"({valid_gt / n * 100:.1f}%)" if n else "Valid KZ GT: 0")

    if valid_gt:
        print(
            f"Valid KZ exact:       {valid_gt_correct}/{valid_gt} "
            f"({valid_gt_correct / valid_gt * 100:.1f}%)"
        )

    print()

    print(f"Average latency:      {avg_latency:.1f} ms")
    print(f"Total time:           {total_latency / 1000:.1f} s")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": str(root),
        "split": args.split,
        "model": args.model,
        "prompt": args.prompt,
        "samples": n,
        "raw_exact": exact_raw,
        "normalized_exact": exact_normalized,
        "extracted_exact": exact_extracted,
        "valid_gt": valid_gt,
        "valid_gt_correct": valid_gt_correct,
        "avg_latency_ms": round(avg_latency, 2),
        "total_time_sec": round(total_latency / 1000, 2),
        "results": results,
    }

    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
