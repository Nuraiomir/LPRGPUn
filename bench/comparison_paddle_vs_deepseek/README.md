# PaddleOCR vs DeepSeek OCR — Comparison Evidence

This directory contains the raw benchmark results, analysis results,
reference pipeline output, and benchmark scripts used to compare:

- PaddleOCR PP-OCRv5 Mobile
- DeepSeek OCR 2

The purpose is to preserve reproducible evidence from the OCR comparison.

---

## 1. Main conclusion from the real-video A/B test

The main real-video A/B test used:

- the same input video;
- the same YOLO11 detector;
- the same detected plate crops;
- the same 129 detected frames;
- PaddleOCR and DeepSeek OCR were evaluated on the same crops.

Result:

| Metric | PaddleOCR | DeepSeek OCR |
|---|---:|---:|
| OCR calls | 129 | 129 |
| Non-empty results | 129 | 0 |
| Empty results | 0 | 129 |
| Median latency | ~3.9 ms | ~332 ms |
| P95 latency | ~4–5 ms | ~320–432 ms |

The episode-level comparison against the existing pipeline reference sequence:

| Episode | Reference | Paddle majority | DeepSeek |
|---|---|---|---|
| 1 | 545BDR05 | 545BDR05 | empty |
| 2 | 633BBT02 | 026BT | empty |
| 3 | 694BPT05 | 694BPT05 | empty |
| 4 | 694BPT03 | 694BPT05 | empty |
| 5 | 694BPT05 | 694BPT05 | empty |

Paddle majority matched 3/5 reference episodes.

Paddle exact frame matches inside reference episodes:
49/128 = 38.28%.

DeepSeek empty:
128/128 = 100%.

---

## 2. Important methodological limitation

The reference sequence is produced by the existing OCR/pipeline:

545BDR05
→ 633BBT02
→ 694BPT05
→ 694BPT03
→ 694BPT05

It is NOT an independently created human ground-truth annotation.

Therefore:

"3/5 episodes" should be described as agreement with the existing
pipeline reference sequence, not as absolute OCR accuracy.

Likewise, 49/128 is not an independently validated OCR accuracy metric.

---

## 3. Raw A/B benchmark

File:

json/ab_yolo11_paddle_vs_deepseek_20260909_171120.json

This is the frame-level A/B result.

Both OCR engines received the same YOLO11 crops.

The benchmark showed:

Paddle:
- 129/129 non-empty
- median latency ~3.87 ms
- P95 ~5.02 ms

DeepSeek:
- 0/129 non-empty
- 129/129 empty
- median latency ~332.1 ms
- P95 ~429.19 ms

Important:
the YOLO worker in this benchmark reported CPUExecutionProvider.
Therefore this benchmark must NOT be used to claim GPU YOLO inference latency.

The OCR comparison itself remains comparable because both OCR engines
received the same crops.

---

## 4. Episode-level analysis

File:

json/ab_episode_analysis_20260909_171120.json

Script:

scripts/analyze_ab_episode.py

The analysis maps the A/B frames to the time intervals defined by
the existing pipeline's switch events.

Reference transitions:

0.1 s:
"" -> 545BDR05

7.7 s:
545BDR05 -> 633BBT02

11.4 s:
633BBT02 -> 694BPT05

11.8 s:
694BPT05 -> 694BPT03

12.1 s:
694BPT03 -> 694BPT05

---

## 5. Full DeepSeek dataset test

File:

json/deepseek_ocr_test_results.json

This contains the larger DeepSeek OCR dataset test.

Previously observed results:

279 test crops.

Raw exact:
2/279 = 0.7%

Normalized exact:
8/279 = 2.9%

Extracted exact:
15/279 = 5.4%

Valid KZ ground-truth samples:
263/279 = 94.3%

Exact among valid GT:
15/263 = 5.7%

Average latency:
~145 ms

Empty outputs:
193/279 = 69.2%

These results are dataset-level DeepSeek results and should be
kept separate from the real-video A/B test.

---

## 6. Comparable 100-image tests

Files:

json/ocr_en_PP-OCRv5_mobile_rec_gw-deepseek-ocr-2_test100.json
json/ocr_gw-deepseek-ocr-2-plate_test100.json
json/ocr_gw-deepseek-ocr-2-up1024_test100.json

Previously observed comparison:

Paddle:
- correct: 71/100
- exact: 83/100
- median latency: ~4.1 ms

DeepSeek x1024:
- correct: 43/100
- exact: 21/100
- median latency: ~347.7 ms

DeepSeek plate prompt:
- correct: 6/100
- exact: 2/100
- median latency: ~149.1 ms

The exact image IDs used in these historical 100-image runs were not
independently re-verified as identical. Therefore these numbers should
not be presented as a strict paired-image A/B accuracy comparison.

---

## 7. DeepSeek preprocessing experiment

The DeepSeek preprocessing pilot tested:

- original;
- upscaled;
- CLAHE.

On the tested pilot set the result was:

0/9 exact for each tested preprocessing path.

This does not prove preprocessing can never help; it only describes
the tested samples.

---

## 8. Reference pipeline

File:

json/reference_results_vehicle_switch_GPU.json

This is the output of the existing real-video vehicle-switch pipeline.

It is included as a reference for:

- detected OCR sequence;
- switch events;
- timing;
- OCR confidence;
- OCR latency;
- normal readings.

It is not independent ground truth.

---

## 9. Scripts

The scripts used for the comparison are preserved under:

scripts/

They are included to make the benchmark process reproducible.

---

## 10. Interpretation

The strongest directly comparable result in this collection is the
real-video A/B test on the same YOLO11 crops.

For that test:

- Paddle produced a non-empty OCR result on every detected crop.
- DeepSeek produced an empty result on every detected crop.
- Paddle was substantially faster.
- DeepSeek was substantially slower.

This result supports the conclusion that, in this tested real-video
scenario, DeepSeek OCR 2 was not a viable drop-in replacement for
the current PaddleOCR OCR stage.

The result should not be generalized to all possible DeepSeek OCR
configurations, models, prompts, image formats, or datasets.
