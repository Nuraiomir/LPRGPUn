# YOLO8 vs YOLO11 — End-to-End Real Video Evaluation

Evaluation dataset:
- 20260908_150800
- 20260908_150904
- 20260909_171120

The contaminated video 20260923_152319 is excluded from the final comparison.

## Results

| Detector | Vehicles | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO8 fair | 13 | 13 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| YOLO11n | 13 | 10 | 1 | 3 | 0.909 | 0.769 | 0.833 |

## Per-video results

### YOLO8 fair

| Video | Cars | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260908_150800 | 4 | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 20260908_150904 | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| 20260909_171120 | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |

Normal plates: 9/9 confirmed.
Square plates: 4/4 confirmed.
Wrong confirmed plate: none.

### YOLO11n

| Video | Cars | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260908_150800 | 4 | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 |
| 20260908_150904 | 6 | 4 | 0 | 2 | 1.000 | 0.667 | 0.800 |
| 20260909_171120 | 3 | 3 | 1 | 0 | 0.750 | 1.000 | 0.857 |

Normal plates: 7/9 confirmed.
Square plates: 3/4 confirmed.
False positive: 694BPT03.
Missed plates: 205BBG05, 979CBB02, 221ZVZ05.

## Conclusion

On the current clean real-video holdout, YOLO8 fair is clearly stronger end-to-end than YOLO11n.

YOLO8 fair:
- F1 = 1.000
- Recall = 1.000
- Precision = 1.000
- 13/13 vehicles confirmed
- 0 false positives
- 0 false negatives

YOLO11n:
- F1 = 0.833
- Recall = 0.769
- Precision = 0.909
- 10/13 vehicles confirmed
- 1 false positive
- 3 false negatives

Therefore YOLO11n should not replace the current YOLO8 fair production candidate based on this evaluation alone.

The next engineering step is to isolate the detector and OCR effects using controlled A/B experiments, especially for the required YOLO11 + DeepSeek OCR architecture.
