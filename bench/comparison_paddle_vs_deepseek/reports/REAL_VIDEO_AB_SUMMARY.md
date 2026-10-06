# Real-video A/B: PaddleOCR vs DeepSeek OCR

## Test

Same video, same YOLO11 detector, same detected crops.

Detected OCR crops: 129.

## Result

| Metric | PaddleOCR | DeepSeek OCR |
|---|---:|---:|
| Calls | 129 | 129 |
| Non-empty | 129 | 0 |
| Empty | 0 | 129 |
| Median latency | 3.87 ms | 332.1 ms |
| P95 latency | 5.02 ms | 429.19 ms |

## Reference sequence

545BDR05
→ 633BBT02
→ 694BPT05
→ 694BPT03
→ 694BPT05

## Episode comparison

| Episode | Reference | Paddle majority | DeepSeek |
|---|---|---|---|
| 1 | 545BDR05 | 545BDR05 | empty |
| 2 | 633BBT02 | 026BT | empty |
| 3 | 694BPT05 | 694BPT05 | empty |
| 4 | 694BPT03 | 694BPT05 | empty |
| 5 | 694BPT05 | 694BPT05 | empty |

Paddle majority agreement: 3/5 episodes.

Paddle exact frame matches: 49/128.

DeepSeek empty: 128/128.

## Caveat

The reference sequence comes from the existing pipeline and is not
independent human ground truth. Therefore these are comparison
metrics against the existing pipeline, not independently verified
absolute OCR accuracy.
