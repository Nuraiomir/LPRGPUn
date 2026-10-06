import json
import re
import statistics
from pathlib import Path

AB = Path("runs/ab_yolo11_paddle_vs_deepseek_20260909_171120.json")
REF = Path("runs/real_video_v18_20260909_171120__best11n_512/results_vehicle_switch_GPU.json")

ab = json.loads(AB.read_text())
ref = json.loads(REF.read_text())

FPS = 60.0021

def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())

def median(xs):
    return statistics.median(xs) if xs else None

def p95(xs):
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(len(xs) - 1, int(len(xs) * 0.95))
    return xs[idx]

rows = [r for r in ab["rows"] if r.get("det") is not None]

# ------------------------------------------------------------
# Build reference episodes from switch events
# ------------------------------------------------------------

events = ref["switch_events"]

episodes = []

for i, ev in enumerate(events):
    start_t = float(ev["time"])
    end_t = float(events[i + 1]["time"]) if i + 1 < len(events) else None

    plate = norm(ev["to"])

    episodes.append({
        "idx": i + 1,
        "plate": plate,
        "start_t": start_t,
        "end_t": end_t,
        "start_frame": round(start_t * FPS),
        "end_frame": round(end_t * FPS) if end_t is not None else None,
    })

# ------------------------------------------------------------
# Compare A/B rows against each episode
# ------------------------------------------------------------

def in_episode(frame, ep):
    if frame < ep["start_frame"]:
        return False

    if ep["end_frame"] is not None and frame >= ep["end_frame"]:
        return False

    return True

def get_text(x):
    if not isinstance(x, dict):
        return ""
    return norm(x.get("text", ""))

print()
print("=" * 120)
print("REAL-VIDEO OCR A/B — EPISODE-LEVEL ANALYSIS")
print("=" * 120)
print(f"Video FPS: {FPS:.3f}")
print(f"A/B rows with detection: {len(rows)}")
print()
print("REFERENCE SEQUENCE:")
for ep in episodes:
    end = f"{ep['end_t']:.1f}s" if ep["end_t"] is not None else "END"
    print(
        f"  {ep['idx']}: {ep['plate']:<12} "
        f"{ep['start_t']:.1f}s -> {end}"
    )

print()
print("=" * 120)
print(
    f"{'EP':>3}  {'REFERENCE':<14} {'FRAMES':>6}  "
    f"{'PADDLE MAJORITY':<18} {'MATCH':>6}  "
    f"{'DEEP MAJORITY':<18} {'EMPTY':>6}  "
    f"{'P_MED':>9} {'P95':>9}  {'D_MED':>9} {'D95':>9}"
)
print("=" * 120)

episode_results = []

for ep in episodes:
    ep_rows = [r for r in rows if in_episode(r["frame_id"], ep)]

    paddle_texts = [get_text(r.get("paddle")) for r in ep_rows]
    deep_texts = [get_text(r.get("deepseek")) for r in ep_rows]

    paddle_nonempty = [x for x in paddle_texts if x]
    deep_nonempty = [x for x in deep_texts if x]

    def majority(xs):
        if not xs:
            return ""
        from collections import Counter
        return Counter(xs).most_common(1)[0][0]

    paddle_majority = majority(paddle_nonempty)
    deep_majority = majority(deep_nonempty)

    paddle_exact = sum(x == ep["plate"] for x in paddle_texts)
    deep_exact = sum(x == ep["plate"] for x in deep_texts)

    paddle_lat = [
        float(r["paddle"]["latency_ms"])
        for r in ep_rows
        if isinstance(r.get("paddle"), dict)
        and r["paddle"].get("latency_ms") is not None
    ]

    deep_lat = [
        float(r["deepseek"]["latency_ms"])
        for r in ep_rows
        if isinstance(r.get("deepseek"), dict)
        and r["deepseek"].get("latency_ms") is not None
    ]

    deep_empty = sum(not x for x in deep_texts)

    match = paddle_majority == ep["plate"]
    deep_match = deep_majority == ep["plate"]

    result = {
        "episode": ep["idx"],
        "reference": ep["plate"],
        "start_t": ep["start_t"],
        "end_t": ep["end_t"],
        "frames": len(ep_rows),
        "paddle_majority": paddle_majority,
        "paddle_majority_match": match,
        "paddle_exact_frames": paddle_exact,
        "paddle_nonempty": len(paddle_nonempty),
        "deepseek_majority": deep_majority,
        "deepseek_majority_match": deep_match,
        "deepseek_exact_frames": deep_exact,
        "deepseek_nonempty": len(deep_nonempty),
        "deepseek_empty": deep_empty,
        "paddle_median_ms": median(paddle_lat),
        "paddle_p95_ms": p95(paddle_lat),
        "deepseek_median_ms": median(deep_lat),
        "deepseek_p95_ms": p95(deep_lat),
    }

    episode_results.append(result)

    print(
        f"{ep['idx']:>3}  "
        f"{ep['plate']:<14} "
        f"{len(ep_rows):>6}  "
        f"{paddle_majority or '-':<18} "
        f"{'YES' if match else 'NO':>6}  "
        f"{deep_majority or '-':<18} "
        f"{deep_empty:>6}  "
        f"{median(paddle_lat) if paddle_lat else 0:>8.2f} "
        f"{p95(paddle_lat) if paddle_lat else 0:>8.2f}  "
        f"{median(deep_lat) if deep_lat else 0:>8.2f} "
        f"{p95(deep_lat) if deep_lat else 0:>8.2f}"
    )

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

paddle_episode_correct = sum(
    x["paddle_majority_match"] for x in episode_results
)

deep_episode_correct = sum(
    x["deepseek_majority_match"] for x in episode_results
)

paddle_total_exact = sum(x["paddle_exact_frames"] for x in episode_results)
paddle_total_frames = sum(x["frames"] for x in episode_results)

deep_total_empty = sum(x["deepseek_empty"] for x in episode_results)
deep_total_frames = sum(x["frames"] for x in episode_results)

print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    f"Paddle episode majority: "
    f"{paddle_episode_correct}/{len(episode_results)}"
)

print(
    f"DeepSeek episode majority: "
    f"{deep_episode_correct}/{len(episode_results)}"
)

print(
    f"Paddle exact frame matches inside reference episodes: "
    f"{paddle_total_exact}/{paddle_total_frames}"
    f" ({100*paddle_total_exact/paddle_total_frames:.2f}%)"
    if paddle_total_frames else
    "Paddle exact frame matches: N/A"
)

print(
    f"DeepSeek empty: "
    f"{deep_total_empty}/{deep_total_frames}"
    f" ({100*deep_total_empty/deep_total_frames:.2f}%)"
    if deep_total_frames else
    "DeepSeek empty: N/A"
)

# ------------------------------------------------------------
# Save machine-readable result
# ------------------------------------------------------------

out = Path("runs/ab_episode_analysis_20260909_171120.json")

payload = {
    "type": "real_video_ocr_ab_episode_analysis",
    "video": ab.get("video"),
    "fps": FPS,
    "ab_source": str(AB),
    "reference_source": str(REF),
    "reference_is_independent_ground_truth": False,
    "episodes": episode_results,
    "summary": {
        "paddle_episode_majority_correct": paddle_episode_correct,
        "paddle_episode_count": len(episode_results),
        "deepseek_episode_majority_correct": deep_episode_correct,
        "deepseek_episode_count": len(episode_results),
        "paddle_exact_frame_matches": paddle_total_exact,
        "paddle_episode_frames": paddle_total_frames,
        "deepseek_empty_frames": deep_total_empty,
        "deepseek_episode_frames": deep_total_frames,
    },
}

out.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print()
print(f"Saved: {out}")
