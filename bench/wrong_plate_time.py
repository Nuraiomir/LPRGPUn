"""Сколько времени на экране был неправильный номер, по уже сохранённым прогонам."""
import json, glob, os, sys

labels = {k: [p["plate"] for p in v] for k, v in json.load(open("bench/labels.json")).items()
          if not k.startswith("_")}

def timeline(path):
    d = json.load(open(path))
    end = max((r["video_time"] for r in d.get("responses", [])), default=0.0)
    ev = [(e["video_time"], e["to"]) for e in d.get("switch_events", [])]
    spans = []
    for i, (t, plate) in enumerate(ev):
        t_end = ev[i + 1][0] if i + 1 < len(ev) else end
        spans.append((plate, t, t_end - t))
    return spans, end

for root in sys.argv[1:]:
    print(f"\n===== {root}")
    print(f"  {'условия':11s} {'видео':16s} {'верно, с':>9s} {'ЧУЖОЙ, с':>9s} {'пусто, с':>9s}  чужие номера")
    tot_ok = tot_bad = 0.0
    for f in sorted(glob.glob(f"{root}/*/*.json")):
        video, variant = os.path.basename(f)[:-5].split("__")
        spans, end = timeline(f)
        ok = sum(d for p, _, d in spans if p in labels[video])
        bad = [(p, d) for p, _, d in spans if p and p not in labels[video]]
        empty = end - ok - sum(d for _, d in bad)
        tot_ok += ok; tot_bad += sum(d for _, d in bad)
        if bad:
            names = ", ".join(f"{p} {d:.1f}с" for p, d in bad)
            print(f"  {variant:11s} {video:16s} {ok:9.1f} {sum(d for _,d in bad):9.1f} {empty:9.1f}  {names}")
    print(f"  ИТОГО правильный номер на экране {tot_ok:.1f} с, чужой {tot_bad:.1f} с")
