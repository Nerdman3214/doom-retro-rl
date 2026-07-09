#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path

src_root = Path("vision_dataset/success_ai_play")
dst_root = Path("vision_dataset/curated_success_keep_fullframes")
dst_root.mkdir(parents=True, exist_ok=True)

# Recent BC-mix/success candidates first.
patterns = [
    "session_ai_success_20260705_*",
    "session_ai_success_20260704_*",
    "session_ai_success_20260703_*",
]

selected = []

def score_session(d):
    actions = d / "actions.csv"
    if not actions.exists():
        return None

    rows = 0
    terminal_lock = 0
    final_wall = 0
    final_health = None

    with actions.open(newline="", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            rows += 1
            source = str(row.get("source", "")).lower()

            if "terminal_use_lock" in source:
                terminal_lock += 1

            if "final_wall_rescue" in source:
                final_wall += 1

            h = row.get("health", "")
            if h not in ("", None):
                try:
                    final_health = float(h)
                except ValueError:
                    pass

    flags = []
    if terminal_lock > 60:
        flags.append("terminal_lock_loop")
    if final_health is not None and final_health < 45:
        flags.append("low_health")
    if rows > 2200:
        flags.append("very_long")

    return {
        "path": d,
        "rows": rows,
        "terminal_lock": terminal_lock,
        "final_wall": final_wall,
        "final_health": final_health,
        "flags": flags,
    }

for pat in patterns:
    for d in sorted(src_root.glob(pat), reverse=True):
        if not d.is_dir():
            continue
        s = score_session(d)
        if not s:
            continue
        if s["flags"]:
            continue
        selected.append(s)

# Keep a manageable full-frame set first.
selected = selected[:80]

manifest = dst_root / "curated_manifest.csv"
with manifest.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "session",
        "source",
        "rows",
        "terminal_lock",
        "final_wall",
        "final_health",
    ])
    w.writeheader()

    for s in selected:
        src = s["path"]
        dst = dst_root / src.name

        print("[curate]", src, "->", dst)

        if not dst.exists():
            shutil.copytree(src, dst)

        w.writerow({
            "session": src.name,
            "source": str(src),
            "rows": s["rows"],
            "terminal_lock": s["terminal_lock"],
            "final_wall": s["final_wall"],
            "final_health": s["final_health"],
        })

print("[curate] selected:", len(selected))
print("[curate] manifest:", manifest)
