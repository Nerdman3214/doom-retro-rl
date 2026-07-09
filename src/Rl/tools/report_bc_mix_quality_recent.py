#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

root = Path("vision_dataset/success_ai_play")
out_dir = Path("quality_reports")
out_dir.mkdir(exist_ok=True)

patterns = sys.argv[1:] or [
    "session_ai_success_20260703_16*",
    "session_ai_success_20260703_22*",
    "session_ai_success_20260703_23*",
]

sessions = []
seen = set()

for pat in patterns:
    for d in sorted(root.glob(pat)):
        if d.is_dir() and d not in seen:
            seen.add(d)
            sessions.append(d)

rows_out = []

for d in sessions:
    actions_csv = d / "actions.csv"
    if not actions_csv.exists():
        continue

    rows = 0
    bc = 0
    final_wall = 0
    final_rescue = 0
    terminal_lock = 0
    terminal_magnet = 0
    combat = 0
    stuck = 0
    final_health = None

    with actions_csv.open(newline="", errors="replace") as f:
        r = csv.DictReader(f)

        for row in r:
            rows += 1

            source = str(row.get("source", "")).lower()

            if "bc_policy" in source:
                bc += 1

            if "final_wall_rescue_slide" in source:
                final_wall += 1

            if "final_rescue_door" in source:
                final_rescue += 1

            if "terminal_use_lock" in source:
                terminal_lock += 1

            if "terminal_magnet" in source:
                terminal_magnet += 1

            if "combat" in source or "shoot" in source:
                combat += 1

            if "stuck" in source:
                stuck += 1

            h = row.get("health", "")
            if h not in ("", None):
                try:
                    final_health = float(h)
                except ValueError:
                    pass

    flags = []
    notes = []

    if bc == 0:
        notes.append("no_bc_seen_in_actions_csv")

    if final_wall > 120:
        flags.append("heavy_final_wall")

    if terminal_lock > 60:
        flags.append("terminal_lock_loop")

    if final_health is not None and final_health < 35:
        flags.append("low_health")

    if rows > 2200:
        flags.append("very_long")

    quality = "clean" if not flags else "messy"

    rows_out.append({
        "session": d.name,
        "quality": quality,
        "flags": ",".join(flags),
        "notes": ",".join(notes),
        "rows": rows,
        "bc": bc,
        "final_wall": final_wall,
        "final_rescue": final_rescue,
        "terminal_lock": terminal_lock,
        "terminal_magnet": terminal_magnet,
        "combat": combat,
        "stuck": stuck,
        "final_health": "" if final_health is None else final_health,
        "path": str(d),
    })

report = out_dir / "bc_mix010_quality_recent.csv"

with report.open("w", newline="") as f:
    fieldnames = [
        "session",
        "quality",
        "flags",
        "notes",
        "rows",
        "bc",
        "final_wall",
        "final_rescue",
        "terminal_lock",
        "terminal_magnet",
        "combat",
        "stuck",
        "final_health",
        "path",
    ]

    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_out)

clean = [r for r in rows_out if r["quality"] == "clean"]
messy = [r for r in rows_out if r["quality"] != "clean"]

print("[recent_quality] patterns:", ", ".join(patterns))
print("[recent_quality] sessions checked:", len(rows_out))
print("[recent_quality] clean:", len(clean))
print("[recent_quality] messy:", len(messy))
print("[recent_quality] report:", report)

print()
print("[recent_quality] messy sessions:")
for r in messy:
    print(
        "MESSY",
        r["session"],
        "flags=", r["flags"],
        "rows=", r["rows"],
        "health=", r["final_health"],
        "final_wall=", r["final_wall"],
        "terminal_lock=", r["terminal_lock"],
        "path=", r["path"],
    )

print()
print("[recent_quality] newest clean sessions:")
for r in clean[-30:]:
    print(
        "CLEAN",
        r["session"],
        "health=", r["final_health"],
        "rows=", r["rows"],
        "final_wall=", r["final_wall"],
        "terminal_lock=", r["terminal_lock"],
    )
