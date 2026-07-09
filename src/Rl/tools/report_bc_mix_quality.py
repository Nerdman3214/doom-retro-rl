#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path("vision_dataset/success_ai_play")
out_dir = Path("quality_reports")
out_dir.mkdir(exist_ok=True)

sessions = sorted(root.glob("session_ai_success_20260703_16*"))

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
            action = str(row.get("action", row.get("active", ""))).lower()

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

    if bc == 0:
        flags.append("no_bc")

    if final_wall > 120:
        flags.append("heavy_final_wall")

    if terminal_lock > 60:
        flags.append("terminal_lock_loop")

    if final_health is not None and final_health < 35:
        flags.append("low_health")

    if rows > 2200:
        flags.append("very_long")

    quality = "clean"
    if flags:
        quality = "messy"

    rows_out.append({
        "session": d.name,
        "quality": quality,
        "flags": ",".join(flags),
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

report = out_dir / "bc_mix010_quality_report.csv"

with report.open("w", newline="") as f:
    fieldnames = [
        "session",
        "quality",
        "flags",
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

print("[bc_quality] sessions checked:", len(rows_out))
print("[bc_quality] clean:", len(clean))
print("[bc_quality] messy:", len(messy))
print("[bc_quality] report:", report)

print()
print("[bc_quality] clean sessions:")
for r in clean[:80]:
    print(
        "CLEAN",
        r["session"],
        "bc=", r["bc"],
        "final_wall=", r["final_wall"],
        "terminal_lock=", r["terminal_lock"],
        "health=", r["final_health"],
        "rows=", r["rows"],
    )

print()
print("[bc_quality] messy sessions:")
for r in messy[:80]:
    print(
        "MESSY",
        r["session"],
        "flags=", r["flags"],
        "bc=", r["bc"],
        "final_wall=", r["final_wall"],
        "terminal_lock=", r["terminal_lock"],
        "health=", r["final_health"],
        "rows=", r["rows"],
    )
