#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path("vision_dataset/success_ai_play")

good = []
bad = []

for d in sorted(root.glob("session_ai_success_*")):
    actions = d / "actions.csv"
    if not actions.exists():
        continue

    wall_loop = 0
    terminal_use = 0
    rescue_cross = 0
    final_health = None
    rows = 0

    with actions.open(newline="", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            rows += 1
            src = row.get("source", "")
            action = row.get("action", "")

            if "final_wall_cross" in src or "final_wall_rescue_slide" in src:
                wall_loop += 1

            if "terminal_use" in src or "terminal_use_lock" in src or action == "use":
                terminal_use += 1

            if "final_rescue_door_cross" in src:
                rescue_cross += 1

            h = row.get("health")
            if h not in (None, ""):
                try:
                    final_health = float(h)
                except ValueError:
                    pass

    reasons = []

    if wall_loop > 120:
        reasons.append(f"wall_loop={wall_loop}")

    if terminal_use > 180:
        reasons.append(f"terminal_use={terminal_use}")

    if final_health is not None and final_health < 20:
        reasons.append(f"low_health={final_health}")

    if rows > 2200:
        reasons.append(f"very_long_run={rows}")

    if reasons:
        bad.append((d.name, reasons, rows, final_health, wall_loop, terminal_use, rescue_cross))
    else:
        good.append((d.name, rows, final_health, wall_loop, terminal_use, rescue_cross))

print("[quality] good:", len(good))
print("[quality] suspicious/bad:", len(bad))
print()

print("[quality] sample good:")
for item in good[:20]:
    print("GOOD", item)

print()
print("[quality] suspicious/bad:")
for name, reasons, rows, health, wall_loop, terminal_use, rescue_cross in bad[:80]:
    print(
        "BAD?",
        name,
        "reasons=", ",".join(reasons),
        "rows=", rows,
        "health=", health,
        "wall_loop=", wall_loop,
        "terminal_use=", terminal_use,
        "rescue_cross=", rescue_cross,
    )
