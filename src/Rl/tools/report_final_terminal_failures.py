#!/usr/bin/env python3
from pathlib import Path
import re

logs = sorted(Path(".").glob("episode_eval_bc010_*"))

print("[logs found]", len(logs))
for logdir in logs[-10:]:
    print("[logdir]", logdir)

print()
print("[terminal/final issue scan]")

patterns = [
    "not success",
    "terminal_use_lock",
    "terminal_signal=False",
    "final_wall_rescue",
    "final_rescue_door",
]

for logdir in logs[-10:]:
    hits = []
    for p in logdir.rglob("*"):
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue

        for pat in patterns:
            if pat in text:
                hits.append((pat, p))
                break

    if hits:
        print()
        print("==", logdir, "==")
        for pat, p in hits[:20]:
            print("[hit]", pat, p)
