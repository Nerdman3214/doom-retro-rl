#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
from pathlib import Path


def parse_log(log_path):
    text = log_path.read_text(errors="replace")
    max_wp = 0
    total_wp = 0
    final_health = None
    last_source = ""
    final_wall_stuck = False

    for line in text.splitlines():
        m = re.search(r"wp=(\d+)/(\d+)", line)
        if m:
            max_wp = max(max_wp, int(m.group(1)))
            total_wp = max(total_wp, int(m.group(2)))

        h = re.search(r"health=([0-9.]+)", line)
        if h:
            final_health = float(h.group(1))

        src = re.search(r"src=([^ ]+)", line)
        if src:
            last_source = src.group(1)

        if "final_wall_cross" in line and "stuck=True" in line:
            final_wall_stuck = True

    terminal_signal = any(token in text for token in [
        "ghost_terminal_face",
        "ghost_terminal_use",
        "ghost_final_use",
        "goal_done",
        "goal_wait_after_use",
        "goal_priority_wait",
    ])

    reached_end = bool(total_wp and max_wp >= total_wp - 2)
    alive = final_health is None or final_health > 0

    completed = terminal_signal and reached_end and alive

    return {
        "episode": log_path.stem.replace("episode_", ""),
        "log": str(log_path),
        "max_wp": max_wp,
        "total_wp": total_wp,
        "completed": int(completed),
        "terminal_signal": int(terminal_signal),
        "final_wall_stuck": int(final_wall_stuck),
        "final_health": final_health,
        "last_source": last_source,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--out-dir", default="vision_dataset/success_ai_logs")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_root = Path(args.out_dir) / eval_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    logs = sorted(eval_dir.glob("episode_*.log"))
    rows = [parse_log(p) for p in logs]

    success_rows = [r for r in rows if r["completed"]]
    fail_rows = [r for r in rows if not r["completed"]]

    manifest = out_root / "success_manifest.csv"
    all_manifest = out_root / "all_eval_manifest.csv"

    fieldnames = [
        "episode", "completed", "max_wp", "total_wp",
        "terminal_signal", "final_wall_stuck",
        "final_health", "last_source", "log",
    ]

    with all_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(success_rows)

    copied_dir = out_root / "logs"
    copied_dir.mkdir(exist_ok=True)

    for r in success_rows:
        src = Path(r["log"])
        if src.exists():
            shutil.copy2(src, copied_dir / src.name)

    total = len(rows)
    wins = len(success_rows)

    print("[collect] eval_dir:", eval_dir)
    print("[collect] total logs:", total)
    print("[collect] successful logs:", wins)
    if total:
        print("[collect] success rate:", f"{wins / total:.2%}")
    print("[collect] wrote:", manifest)
    print("[collect] wrote:", all_manifest)
    print("[collect] copied logs to:", copied_dir)


if __name__ == "__main__":
    main()
