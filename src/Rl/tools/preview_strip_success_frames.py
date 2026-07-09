#!/usr/bin/env python3
import csv
from pathlib import Path

active_root = Path("vision_dataset/success_ai_play")
curated_manifest = Path("vision_dataset/curated_success_keep_fullframes/curated_manifest.csv")

image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

keep_sessions = set()

if curated_manifest.exists():
    with curated_manifest.open(newline="", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("session", "").strip()
            if name:
                keep_sessions.add(name)

delete_files = 0
delete_bytes = 0
delete_sessions = 0
keep_sessions_seen = 0

for session in sorted(active_root.glob("session_*")):
    if not session.is_dir():
        continue

    if session.name in keep_sessions:
        keep_sessions_seen += 1
        continue

    session_image_files = []
    session_image_bytes = 0

    for p in session.rglob("*"):
        if p.is_file() and p.suffix.lower() in image_exts:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            session_image_files.append(p)
            session_image_bytes += size

    if session_image_files:
        delete_sessions += 1
        delete_files += len(session_image_files)
        delete_bytes += session_image_bytes

print("[preview] active root:", active_root)
print("[preview] curated keep sessions:", len(keep_sessions))
print("[preview] curated sessions found in active:", keep_sessions_seen)
print("[preview] sessions whose frames could be stripped:", delete_sessions)
print("[preview] image files that would be deleted:", delete_files)
print("[preview] space that would be freed GB:", round(delete_bytes / 1024**3, 2))

print()
print("[preview] Nothing was deleted.")
