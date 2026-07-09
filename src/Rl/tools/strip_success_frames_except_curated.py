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

if not keep_sessions:
    raise SystemExit("[stop] curated keep list is empty; refusing to delete frames")

print("[guard] curated keep sessions:", len(keep_sessions))
print("[guard] active root:", active_root)
print()
confirm = input("Type DELETE_FRAMES to delete non-curated image frames: ").strip()

if confirm != "DELETE_FRAMES":
    raise SystemExit("[cancelled] no frames deleted")

deleted_files = 0
deleted_bytes = 0
stripped_sessions = 0
kept_sessions = 0

for session in sorted(active_root.glob("session_*")):
    if not session.is_dir():
        continue

    if session.name in keep_sessions:
        kept_sessions += 1
        print("[keep full frames]", session.name)
        continue

    session_deleted = 0
    session_bytes = 0

    for p in session.rglob("*"):
        if p.is_file() and p.suffix.lower() in image_exts:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0

            p.unlink()
            session_deleted += 1
            session_bytes += size

    if session_deleted:
        stripped_sessions += 1
        deleted_files += session_deleted
        deleted_bytes += session_bytes
        print(
            "[stripped]",
            session.name,
            "files=",
            session_deleted,
            "freed_GB=",
            round(session_bytes / 1024**3, 3),
        )

print()
print("[done] kept full-frame sessions:", kept_sessions)
print("[done] stripped sessions:", stripped_sessions)
print("[done] deleted image files:", deleted_files)
print("[done] freed GB:", round(deleted_bytes / 1024**3, 2))
