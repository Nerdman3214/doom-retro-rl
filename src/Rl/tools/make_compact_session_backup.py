#!/usr/bin/env python3
import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
META_EXTS = {".csv", ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml"}

DEFAULT_ROOTS = [
    "vision_dataset/success_ai_play_quarantine_bad",
    "vision_dataset/success_ai_play_quarantine_bc_messy",
]

def file_sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_read_actions(path, max_rows=None):
    rows = []
    if not path.exists():
        return rows

    with path.open(newline="", errors="replace") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(dict(row))
    return rows

def dir_size_bytes(path):
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total

def main():
    roots = [Path(p) for p in (sys.argv[1:] or DEFAULT_ROOTS)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("storage_backups") / f"compact_session_backup_{stamp}"
    meta_root = out_root / "metadata_files"
    meta_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    pickle_data = {
        "created_at": stamp,
        "roots": [str(r) for r in roots],
        "sessions": [],
    }

    for root in roots:
        if not root.exists():
            print("[skip missing]", root)
            continue

        for session in sorted(root.glob("session_*")):
            if not session.is_dir():
                continue

            print("[scan]", session)

            image_count = 0
            image_bytes = 0
            metadata_count = 0
            metadata_bytes = 0
            copied_meta = []

            for p in session.rglob("*"):
                if not p.is_file():
                    continue

                ext = p.suffix.lower()
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0

                if ext in IMAGE_EXTS:
                    image_count += 1
                    image_bytes += size

                if ext in META_EXTS:
                    metadata_count += 1
                    metadata_bytes += size

                    rel = p.relative_to(session)
                    dst = meta_root / session.name / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
                    copied_meta.append(str(rel))

            actions_path = session / "actions.csv"
            action_rows = safe_read_actions(actions_path)

            # Lightweight quality signals from actions.csv.
            final_health = None
            terminal_lock = 0
            final_wall = 0
            bc_policy = 0
            rows = 0

            for row in action_rows:
                rows += 1
                source = str(row.get("source", "")).lower()
                if "terminal_use_lock" in source:
                    terminal_lock += 1
                if "final_wall_rescue" in source:
                    final_wall += 1
                if "bc_policy" in source:
                    bc_policy += 1

                h = row.get("health", "")
                if h not in ("", None):
                    try:
                        final_health = float(h)
                    except ValueError:
                        pass

            record = {
                "session": session.name,
                "path": str(session),
                "root": str(root),
                "size_bytes": dir_size_bytes(session),
                "image_count": image_count,
                "image_bytes": image_bytes,
                "metadata_count": metadata_count,
                "metadata_bytes": metadata_bytes,
                "actions_rows": rows,
                "bc_policy_rows": bc_policy,
                "terminal_lock_rows": terminal_lock,
                "final_wall_rows": final_wall,
                "final_health": final_health,
                "copied_metadata_files": copied_meta,
            }

            if actions_path.exists():
                record["actions_sha256"] = file_sha256(actions_path)

            manifest.append(record)
            pickle_data["sessions"].append({
                **record,
                "actions": action_rows,
            })

    manifest_csv = out_root / "manifest.csv"
    manifest_json = out_root / "manifest.json"
    pickle_gz = out_root / "sessions_metadata.pkl.gz"

    fields = [
        "session",
        "path",
        "root",
        "size_bytes",
        "image_count",
        "image_bytes",
        "metadata_count",
        "metadata_bytes",
        "actions_rows",
        "bc_policy_rows",
        "terminal_lock_rows",
        "final_wall_rows",
        "final_health",
        "actions_sha256",
    ]

    with manifest_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)

    manifest_json.write_text(json.dumps(manifest, indent=2))

    with gzip.open(pickle_gz, "wb", compresslevel=9) as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("[backup] output:", out_root)
    print("[backup] manifest:", manifest_csv)
    print("[backup] json:", manifest_json)
    print("[backup] pickle:", pickle_gz)
    print("[backup] sessions:", len(manifest))
    print("[backup] image GB represented:", round(sum(r["image_bytes"] for r in manifest) / 1024**3, 2))

if __name__ == "__main__":
    main()
