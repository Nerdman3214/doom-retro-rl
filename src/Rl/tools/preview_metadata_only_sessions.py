#!/usr/bin/env python3
from pathlib import Path

root = Path("vision_dataset/success_ai_play")
image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

fullframe = []
metadata_only = []

for d in sorted(root.glob("session_*")):
    if not d.is_dir():
        continue

    has_image = any(
        p.is_file() and p.suffix.lower() in image_exts
        for p in d.rglob("*")
    )

    if has_image:
        fullframe.append(d)
    else:
        metadata_only.append(d)

print("[preview] root:", root)
print("[preview] full-frame sessions:", len(fullframe))
print("[preview] metadata-only sessions:", len(metadata_only))
print()

print("[preview] first metadata-only sessions:")
for d in metadata_only[:30]:
    print(d)

print()
print("[preview] first full-frame sessions:")
for d in fullframe[:30]:
    print(d)

print()
print("[preview] Nothing was moved.")
