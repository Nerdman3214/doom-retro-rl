from pathlib import Path

p = Path("supervised/test_human_policy_advisor.py")
text = p.read_text()

backup = Path("supervised/test_human_policy_advisor.py.backup_before_import_path_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = """import csv
from pathlib import Path

import cv2
"""

new = """import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
"""

if old not in text:
    raise SystemExit("Could not find import block to patch.")

text = text.replace(old, new, 1)

# Remove duplicate ROOT assignment if it exists later.
text = text.replace(
    """

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "vision_dataset" / "raw_human_play"
""",
    """

DATASET_ROOT = ROOT / "vision_dataset" / "raw_human_play"
""",
    1,
)

p.write_text(text)
print("Fixed import path for advisor test.")
