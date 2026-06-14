from pathlib import Path

p = Path("supervised/test_human_policy_predictions.py")
text = p.read_text()

backup = Path("supervised/test_human_policy_predictions.py.backup_before_import_path_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = """import argparse
import csv
from pathlib import Path

import cv2
"""

new = """import argparse
import csv
import sys
from pathlib import Path

# Allow running this file directly:
#   python supervised/test_human_policy_predictions.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
"""

if old not in text:
    raise SystemExit("Could not find import block to patch.")

text = text.replace(old, new, 1)

# Remove duplicate ROOT assignment later if present.
text = text.replace(
    """

ROOT = Path(__file__).resolve().parents[1]

ACTION_NAMES = [
""",
    """

ACTION_NAMES = [
""",
    1,
)

p.write_text(text)
print("Fixed import path for supervised prediction tester.")
