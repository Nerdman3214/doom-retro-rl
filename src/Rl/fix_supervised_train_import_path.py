from pathlib import Path

p = Path("supervised/train_human_policy.py")
text = p.read_text()

backup = Path("supervised/train_human_policy.py.backup_before_import_path_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = """import argparse
from pathlib import Path

import torch
"""

new = """import argparse
import sys
from pathlib import Path

# Allow running this file directly:
#   python supervised/train_human_policy.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
"""

if old not in text:
    raise SystemExit("Could not find import block to patch.")

text = text.replace(old, new, 1)

# Remove duplicate ROOT assignment later if present.
text = text.replace(
    """

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "supervised_models"
""",
    """

MODEL_DIR = ROOT / "supervised_models"
""",
    1,
)

p.write_text(text)
print("Fixed import path for supervised training script.")
