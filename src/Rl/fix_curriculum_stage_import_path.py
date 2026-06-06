from pathlib import Path

p = Path("training/train_vizdoom_curriculum_stage.py")
text = p.read_text()

backup = Path("training/train_vizdoom_curriculum_stage.py.backup_before_import_path_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = """import argparse
import os
from pathlib import Path
"""

new = """import argparse
import os
import sys
from pathlib import Path

# Allow running this file directly:
#   python training/train_vizdoom_curriculum_stage.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""

if old not in text:
    raise SystemExit("Could not find import block to patch.")

text = text.replace(old, new, 1)

# Remove duplicate ROOT assignment later if present.
text = text.replace(
    """
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
""",
    """
CHECKPOINT_DIR = ROOT / "checkpoints"
""",
    1,
)

p.write_text(text)
print("Fixed import path for curriculum training script.")
