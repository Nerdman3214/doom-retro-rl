from pathlib import Path

p = Path("training/train_behavior_judge.py")
text = p.read_text()

backup = Path("training/train_behavior_judge.py.backup_before_import_path_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

insert = '''import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

'''

# Replace the first Path import area cleanly.
if "if str(ROOT) not in sys.path:" not in text:
    text = text.replace("import json\n", "import json\n" + insert, 1)

# Remove duplicate ROOT line if it exists later.
text = text.replace('ROOT = Path(__file__).resolve().parents[1]\nDATA_ROOT = ROOT / "vision_dataset_v2" / "skill_labeled_clips"\n', 'DATA_ROOT = ROOT / "vision_dataset_v2" / "skill_labeled_clips"\n')

p.write_text(text)
print("Fixed behavior judge import path.")
