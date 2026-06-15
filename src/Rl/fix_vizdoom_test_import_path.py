from pathlib import Path

p = Path("scripts/test_vizdoom_recovery_policy.py")
text = p.read_text()

import_block = '''import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

'''

# Add import path fix after normal imports start, before navigation import.
if "PROJECT_ROOT = Path(__file__).resolve().parents[1]" not in text:
    text = text.replace(
        "import time\n",
        "import time\n" + import_block,
        1,
    )

p.write_text(text)
print("Patched scripts/test_vizdoom_recovery_policy.py import path.")
