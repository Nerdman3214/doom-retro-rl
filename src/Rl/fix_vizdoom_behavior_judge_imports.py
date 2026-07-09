from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_behavior_judge_import_fix")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add torch import if missing.
if "import torch" not in text:
    # Put it near the top after common imports.
    if "import os\n" in text:
        text = text.replace("import os\n", "import os\nimport torch\n", 1)
    else:
        text = "import torch\n" + text

# Add numpy import if missing.
if "import numpy as np" not in text:
    if "import torch\n" in text:
        text = text.replace("import torch\n", "import torch\nimport numpy as np\n", 1)
    else:
        text = "import numpy as np\n" + text

# Add deque import if missing.
if "from collections import deque" not in text:
    if "import os\n" in text:
        text = text.replace("import os\n", "import os\nfrom collections import deque\n", 1)
    else:
        text = "from collections import deque\n" + text

p.write_text(text)
print("Fixed torch/numpy/deque imports in env/vizdoom_env.py.")
