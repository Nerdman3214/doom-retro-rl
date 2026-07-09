from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_force_skill")
backup.write_text(text)
print(f"Backup saved to {backup}")

# 1. Add os import if needed.
if "import os" not in text:
    text = text.replace("import random", "import os\nimport random", 1)

# 2. Insert set_doom_skill immediately before game.init().
if "VIZDOOM_SKILL" not in text:
    pattern = re.compile(r"(?P<indent>\s*)self\.game\.init\(\)")
    m = pattern.search(text)
    if not m:
        pattern = re.compile(r"(?P<indent>\s*)game\.init\(\)")
        m = pattern.search(text)

    if not m:
        raise SystemExit("Could not find game.init() or self.game.init() in env/vizdoom_env.py")

    indent = m.group("indent")
    block = (
        f"{indent}# Force Doom skill before game.init(). 1=easiest, 3=normal.\n"
        f"{indent}try:\n"
        f"{indent}    skill = int(os.environ.get('VIZDOOM_SKILL', '3'))\n"
        f"{indent}    skill = max(1, min(5, skill))\n"
        f"{indent}    self.game.set_doom_skill(skill)\n"
        f"{indent}    print(f'[vizdoom_env] doom_skill={{skill}} (1=easiest, 3=normal)')\n"
        f"{indent}except Exception as e:\n"
        f"{indent}    print(f'[vizdoom_env] could not set doom skill: {{e}}')\n"
    )

    text = text[:m.start()] + block + text[m.start():]
    print("Inserted VIZDOOM_SKILL hook before game.init().")
else:
    print("VIZDOOM_SKILL hook already present.")

p.write_text(text)
print("Patch complete.")
