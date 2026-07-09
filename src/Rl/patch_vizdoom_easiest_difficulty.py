from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_easiest_difficulty")
backup.write_text(text)
print(f"Backup saved to {backup}")

if "game.set_doom_skill(1)" in text or "self.game.set_doom_skill(1)" in text:
    print("Doom skill already appears to be set to 1.")
    raise SystemExit

# Common init patterns.
patterns = [
    "        game.init()\n",
    "        self.game.init()\n",
]

for pat in patterns:
    if pat in text:
        prefix = "        game" if pat.strip() == "game.init()" else "        self.game"
        insert = f'''{prefix}.set_doom_skill(1)  # 1 = easiest / I'm Too Young To Die
'''
        text = text.replace(pat, insert + pat, 1)
        p.write_text(text)
        print("Inserted Doom skill 1 before game.init().")
        break
else:
    raise SystemExit("Could not find game.init() or self.game.init(). Paste the game setup section.")
