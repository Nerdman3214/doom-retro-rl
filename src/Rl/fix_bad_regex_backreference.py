from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_fix_bad_backreference")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Remove accidental literal regex backreference line created by the previous patch.
text = text.replace("        \\1\n", "")
text = text.replace("    \\1\n", "")
text = text.replace("\\1\n", "")

# If the print was inserted without the actual reset lines nearby, remove the print too.
# We will re-add the reset debug safely later after compile succeeds.
text = text.replace('        print("[route_reset] route progress reset")\n', "")

p.write_text(text)
print("Removed accidental literal \\1 backreference.")
