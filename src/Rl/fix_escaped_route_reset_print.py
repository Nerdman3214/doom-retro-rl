from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_fix_escaped_route_reset_print")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Fix escaped quote version that was accidentally written into Python source.
text = text.replace('print(\\"[route_reset] route progress reset\\")', 'print("[route_reset] route progress reset")')

# Also remove any remaining accidental backreference fragments.
text = text.replace("        \\\\1\n", "")
text = text.replace("    \\\\1\n", "")
text = text.replace("\\\\1\n", "")
text = text.replace("        \\1\n", "")
text = text.replace("    \\1\n", "")
text = text.replace("\\1\n", "")

p.write_text(text)
print("Fixed escaped route reset print and cleaned backreference fragments.")
