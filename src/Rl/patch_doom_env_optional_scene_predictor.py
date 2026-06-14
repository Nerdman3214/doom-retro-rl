from pathlib import Path

p = Path("env/doom_env.py")
text = p.read_text()

backup = Path("env/doom_env.py.backup_before_optional_scene_predictor")
backup.write_text(text)
print(f"Backup saved to {backup}")

old = "from vision.scene_predictor import ScenePredictor\n"

new = """try:
    from vision.scene_predictor import ScenePredictor
except Exception as e:
    print(f"[scene_vision] ScenePredictor disabled: {e}")
    ScenePredictor = None
"""

if old not in text:
    raise SystemExit("Could not find ScenePredictor import line.")

text = text.replace(old, new, 1)

p.write_text(text)
print("Made ScenePredictor optional.")
