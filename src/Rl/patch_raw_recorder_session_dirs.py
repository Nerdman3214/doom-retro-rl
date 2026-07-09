from pathlib import Path

p = Path("recording/record_vizdoom_raw_human_play.py")
text = p.read_text()

backup = Path("recording/record_vizdoom_raw_human_play.py.backup_before_session_dirs")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add datetime import.
text = text.replace(
    "import csv\nimport sys\nimport time\n",
    "import csv\nimport sys\nimport time\nfrom datetime import datetime\n",
    1,
)

old = """OUTPUT_DIR = ROOT / "vision_dataset" / "raw_human_play"
FRAME_DIR = OUTPUT_DIR / "frames"
CSV_PATH = OUTPUT_DIR / "actions.csv"

FRAME_DIR.mkdir(parents=True, exist_ok=True)
"""

new = """DATASET_ROOT = ROOT / "vision_dataset" / "raw_human_play"
SESSION_NAME = datetime.now().strftime("session_%Y%m%d_%H%M%S")
OUTPUT_DIR = DATASET_ROOT / SESSION_NAME
FRAME_DIR = OUTPUT_DIR / "frames"
CSV_PATH = OUTPUT_DIR / "actions.csv"

FRAME_DIR.mkdir(parents=True, exist_ok=True)
"""

if old not in text:
    raise SystemExit("Could not find old OUTPUT_DIR block.")

text = text.replace(old, new, 1)

# Make Ctrl+C clean if not already patched.
old2 = """            # 20 FPS-ish recording.
            clock.tick(20)

    finally:
"""

new2 = """            # 20 FPS-ish recording.
            clock.tick(20)

    except KeyboardInterrupt:
        print("")
        print("[raw_record] Ctrl+C received. Saving dataset...")

    finally:
"""

if old2 in text and "Ctrl+C received. Saving dataset" not in text:
    text = text.replace(old2, new2, 1)

p.write_text(text)
print("Recorder now saves each run into a timestamped session folder.")
