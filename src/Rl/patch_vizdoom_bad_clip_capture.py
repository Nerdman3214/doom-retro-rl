from pathlib import Path

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_bad_clip_capture")
backup.write_text(text)
print(f"Backup saved to {backup}")

# Add imports.
if "from PIL import Image" not in text:
    text = text.replace("from pathlib import Path\n", "from pathlib import Path\nfrom PIL import Image\n", 1)

# Add init fields after action prior fields if possible.
marker = "        self.action_prior_min_confidence = 0.40\n"
init_block = """        self.capture_bad_behavior_clips = True
        self.bad_clip_capture_dir = ROOT / "vision_dataset_v2" / "manual_clip_sources" / "agent_bad_auto"
        self.bad_clip_capture_dir.mkdir(parents=True, exist_ok=True)
        self._bad_clip_capture_count = 0
        self._bad_clip_capture_limit = 500
"""

if "self.capture_bad_behavior_clips = True" not in text:
    if marker not in text:
        raise SystemExit("Could not find action_prior_min_confidence marker.")
    text = text.replace(marker, marker + init_block, 1)

# Add helper method before action_prior_advice_reward.
helper = '''
    def maybe_capture_bad_behavior_frame(self, reason="unknown"):
        """
        Save occasional frames from bad/stuck behavior for later skill labeling.
        This is lightweight and capped so it will not fill the disk quickly.
        """
        if not getattr(self, "capture_bad_behavior_clips", False):
            return

        count = getattr(self, "_bad_clip_capture_count", 0)
        limit = getattr(self, "_bad_clip_capture_limit", 500)

        if count >= limit:
            return

        # Save at a low frequency.
        step = getattr(self, "step_count", getattr(self, "episode_step", count))
        if step % 5 != 0:
            return

        state = self.game.get_state() if getattr(self, "game", None) is not None else None
        if state is None:
            return

        frame = state.screen_buffer

        try:
            import numpy as np

            arr = frame
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.ndim == 2:
                img = Image.fromarray(arr.astype("uint8"))
            else:
                img = Image.fromarray(arr.astype("uint8"))

            out_dir = self.bad_clip_capture_dir / "frames"
            out_dir.mkdir(parents=True, exist_ok=True)

            safe_reason = str(reason).replace("/", "_").replace(" ", "_")
            out_path = out_dir / f"bad_{count:06d}_{safe_reason}.png"
            img.save(out_path)

            self._bad_clip_capture_count = count + 1

            if self._bad_clip_capture_count % 50 == 0:
                print(f"[bad_clip_capture] saved={self._bad_clip_capture_count} dir={out_dir}")

        except Exception as e:
            if count < 3:
                print(f"[bad_clip_capture] failed: {e}")

'''

if "def maybe_capture_bad_behavior_frame" not in text:
    insert_before = "    def action_prior_advice_reward"
    if insert_before not in text:
        raise SystemExit("Could not find action_prior_advice_reward insertion point.")
    text = text.replace(insert_before, helper + "\n" + insert_before, 1)

# Add calls after common bad behavior prints/events.
# We use simple string anchors from the logs.
anchors = [
    'self.reward_manager.add("no_position_change", -penalty)',
    'self.reward_manager.add("no_goal_distance_progress", -penalty)',
]

for anchor in anchors:
    if anchor in text:
        replacement = anchor + '\n                self.maybe_capture_bad_behavior_frame("stuck_or_no_progress")'
        if replacement not in text:
            text = text.replace(anchor, replacement, 1)

p.write_text(text)
print("Patched ViZDoom bad behavior frame capture.")
