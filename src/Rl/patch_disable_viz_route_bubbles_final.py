from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_disable_route_bubbles_final")
backup.write_text(text)
print(f"Backup saved to {backup}")

replacement = '''    def route_progress_reward(self, game_state):
        """
        Route/corridor bubble rewards disabled.

        Reason:
        The agent already reaches early corridor zones.
        We do not want it farming/printing spawn_exit_lane,
        sloped_corridor_entry, sloped_corridor_mid, or corridor_left_turn.
        """
        return 0.0

'''

text, count = re.subn(
    r"    def route_progress_reward\(self, game_state\):.*?(?=\n    def goal_heading_reward\(self, game_state\):)",
    replacement,
    text,
    flags=re.DOTALL,
)

if count != 1:
    raise SystemExit(f"Expected to replace route_progress_reward once, replaced {count}")

# Remove route-zone progress block inside _compute_reward too.
text = re.sub(
    r"            # Sequential route zones\..*?            # Penalize old east/right drift\.",
    "            # Route/corridor zones disabled for exit-first training.\\n\\n            # Penalize old east/right drift.",
    text,
    flags=re.DOTALL,
)

p.write_text(text)
print("Disabled ViZDoom route/corridor bubble progress.")
