from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_post_corridor_route")
backup.write_text(text)
print(f"Backup saved to {backup}")

# ------------------------------------------------------------
# 1. Make sure forced_route_reward knows post_corridor.
# ------------------------------------------------------------
if '"post_corridor": 0.60' not in text:
    text = text.replace(
        '"corridor_left_turn": 0.40,',
        '"corridor_left_turn": 0.40,\n                "post_corridor": 0.60,',
        1,
    )

# ------------------------------------------------------------
# 2. Add a defensive helper that injects post_corridor into
#    self.level_guide["route_zones"] after corridor_left_turn.
# ------------------------------------------------------------
helper = '''
    def ensure_post_corridor_route_zone(self):
        """
        Add the next navigation milestone after corridor_left_turn.

        This keeps Stage 01 from stopping its guidance at the left turn.
        The exact coordinate is intentionally a little generous; it should
        reward the agent for continuing out of the left-turn/corridor area.
        """
        if not hasattr(self, "level_guide") or self.level_guide is None:
            return

        route_zones = self.level_guide.setdefault("route_zones", [])
        if not isinstance(route_zones, list):
            return

        names = [str(z.get("name", "")) for z in route_zones if isinstance(z, dict)]

        if "post_corridor" in names:
            return

        post_zone = {
            "name": "post_corridor",
            "x": -340.0,
            "y": 560.0,
            "radius": 170.0,
            "reward": 0.60,
        }

        insert_at = len(route_zones)
        for i, zone in enumerate(route_zones):
            if not isinstance(zone, dict):
                continue
            if str(zone.get("name", "")) == "corridor_left_turn":
                insert_at = i + 1
                break

        route_zones.insert(insert_at, post_zone)
        print("[route_guide] added post_corridor route zone after corridor_left_turn")

'''

if "def ensure_post_corridor_route_zone(self):" not in text:
    insert_before = "    def route_progress_reward(self, game_state):"
    if insert_before not in text:
        raise SystemExit("Could not find route_progress_reward insertion point.")
    text = text.replace(insert_before, helper + "\n" + insert_before, 1)

# ------------------------------------------------------------
# 3. Call helper after level_guide is created/loaded.
#    Easiest safe place: at start of route_progress_reward.
# ------------------------------------------------------------
old = '''    def route_progress_reward(self, game_state):
        x = game_state.get("x")
        y = game_state.get("y")
'''

new = '''    def route_progress_reward(self, game_state):
        self.ensure_post_corridor_route_zone()

        x = game_state.get("x")
        y = game_state.get("y")
'''

if "self.ensure_post_corridor_route_zone()" not in text:
    if old not in text:
        raise SystemExit("Could not patch route_progress_reward start.")
    text = text.replace(old, new, 1)

p.write_text(text)
print("Patched post_corridor route milestone.")
