from pathlib import Path
import re

p = Path("env/vizdoom_env.py")
text = p.read_text()

backup = Path("env/vizdoom_env.py.backup_before_canonical_stage01_route")
backup.write_text(text)
print(f"Backup saved to {backup}")

new_helper = '''    def ensure_post_corridor_route_zone(self):
        """
        Canonical ordered Stage 01 route.

        This replaces duplicate/loose route zones with one clean ordered list:
        spawn_exit_lane -> sloped_corridor_entry -> sloped_corridor_mid
        -> corridor_left_turn -> post_corridor

        Important:
        post_corridor is intentionally farther forward and tighter so it
        cannot trigger at the same place as corridor_left_turn.
        """
        if not hasattr(self, "level_guide") or self.level_guide is None:
            return

        canonical_route = [
            {
                "name": "spawn_exit_lane",
                "x": -416.0,
                "y": 256.0,
                "radius": 60.0,
                "reward": 0.10,
            },
            {
                "name": "sloped_corridor_entry",
                "x": -400.0,
                "y": 285.0,
                "radius": 80.0,
                "reward": 0.20,
            },
            {
                "name": "sloped_corridor_mid",
                "x": -340.0,
                "y": 370.0,
                "radius": 85.0,
                "reward": 0.30,
            },
            {
                "name": "corridor_left_turn",
                "x": -360.0,
                "y": 410.0,
                "radius": 80.0,
                "reward": 0.40,
            },
            {
                "name": "post_corridor",
                "x": -340.0,
                "y": 560.0,
                "radius": 75.0,
                "reward": 0.60,
            },
        ]

        current = self.level_guide.get("route_zones", [])
        current_names = [str(z.get("name", "")) for z in current if isinstance(z, dict)]
        canonical_names = [z["name"] for z in canonical_route]

        if current_names != canonical_names:
            self.level_guide["route_zones"] = canonical_route
            print("[route_guide] replaced route_zones with canonical ordered Stage 01 route")

'''

pattern = r"    def ensure_post_corridor_route_zone\(self\):\n.*?(?=\n    def route_progress_reward\(self, game_state\):)"

new_text, count = re.subn(pattern, new_helper + "\n", text, count=1, flags=re.S)

if count != 1:
    raise SystemExit(f"Failed to replace ensure_post_corridor_route_zone. replacements={count}")

p.write_text(new_text)
print("Patched canonical ordered Stage 01 route zones.")
