from pathlib import Path

p = Path("sensory/world_state.py")
text = p.read_text()

old = '''    if situation in {
        "stuck_or_looping",
        "spawn_wall_zone",
        "front_blocked",
        "secret_side_area",
        "right_route_area",
    }:
        return "unstuck"
'''

new = '''    if situation in {
        "stuck_or_looping",
        "spawn_wall_zone",
        "front_blocked",
        "secret_side_area",
        "right_route_area",
        "half_wall_rail",
        "side_rail",
    }:
        return "unstuck"
'''

if old in text:
    text = text.replace(old, new)
else:
    print("Exact situation block not found. Check sensory/world_state.py manually.")

p.write_text(text)
print("Updated world_state rail handling.")
