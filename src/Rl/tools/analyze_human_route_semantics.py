#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def fnum(row, key, default=0.0):
    try:
        v = row.get(key, default)
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


def inum(row, key, default=0):
    try:
        v = row.get(key, default)
        if v in ("", None):
            return default
        return int(float(v))
    except Exception:
        return default


def text_has(row, *words):
    text = " ".join(str(row.get(k, "")) for k in ["action", "buttons", "active"]).lower()
    return any(w.lower() in text for w in words)


def angle_delta(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def dist2(a, b):
    return math.hypot(fnum(a, "x") - fnum(b, "x"), fnum(a, "y") - fnum(b, "y"))


def event_reason(kind, row, prev=None, final=False):
    if kind == "spawn":
        return "Start of the route. Establish the first forward direction from spawn."
    if kind == "shoot":
        return "Shoot because an enemy or threat is on/near the route. This protects progress and prevents dying."
    if kind == "kill":
        return "Enemy was killed here. This confirms shooting was useful for clearing the route."
    if kind == "use":
        if final:
            return "Press use here because this appears to be the final door/button needed to complete the level."
        return "Press use here because this is likely a door, switch, or route gate."
    if kind == "turn":
        return "Turn/steer here because the route changes direction; do not keep walking straight into a wall."
    if kind == "hurt":
        return "Health dropped here. Treat this as danger; shooting or faster movement may be needed."
    if kind == "z_change":
        return "Z/floor height changed here. This marks stairs, lift, drop, or floor transition."
    if kind == "finish":
        return "End of the demonstration. Treat the final position/actions as the completion target."
    return "Important route event."


def add_event(events, kind, i, row, reason, extra=None):
    events.append({
        "idx": i,
        "frame": row.get("frame", str(i)),
        "kind": kind,
        "x": fnum(row, "x"),
        "y": fnum(row, "y"),
        "z": fnum(row, "z"),
        "angle": fnum(row, "angle"),
        "health": fnum(row, "health"),
        "ammo": fnum(row, "ammo"),
        "kill_count": inum(row, "kill_count"),
        "item_count": inum(row, "item_count"),
        "action": row.get("action", ""),
        "buttons": row.get("buttons", ""),
        "reason": reason,
        "extra": extra or "",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-session", required=True)
    ap.add_argument("--turn-threshold-deg", type=float, default=45)
    ap.add_argument("--min-event-gap", type=int, default=8)
    args = ap.parse_args()

    route = Path(args.route_session)
    actions = route / "actions.csv"

    if not actions.exists():
        raise SystemExit(f"Missing actions.csv: {actions}")

    rows = list(csv.DictReader(actions.open()))
    if not rows:
        raise SystemExit("actions.csv has no rows")

    events = []
    last_event_idx = {
        "shoot": -99999,
        "use": -99999,
        "turn": -99999,
        "hurt": -99999,
        "z_change": -99999,
    }

    add_event(events, "spawn", 0, rows[0], event_reason("spawn", rows[0]))

    total_shoot_rows = 0
    total_use_rows = 0
    total_kills = 0
    total_health_lost = 0.0

    final_zone_start = int(len(rows) * 0.82)

    for i in range(1, len(rows)):
        row = rows[i]
        prev = rows[i - 1]

        ammo_now = fnum(row, "ammo")
        ammo_prev = fnum(prev, "ammo")
        health_now = fnum(row, "health")
        health_prev = fnum(prev, "health")
        kills_now = inum(row, "kill_count")
        kills_prev = inum(prev, "kill_count")
        z_now = fnum(row, "z")
        z_prev = fnum(prev, "z")
        angle_now = fnum(row, "angle")
        angle_prev = fnum(prev, "angle")

        shoot = text_has(row, "shoot", "attack", "fire") or ammo_now < ammo_prev
        use = text_has(row, "use")
        killed = kills_now > kills_prev
        hurt = health_now < health_prev
        z_change = abs(z_now - z_prev) >= 16
        turn = abs(angle_delta(angle_now, angle_prev)) >= args.turn_threshold_deg

        if shoot:
            total_shoot_rows += 1

        if use:
            total_use_rows += 1

        if killed:
            total_kills += kills_now - kills_prev

        if hurt:
            total_health_lost += health_prev - health_now

        if shoot and i - last_event_idx["shoot"] >= args.min_event_gap:
            add_event(
                events,
                "shoot",
                i,
                row,
                event_reason("shoot", row, prev),
                extra=f"ammo {ammo_prev:g}->{ammo_now:g}",
            )
            last_event_idx["shoot"] = i

        if killed:
            add_event(
                events,
                "kill",
                i,
                row,
                event_reason("kill", row, prev),
                extra=f"kills {kills_prev}->{kills_now}",
            )

        if use and i - last_event_idx["use"] >= args.min_event_gap:
            final = i >= final_zone_start
            add_event(
                events,
                "use",
                i,
                row,
                event_reason("use", row, prev, final=final),
                extra="final_area=1" if final else "final_area=0",
            )
            last_event_idx["use"] = i

        if turn and i - last_event_idx["turn"] >= args.min_event_gap:
            # Avoid marking tiny jitter turns as lessons if barely moved.
            if dist2(row, prev) >= 1.0:
                add_event(
                    events,
                    "turn",
                    i,
                    row,
                    event_reason("turn", row, prev),
                    extra=f"angle_delta={angle_delta(angle_now, angle_prev):.1f}",
                )
                last_event_idx["turn"] = i

        if hurt and i - last_event_idx["hurt"] >= args.min_event_gap:
            add_event(
                events,
                "hurt",
                i,
                row,
                event_reason("hurt", row, prev),
                extra=f"health {health_prev:g}->{health_now:g}",
            )
            last_event_idx["hurt"] = i

        if z_change and i - last_event_idx["z_change"] >= args.min_event_gap:
            add_event(
                events,
                "z_change",
                i,
                row,
                event_reason("z_change", row, prev),
                extra=f"z {z_prev:g}->{z_now:g}",
            )
            last_event_idx["z_change"] = i

    add_event(events, "finish", len(rows) - 1, rows[-1], event_reason("finish", rows[-1]))

    story_path = route / "route_story.md"
    events_csv_path = route / "route_events.csv"
    lessons_path = route / "route_lessons.jsonl"

    with events_csv_path.open("w", newline="") as f:
        fieldnames = [
            "idx", "frame", "kind", "x", "y", "z", "angle",
            "health", "ammo", "kill_count", "item_count",
            "action", "buttons", "reason", "extra",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in events:
            w.writerow(e)

    with lessons_path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    with story_path.open("w") as f:
        f.write("# Human Route Lesson\n\n")
        f.write(f"Route session: `{route}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Frames/actions recorded: **{len(rows)}**\n")
        f.write(f"- Important events detected: **{len(events)}**\n")
        f.write(f"- Use rows: **{total_use_rows}**\n")
        f.write(f"- Shoot rows: **{total_shoot_rows}**\n")
        f.write(f"- Kills gained: **{total_kills}**\n")
        f.write(f"- Health lost: **{total_health_lost:.1f}**\n")
        f.write(f"- Start position: `({fnum(rows[0], 'x'):.1f}, {fnum(rows[0], 'y'):.1f}, {fnum(rows[0], 'z'):.1f})`\n")
        f.write(f"- Final position: `({fnum(rows[-1], 'x'):.1f}, {fnum(rows[-1], 'y'):.1f}, {fnum(rows[-1], 'z'):.1f})`\n\n")

        f.write("## What the AI should notice\n\n")
        f.write("- Movement is not random: Steven follows a path toward doors, rooms, and the final button.\n")
        f.write("- Shooting is tied to route safety: when an enemy blocks or threatens the route, shooting helps progress.\n")
        f.write("- Use is special: it usually means a door, switch, or final button, not a movement action to spam forever.\n")
        f.write("- Turns matter: when the path bends, the agent should face the next route direction before moving forward.\n")
        f.write("- Z changes matter: floor height transitions indicate stairs, lifts, drops, or different reachable areas.\n")
        f.write("- The final use event matters most: successful completion should be saved as high-value training data.\n\n")

        f.write("## Timeline of important events\n\n")
        for e in events:
            f.write(
                f"- **{e['kind']}** at idx `{e['idx']}` frame `{e['frame']}` "
                f"pos=({e['x']:.1f}, {e['y']:.1f}, {e['z']:.1f}) "
                f"angle={e['angle']:.1f}: {e['reason']}"
            )
            if e["extra"]:
                f.write(f" `{e['extra']}`")
            f.write("\n")

        f.write("\n## Training lesson\n\n")
        f.write("This route should be used as a successful demonstration only if the run completes the level. ")
        f.write("The agent should learn the sequence: follow route, clear route enemies, use doors/switches briefly, ")
        f.write("cross after use, then face and press the final button.\n")

    print("[analyze] wrote:", story_path)
    print("[analyze] wrote:", events_csv_path)
    print("[analyze] wrote:", lessons_path)
    print()
    print("[analyze] quick story:")
    for e in events[:25]:
        print(f"- {e['kind']} idx={e['idx']} frame={e['frame']} reason={e['reason']}")

    if len(events) > 25:
        print(f"... {len(events) - 25} more events")


if __name__ == "__main__":
    main()
