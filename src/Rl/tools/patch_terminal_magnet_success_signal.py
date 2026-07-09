#!/usr/bin/env python3
from pathlib import Path

p = Path("tools/watch_vizdoom_z_breadcrumb_agent.py")
s = p.read_text()

backup = p.with_suffix(".py.before_terminal_magnet_success_signal")
backup.write_text(s)
print("[patch] backup saved:", backup)

old = '''                src_text = str(source)
                if any(tok in src_text for tok in [
                    "ghost_terminal",
                    "ghost_final_use",
                    "goal_done",
                    "goal_wait_after_use",
                    "goal_priority_wait",
                    "final_wall_use_once",
                    "goal_use",
                ]):
                    ai_record_terminal_signal = True
'''

new = '''                src_text = str(source)

                # Terminal/completion signal:
                # Older finish controllers used ghost_final_use/goal_use names.
                # The newer final controller uses terminal_magnet_* and
                # terminal_use_lock* names. Without this bridge, runs can reach
                # wp=260/261 near the terminal but still be discarded as
                # terminal_signal=False.
                terminal_signal_source = any(tok in src_text for tok in [
                    "ghost_terminal",
                    "ghost_final_use",
                    "goal_done",
                    "goal_wait_after_use",
                    "goal_priority_wait",
                    "final_wall_use_once",
                    "goal_use",
                    "terminal_magnet_use",
                    "terminal_use_lock",
                    "terminal_use_lock_start",
                ])

                # Conservative fallback for the exact near-finish failure mode:
                # terminal_magnet_approach/final_face at wp260 should count as
                # a terminal candidate only when already in the final waypoint
                # window and physically inside the terminal magnet radius.
                terminal_d2_now = float(locals().get("terminal_d2", 999999.0))
                near_terminal_finish_signal = (
                    int(wp_idx) >= max(0, len(waypoints) - 2)
                    and terminal_d2_now <= float(args.terminal_magnet_radius)
                    and any(tok in src_text for tok in [
                        "terminal_magnet_approach",
                        "terminal_magnet_final_face",
                        "terminal_magnet_face",
                    ])
                )

                if terminal_signal_source or near_terminal_finish_signal:
                    ai_record_terminal_signal = True
'''

if old not in s:
    raise SystemExit("[patch] target block not found; file may have changed")

s = s.replace(old, new)

p.write_text(s)
print("[patch] terminal magnet success signal installed")
