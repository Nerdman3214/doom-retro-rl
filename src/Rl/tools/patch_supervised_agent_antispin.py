#!/usr/bin/env python3
from pathlib import Path

p = Path("training/train_supervised_agent.py")
s = p.read_text()

backup = p.with_suffix(".py.before_antispin_guard")
backup.write_text(s)
print("[patch] backup saved:", backup)

# 1) Add CLI args after mouse-y-scale.
old_args = '''    parser.add_argument("--mouse-deadzone", type=float, default=0.35)
    parser.add_argument("--mouse-y-scale", type=float, default=0.35)
'''

new_args = '''    parser.add_argument("--mouse-deadzone", type=float, default=0.35)
    parser.add_argument("--mouse-y-scale", type=float, default=0.35)

    # Anti-spin guard: prevents tiny repeated same-direction mouse drift
    # from becoming a constant circle while the agent holds move_forward.
    parser.add_argument("--anti-spin", action="store_true")
    parser.add_argument("--anti-spin-min-dx", type=float, default=0.08)
    parser.add_argument("--anti-spin-max-same", type=int, default=8)
    parser.add_argument("--anti-spin-dampen", type=float, default=0.25)
    parser.add_argument("--turn-rate-limit", type=float, default=0.0)
'''

if old_args in s and "--anti-spin" not in s:
    s = s.replace(old_args, new_args, 1)
    print("[patch] added anti-spin arguments")
elif "--anti-spin" in s:
    print("[patch] anti-spin arguments already exist")
else:
    raise SystemExit("[patch] could not find mouse args block")

# 2) Add state variables after mouse accumulator.
old_state = '''    mouse_accum_x = 0.0
    mouse_accum_y = 0.0
'''

new_state = '''    mouse_accum_x = 0.0
    mouse_accum_y = 0.0

    same_turn_sign = 0
    same_turn_frames = 0
    turn_rate_window_start = time.time()
    turn_rate_accum = 0.0
'''

if old_state in s and "same_turn_frames = 0" not in s:
    s = s.replace(old_state, new_state, 1)
    print("[patch] added anti-spin state")
elif "same_turn_frames = 0" in s:
    print("[patch] anti-spin state already exists")
else:
    raise SystemExit("[patch] could not find mouse accumulator block")

# 3) Insert guard after no_mouse_on_noop block and before do_shoot.
old_guard_anchor = '''                if args.no_mouse_on_noop and raw_keyboard_label == "no_op":
                    dx = 0.0
                    dy = 0.0
                    mouse_accum_x = 0.0
                    mouse_accum_y = 0.0

                do_shoot = shoot_prob >= args.shoot_threshold
'''

new_guard = '''                if args.no_mouse_on_noop and raw_keyboard_label == "no_op":
                    dx = 0.0
                    dy = 0.0
                    mouse_accum_x = 0.0
                    mouse_accum_y = 0.0

                if args.anti_spin:
                    # Track repeated same-direction turning while moving.
                    if dx > args.anti_spin_min_dx:
                        turn_sign = 1
                    elif dx < -args.anti_spin_min_dx:
                        turn_sign = -1
                    else:
                        turn_sign = 0

                    if movement_like and turn_sign != 0:
                        if turn_sign == same_turn_sign:
                            same_turn_frames += 1
                        else:
                            same_turn_sign = turn_sign
                            same_turn_frames = 1
                    else:
                        same_turn_sign = 0
                        same_turn_frames = 0

                    # If it keeps turning the same direction for too long,
                    # damp both current dx and accumulated fractional dx.
                    if same_turn_frames > args.anti_spin_max_same:
                        dx *= args.anti_spin_dampen
                        mouse_accum_x *= args.anti_spin_dampen

                    # Optional per-second turn budget.
                    if args.turn_rate_limit > 0:
                        now_rate = time.time()
                        if now_rate - turn_rate_window_start >= 1.0:
                            turn_rate_window_start = now_rate
                            turn_rate_accum = 0.0

                        proposed = turn_rate_accum + dx
                        if abs(proposed) > args.turn_rate_limit:
                            remaining = max(0.0, args.turn_rate_limit - abs(turn_rate_accum))
                            dx = math.copysign(remaining, dx) if dx != 0 else 0.0

                        turn_rate_accum += dx

                do_shoot = shoot_prob >= args.shoot_threshold
'''

if old_guard_anchor in s and "if args.anti_spin:" not in s:
    s = s.replace(old_guard_anchor, new_guard, 1)
    print("[patch] inserted anti-spin guard")
elif "if args.anti_spin:" in s:
    print("[patch] anti-spin guard already exists")
else:
    raise SystemExit("[patch] could not find no_mouse_on_noop block")

p.write_text(s)
print("[patch] wrote:", p)
