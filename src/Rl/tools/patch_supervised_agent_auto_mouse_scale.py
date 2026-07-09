#!/usr/bin/env python3
from pathlib import Path

p = Path("training/train_supervised_agent.py")
s = p.read_text()

backup = p.with_suffix(".py.before_auto_mouse_scale_fix")
backup.write_text(s)
print("[patch] backup saved:", backup)

# 1) Remove the misplaced block that was added before the try/loop.
bad_start = s.find("    now = time.time()\n    dt = max(1e-6, now - last_step_time)")
bad_end = s.find("\n    try:\n        with mss.mss() as sct:", bad_start)

if bad_start != -1 and bad_end != -1:
    s = s[:bad_start] + s[bad_end + 1:]
    print("[patch] removed misplaced pre-loop auto mouse block")
else:
    print("[patch] misplaced pre-loop block not found; continuing")

# 2) Add loop timing state before the try block.
old_init = '''    mouse_accum_x = 0.0
    mouse_accum_y = 0.0

    try:
'''

new_init = '''    mouse_accum_x = 0.0
    mouse_accum_y = 0.0

    last_step_time = time.time()
    auto_mouse_scale = 1.0

    try:
'''

if old_init in s and "auto_mouse_scale = 1.0" not in s:
    s = s.replace(old_init, new_init, 1)
    print("[patch] added last_step_time and auto_mouse_scale init")
elif "auto_mouse_scale = 1.0" in s:
    print("[patch] auto mouse init already exists")
else:
    raise SystemExit("[patch] could not find mouse_accum init block")

# 3) Add FPS measurement inside the while loop right after loop_start.
old_loop = '''            while True:
                loop_start = time.time()

                if args.seconds > 0 and loop_start - start_time >= args.seconds:
'''

new_loop = '''            while True:
                loop_start = time.time()

                dt = max(1e-6, loop_start - last_step_time)
                last_step_time = loop_start
                measured_fps = 1.0 / dt
                raw_auto_scale = args.mouse_reference_fps / max(1.0, measured_fps)
                raw_auto_scale = max(args.mouse_scale_min, min(args.mouse_scale_max, raw_auto_scale))
                auto_mouse_scale = (
                    args.mouse_scale_smoothing * auto_mouse_scale
                    + (1.0 - args.mouse_scale_smoothing) * raw_auto_scale
                )

                if args.seconds > 0 and loop_start - start_time >= args.seconds:
'''

if old_loop in s and "measured_fps = 1.0 / dt" not in s:
    s = s.replace(old_loop, new_loop, 1)
    print("[patch] added inside-loop FPS measurement")
elif "measured_fps = 1.0 / dt" in s:
    print("[patch] inside-loop FPS measurement already exists")
else:
    raise SystemExit("[patch] could not find while-loop start block")

# 4) Apply auto scale to raw mouse values after model output and before clamp.
old_mouse = '''                raw_dx = float(mouse_xy[0]) * mouse_scale * args.mouse_gain
                raw_dy = float(mouse_xy[1]) * mouse_scale * args.mouse_gain

                # Clamp hard flicks.
'''

new_mouse = '''                raw_dx = float(mouse_xy[0]) * mouse_scale * args.mouse_gain
                raw_dy = float(mouse_xy[1]) * mouse_scale * args.mouse_gain

                if args.auto_mouse_fps_scale:
                    raw_dx *= auto_mouse_scale
                    raw_dy *= auto_mouse_scale

                # Clamp hard flicks.
'''

if old_mouse in s and "raw_dx *= auto_mouse_scale" not in s:
    s = s.replace(old_mouse, new_mouse, 1)
    print("[patch] applied auto scale before mouse clamp")
elif "raw_dx *= auto_mouse_scale" in s:
    print("[patch] auto scale application already exists")
else:
    raise SystemExit("[patch] could not find raw mouse block")

p.write_text(s)
print("[patch] wrote:", p)
