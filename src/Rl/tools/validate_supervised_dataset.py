from pathlib import Path
import csv
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "supervised_doomretro_v1"
RAW_RUNS = DATASET / "raw_runs"
REPORTS = DATASET / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

KEYBOARD_COLS = [
    "keyboard_action",
    "kbd_action",
    "kbd",
    "action",
]

MOUSE_DX_COLS = ["mouse_dx", "dx", "mouse_x"]
MOUSE_DY_COLS = ["mouse_dy", "dy", "mouse_y"]
BUTTON_COLS = ["mouse_buttons", "buttons", "button"]


def pick_col(row, choices, default=""):
    for col in choices:
        if col in row:
            return row.get(col, default)
    return default


def to_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def validate_run(run_dir: Path):
    actions_csv = run_dir / "actions.csv"
    frames_dir = run_dir / "frames"

    result = {
        "run_name": run_dir.name,
        "ok": True,
        "problems": [],
        "rows": 0,
        "frames": 0,
        "missing_actions_csv": False,
        "missing_frames_dir": False,
        "no_op_percent": 0.0,
        "shoot_rows": 0,
        "use_rows": 0,
        "move_forward_rows": 0,
        "strafe_rows": 0,
        "mouse_nonzero_rows": 0,
        "keyboard_counts": {},
    }

    if not actions_csv.exists():
        result["ok"] = False
        result["missing_actions_csv"] = True
        result["problems"].append("missing actions.csv")
        return result

    if not frames_dir.exists():
        result["ok"] = False
        result["missing_frames_dir"] = True
        result["problems"].append("missing frames directory")
        return result

    frame_files = sorted(frames_dir.glob("*.png"))
    result["frames"] = len(frame_files)

    with actions_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    result["rows"] = len(rows)

    if result["rows"] == 0:
        result["ok"] = False
        result["problems"].append("actions.csv has 0 rows")

    if result["frames"] == 0:
        result["ok"] = False
        result["problems"].append("frames folder has 0 png files")

    if result["rows"] and result["frames"]:
        diff = abs(result["rows"] - result["frames"])
        if diff > 3:
            result["ok"] = False
            result["problems"].append(f"row/frame mismatch: rows={result['rows']} frames={result['frames']}")

    keyboard_counter = Counter()
    no_op_count = 0
    shoot_rows = 0
    use_rows = 0
    move_forward_rows = 0
    strafe_rows = 0
    mouse_nonzero_rows = 0

    for row in rows:
        kbd = pick_col(row, KEYBOARD_COLS, "unknown")
        buttons = pick_col(row, BUTTON_COLS, "")
        dx = to_float(pick_col(row, MOUSE_DX_COLS, 0))
        dy = to_float(pick_col(row, MOUSE_DY_COLS, 0))

        keyboard_counter[kbd] += 1

        if kbd == "no_op":
            no_op_count += 1

        if "left" in buttons:
            shoot_rows += 1

        if "use" in kbd or "e" in kbd:
            use_rows += 1

        if "move_forward" in kbd or "w" in kbd:
            move_forward_rows += 1

        if "strafe_left" in kbd or "strafe_right" in kbd or "a" in kbd or "d" in kbd:
            strafe_rows += 1

        if abs(dx) > 0.001 or abs(dy) > 0.001:
            mouse_nonzero_rows += 1

    if result["rows"] > 0:
        result["no_op_percent"] = round((no_op_count / result["rows"]) * 100.0, 2)

    result["shoot_rows"] = shoot_rows
    result["use_rows"] = use_rows
    result["move_forward_rows"] = move_forward_rows
    result["strafe_rows"] = strafe_rows
    result["mouse_nonzero_rows"] = mouse_nonzero_rows
    result["keyboard_counts"] = dict(keyboard_counter.most_common())

    if result["rows"] < 100:
        result["problems"].append("very short run")
        result["ok"] = False

    if result["no_op_percent"] > 60:
        result["problems"].append(f"high no_op percent: {result['no_op_percent']}%")

    if result["move_forward_rows"] == 0:
        result["problems"].append("no move_forward rows")

    if result["mouse_nonzero_rows"] == 0:
        result["problems"].append("no mouse movement detected")

    return result


def main():
    if not RAW_RUNS.exists():
        raise SystemExit(f"Missing raw runs folder: {RAW_RUNS}")

    run_dirs = sorted([p for p in RAW_RUNS.iterdir() if p.is_dir() and p.name.startswith("run_")])

    if not run_dirs:
        raise SystemExit(f"No runs found in: {RAW_RUNS}")

    results = [validate_run(run_dir) for run_dir in run_dirs]

    summary = {
        "raw_runs_dir": str(RAW_RUNS),
        "num_runs": len(results),
        "ok_runs": sum(1 for r in results if r["ok"]),
        "problem_runs": sum(1 for r in results if not r["ok"]),
        "total_rows": sum(r["rows"] for r in results),
        "total_frames": sum(r["frames"] for r in results),
        "runs": results,
    }

    report_json = REPORTS / "supervised_dataset_validation.json"
    report_txt = REPORTS / "supervised_dataset_validation.txt"

    report_json.write_text(json.dumps(summary, indent=2))

    with report_txt.open("w") as f:
        f.write("Supervised Doom Retro Dataset Validation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Runs:         {summary['num_runs']}\n")
        f.write(f"OK runs:      {summary['ok_runs']}\n")
        f.write(f"Problem runs: {summary['problem_runs']}\n")
        f.write(f"Total rows:   {summary['total_rows']}\n")
        f.write(f"Total frames: {summary['total_frames']}\n")
        f.write("\n")

        for r in results:
            status = "OK" if r["ok"] else "CHECK"
            f.write(f"{status} {r['run_name']} rows={r['rows']} frames={r['frames']} no_op={r['no_op_percent']}%\n")
            f.write(f"  move_forward={r['move_forward_rows']} strafe={r['strafe_rows']} shoot={r['shoot_rows']} use={r['use_rows']} mouse_move={r['mouse_nonzero_rows']}\n")
            if r["problems"]:
                for problem in r["problems"]:
                    f.write(f"  - {problem}\n")
            f.write("\n")

    print("=" * 80)
    print("[validate] DONE")
    print(f"[validate] runs:         {summary['num_runs']}")
    print(f"[validate] ok runs:      {summary['ok_runs']}")
    print(f"[validate] problem runs: {summary['problem_runs']}")
    print(f"[validate] total rows:   {summary['total_rows']}")
    print(f"[validate] total frames: {summary['total_frames']}")
    print(f"[validate] report txt:   {report_txt}")
    print(f"[validate] report json:  {report_json}")
    print("=" * 80)

    problem_runs = [r for r in results if r["problems"]]
    if problem_runs:
        print("[validate] Runs with notes/problems:")
        for r in problem_runs:
            print(f"  {r['run_name']}: {', '.join(r['problems'])}")


if __name__ == "__main__":
    main()