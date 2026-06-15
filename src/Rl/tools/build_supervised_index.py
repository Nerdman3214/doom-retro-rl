from pathlib import Path
import csv
import json
import random

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "supervised_doomretro_v1"
RAW_RUNS = DATASET / "raw_runs"
INDEX_DIR = DATASET / "index"

EXCLUDE_FILE = INDEX_DIR / "exclude_runs.txt"
WEIGHTS_FILE = INDEX_DIR / "run_weights.csv"

FULL_INDEX = INDEX_DIR / "supervised_index.csv"
TRAIN_INDEX = INDEX_DIR / "train_index.csv"
VAL_INDEX = INDEX_DIR / "val_index.csv"
SUMMARY_TXT = INDEX_DIR / "index_summary.txt"
SUMMARY_JSON = INDEX_DIR / "index_summary.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def read_excludes():
    if not EXCLUDE_FILE.exists():
        return set()

    excluded = set()
    for line in EXCLUDE_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            excluded.add(line)
    return excluded


def read_weights():
    weights = {}

    if not WEIGHTS_FILE.exists():
        return weights

    with WEIGHTS_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = row.get("run_name", "").strip()
            if not run_name:
                continue

            try:
                weight = float(row.get("weight", "1.0"))
            except Exception:
                weight = 1.0

            reason = row.get("reason", "")
            weights[run_name] = {
                "weight": weight,
                "reason": reason,
            }

    return weights


def read_rows(actions_csv):
    with actions_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def numeric_frame_key(path):
    # frame_000123.png -> 123
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    excluded = read_excludes()
    weights = read_weights()

    run_dirs = sorted(
        p for p in RAW_RUNS.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )

    usable_runs = []
    skipped_runs = []

    for run_dir in run_dirs:
        run_name = run_dir.name

        if run_name in excluded:
            skipped_runs.append({
                "run_name": run_name,
                "reason": "excluded by exclude_runs.txt",
            })
            continue

        actions_csv = run_dir / "actions.csv"
        frames_dir = run_dir / "frames"

        if not actions_csv.exists():
            skipped_runs.append({
                "run_name": run_name,
                "reason": "missing actions.csv",
            })
            continue

        if not frames_dir.exists():
            skipped_runs.append({
                "run_name": run_name,
                "reason": "missing frames directory",
            })
            continue

        rows, action_fields = read_rows(actions_csv)
        frames = sorted(frames_dir.glob("*.png"), key=numeric_frame_key)

        row_count = len(rows)
        frame_count = len(frames)
        diff = abs(row_count - frame_count)

        if row_count < 100:
            skipped_runs.append({
                "run_name": run_name,
                "reason": f"too short: rows={row_count}",
            })
            continue

        if diff > 3:
            skipped_runs.append({
                "run_name": run_name,
                "reason": f"row/frame mismatch: rows={row_count} frames={frame_count}",
            })
            continue

        usable_runs.append({
            "run_name": run_name,
            "run_dir": run_dir,
            "actions_csv": actions_csv,
            "frames": frames,
            "rows": rows,
            "action_fields": action_fields,
            "row_count": row_count,
            "frame_count": frame_count,
            "usable_pairs": min(row_count, frame_count),
        })

    if not usable_runs:
        raise SystemExit("No usable runs found. Check validation report and exclude list.")

    rng = random.Random(42)
    shuffled = usable_runs[:]
    rng.shuffle(shuffled)

    val_count = max(1, round(len(shuffled) * 0.10))
    val_run_names = {r["run_name"] for r in shuffled[:val_count]}

    all_action_fields = []
    seen_fields = set()

    for run in usable_runs:
        for field in run["action_fields"]:
            if field not in seen_fields:
                seen_fields.add(field)
                all_action_fields.append(field)

    base_fields = [
        "split",
        "run_name",
        "frame_index",
        "frame_path",
        "actions_csv",
        "run_weight",
        "run_note",
    ]

    output_fields = base_fields + all_action_fields

    all_records = []
    train_records = []
    val_records = []

    run_summaries = []

    for run in usable_runs:
        run_name = run["run_name"]
        split = "val" if run_name in val_run_names else "train"

        weight_info = weights.get(run_name, {})
        run_weight = weight_info.get("weight", 1.0)
        run_note = weight_info.get("reason", "")

        n = run["usable_pairs"]

        for i in range(n):
            action_row = run["rows"][i]
            frame_path = run["frames"][i]

            record = {
                "split": split,
                "run_name": run_name,
                "frame_index": i,
                "frame_path": str(frame_path),
                "actions_csv": str(run["actions_csv"]),
                "run_weight": run_weight,
                "run_note": run_note,
            }

            for field in all_action_fields:
                record[field] = action_row.get(field, "")

            all_records.append(record)

            if split == "train":
                train_records.append(record)
            else:
                val_records.append(record)

        run_summaries.append({
            "run_name": run_name,
            "split": split,
            "rows": run["row_count"],
            "frames": run["frame_count"],
            "usable_pairs": n,
            "weight": run_weight,
            "note": run_note,
        })

    write_csv(FULL_INDEX, all_records, output_fields)
    write_csv(TRAIN_INDEX, train_records, output_fields)
    write_csv(VAL_INDEX, val_records, output_fields)

    summary = {
        "raw_runs_dir": str(RAW_RUNS),
        "usable_runs": len(usable_runs),
        "skipped_runs": skipped_runs,
        "train_runs": sum(1 for r in run_summaries if r["split"] == "train"),
        "val_runs": sum(1 for r in run_summaries if r["split"] == "val"),
        "total_pairs": len(all_records),
        "train_pairs": len(train_records),
        "val_pairs": len(val_records),
        "full_index": str(FULL_INDEX),
        "train_index": str(TRAIN_INDEX),
        "val_index": str(VAL_INDEX),
        "runs": run_summaries,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    with SUMMARY_TXT.open("w") as f:
        f.write("Supervised Doom Retro Index Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Usable runs:  {summary['usable_runs']}\n")
        f.write(f"Train runs:   {summary['train_runs']}\n")
        f.write(f"Val runs:     {summary['val_runs']}\n")
        f.write(f"Total pairs:  {summary['total_pairs']}\n")
        f.write(f"Train pairs:  {summary['train_pairs']}\n")
        f.write(f"Val pairs:    {summary['val_pairs']}\n")
        f.write("\nIndexes:\n")
        f.write(f"  full:  {FULL_INDEX}\n")
        f.write(f"  train: {TRAIN_INDEX}\n")
        f.write(f"  val:   {VAL_INDEX}\n")
        f.write("\nSkipped runs:\n")
        if skipped_runs:
            for skipped in skipped_runs:
                f.write(f"  - {skipped['run_name']}: {skipped['reason']}\n")
        else:
            f.write("  none\n")

        f.write("\nWeighted/noisy runs:\n")
        weighted = [r for r in run_summaries if float(r["weight"]) != 1.0]
        if weighted:
            for r in weighted:
                f.write(f"  - {r['run_name']}: weight={r['weight']} note={r['note']}\n")
        else:
            f.write("  none\n")

        f.write("\nValidation split runs:\n")
        for r in run_summaries:
            if r["split"] == "val":
                f.write(f"  - {r['run_name']} pairs={r['usable_pairs']}\n")

    print("=" * 80)
    print("[index] DONE")
    print(f"[index] usable runs: {summary['usable_runs']}")
    print(f"[index] train runs:  {summary['train_runs']}")
    print(f"[index] val runs:    {summary['val_runs']}")
    print(f"[index] total pairs: {summary['total_pairs']}")
    print(f"[index] train pairs: {summary['train_pairs']}")
    print(f"[index] val pairs:   {summary['val_pairs']}")
    print(f"[index] summary:     {SUMMARY_TXT}")
    print("=" * 80)

    if skipped_runs:
        print("[index] skipped runs:")
        for skipped in skipped_runs:
            print(f"  {skipped['run_name']}: {skipped['reason']}")


if __name__ == "__main__":
    main()