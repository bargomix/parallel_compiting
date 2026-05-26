#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMPLEMENTATION = "task3_2_1"
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
CSV_PATH = RESULTS_DIR / "timing_results.csv"
REQUIRED_FIELDS = [
    "implementation",
    "clients",
    "tasks_per_client",
    "server_workers",
    "init_time_s",
    "work_time_s",
    "stop_time_s",
    "total_time_s",
]


def mean(values):
    return sum(values) / len(values)


def unique_path(path):
    if not path.exists():
        return path

    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Too many existing plot files for {path.name}")


def load_rows():
    rows = []

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Bad CSV header in {CSV_PATH}. Missing fields: {', '.join(missing)}")

        for row in reader:
            if row.get("implementation") != IMPLEMENTATION:
                continue

            rows.append({
                "tasks_per_client": int(row["tasks_per_client"]),
                "server_workers": int(row["server_workers"]),
                "init_time_s": float(row["init_time_s"]),
                "work_time_s": float(row["work_time_s"]),
                "stop_time_s": float(row["stop_time_s"]),
                "total_time_s": float(row["total_time_s"]),
            })
    return rows


def grouped_mean_by_workers(rows, task_count, metric):
    grouped = defaultdict(list)
    for row in rows:
        if row["tasks_per_client"] == task_count:
            grouped[row["server_workers"]].append(row[metric])

    xs = sorted(grouped)
    ys = [mean(grouped[x]) for x in xs]
    return xs, ys


def draw_metric(ax, rows, metric, title):
    task_values = sorted({row["tasks_per_client"] for row in rows})

    for task_count in task_values:
        xs, ys = grouped_mean_by_workers(rows, task_count, metric)
        line, = ax.plot(xs, ys, marker="o", linewidth=2, label=f"{task_count} tasks")
        color = line.get_color()

        for row in rows:
            if row["tasks_per_client"] == task_count:
                ax.scatter(row["server_workers"], row[metric], color=color, alpha=0.25, s=24)

    ax.set_title(title)
    ax.set_xlabel("Server workers")
    ax.set_ylabel("Time, seconds")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV file not found: {CSV_PATH}")

    rows = load_rows()
    if not rows:
        raise SystemExit(f"No rows for {IMPLEMENTATION} in {CSV_PATH}")

    PLOTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    draw_metric(axes[0], rows, "work_time_s", "Work time by workers")
    draw_metric(axes[1], rows, "total_time_s", "Total time by workers")

    workers = sorted({row["server_workers"] for row in rows})
    workers_label = "-".join(str(x) for x in workers)
    task_values = sorted({row["tasks_per_client"] for row in rows})
    task_label = "_".join(str(x) for x in task_values)

    fig.suptitle(f"{IMPLEMENTATION}: timing by server workers")
    fig.tight_layout()

    output = PLOTS_DIR / f"{IMPLEMENTATION}_workers_{workers_label}_tasks_{task_label}.png"
    output = unique_path(output)
    plt.savefig(output, dpi=160)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
