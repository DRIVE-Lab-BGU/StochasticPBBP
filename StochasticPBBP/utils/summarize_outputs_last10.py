"""Summarize output CSV files over the last N iterations.

This utility scans ``outputs/*/run_logs`` for:
- ``returns_*.csv``
- ``*_ppo1.csv``
- ``eval_returns_*.csv``

For each file it sorts by ``iteration``, takes the last N rows, and computes:
- the average of the ``mean`` column
- the average of the ``std`` column

It then writes a wide CSV where each instance is a column.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import fmean


GRADIENT_PATTERN = re.compile(
    r"^returns_.*_noise_(?P<noise_method>gradient2noise)"
    r"(?P<noise_level>[\d.]+)_alpha(?P<alpha>[\d.]+)"
    r"_smin(?P<smin>[\d.]+)_exact\.csv$"
)
CONSTANT_PATTERN = re.compile(
    r"^returns_.*_noise_(?P<noise_method>constant)"
    r"(?P<noise_level>[\d.]+)_exact\.csv$"
)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    default_outputs_dir = script_root / "outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Summarize returns CSV files by averaging the last N rows of the "
            "'mean' and 'std' columns."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=default_outputs_dir,
        help="Directory containing instance folders such as hvac_4/run_logs.",
    )
    parser.add_argument(
        "--tail-count",
        type=int,
        default=5,
        help="How many final rows to average from each CSV file.",
    )
    parser.add_argument(
        "--value-mode",
        choices=("combined", "mean", "std"),
        default="combined",
        help=(
            "combined writes 'std ± mean', mean writes only the averaged "
            "'mean', and std writes only the averaged 'std'."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Destination CSV path. Defaults depend on --value-mode: "
            "summary_last10_mean_pm_std.csv, summary_last10_mean_only.csv, "
            "or summary_last10_std_only.csv."
        ),
    )
    return parser.parse_args()


def parse_case_metadata(filename: str) -> dict[str, str] | None:
    match = GRADIENT_PATTERN.match(filename)
    if match:
        return {
            "noise_method": match.group("noise_method"),
            "noise_level": match.group("noise_level"),
            "alpha": match.group("alpha"),
            "smin": match.group("smin"),
        }

    match = CONSTANT_PATTERN.match(filename)
    if match:
        return {
            "noise_method": match.group("noise_method"),
            "noise_level": match.group("noise_level"),
            "alpha": "NA",
            "smin": "NA",
        }

    return None


def compute_last_n_stats(csv_path: Path, tail_count: int) -> tuple[float, float]:
    rows: list[tuple[float, float, float]] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    float(row["iteration"]),
                    float(row["mean"]),
                    float(row["std"]),
                )
            )

    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    rows.sort(key=lambda item: item[0])
    last_rows = rows[-tail_count:]
    mean_avg = fmean(mean_value for _, mean_value, _ in last_rows)
    std_avg = fmean(std_value for _, _, std_value in last_rows)
    return mean_avg, std_avg


def instance_dirs(outputs_dir: Path) -> list[Path]:
    return sorted(path for path in outputs_dir.iterdir() if (path / "run_logs").is_dir())


def numeric_sort_value(value: str) -> tuple[int, float | str]:
    if value == "NA":
        return (1, value)
    return (0, float(value))


def format_output_value(value: object) -> str:
    if value == "NA":
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"

    text = str(value)
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def format_instance_value(mean_value: float, std_value: float, value_mode: str) -> object:
    lrm = "\u200e"
    if value_mode == "combined":
        return f"{lrm}{std_value:.2f} ± {lrm}{mean_value:.2f}"
    if value_mode == "mean":
        return mean_value
    if value_mode == "std":
        return std_value
    raise ValueError(f"Unsupported value mode: {value_mode}")


def build_summary_rows(
    outputs_dir: Path, tail_count: int, value_mode: str
) -> tuple[list[str], list[dict[str, object]]]:
    instances = [path.name for path in instance_dirs(outputs_dir)]
    summary_by_case: dict[tuple[str, str, str, str], dict[str, object]] = {}
    baseline_rows: dict[str, dict[str, object]] = {
        "ppo": {"noise_method": "ppo", "noise_level": "NA", "alpha": "NA", "smin": "NA"},
        "eval": {"noise_method": "eval", "noise_level": "NA", "alpha": "NA", "smin": "NA"},
    }

    for instance_dir in instance_dirs(outputs_dir):
        instance = instance_dir.name
        run_logs_dir = instance_dir / "run_logs"

        for csv_path in sorted(run_logs_dir.glob("returns_*.csv")):
            metadata = parse_case_metadata(csv_path.name)
            if metadata is None:
                continue

            case_key = (
                metadata["noise_method"],
                metadata["noise_level"],
                metadata["alpha"],
                metadata["smin"],
            )
            case_row = summary_by_case.setdefault(
                case_key,
                {
                    "noise_method": metadata["noise_method"],
                    "noise_level": metadata["noise_level"],
                    "alpha": metadata["alpha"],
                    "smin": metadata["smin"],
                },
            )
            mean_value, std_value = compute_last_n_stats(csv_path, tail_count)
            case_row[instance] = format_instance_value(mean_value, std_value, value_mode)

        baseline_file_groups = {
            "ppo": sorted(run_logs_dir.glob("*_ppo1.csv")),
            "eval": sorted(run_logs_dir.glob("eval_returns_*.csv")),
        }
        for baseline_name, baseline_files in baseline_file_groups.items():
            if not baseline_files:
                continue

            mean_value, std_value = compute_last_n_stats(baseline_files[0], tail_count)
            baseline_rows[baseline_name][instance] = format_instance_value(
                mean_value, std_value, value_mode
            )

    summary_rows = []
    for row in summary_by_case.values():
        for instance in instances:
            row.setdefault(instance, "NA")
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda row: (
            str(row["noise_method"]),
            numeric_sort_value(str(row["noise_level"])),
            numeric_sort_value(str(row["alpha"])),
            numeric_sort_value(str(row["smin"])),
        )
    )

    ordered_baseline_rows = []
    for baseline_name in ("ppo", "eval"):
        row = baseline_rows[baseline_name]
        for instance in instances:
            row.setdefault(instance, "NA")
        ordered_baseline_rows.append(row)

    return instances, summary_rows + ordered_baseline_rows


def write_summary_csv(output_file: Path, instances: list[str], rows: list[dict[str, object]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["noise_method", "noise_level", "alpha", "smin", *instances]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted_row = {
                fieldname: format_output_value(row.get(fieldname, "NA"))
                for fieldname in fieldnames
            }
            writer.writerow(formatted_row)


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    default_output_name_by_mode = {
        "combined": "summary_last10_mean_pm_std.csv",
        "mean": "summary_last10_mean_only.csv",
        "std": "summary_last10_std_only.csv",
    }
    output_file = (
        args.output_file.resolve()
        if args.output_file is not None
        else outputs_dir / default_output_name_by_mode[args.value_mode]
    )

    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    if args.tail_count <= 0:
        raise ValueError("--tail-count must be a positive integer")

    instances, rows = build_summary_rows(outputs_dir, args.tail_count, args.value_mode)
    if not rows:
        raise ValueError(f"No matching returns CSV files found under {outputs_dir}")

    write_summary_csv(output_file, instances, rows)
    print(f"Wrote {len(rows)} summary rows to {output_file}")


if __name__ == "__main__":
    main()
