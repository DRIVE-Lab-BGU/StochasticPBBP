from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _load_summary_csv(csv_path: Path) -> Dict[str, Any]:
    iterations: List[int] = []
    mean_values: List[float] = []
    std_values: List[float] = []

    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required_fields = {'iteration', 'mean_over_seeds', 'std_over_seeds'}
        if not required_fields.issubset(fieldnames):
            raise ValueError(
                f'{csv_path} is not a summary csv. Missing fields: '
                f'{sorted(required_fields - fieldnames)!r}.'
            )
        for row in reader:
            iterations.append(int(row['iteration']))
            mean_values.append(float(row['mean_over_seeds']))
            std_values.append(float(row['std_over_seeds']))

    return {
        'label': csv_path.stem,
        'iterations': iterations,
        'mean_values': mean_values,
        'std_values': std_values,
    }


def _plot_mean_with_std_band(ax: Any,
                             results: Dict[str, Any],
                             *,
                             color: str,
                             linestyle: str='-') -> None:
    mean_values = results['mean_values']
    std_values = results['std_values']
    lower = [mean - std for mean, std in zip(mean_values, std_values)]
    upper = [mean + std for mean, std in zip(mean_values, std_values)]

    ax.plot(
        results['iterations'],
        mean_values,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=results['label'],
    )
    ax.fill_between(
        results['iterations'],
        lower,
        upper,
        color=color,
        alpha=0.18,
    )


def plot_output_folder_summary(output_folder: Path | str,
                               *,
                               output_path: Optional[Path | str]=None,
                               title: Optional[str]=None) -> Path:
    folder = Path(output_folder)
    csv_paths = sorted(path for path in folder.glob('*.csv') if not path.name.endswith('.tmp.csv'))
    results: List[Dict[str, Any]] = []
    for csv_path in csv_paths:
        try:
            results.append(_load_summary_csv(csv_path))
        except ValueError:
            continue

    if not results:
        raise ValueError(f'No summary csv files with mean/std columns were found in {folder}.')

    plt.switch_backend('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])

    for index, result in enumerate(results):
        _plot_mean_with_std_band(
            ax,
            result,
            color=color_cycle[index % len(color_cycle)],
            linestyle='-',
        )

    ax.set_ylabel('Evaluation return')
    ax.set_title(title or f'Evaluation mean/std summary from {folder.name}')
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    target_path = folder / 'results_plot.png' if output_path is None else Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path)
    plt.close(fig)
    return target_path


def collapse_history_to_iterations(
    history: List[Dict[str, Any]],
    *,
    label: str,
    seed: int,
) -> Tuple[List[int], List[float]]:
    if not history:
        raise ValueError(f'No training history was returned for {label}, seed={seed}.')

    returns_by_iteration: Dict[int, float] = {}
    chunks_by_iteration: Dict[int, int] = {}
    expected_chunks: Optional[int] = None

    for item in history:
        iteration = int(item['iteration'])
        num_chunks = int(item['num_chunks'])

        if expected_chunks is None:
            expected_chunks = num_chunks
        elif expected_chunks != num_chunks:
            raise ValueError(
                f'Inconsistent num_chunks for {label}, seed={seed}: '
                f'expected {expected_chunks}, got {num_chunks}.'
            )

        returns_by_iteration.setdefault(iteration, 0.0)
        returns_by_iteration[iteration] += float(item['return'])
        chunks_by_iteration[iteration] = chunks_by_iteration.get(iteration, 0) + 1

    iteration_axis = sorted(returns_by_iteration)
    expected_iteration_axis = list(range(1, len(history) + 1))
    if iteration_axis != expected_iteration_axis:
        raise ValueError(
            f'Expected iterations {expected_iteration_axis[:3]}...{expected_iteration_axis[-3:]}, '
            f'got {iteration_axis[:3]}...{iteration_axis[-3:]} '
            f'for {label}, seed={seed}.'
        )

    assert expected_chunks is not None
    for iteration in iteration_axis:
        seen_chunks = chunks_by_iteration[iteration]
        if seen_chunks != expected_chunks:
            raise ValueError(
                f'Expected {expected_chunks} chunks for iteration {iteration} in '
                f'{label}, seed={seed}, got {seen_chunks}.'
            )

    return iteration_axis, [returns_by_iteration[iteration] for iteration in iteration_axis]


def plot_csv_curves_from_folder(folder_path, output_file=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, file_name)
        iterations = []
        returns = []
        stds = []

        with open(file_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                iterations.append(float(row["iteration"]))
                returns.append(float(row["mean"]))
                stds.append(float(row["std"]))

        lower = [r - s for r, s in zip(returns, stds)]
        upper = [r + s for r, s in zip(returns, stds)]
        label = os.path.splitext(file_name)[0]

        ax.plot(iterations, returns, label=label)
        ax.fill_between(iterations, lower, upper, alpha=0.2)

    ax.set_xlabel("iterations")
    ax.set_ylabel("returns")
    ax.set_title("Returns with std envelope")
    ax.legend()
    ax.grid(True)

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()