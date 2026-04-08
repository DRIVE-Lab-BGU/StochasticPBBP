from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

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
