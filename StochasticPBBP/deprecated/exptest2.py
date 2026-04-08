from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pyRDDLGym
import torch

from StochasticPBBP.core.Logic import FuzzyLogic, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftRounding,SoftControlFlow
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.core.Train import Train
from StochasticPBBP.utils.Noise import AdditiveNoise, AdditiveNoiseFactory
from StochasticPBBP.utils.Policies import StationaryMarkov

PACKAGE_ROOT = Path(__file__).resolve().parent


def collapse_history_to_iterations(
    history: List[Dict[str, Any]],
    *,
    iterations: int,
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
    expected_iteration_axis = list(range(1, iterations + 1))
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


def run_experiment(
    *,
    env: Any,
    template_rollout: TorchRollout,
    observation_template: Dict[str, Any],
    hidden_sizes: Tuple[int, ...],
    horizon: int,
    iterations: int,
    seeds: Sequence[int],
    noise_value: AdditiveNoise,
    label: str,
    batch_size: int=1,
) -> Dict[str, Any]:
    all_training_returns: List[List[float]] = []
    iteration_axis: List[int] = []

    for seed in seeds:
        torch.manual_seed(seed)

        policy = StationaryMarkov(
            observation_template=observation_template,
            action_template=template_rollout.noop_actions,
            action_space=env.action_space,
            hidden_sizes=hidden_sizes,
        )

        fuzzy = FuzzyLogic(
            tnorm=ProductTNorm(),
            comparison=SigmoidComparison(weight=200.0),
            rounding=SoftRounding(weight=200.0),
            control=SoftControlFlow(weight=200.0),
            sampling=SoftRandomSampling(
                poisson_max_bins=100,
                binomial_max_bins=100,
                bernoulli_gumbel_softmax=False),
        )

        trainer = Train(
            horizon=horizon,
            model=env.model,
            action_space=env.action_space,
            policy=policy,
            lr=0.01,
            hidden_sizes=hidden_sizes,
            batch_size=horizon,
            seed=seed,
            additive_noise=noise_value,
            logic=fuzzy,
        )

        history, trained_policy = trainer.train_trajectory(
            iterations=iterations,
            print_every=100,
            batch_size=batch_size,
            additive_noise=trainer.default_additive_noise,
        )
        del trained_policy

        seed_iterations, seed_returns = collapse_history_to_iterations(
            history,
            iterations=iterations,
            label=label,
            seed=seed,
        )
        if not iteration_axis:
            iteration_axis = seed_iterations
        elif iteration_axis != seed_iterations:
            raise ValueError('Training iterations are inconsistent across seeds.')

        all_training_returns.append(seed_returns)
        print(
            f"{label}: seed={seed:2d} "
            f"start_return={seed_returns[0]:10.4f} "
            f"final_return={seed_returns[-1]:10.4f}"
        )

    returns_tensor = torch.tensor(all_training_returns, dtype=torch.float32)
    mean_training_returns = returns_tensor.mean(dim=0)
    std_training_returns = returns_tensor.std(dim=0, unbiased=False)
    var_training_returns = returns_tensor.var(dim=0, unbiased=False)

    print(f"=== {label} ===")
    print(
        f"final mean train return={mean_training_returns[-1].item():.4f} "
        f"std={std_training_returns[-1].item():.4f} "
        f"over {len(seeds)} seeds"
    )

    return {
        'label': label,
        'iterations': iteration_axis,
        'all_training_returns': all_training_returns,
        'mean_training_returns': mean_training_returns.tolist(),
        'std_training_returns': std_training_returns.tolist(),
        'var_training_returns': var_training_returns.tolist(),
    }


def run_timed_experiment(
    *,
    env: Any,
    template_rollout: TorchRollout,
    observation_template: Dict[str, Any],
    hidden_sizes: Tuple[int, ...],
    horizon: int,
    iterations: int,
    seeds: Sequence[int],
    noise_kwargs: Dict[str, Any],
    label: str,
    summary_label: str,
    elapsed_label: str,
    batch_size: int,
) -> Dict[str, Any]:
    start = time.perf_counter()
    print(summary_label)
    noise = AdditiveNoiseFactory.create(source=template_rollout, **noise_kwargs)
    results = run_experiment(
        env=env,
        template_rollout=template_rollout,
        observation_template=observation_template,
        hidden_sizes=hidden_sizes,
        horizon=horizon,
        iterations=iterations,
        seeds=seeds,
        noise_value=noise,
        label=label,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.6f} seconds {elapsed_label}\n\n")
    return results


def plot_mean_with_std_band(
    ax: Any,
    results: Dict[str, Any],
    *,
    color: str,
    linestyle: str='-',
) -> None:
    mean_values = results['mean_training_returns']
    std_values = results['std_training_returns']
    lower = [mean - std for (mean, std) in zip(mean_values, std_values)]
    upper = [mean + std for (mean, std) in zip(mean_values, std_values)]

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


def plot_results(
    plots: Sequence[Dict[str, Any]],
    colors: Sequence[str],
    *,
    seeds: Sequence[int],
    output_path: Path,
) -> None:
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    for plot_data, color in zip(plots, colors):
        plot_mean_with_std_band(
            ax,
            plot_data,
            color=color,
            linestyle='-',
        )

    ax.set_ylabel('Training return')
    ax.set_title(f'Torch with/without noise over {len(seeds)} seeds')
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    print(f'Plot saved to {output_path}')


def main() -> None:
    # domain = PACKAGE_ROOT / 'problems' / 'hvac' / 'domain.rddl'
    # instance = PACKAGE_ROOT / 'problems' / 'hvac' / 'instance_2.rddl'
    # domain = PACKAGE_ROOT / 'problems' / 'hvac' / 'domain_c.rddl'
    # instance = PACKAGE_ROOT / 'problems' / 'hvac' / 'instance_2c.rddl'
    domain = PACKAGE_ROOT / 'problems' / 'reservoir' / 'domain.rddl'
    instance = PACKAGE_ROOT / 'problems' / 'reservoir' / 'instance_3.rddl'
    # domain = PACKAGE_ROOT / 'problems' / 'navigation' / 'domain.rddl'
    # instance = PACKAGE_ROOT / 'problems' / 'navigation' / 'instance_1.rddl'
    # domain = PACKAGE_ROOT / 'problems' / 'powergen' / 'domain.rddl'
    # instance = PACKAGE_ROOT / 'problems' / 'powergen' / 'instance_1.rddl'

    print(f'DOMAIN={domain}')
    print(f'INSTANCE={instance}')

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    horizon = 500
    hidden_sizes = (12, 12)
    iterations = 1000
    num_seeds = 1
    seed_offset = 112
    seeds = [seed_offset + 2 * index for index in range(num_seeds)]

    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())
    _, observation_template, _ = template_rollout.reset()

    experiment_specs = [
        {
            'noise_kwargs': {'noise_type': 'constant', 'std': 0.0},
            'label': 'torch | no noise ',
            'summary_label': f'=== Torch Monte Carlo horizon {horizon:3d}, no exploration noise ===',
            'elapsed_label': 'torch no noise',
            'color': 'green',
        },
        {
            'noise_kwargs': {'noise_type': 'constant', 'std': 1.0},
            'label': 'torch | noise=1.0 ',
            'summary_label': (
                f'=== Torch Monte Carlo horizon {horizon:3d}, constant exploration noise 1.0 ==='
            ),
            'elapsed_label': 'torch noise = 1.0',
            'color': 'black',
        },
        {
            'noise_kwargs': {'noise_type': 'constant', 'std': 3.0},
            'label': 'torch | noise=3.0 ',
            'summary_label': (
                f'=== Torch Monte Carlo horizon {horizon:3d}, constant exploration noise 3.0 ==='
            ),
            'elapsed_label': 'torch noise = 3.0',
            'color': 'blue',
        },
        # {
        #     'noise_kwargs': {'noise_type': 'constant', 'std': 5.0},
        #     'label': 'torch | noise=5.0 ',
        #     'summary_label': (
        #         f'=== Torch Monte Carlo horizon {horizon:3d}, constant exploration noise 5.0 ==='
        #     ),
        #     'elapsed_label': 'torch noise = 5.0',
        #     'color': 'yellow',
        # },
        # {
        #     'noise_kwargs': {
        #         'noise_type': 'decay',
        #         'start_std': 1.0,
        #         'end_std': 0.0,
        #         'num_iterations': horizon,
        #     },
        #     'label': 'torch | noise=decay ',
        #     'summary_label': (
        #         f'=== Torch Monte Carlo horizon {horizon:3d}, decay exploration noise 1.0->0.0 ==='
        #     ),
        #     'elapsed_label': 'torch noise = decay ',
        #     'color': 'red',
        # },
    ]

    plots: List[Dict[str, Any]] = []
    colors: List[str] = []
    for spec in experiment_specs:
        results = run_timed_experiment(
            env=env,
            template_rollout=template_rollout,
            observation_template=observation_template,
            hidden_sizes=hidden_sizes,
            horizon=horizon,
            iterations=iterations,
            seeds=seeds,
            noise_kwargs=spec['noise_kwargs'],
            label=spec['label'],
            summary_label=spec['summary_label'],
            elapsed_label=spec['elapsed_label'],
            batch_size=horizon,
        )
        plots.append(results)
        colors.append(spec['color'])

    plot_results(
        plots,
        colors,
        seeds=seeds,
        output_path=PACKAGE_ROOT / 'results_plot.png',
    )


if __name__ == '__main__':
    main()
