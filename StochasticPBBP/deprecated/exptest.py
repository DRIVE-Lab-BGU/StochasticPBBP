from __future__ import annotations
import time
import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pyRDDLGym
import matplotlib.pyplot as plt

PACKAGE_ROOT = Path(__file__).resolve().parent
print(f"PACKAGE_ROOT={PACKAGE_ROOT}")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.Rollout import TorchRollout
from core.Train import Train
from core.Logic import FuzzyLogic
from utils.Noise import AdditiveNoiseFactory, AdditiveNoise
from utils.Policies import TensorDict, StationaryMarkov


def main() -> None:
    # domain = PACKAGE_ROOT / "problems" / "hvac" / "domain.rddl"
    # instance = PACKAGE_ROOT / "problems" / "hvac" / "instance_2.rddl"
    # domain = PACKAGE_ROOT / "problems" / "hvac" / "domain_c.rddl"
    # instance = PACKAGE_ROOT / "problems" / "hvac" / "instance_2c.rddl"
    domain = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
    instance = PACKAGE_ROOT / "problems" / "reservoir" / "instance_3.rddl"
    # domain = PACKAGE_ROOT / "problems" / "navigation" / "domain.rddl"
    # instance = PACKAGE_ROOT / "problems" / "navigation" / "instance_1.rddl"
    # domain = PACKAGE_ROOT / "problems" / "powergen" / "domain.rddl"
    # instance = PACKAGE_ROOT / "problems" / "powergen" / "instance_1.rddl"

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    horizon = 100
    hidden_sizes = (12, 12)
    iterations = 1000
    num_seeds = 1
    seed_offset = 112
    seeds = [seed_offset + 2*index for index in range(num_seeds)]

    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())
    _, observation_template, _ = template_rollout.reset()

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

    
    def run_experiment(noise_value: AdditiveNoise, label: str, batch_size: int = 1) -> Dict[str, Any]:
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
            )

            history, trained_policy = trainer.train_trajectory(
                iterations=iterations,
                print_every=0,
                batch_size=batch_size,
                additive_noise=trainer.default_additive_noise,
            )
            del trained_policy

            seed_iterations, seed_returns = collapse_history_to_iterations(
                history,
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


    plots = []
    colors = []
    ###########################

    #   torch without noise   #
    #      constant noise 0   #

    ###########################
    start = time.perf_counter()
    print(f"=== Torch Monte Carlo horizon {horizon:3d}, no exploration noise ===")
    noise = AdditiveNoiseFactory.create(
            noise_type='constant',
            std=0.0,
            source=template_rollout,
    )
    results_no_noise = run_experiment(
        noise_value=noise,
        label='torch | no noise ',
        batch_size=horizon,
    )
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch no noise\n\n")
    plots.append(results_no_noise)
    colors.append('green')

    ###########################

    #    torch with noise     #
    #    constant noise 1     #

    ###########################
    start = time.perf_counter()
    print(f"=== Torch Monte Carlo horizon {horizon:3d}, constant exploration noise 1.0 ===")
    noise = AdditiveNoiseFactory.create(
        noise_type='constant',
        std=1.0,
        source=template_rollout,
    )
    results_with_noise1 = run_experiment(
        noise_value=noise,
        label='torch | noise=1.0 ',
        batch_size=horizon,
    )
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch noise = 1.0\n\n")
    plots.append(results_with_noise1)
    colors.append('black')

    ###########################

    #    torch with noise     #
    #    constant noise 3     #

    ###########################
    start = time.perf_counter()
    print(f"=== Torch Monte Carlo horizon {horizon:3d}, constant exploration noise 3.0 ===")
    noise = AdditiveNoiseFactory.create(
        noise_type='constant',
        std=3.0,
        source=template_rollout,
    )
    results_with_noise3 = run_experiment(
        noise_value=noise,
        label='torch | noise=3.0 ',
        batch_size=horizon,
    )
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch noise = 3.0\n\n")
    plots.append(results_with_noise3)
    colors.append('blue')

    ###########################

    #    torch with noise     #
    #    constant noise 5     #

    ###########################
    # start = time.perf_counter()
    # print(f"=== Torch Monte Carlo horizon {horizon:3d}, constant exploration noise 5.0 ===")
    # noise = AdditiveNoiseFactory.create(
    #     noise_type='constant',
    #     std=5.0,
    #     source=template_rollout,
    # )
    # results_with_noise5 = run_experiment(
    #     noise_value=noise,
    #     label='torch | noise=5.0 ',
    #     batch_size=horizon,
    # )
    # end = time.perf_counter()
    # elapsed = end - start
    # print(f"Elapsed time: {elapsed:.6f} seconds torch noise = 5.0\n\n")
    # plots.append(results_with_noise5)
    # colors.append('yellow')

    ###########################

    #    torch with noise     #
    #    decay noise 5->1     #

    ###########################
    # start = time.perf_counter()
    # print(f"=== Torch Monte Carlo horizon {horizon:3d}, decay exploration noise 5.0->1.0 ===")
    # noise = AdditiveNoiseFactory.create(
    #     noise_type='decay',
    #     start_std=5.0,
    #     end_std=0.0,
    #     num_iterations=horizon,
    #     source=template_rollout,
    # )
    # results_with_noise_decay = run_experiment(
    #     noise_value=noise,
    #     label='torch | noise=decay ',
    #     batch_size=horizon,
    # )
    # end = time.perf_counter()
    # elapsed = end - start
    # print(f"Elapsed time: {elapsed:.6f} seconds torch noise = decay \n\n")
    # plots.append(results_with_noise_decay)
    # colors.append('red')


    plt.switch_backend("Agg")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    def plot_mean_with_std_band(
        ax,
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

    for p, c in zip(plots, colors):
        plot_mean_with_std_band(
            ax,
            p,
            color=c,
            linestyle='-',
        )

    ax.set_ylabel('Training return')
    ax.set_title(
        f'Torch with/without noise over {len(seeds)} seeds'
    )
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    output_path = PACKAGE_ROOT / "results_plot.png"
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == '__main__':
    main()
