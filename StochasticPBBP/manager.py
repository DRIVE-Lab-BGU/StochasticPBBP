from __future__ import annotations

import csv
import os
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict
import time
import numpy as np

import pyRDDLGym

import torch

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "stochasticpbbp-matplotlib"),
)


from StochasticPBBP.core.Train import Train
from StochasticPBBP.core.R2Trainer import R2Trainer
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Policies import MBDPOPolicy, NeuralStateFeedbackPolicy, TO
from StochasticPBBP.utils.helper import collapse_history_to_iterations
from StochasticPBBP.utils.seeder import FibonacciSeeder
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory, NoiseInfo
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow
from StochasticPBBP.utils.logger import CSVLogger



class ExperimentManager:
    def __init__(self,domain: str,
                 instance: str,
                 seed: int=42,
                 seeds: int=1,
                 eval_seed: int=42,
                 eval_seeds: int=1,
                 horizon: int=100,
                 arch: Tuple[int, ...]= None,
                 fuzzy_weight: float=50.0,
                 learning_rate: float=0.01,
                 noise: Optional[NoiseInfo]=None,
                 exact_eval_mode=False,
                 output_folder=None,
                 policy_type: str='neural') -> None:
        self.env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
        self.env.horizon = horizon
        self.horizon = horizon
        normalized_policy_type = policy_type.strip().lower()
        if normalized_policy_type not in {'neural', 'to'}:
            raise ValueError(
                f'policy_type must be "neural" or "to", got {policy_type!r}.'
            )
        self.policy_type = normalized_policy_type
        if arch is None:
            self.arch = (12,12)
            print("[INFO] No architecture specified, using default (12, 12)")
        else:
            self.arch = arch
        self.lr = learning_rate
        self.seed = seed
        self.seeds = seeds
        self.eval_seed = eval_seed
        self.eval_seeds = eval_seeds
        self.exact_eval_mode = exact_eval_mode
        self.logger = None
        self.output_folder = output_folder
        self.last_sigma_artifacts: Optional[Dict[str, Any]] = None

        # i dont wnat. different seeds for each run i want to be able to reproduce 
        # the same results with the same seed.
        #seed_start = int(str(time.time_ns())[10:13])

        self.train_seeder = FibonacciSeeder(self.seed)
        self.eval_seeder = FibonacciSeeder(self.eval_seed)
        self.noise = dict(noise) if noise is not None else {"type": "constant", "value": 0.0}
        self.noise.setdefault("final", float(self.noise["value"]))
        self.noise.setdefault("alpha", 0.1)
        torch.manual_seed(seed)

        self.template_rollout = TorchRollout(self.env.model, horizon=self.horizon)
        _, self.observation_template, _ = self.template_rollout.reset()

        self.logic = FuzzyLogic(
            tnorm=ProductTNorm(),
            comparison=SigmoidComparison(weight=fuzzy_weight),
            rounding=SoftRounding(weight=fuzzy_weight),
            control=SoftControlFlow(weight=fuzzy_weight),
            sampling=SoftRandomSampling(
                poisson_max_bins=100,
                binomial_max_bins=100,
                bernoulli_gumbel_softmax=False
            )
        )

    def run_experiment(self, iterations: int=100, log_frequency: int=10) -> None:
        # iterations_axis: List[int] = []
        all_returns: List[List[float]] = []
        all_eval_returns: List[List[float]] = []
        all_sigma_matrices: List[np.ndarray] = []
        i = 1
        for seed in range(self.seeds):
            print("[INFO] Starting experiment {}, running {} iterations, with noise {}".format(i, iterations,
                                                                                               self.noise["value"]))
            iterations_i, returns_i, eval_iterations_i, eval_returns_i, policy, sigma_matrix_i = self._run_single_experiment(
                iterations=iterations, log_frequency=log_frequency)
            all_returns.append(returns_i)
            all_eval_returns.append(eval_returns_i)
            if sigma_matrix_i is not None:
                all_sigma_matrices.append(sigma_matrix_i)
            i = i + 1
        if self.exact_eval_mode:
            iterations_axis = eval_iterations_i
            mean, std = self._average_over_returns(all_eval_returns)
        else:
            iterations_axis = iterations_i
            mean, std = self._average_over_returns(all_returns)
        self.last_sigma_artifacts = self._build_sigma_artifacts(all_sigma_matrices)
        return iterations_axis, mean, std

    def log(self, file_name, iterations, returns, stds):
        data = {}
        headers = ["iteration", "mean", "std"]
        data[headers[0]] = iterations
        data[headers[1]] = returns
        data[headers[2]] = stds

        if self.logger is None:
            csv_output_file = os.path.join(self.output_folder, "run_logs", file_name)
            self.logger = CSVLogger(csv_file_name=csv_output_file)
        self.logger.write_CSV(data=data, headers=headers)
        pass

    def _average_over_returns(self, returns: List[List[float]]) -> List[float]:
        rows_avg = np.mean(returns, axis=0)
        rows_std = np.std(returns, axis=0)
        return rows_avg, rows_std

    @staticmethod
    def _history_to_sigma_rows(
        history: Sequence[Dict[str, Any]],
        *,
        iteration_offset: int,
    ) -> List[Tuple[int, List[float]]]:
        sigma_rows: List[Tuple[int, List[float]]] = []
        for item in history:
            sigma_profile = item.get('sigma_profile')
            if not isinstance(sigma_profile, list):
                continue
            sigma_rows.append((
                iteration_offset + int(item['iteration']),
                [float(value) for value in sigma_profile],
            ))
        return sigma_rows

    def _build_sigma_artifacts(
        self,
        sigma_matrices: Sequence[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        if not sigma_matrices:
            return None

        reference_shape = sigma_matrices[0].shape
        if len(reference_shape) != 2:
            raise ValueError(
                f'Sigma matrix must be rank-2, got shape={reference_shape}.'
            )
        for sigma_matrix in sigma_matrices[1:]:
            if sigma_matrix.shape != reference_shape:
                raise ValueError(
                    'All sigma matrices must have the same shape, got '
                    f'{reference_shape} and {sigma_matrix.shape}.'
                )

        stacked = np.stack(sigma_matrices, axis=0)
        mean_sigma_matrix = np.mean(stacked, axis=0)
        iterations_axis = list(range(1, reference_shape[0] + 1))

        mean_sigma: List[float] = []
        min_sigma: List[float] = []
        max_sigma: List[float] = []
        for row in mean_sigma_matrix:
            finite_row = row[np.isfinite(row)]
            if finite_row.size == 0:
                mean_sigma.append(float('nan'))
                min_sigma.append(float('nan'))
                max_sigma.append(float('nan'))
                continue
            mean_sigma.append(float(np.mean(finite_row)))
            min_sigma.append(float(np.min(finite_row)))
            max_sigma.append(float(np.max(finite_row)))

        timesteps_axis = list(range(reference_shape[1]))
        mean_sigma_per_timestep: List[float] = []
        for timestep_index in timesteps_axis:
            timestep_values = mean_sigma_matrix[:, timestep_index]
            finite_values = timestep_values[np.isfinite(timestep_values)]
            if finite_values.size == 0:
                mean_sigma_per_timestep.append(float('nan'))
                continue
            mean_sigma_per_timestep.append(float(np.mean(finite_values)))

        return {
            'iterations': iterations_axis,
            'timesteps': timesteps_axis,
            'mean_sigma_matrix': mean_sigma_matrix,
            'mean_sigma_per_iteration': mean_sigma,
            'min_sigma_per_iteration': min_sigma,
            'max_sigma_per_iteration': max_sigma,
            'mean_sigma_per_timestep': mean_sigma_per_timestep,
            'seed_count': len(sigma_matrices),
        }

    def save_sigma_artifacts(self, *, file_stem: str) -> Optional[Dict[str, str]]:
        if self.last_sigma_artifacts is None:
            return None

        output_dir = os.path.join(self.output_folder, "run_logs")
        os.makedirs(output_dir, exist_ok=True)

        matrix_path = os.path.join(output_dir, f"{file_stem}_sigma_matrix.csv")
        summary_path = os.path.join(output_dir, f"{file_stem}_sigma_summary.csv")
        heatmap_path = os.path.join(output_dir, f"{file_stem}_sigma_heatmap.png")
        iteration_plot_path = os.path.join(output_dir, f"{file_stem}_sigma_mean_by_iteration.png")
        timestep_mean_path = os.path.join(output_dir, f"{file_stem}_sigma_mean_by_timestep.csv")
        timestep_plot_path = os.path.join(output_dir, f"{file_stem}_sigma_mean_by_timestep.png")

        self._write_sigma_matrix_csv(
            matrix_path=matrix_path,
            iterations=self.last_sigma_artifacts['iterations'],
            sigma_matrix=self.last_sigma_artifacts['mean_sigma_matrix'],
        )
        self._write_sigma_summary_csv(
            summary_path=summary_path,
            iterations=self.last_sigma_artifacts['iterations'],
            mean_sigma=self.last_sigma_artifacts['mean_sigma_per_iteration'],
            min_sigma=self.last_sigma_artifacts['min_sigma_per_iteration'],
            max_sigma=self.last_sigma_artifacts['max_sigma_per_iteration'],
        )
        self._save_sigma_heatmap(
            heatmap_path=heatmap_path,
            iterations=self.last_sigma_artifacts['iterations'],
            sigma_matrix=self.last_sigma_artifacts['mean_sigma_matrix'],
            seed_count=int(self.last_sigma_artifacts['seed_count']),
        )
        self._save_sigma_iteration_plot(
            plot_path=iteration_plot_path,
            iterations=self.last_sigma_artifacts['iterations'],
            mean_sigma=self.last_sigma_artifacts['mean_sigma_per_iteration'],
            seed_count=int(self.last_sigma_artifacts['seed_count']),
        )
        self._write_sigma_timestep_mean_csv(
            csv_path=timestep_mean_path,
            timesteps=self.last_sigma_artifacts['timesteps'],
            mean_sigma=self.last_sigma_artifacts['mean_sigma_per_timestep'],
        )
        self._save_sigma_timestep_plot(
            plot_path=timestep_plot_path,
            timesteps=self.last_sigma_artifacts['timesteps'],
            mean_sigma=self.last_sigma_artifacts['mean_sigma_per_timestep'],
            seed_count=int(self.last_sigma_artifacts['seed_count']),
        )

        return {
            'matrix_csv': matrix_path,
            'summary_csv': summary_path,
            'heatmap_png': heatmap_path,
            'iteration_mean_png': iteration_plot_path,
            'timestep_mean_csv': timestep_mean_path,
            'timestep_mean_png': timestep_plot_path,
        }

    @staticmethod
    def _write_sigma_matrix_csv(
        *,
        matrix_path: str,
        iterations: Sequence[int],
        sigma_matrix: np.ndarray,
    ) -> None:
        horizon = int(sigma_matrix.shape[1])
        headers = ['iteration'] + [f'timestep_{step_index}' for step_index in range(horizon)]
        with open(matrix_path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for iteration, row in zip(iterations, sigma_matrix):
                writer.writerow([int(iteration), *map(float, row.tolist())])

    @staticmethod
    def _write_sigma_summary_csv(
        *,
        summary_path: str,
        iterations: Sequence[int],
        mean_sigma: Sequence[float],
        min_sigma: Sequence[float],
        max_sigma: Sequence[float],
    ) -> None:
        with open(summary_path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['iteration', 'mean_sigma', 'min_sigma', 'max_sigma'])
            for row in zip(iterations, mean_sigma, min_sigma, max_sigma):
                iteration, mean_value, min_value, max_value = row
                writer.writerow([
                    int(iteration),
                    float(mean_value),
                    float(min_value),
                    float(max_value),
                ])

    @staticmethod
    def _save_sigma_heatmap(
        *,
        heatmap_path: str,
        iterations: Sequence[int],
        sigma_matrix: np.ndarray,
        seed_count: int,
    ) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "stochasticpbbp-matplotlib"),
        )
        import matplotlib.pyplot as plt

        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(12, 6))
        x_min = float(iterations[0]) - 0.5 if iterations else -0.5
        x_max = float(iterations[-1]) + 0.5 if iterations else float(sigma_matrix.shape[0]) - 0.5
        image = ax.imshow(
            sigma_matrix.T,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            extent=(x_min, x_max, -0.5, float(sigma_matrix.shape[1]) - 0.5),
        )
        ax.set_xlabel('Iteration', fontsize=20)
        ax.set_ylabel('Timestep', fontsize=20)
        ax.set_title(
            'Gradient2Noise sigma heatmap'
            if seed_count == 1 else
            f'Gradient2Noise sigma heatmap | mean across {seed_count} seeds'
        )
        ax.set_xlim(x_min, x_max)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label('sigma')
        fig.tight_layout()
        fig.savefig(heatmap_path)
        plt.close(fig)

    @staticmethod
    def _write_sigma_timestep_mean_csv(
        *,
        csv_path: str,
        timesteps: Sequence[int],
        mean_sigma: Sequence[float],
    ) -> None:
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestep', 'mean_sigma'])
            for timestep, mean_value in zip(timesteps, mean_sigma):
                writer.writerow([int(timestep), float(mean_value)])

    @staticmethod
    def _save_sigma_timestep_plot(
        *,
        plot_path: str,
        timesteps: Sequence[int],
        mean_sigma: Sequence[float],
        seed_count: int,
    ) -> None:
        import matplotlib.pyplot as plt

        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timesteps, mean_sigma, color='tab:green', linewidth=2.0)
        ax.set_xlabel('Timestep', fontsize=20)
        ax.set_ylabel('Mean sigma', fontsize=20)
        ax.set_title(
            'Gradient2Noise mean sigma by timestep'
            if seed_count == 1 else
            f'Gradient2Noise mean sigma by timestep | mean across {seed_count} seeds'
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    @staticmethod
    def _save_sigma_iteration_plot(
        *,
        plot_path: str,
        iterations: Sequence[int],
        mean_sigma: Sequence[float],
        seed_count: int,
    ) -> None:
        import matplotlib.pyplot as plt

        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iterations, mean_sigma, color='tab:blue', linewidth=2.0)
        ax.set_xlabel('Iteration', fontsize=20)
        ax.set_ylabel('Mean sigma', fontsize=20)
        ax.set_title(
            'Gradient2Noise mean sigma by iteration'
            if seed_count == 1 else
            f'Gradient2Noise mean sigma by iteration | mean across {seed_count} seeds'
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    def _build_additive_noise(self, iterations: int):
        return AdditiveNoiseFactory.create(
            noise_type=self.noise["type"],
            std=self.noise["value"],
            start_std=self.noise["value"],
            end_std=self.noise["final"],
            alpha = self.noise["alpha"],
            num_iterations=iterations,
            source=self.template_rollout,
        )

    def _build_trainer(self, *, policy, iterations: int):
        additive_noise = self._build_additive_noise(iterations)
        if self.noise["type"] != "gradient2noise":
            return Train(
                horizon=self.horizon,
                model=self.env.model,
                action_space=self.env.action_space,
                policy=policy,
                logic=self.logic,
                lr=self.lr,
                hidden_sizes=self.arch,
                batch_size=self.horizon,
                seed=self.seed,
                additive_noise=additive_noise,
            )

        analysis_additive_noise = AdditiveNoiseFactory.create(
            noise_type='constant',
            std=0.0,
            source=self.template_rollout,
        )
        return R2Trainer(
            model=self.env.model,
            action_space=self.env.action_space,
            policy=policy,
            horizon=self.horizon,
            hidden_sizes=self.arch,
            additive_noise=additive_noise,
            analysis_additive_noise=analysis_additive_noise,
            logic=self.logic,
            lr=self.lr,
            seed=self.seed,
        )

    def _history_to_iterations(self, history):
        if not history:
            return [], []
        first_item = history[0]
        if "analysis_return" in first_item:
            return (
                [int(item["iteration"]) for item in history],
                [float(item["analysis_return"]) for item in history],
            )
        return collapse_history_to_iterations(
            history,
            label="",
            seed=self.seed,
        )

    def _build_policy(self):
        if self.policy_type == 'neural':
            return NeuralStateFeedbackPolicy(
                observation_template=self.observation_template,
                action_template=self.template_rollout.noop_actions,
                hidden_sizes=self.arch,
                action_space=self.env.action_space,
                seed=next(self.train_seeder),
            )
        if self.policy_type == 'to':
            return TO(
                action_template=self.template_rollout.noop_actions,
                horizon=self.horizon,)
        raise RuntimeError(f'Unsupported policy_type={self.policy_type!r}.')

    def _run_single_experiment(self, iterations: int=100, log_frequency: int=10) -> None:
        policy = self._build_policy()
        trainer = self._build_trainer(policy=policy, iterations=iterations)
        eval_returns = []
        eval_iterations = []
        all_train_iterations = []
        all_train_returns = []
        sigma_rows: List[Tuple[int, List[float]]] = []
        chunks = -(-iterations // log_frequency)
        to_go = iterations
        print_iter = 0

        # log = log_frequency
        # evaluate policy at beginning
        if self.exact_eval_mode:
            # log = 0
            self.eval_seeder.reset()
            result = policy.evaluate(self.env, episodes=self.eval_seeds, seed_generator=self.eval_seeder)
            eval_returns.append(result['mean'])
            eval_iterations.append(print_iter)
            print('[INFO] iter={:4d}, steps={:3d}, discounted return={:.2f}, std={:.2f}'.format(print_iter, self.horizon, result['mean'],
                                                                                  result['std']))

        all_train_iterations.append(0)

        # execute training with evaluation on pyrddlgym
        for i in range(chunks):
            to_run = min(to_go, log_frequency)
            history, trained_policy = trainer.train_trajectory(
                iterations=to_run,
                print_every=0,
                batch_size=self.horizon,  # why again?
            )
            to_go = to_go - log_frequency
            sigma_rows.extend(
                self._history_to_sigma_rows(
                    history,
                    iteration_offset=all_train_iterations[-1],
                )
            )
            train_iterations, train_returns = self._history_to_iterations(history)

            # evaluate policy
            if self.exact_eval_mode:
                self.eval_seeder.reset()
                result = policy.evaluate(self.env, episodes=self.eval_seeds, seed_generator=self.eval_seeder)
                eval_returns.append(result['mean'])

                print_iter = print_iter + to_run
                eval_iterations.append(print_iter)
                print('[INFO] iter={:4d}, steps={:3d}, discounted return={:.2f}, std={:.2f}'.format(print_iter, self.horizon, result['mean'], result['std']))
            else:
                print('[INFO] iter={:4d}, steps={:3d}, discounted return={:.2f}'.format(all_train_iterations[-1]+train_iterations[-1], self.horizon,
                                                                                      train_returns[0]))

            all_train_returns.extend(train_returns)
            all_train_iterations.extend(list(map(lambda x: x + all_train_iterations[-1], train_iterations)))
        sigma_matrix = None
        if sigma_rows:
            sigma_rows.sort(key=lambda row: row[0])
            sigma_matrix = np.asarray([row for (_, row) in sigma_rows], dtype=np.float64)

        return all_train_iterations[1:], all_train_returns, eval_iterations, eval_returns, policy, sigma_matrix
