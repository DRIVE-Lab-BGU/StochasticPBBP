from __future__ import annotations

import csv
import secrets
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyRDDLGym
import torch
from torch import nn

from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.core.Train import Train
from StochasticPBBP.utils.Noise import AdditiveNoise, NoAdditiveNoise
from StochasticPBBP.utils.Policies import StationaryMarkov, NeuralStateFeedbackPolicy, MBDPOPolicy, state2action

PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_ROOT / 'outputs'

TEMP_CSV_FIELDS = [
    'iteration',
    'return',
]


def write_csv_rows(csv_path: Path,
                   rows: Iterable[Dict[str, Any]],
                   *,
                   fieldnames: Sequence[str],
                   append: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    mode = 'a' if append else 'w'
    with csv_path.open(mode, newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not append or not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ExperimentManager:
    def __init__(self,
                 domain: Path | str,
                 instance: Path | str,
                 *,
                 horizon: int,
                 hidden_sizes: Tuple[int, ...]=(64, 64),
                 lr: float=1e-2,
                 batch_size: Optional[int]=None,
                 batch_num: int=1,
                 grad_clip_norm: float=100.0,
                 seed: Optional[int]=None,
                 output_dir: Optional[Path | str]=None,
                 evaluation_seeds: Optional[Sequence[int]]=None,
                 exact_evaluation: bool=True,
                 debug_logging: bool=False,
                 policy: Optional[nn.Module]=None,
                 logic: Optional[Any]=None,
                 additive_noise: Optional[AdditiveNoise]=None,
                 vectorized: bool=True) -> None:
        self.domain = Path(domain)
        self.instance = Path(instance)
        self.horizon = int(horizon)
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = float(lr)
        self.batch_size = batch_size
        self.batch_num = int(batch_num)
        self.grad_clip_norm = float(grad_clip_norm)
        self.seed = self._resolve_training_seed(seed)
        self.output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
        self.evaluation_seeds = [int(local_seed) for local_seed in evaluation_seeds] if evaluation_seeds else [self.seed]
        self.exact_evaluation = bool(exact_evaluation)
        self.debug_logging = bool(debug_logging)
        self.logic = logic
        self.default_additive_noise = additive_noise
        self.vectorized = vectorized

        self.env = pyRDDLGym.make(
            domain=str(self.domain),
            instance=str(self.instance),
            vectorized=self.vectorized,
        )
        self.template_rollout = TorchRollout(self.env.model, horizon=self.horizon, logic=self.logic)
        _, self.observation_template, _ = self.template_rollout.reset()

        self.policy = policy if policy is not None else self._build_policy()
        self.trainer = Train(
            horizon=self.horizon,
            model=self.env.model,
            action_space=self.env.action_space,
            policy=self.policy,
            lr=self.lr,
            hidden_sizes=self.hidden_sizes,
            batch_size=self.batch_size,
            batch_num=self.batch_num,
            # grad_clip_norm=self.grad_clip_norm,
            seed=self.seed,
            additive_noise=self.default_additive_noise,
            logic=self.logic,
        )

        self._completed_iterations = 0
        self._train_call_count = 0
        self._history: List[Dict[str, Any]] = []

    # def _build_policy(self) -> MBDPOPolicy:
    #     return NeuralStateFeedbackPolicy(
    #         observation_template=self.observation_template,
    #         action_template=self.template_rollout.noop_actions,
    #         action_space=self.env.action_space,
    #         hidden_sizes=self.hidden_sizes
    #     )

    def _build_policy(self) -> nn.Module:
        return state2action(
            observation_template=self.observation_template,
            action_template=self.template_rollout.noop_actions,
            hidden_sizes=self.hidden_sizes
        )

    # def _build_policy(self) -> StationaryMarkov:
    #     return StationaryMarkov(
    #         observation_template=self.observation_template,
    #         action_template=self.template_rollout.noop_actions,
    #         action_space=self.env.action_space,
    #         hidden_sizes=self.hidden_sizes,
    #     )

    @staticmethod
    def _resolve_training_seed(seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        return int(secrets.randbelow(2**31 - 1))

    def _spawn_for_seed(self, seed: int) -> 'ExperimentManager':
        return ExperimentManager(
            domain=self.domain,
            instance=self.instance,
            horizon=self.horizon,
            hidden_sizes=self.hidden_sizes,
            lr=self.lr,
            batch_size=self.batch_size,
            batch_num=self.batch_num,
            grad_clip_norm=self.grad_clip_norm,
            seed=seed,
            output_dir=self.output_dir,
            evaluation_seeds=self.evaluation_seeds,
            exact_evaluation=self.exact_evaluation,
            debug_logging=self.debug_logging,
            logic=self.logic,
            additive_noise=self.default_additive_noise,
            vectorized=self.vectorized,
        )

    @staticmethod
    def _offset_history(history: Sequence[Dict[str, Any]],
                        *,
                        iteration_offset: int) -> List[Dict[str, Any]]:
        adjusted: List[Dict[str, Any]] = []
        for item in history:
            local_item = deepcopy(item)
            local_item['iteration'] = float(int(local_item['iteration']) + iteration_offset)
            adjusted.append(local_item)
        return adjusted

    @staticmethod
    def _returns_by_iteration(history: Sequence[Dict[str, Any]]) -> Dict[int, float]:
        returns_by_iteration: Dict[int, float] = {}
        for item in history:
            iteration = int(item['iteration'])
            returns_by_iteration.setdefault(iteration, 0.0)
            returns_by_iteration[iteration] += float(item['return'])
        return returns_by_iteration

    def _iteration_rows_to_csv(self,
                               history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        returns_by_iteration = self._returns_by_iteration(history)
        rows: List[Dict[str, Any]] = []
        for iteration in sorted(returns_by_iteration):
            rows.append({
                'iteration': iteration,
                'return': returns_by_iteration[iteration],
            })
        return rows

    @staticmethod
    def _seed_column_name(seed: int) -> str:
        return f'seed_{seed}'

    @staticmethod
    def _describe_noise(additive_noise: Optional[AdditiveNoise]) -> str:
        if additive_noise is None:
            return 'default'
        return type(additive_noise).__name__

    def _debug_run_dir(self, csv_name: str) -> Path:
        return self.output_dir / 'debug_logs' / f'{Path(csv_name).stem}_seed_{self.seed}'

    @staticmethod
    def _gradient_norm_by_parameter(policy: nn.Module) -> Dict[str, float]:
        norms: Dict[str, float] = {}
        total = 0.0
        for name, parameter in policy.named_parameters():
            if parameter.grad is None:
                continue
            grad_norm = float(parameter.grad.detach().norm().item())
            norms[name] = grad_norm
            total += grad_norm * grad_norm
        norms['total_gradient_norm'] = total ** 0.5
        return norms

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace('.', '_').replace('/', '_')

    def _write_debug_csv(self, csv_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        write_csv_rows(
            csv_path,
            rows,
            fieldnames=list(rows[0].keys()),
            append=False,
        )

    def _plot_gradient_norms(self,
                             debug_dir: Path,
                             gradient_rows: Sequence[Dict[str, Any]]) -> None:
        if not gradient_rows:
            return
        iterations = [int(row['iteration']) for row in gradient_rows]
        ignored = {'iteration', 'total_accumulated_reward', 'eval_mean_return', 'eval_std_return'}
        parameter_keys = [key for key in gradient_rows[0].keys() if key not in ignored]
        plt.switch_backend('Agg')
        for key in parameter_keys:
            values = [float(row.get(key, 0.0)) for row in gradient_rows]
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(iterations, values, linewidth=2.0)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Gradient norm')
            ax.set_title(key)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(debug_dir / f'gradient_norm_{self._sanitize_name(key)}.png')
            plt.close(fig)

    def _plot_weight_histograms(self,
                                debug_dir: Path,
                                weight_snapshots: Sequence[Dict[str, Any]]) -> None:
        if not weight_snapshots:
            return
        plt.switch_backend('Agg')
        parameter_names = list(weight_snapshots[0]['weights'].keys())
        for parameter_name in parameter_names:
            snapshot_values = [
                np.asarray(snapshot['weights'][parameter_name], dtype=float)
                for snapshot in weight_snapshots
            ]
            if not snapshot_values:
                continue
            flat_values = np.concatenate(snapshot_values)
            if flat_values.size == 0:
                continue
            bin_edges = np.histogram_bin_edges(flat_values, bins=30)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            for snapshot in weight_snapshots:
                iteration = int(snapshot['iteration'])
                counts, _ = np.histogram(snapshot['weights'][parameter_name], bins=bin_edges)
                xs = bin_centers
                ys = np.full_like(xs, float(iteration), dtype=float)
                zs = np.zeros_like(xs, dtype=float)
                ax.bar3d(
                    xs,
                    ys,
                    zs,
                    bin_width,
                    max(1.0, 0.6 * weight_snapshots[-1]['iteration'] / max(len(weight_snapshots), 1)),
                    counts.astype(float),
                    shade=True,
                    alpha=0.75,
                )
            ax.set_xlabel('Weight value')
            ax.set_ylabel('Iteration')
            ax.set_zlabel('Count')
            ax.set_title(parameter_name)
            fig.tight_layout()
            fig.savefig(debug_dir / f'weights_hist_3d_{self._sanitize_name(parameter_name)}.png')
            plt.close(fig)

    def _write_weight_snapshots(self,
                                debug_dir: Path,
                                weight_snapshots: Sequence[Dict[str, Any]]) -> None:
        rows: List[Dict[str, Any]] = []
        for snapshot in weight_snapshots:
            iteration = int(snapshot['iteration'])
            for parameter, values in snapshot['weights'].items():
                flattened = np.asarray(values, dtype=float).reshape(-1)
                rows.append({
                    'iteration': iteration,
                    'parameter': parameter,
                    'mean': float(flattened.mean()),
                    'std': float(flattened.std()),
                    'min': float(flattened.min()),
                    'max': float(flattened.max()),
                    'numel': int(flattened.size),
                })
        self._write_debug_csv(debug_dir / 'weight_stats.csv', rows)

    def _capture_weight_snapshot(self) -> Dict[str, Any]:
        weights: Dict[str, Any] = {}
        for name, parameter in self.policy.named_parameters():
            if not name.endswith('weight'):
                continue
            weights[name] = parameter.detach().cpu().reshape(-1).numpy().copy()
        return {
            'iteration': self._completed_iterations,
            'weights': weights,
        }

    def _write_debug_metadata(self,
                              debug_dir: Path,
                              gradient_rows: Sequence[Dict[str, Any]]) -> None:
        rows: List[Dict[str, Any]] = []
        for row in gradient_rows:
            rows.append({
                'iteration': int(row['iteration']),
                'total_accumulated_reward': float(row['total_accumulated_reward']),
                'eval_mean_return': float(row['eval_mean_return']),
                'eval_std_return': float(row['eval_std_return']),
            })
        self._write_debug_csv(debug_dir / 'debug_metrics.csv', rows)

    def _finalize_debug_logs(self,
                             csv_name: str,
                             gradient_rows: Sequence[Dict[str, Any]],
                             weight_snapshots: Sequence[Dict[str, Any]]) -> None:
        debug_dir = self._debug_run_dir(csv_name)
        debug_dir.mkdir(parents=True, exist_ok=True)
        self._write_debug_csv(debug_dir / 'gradient_norms.csv', gradient_rows)
        self._write_debug_metadata(debug_dir, gradient_rows)
        self._write_weight_snapshots(debug_dir, weight_snapshots)
        self._plot_gradient_norms(debug_dir, gradient_rows)
        self._plot_weight_histograms(debug_dir, weight_snapshots)

    @staticmethod
    def _temp_csv_path(summary_csv_path: Path, seed: int) -> Path:
        return summary_csv_path.with_name(f'{summary_csv_path.stem}_seed_{seed}.tmp.csv')

    @staticmethod
    def _read_temp_seed_csv(csv_path: Path) -> Dict[int, float]:
        rows_by_iteration: Dict[int, float] = {}
        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows_by_iteration[int(row['iteration'])] = float(row['return'])
        return rows_by_iteration

    def _write_seed_temp_csv(self,
                             summary_csv_path: Path,
                             *,
                             seed: int,
                             history: Sequence[Dict[str, Any]]) -> Path:
        temp_path = self._temp_csv_path(summary_csv_path, seed)
        write_csv_rows(
            temp_path,
            self._iteration_rows_to_csv(history),
            fieldnames=TEMP_CSV_FIELDS,
            append=False,
        )
        return temp_path

    def _write_summary_csv(self,
                           summary_csv_path: Path,
                           *,
                           temp_paths: Sequence[Path],
                           seeds: Sequence[int]) -> None:
        per_seed_returns = {
            int(seed): self._read_temp_seed_csv(temp_path)
            for seed, temp_path in zip(seeds, temp_paths)
        }
        iteration_axis = sorted(next(iter(per_seed_returns.values()))) if per_seed_returns else []
        fieldnames = ['iteration'] + [self._seed_column_name(int(seed)) for seed in seeds] + [
            'mean_over_seeds',
            'std_over_seeds',
        ]
        rows: List[Dict[str, Any]] = []
        for iteration in iteration_axis:
            iteration_values = [per_seed_returns[int(seed)][iteration] for seed in seeds]
            iteration_tensor = torch.tensor(iteration_values, dtype=torch.float32)
            row: Dict[str, Any] = {
                'iteration': iteration,
                'mean_over_seeds': float(iteration_tensor.mean().item()),
                'std_over_seeds': float(iteration_tensor.std(unbiased=False).item()),
            }
            for seed, value in zip(seeds, iteration_values):
                row[self._seed_column_name(int(seed))] = float(value)
            rows.append(row)
        write_csv_rows(summary_csv_path, rows, fieldnames=fieldnames, append=False)

    @staticmethod
    def _cleanup_temp_csvs(temp_paths: Sequence[Path]) -> None:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()

    def collapse_history_to_iterations(self,
                                       history: List[Dict[str, Any]],
                                       *,
                                       iterations: int) -> Tuple[List[int], List[float]]:
        if not history:
            return [], []

        returns_by_iteration = self._returns_by_iteration(history)

        iteration_axis = sorted(returns_by_iteration)
        expected_iteration_axis = list(range(1, iterations + 1))
        if iteration_axis != expected_iteration_axis:
            raise ValueError(
                f'Expected iterations {expected_iteration_axis[:3]}...{expected_iteration_axis[-3:]}, '
                f'got {iteration_axis[:3]}...{iteration_axis[-3:]}.'
            )

        return iteration_axis, [returns_by_iteration[iteration] for iteration in iteration_axis]

    def Train(self,
              iterations: int,
              *,
              csv_name: str,
              log_every: int=0,
              batch_size: Optional[int]=None,
              batch_num: Optional[int]=None,
              additive_noise: Optional[AdditiveNoise]=None,
              write_csv: bool=True) -> Dict[str, Any]:
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        effective_batch_num = self.batch_num if batch_num is None else batch_num
        effective_noise = self.default_additive_noise if additive_noise is None else additive_noise
        training_csv_path = self.output_dir / 'training_logs' / csv_name
        evaluation_csv_path = self.output_dir / csv_name
        effective_log_every = iterations if log_every <= 0 else int(log_every)
        print(
            f"[ExperimentManager] train start "
            f"iterations={iterations} horizon={self.horizon} "
            f"batch_size={effective_batch_size} batch_num={effective_batch_num} "
            f"grad_clip_norm={self.grad_clip_norm} "
            f"noise={self._describe_noise(effective_noise)} log_every={effective_log_every} "
            f"evaluation_mode={'exact' if self.exact_evaluation else 'approx'} "
            f"debug_logging={self.debug_logging} "
            f"evaluation_seeds={self.evaluation_seeds} "
            f"train_csv={training_csv_path} eval_csv={evaluation_csv_path}"
        )
        start_time = time.perf_counter()
        remaining_iterations = int(iterations)
        chunk_histories: List[Dict[str, Any]] = []
        evaluation_rows: List[Dict[str, Any]] = []
        gradient_rows: List[Dict[str, Any]] = []
        weight_snapshots: List[Dict[str, Any]] = []

        while remaining_iterations > 0:
            current_chunk = min(effective_log_every, remaining_iterations)
            history, trained_policy = self.trainer.train_trajectory(
                iterations=current_chunk,
                print_every=0,
                batch_size=batch_size,
                batch_num=batch_num,
                additive_noise=additive_noise,
            )
            self.policy = trained_policy

            self._train_call_count += 1
            offset_history = self._offset_history(
                history,
                iteration_offset=self._completed_iterations,
            )
            self._completed_iterations += current_chunk
            self._history.extend(offset_history)
            chunk_histories.extend(offset_history)

            eval_mean, eval_std = self.Evaluate()
            evaluation_rows.append({
                'iteration': self._completed_iterations,
                self._seed_column_name(self.seed): eval_mean,
                'mean_over_seeds': eval_mean,
                'std_over_seeds': eval_std,
            })
            latest_training_return = self._returns_by_iteration(self._history)[self._completed_iterations]
            if self.debug_logging:
                gradient_row = {
                    'iteration': self._completed_iterations,
                    'total_accumulated_reward': float(latest_training_return),
                    'eval_mean_return': float(eval_mean),
                    'eval_std_return': float(eval_std),
                }
                gradient_row.update(self._gradient_norm_by_parameter(self.policy))
                gradient_rows.append(gradient_row)
                weight_snapshots.append(self._capture_weight_snapshot())
            remaining_iterations -= current_chunk

        if write_csv:
            write_csv_rows(
                training_csv_path,
                (
                    {
                        'iteration': row['iteration'],
                        self._seed_column_name(self.seed): row['return'],
                        'mean_over_seeds': row['return'],
                        'std_over_seeds': 0.0,
                    }
                    for row in self._iteration_rows_to_csv(chunk_histories)
                ),
                fieldnames=[
                    'iteration',
                    self._seed_column_name(self.seed),
                    'mean_over_seeds',
                    'std_over_seeds',
                ],
                append=self._train_call_count > len(evaluation_rows),
            )
            write_csv_rows(
                evaluation_csv_path,
                evaluation_rows,
                fieldnames=[
                    'iteration',
                    self._seed_column_name(self.seed),
                    'mean_over_seeds',
                    'std_over_seeds',
                ],
                append=self._completed_iterations > len(evaluation_rows),
            )

        iteration_axis, training_returns = self.collapse_history_to_iterations(
            self._history,
            iterations=self._completed_iterations,
        )
        if self.debug_logging:
            self._finalize_debug_logs(csv_name, gradient_rows, weight_snapshots)
        evaluation_iterations = [int(row['iteration']) for row in evaluation_rows]
        evaluation_returns = [float(row['mean_over_seeds']) for row in evaluation_rows]
        elapsed = time.perf_counter() - start_time
        if training_returns:
            print(
                f"[ExperimentManager] train done "
                f"total_iterations={self._completed_iterations} "
                f"last_train_return={training_returns[-1]:.4f} "
                f"last_eval_return={evaluation_returns[-1]:.4f} "
                f"elapsed={elapsed:.2f}s"
            )
        else:
            print(
                f"[ExperimentManager] train done "
                f"total_iterations={self._completed_iterations} "
                f"elapsed={elapsed:.2f}s"
            )
        return {
            'policy': self.policy,
            'history': chunk_histories,
            'all_history': list(self._history),
            'iterations': iteration_axis,
            'training_returns': training_returns,
            'training_csv_path': training_csv_path,
            'evaluation_csv_path': evaluation_csv_path,
            'evaluation_iterations': evaluation_iterations,
            'evaluation_returns': evaluation_returns,
        }

    def get_trained_policy(self) -> nn.Module:
        return self.policy

    def _observation_to_tensor_dict(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tensor_observation: Dict[str, torch.Tensor] = {}
        for name, template in self.observation_template.items():
            if name not in observation:
                raise KeyError(f'Missing observation fluent <{name}> during evaluation.')
            reference = template if isinstance(template, torch.Tensor) else torch.as_tensor(template)
            tensor = torch.as_tensor(
                observation[name],
                dtype=torch.float32,
                device=reference.device,
            )
            if tuple(tensor.shape) != tuple(reference.shape):
                raise ValueError(
                    f'Observation <{name}> must have shape {tuple(reference.shape)}, '
                    f'got {tuple(tensor.shape)}.'
                )
            tensor_observation[name] = tensor
        return tensor_observation

    @staticmethod
    def _action_to_env_dict(action: Dict[str, Any]) -> Dict[str, Any]:
        env_action: Dict[str, Any] = {}
        for name, value in action.items():
            if isinstance(value, torch.Tensor):
                detached = value.detach().cpu()
                env_action[name] = detached.item() if detached.numel() == 1 else detached.numpy()
            else:
                env_action[name] = value
        return env_action

    @staticmethod
    def _unpack_reset(reset_result: Any) -> Dict[str, Any]:
        if isinstance(reset_result, tuple):
            if not reset_result:
                raise ValueError('env.reset() returned an empty tuple.')
            return reset_result[0]
        return reset_result

    @staticmethod
    def _unpack_step(step_result: Any) -> Tuple[Dict[str, Any], float, bool]:
        if not isinstance(step_result, tuple):
            raise ValueError('env.step(action) must return a tuple.')
        if len(step_result) == 5:
            observation, reward, terminated, truncated, _ = step_result
            done = bool(terminated) or bool(truncated)
            return observation, float(reward), done
        if len(step_result) == 4:
            observation, reward, done, _ = step_result
            return observation, float(reward), bool(done)
        if len(step_result) == 3:
            observation, reward, done = step_result
            return observation, float(reward), bool(done)
        raise ValueError(f'Unsupported env.step(action) return format of length {len(step_result)}.')

    def _evaluation_policy(self,
                           observation: Dict[str, torch.Tensor],
                           step: int,
                           policy_state: Any) -> Dict[str, Any]:
        del step, policy_state
        if hasattr(self.policy, 'sample_action'):
            return self.policy.sample_action(observation)
        with torch.no_grad():
            return self.policy(observation, step=None, policy_state=None)

    def EvaluateExact(self,
                      evaluation_seeds: Optional[Sequence[int]]=None,
                      *,
                      horizon: Optional[int]=None) -> Tuple[float, float]:
        effective_eval_seeds = self.evaluation_seeds if evaluation_seeds is None else [
            int(local_seed) for local_seed in evaluation_seeds
        ]
        if not effective_eval_seeds:
            raise ValueError('evaluation_seeds must contain at least one seed.')

        evaluation_horizon = self.horizon if horizon is None else int(horizon)
        print(
            f"[ExperimentManager] eval start "
            f"iteration={self._completed_iterations} "
            f"evaluation_seeds={effective_eval_seeds} horizon={evaluation_horizon}"
        )
        start_time = time.perf_counter()
        was_training = self.policy.training
        self.policy.eval()
        returns: List[float] = []

        try:
            for evaluation_seed in effective_eval_seeds:
                env = pyRDDLGym.make(
                    domain=str(self.domain),
                    instance=str(self.instance),
                    vectorized=self.vectorized,
                )
                try:
                    observation = self._unpack_reset(env.reset(seed=int(evaluation_seed)))
                    total_return = 0.0

                    for step in range(evaluation_horizon):
                        tensor_observation = self._observation_to_tensor_dict(observation)
                        action = self._evaluation_policy(tensor_observation, step, None)
                        observation, reward, done = self._unpack_step(
                            env.step(self._action_to_env_dict(action))
                        )
                        total_return += reward
                        if done:
                            break

                    returns.append(total_return)
                finally:
                    close_fn = getattr(env, 'close', None)
                    if callable(close_fn):
                        close_fn()
        finally:
            if was_training:
                self.policy.train()

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        mean_return = float(returns_tensor.mean().item())
        std_return = float(returns_tensor.std(unbiased=False).item())
        elapsed = time.perf_counter() - start_time
        print(
            f"[ExperimentManager] eval done "
            f"mean_return={mean_return:.4f} std_return={std_return:.4f} "
            f"elapsed={elapsed:.2f}s"
        )
        return mean_return, std_return

    def Evaluate(self,
                 evaluation_seeds: Optional[Sequence[int]]=None,
                 *,
                 horizon: Optional[int]=None) -> Tuple[float, float]:
        if self.exact_evaluation:
            return self.EvaluateExact(evaluation_seeds=evaluation_seeds, horizon=horizon)
        return self.EvaluateApprox(evaluation_seeds=evaluation_seeds, horizon=horizon)

    def EvaluateApprox(self,
                       evaluation_seeds: Optional[Sequence[int]]=None,
                       *,
                       horizon: Optional[int]=None) -> Tuple[float, float]:
        effective_eval_seeds = self.evaluation_seeds if evaluation_seeds is None else [
            int(local_seed) for local_seed in evaluation_seeds
        ]
        if not effective_eval_seeds:
            raise ValueError('evaluation_seeds must contain at least one seed.')

        evaluation_horizon = self.horizon if horizon is None else int(horizon)
        print(
            f"[ExperimentManager] approx eval start "
            f"iteration={self._completed_iterations} "
            f"evaluation_seeds={effective_eval_seeds} horizon={evaluation_horizon}"
        )
        start_time = time.perf_counter()
        was_training = self.policy.training
        self.policy.eval()
        returns: List[float] = []

        try:
            for evaluation_seed in effective_eval_seeds:
                torch.manual_seed(int(evaluation_seed))
                self.trainer.rollout.cell.key.manual_seed(int(evaluation_seed))
                with torch.no_grad():
                    trace = self.trainer.rollout(
                        policy=self._evaluation_policy,
                        steps=evaluation_horizon,
                        iteration=self._completed_iterations,
                        additive_noise=NoAdditiveNoise(),
                    )
                returns.append(float(trace.return_.detach()))
        finally:
            if was_training:
                self.policy.train()

        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        mean_return = float(returns_tensor.mean().item())
        std_return = float(returns_tensor.std(unbiased=False).item())
        elapsed = time.perf_counter() - start_time
        print(
            f"[ExperimentManager] approx eval done "
            f"mean_return={mean_return:.4f} std_return={std_return:.4f} "
            f"elapsed={elapsed:.2f}s"
        )
        return mean_return, std_return

    def run_seed_experiment(self,
                            seeds: Optional[Sequence[int]]=None,
                            *,
                            iterations: int,
                            csv_name: str,
                            log_every: int=0,
                            batch_size: Optional[int]=None,
                            batch_num: Optional[int]=None,
                            additive_noise: Optional[AdditiveNoise]=None,
                            evaluate_runs: int=0) -> Dict[str, Any]:
        if seeds is None:
            raise ValueError('seeds must be provided for run_seed_experiment().')
        effective_seeds = [int(local_seed) for local_seed in seeds]
        if not effective_seeds:
            raise ValueError('seeds must contain at least one seed.')

        all_training_returns: List[List[float]] = []
        evaluation_stats: List[Tuple[float, float]] = []
        iteration_axis: List[int] = []
        summary_csv_path = OUTPUT_DIR / csv_name
        temp_paths: List[Path] = []

        for seed in effective_seeds:
            manager = self._spawn_for_seed(int(seed))
            train_result = manager.Train(
                iterations,
                csv_name=csv_name,
                log_every=log_every,
                batch_size=batch_size,
                batch_num=batch_num,
                additive_noise=additive_noise,
                write_csv=False,
            )
            temp_paths.append(
                manager._write_seed_temp_csv(
                    summary_csv_path,
                    seed=int(seed),
                    history=train_result['history'],
                )
            )
            seed_iterations, seed_returns = manager.collapse_history_to_iterations(
                train_result['history'],
                iterations=iterations,
            )
            if not iteration_axis:
                iteration_axis = seed_iterations
            elif iteration_axis != seed_iterations:
                raise ValueError('Training iterations are inconsistent across seeds.')
            all_training_returns.append(seed_returns)

            if evaluate_runs > 0:
                evaluation_stats.append(manager.Evaluate(effective_seeds[:evaluate_runs]))


        returns_tensor = torch.tensor(all_training_returns, dtype=torch.float32)
        self._write_summary_csv(summary_csv_path, temp_paths=temp_paths, seeds=effective_seeds)
        self._cleanup_temp_csvs(temp_paths)

        result: Dict[str, Any] = {
            'seeds': list(effective_seeds),
            'iterations': iteration_axis,
            'all_training_returns': all_training_returns,
            'mean_training_returns': returns_tensor.mean(dim=0).tolist(),
            'std_training_returns': returns_tensor.std(dim=0, unbiased=False).tolist(),
            'csv_path': summary_csv_path,
        }

        if evaluation_stats:
            eval_means = torch.tensor([item[0] for item in evaluation_stats], dtype=torch.float32)
            eval_stds = torch.tensor([item[1] for item in evaluation_stats], dtype=torch.float32)
            result['evaluation_mean_return'] = float(eval_means.mean().item())
            result['evaluation_std_return'] = float(eval_means.std(unbiased=False).item())
            result['evaluation_seed_stds'] = eval_stds.tolist()

        return result


class MultiSeedExperimentManager:
    def __init__(self,
                 domain: Path | str,
                 instance: Path | str,
                 *,
                 num_random_policies: int,
                 evaluation_seeds: Sequence[int],
                 horizon: int,
                 hidden_sizes: Tuple[int, ...]=(64, 64),
                 lr: float=1e-2,
                 batch_size: Optional[int]=None,
                 batch_num: int=1,
                 grad_clip_norm: float=100.0,
                 output_dir: Optional[Path | str]=None,
                 verbose: bool=False,
                 exact_evaluation: bool=True,
                 debug_logging: bool=False,
                 logic: Optional[Any]=None,
                 additive_noise: Optional[AdditiveNoise]=None,
                 vectorized: bool=True) -> None:
        self.domain = Path(domain)
        self.instance = Path(instance)
        self.num_random_policies = int(num_random_policies)
        if self.num_random_policies < 1:
            raise ValueError('num_random_policies must be at least one.')
        self.evaluation_seeds = [int(seed) for seed in evaluation_seeds]
        if not self.evaluation_seeds:
            raise ValueError('evaluation_seeds must contain at least one seed.')
        self.horizon = int(horizon)
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = float(lr)
        self.batch_size = batch_size
        self.batch_num = int(batch_num)
        self.grad_clip_norm = float(grad_clip_norm)
        self.output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
        self.verbose = bool(verbose)
        self.exact_evaluation = bool(exact_evaluation)
        self.debug_logging = bool(debug_logging)
        self.logic = logic
        self.default_additive_noise = additive_noise
        self.vectorized = vectorized
        self.training_seeds = self._generate_training_seeds(self.num_random_policies)

    @staticmethod
    def _generate_training_seeds(num_random_policies: int) -> List[int]:
        seeds: List[int] = []
        seen = set()
        while len(seeds) < num_random_policies:
            seed = int(secrets.randbelow(2**31 - 1))
            if seed in seen:
                continue
            seen.add(seed)
            seeds.append(seed)
        return seeds

    def _spawn_manager(self, seed: int) -> ExperimentManager:
        return ExperimentManager(
            domain=self.domain,
            instance=self.instance,
            horizon=self.horizon,
            hidden_sizes=self.hidden_sizes,
            lr=self.lr,
            batch_size=self.batch_size,
            batch_num=self.batch_num,
            grad_clip_norm=self.grad_clip_norm,
            seed=seed,
            output_dir=self.output_dir,
            evaluation_seeds=self.evaluation_seeds,
            exact_evaluation=self.exact_evaluation,
            debug_logging=self.debug_logging,
            logic=self.logic,
            additive_noise=self.default_additive_noise,
            vectorized=self.vectorized,
        )

    @staticmethod
    def _seed_column_name(seed: int) -> str:
        return f'seed_{seed}'

    def _temp_csv_path(self, summary_csv_path: Path, seed: int) -> Path:
        return summary_csv_path.with_name(f'{summary_csv_path.stem}_seed_{seed}.tmp.csv')

    def _write_seed_temp_csv(self,
                             summary_csv_path: Path,
                             *,
                             seed: int,
                             iterations: Sequence[int],
                             returns: Sequence[float]) -> Path:
        rows = [
            {'iteration': int(iteration), 'return': float(return_)}
            for iteration, return_ in zip(iterations, returns)
        ]
        temp_path = self._temp_csv_path(summary_csv_path, seed)
        write_csv_rows(
            temp_path,
            rows,
            fieldnames=TEMP_CSV_FIELDS,
            append=False,
        )
        return temp_path

    @staticmethod
    def _read_temp_seed_csv(csv_path: Path) -> Dict[int, float]:
        rows_by_iteration: Dict[int, float] = {}
        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows_by_iteration[int(row['iteration'])] = float(row['return'])
        return rows_by_iteration

    def _write_summary_csv(self,
                           summary_csv_path: Path,
                           *,
                           temp_paths: Sequence[Path],
                           training_seeds: Sequence[int]) -> None:
        per_seed_returns = {
            seed: self._read_temp_seed_csv(temp_path)
            for seed, temp_path in zip(training_seeds, temp_paths)
        }
        iteration_axis = sorted(next(iter(per_seed_returns.values()))) if per_seed_returns else []
        fieldnames = (
            ['iteration']
            + [self._seed_column_name(seed) for seed in training_seeds]
            + ['mean_over_seeds', 'std_over_seeds']
        )
        rows: List[Dict[str, Any]] = []
        for iteration in iteration_axis:
            values = [per_seed_returns[seed][iteration] for seed in training_seeds]
            values_tensor = torch.tensor(values, dtype=torch.float32)
            row: Dict[str, Any] = {
                'iteration': int(iteration),
                'mean_over_seeds': float(values_tensor.mean().item()),
                'std_over_seeds': float(values_tensor.std(unbiased=False).item()),
            }
            for seed, value in zip(training_seeds, values):
                row[self._seed_column_name(seed)] = float(value)
            rows.append(row)
        write_csv_rows(summary_csv_path, rows, fieldnames=fieldnames, append=False)

    @staticmethod
    def _cleanup_temp_csvs(temp_paths: Sequence[Path]) -> None:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()

    def Train(self,
              iterations: int,
              *,
              csv_name: str,
              log_every: int=0,
              batch_size: Optional[int]=None,
              batch_num: Optional[int]=None,
              additive_noise: Optional[AdditiveNoise]=None) -> Dict[str, Any]:
        training_summary_csv_path = self.output_dir / 'training_logs' / csv_name
        evaluation_summary_csv_path = self.output_dir / csv_name
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        effective_batch_num = self.batch_num if batch_num is None else batch_num
        effective_noise = self.default_additive_noise if additive_noise is None else additive_noise
        effective_log_every = iterations if log_every <= 0 else int(log_every)
        self.training_seeds = self._generate_training_seeds(self.num_random_policies)
        print(
            f"[MultiSeedExperimentManager] experiment start "
            f"num_random_policies={self.num_random_policies} "
            f"evaluation_mode={'exact' if self.exact_evaluation else 'approx'} "
            f"evaluation_seeds={self.evaluation_seeds} iterations={iterations} horizon={self.horizon} "
            f"batch_size={effective_batch_size} batch_num={effective_batch_num} "
            f"grad_clip_norm={self.grad_clip_norm} "
            f"noise={type(effective_noise).__name__ if effective_noise is not None else 'default'} "
            f"log_every={effective_log_every} verbose={self.verbose} "
            f"debug_logging={self.debug_logging} "
            f"training_csv={training_summary_csv_path} eval_csv={evaluation_summary_csv_path}"
        )
        start_time = time.perf_counter()
        training_temp_paths: List[Path] = []
        evaluation_temp_paths: List[Path] = []
        all_training_returns: List[List[float]] = []
        all_evaluation_returns: List[List[float]] = []
        iteration_axis: List[int] = []
        evaluation_iteration_axis: List[int] = []

        for policy_index, seed in enumerate(self.training_seeds, start=1):
            print(
                f"[MultiSeedExperimentManager] policy start "
                f"policy_run={policy_index}/{self.num_random_policies}"
            )
            manager = self._spawn_manager(seed)
            train_result = manager.Train(
                iterations,
                csv_name=f'{evaluation_summary_csv_path.stem}_seed_{seed}.csv',
                log_every=effective_log_every,
                batch_size=batch_size,
                batch_num=batch_num,
                additive_noise=additive_noise,
                write_csv=False,
            )
            seed_iterations = train_result['iterations']
            seed_returns = train_result['training_returns']
            seed_evaluation_iterations = train_result['evaluation_iterations']
            seed_evaluation_returns = train_result['evaluation_returns']
            if not iteration_axis:
                iteration_axis = list(seed_iterations)
            elif iteration_axis != seed_iterations:
                raise ValueError('Training iterations are inconsistent across seeds.')
            if not evaluation_iteration_axis:
                evaluation_iteration_axis = list(seed_evaluation_iterations)
            elif evaluation_iteration_axis != seed_evaluation_iterations:
                raise ValueError('Evaluation iterations are inconsistent across seeds.')
            all_training_returns.append(list(seed_returns))
            all_evaluation_returns.append(list(seed_evaluation_returns))
            if seed_returns:
                print(
                    f"[MultiSeedExperimentManager] policy done "
                    f"policy_run={policy_index}/{self.num_random_policies} "
                    f"final_train_return={seed_returns[-1]:.4f} "
                    f"final_eval_return={seed_evaluation_returns[-1]:.4f}"
                )
            training_temp_paths.append(
                self._write_seed_temp_csv(
                    training_summary_csv_path,
                    seed=seed,
                    iterations=seed_iterations,
                    returns=seed_returns,
                )
            )
            evaluation_temp_paths.append(
                self._write_seed_temp_csv(
                    evaluation_summary_csv_path,
                    seed=seed,
                    iterations=seed_evaluation_iterations,
                    returns=seed_evaluation_returns,
                )
            )
            completed_training_seeds = self.training_seeds[:len(training_temp_paths)]
            self._write_summary_csv(
                training_summary_csv_path,
                temp_paths=training_temp_paths,
                training_seeds=completed_training_seeds,
            )
            self._write_summary_csv(
                evaluation_summary_csv_path,
                temp_paths=evaluation_temp_paths,
                training_seeds=completed_training_seeds,
            )

        self._cleanup_temp_csvs(training_temp_paths)
        self._cleanup_temp_csvs(evaluation_temp_paths)

        returns_tensor = torch.tensor(all_training_returns, dtype=torch.float32)
        evaluation_tensor = torch.tensor(all_evaluation_returns, dtype=torch.float32)
        result = {
            'training_seeds': list(self.training_seeds),
            'evaluation_seeds': list(self.evaluation_seeds),
            'iterations': iteration_axis,
            'evaluation_iterations': evaluation_iteration_axis,
            'all_training_returns': all_training_returns,
            'all_evaluation_returns': all_evaluation_returns,
            'mean_training_returns': returns_tensor.mean(dim=0).tolist(),
            'std_training_returns': returns_tensor.std(dim=0, unbiased=False).tolist(),
            'mean_evaluation_returns': evaluation_tensor.mean(dim=0).tolist(),
            'std_evaluation_returns': evaluation_tensor.std(dim=0, unbiased=False).tolist(),
            'training_csv_path': training_summary_csv_path,
            'evaluation_csv_path': evaluation_summary_csv_path,
        }
        elapsed = time.perf_counter() - start_time
        if result['mean_training_returns'] and result['mean_evaluation_returns']:
            print(
                f"[MultiSeedExperimentManager] experiment done "
                f"final_mean_train_return={result['mean_training_returns'][-1]:.4f} "
                f"final_mean_eval_return={result['mean_evaluation_returns'][-1]:.4f} "
                f"final_eval_std={result['std_evaluation_returns'][-1]:.4f} "
                f"elapsed={elapsed:.2f}s eval_csv={evaluation_summary_csv_path}"
            )
        else:
            print(
                f"[MultiSeedExperimentManager] experiment done "
                f"elapsed={elapsed:.2f}s "
                f"eval_csv={evaluation_summary_csv_path}"
            )
        return result
