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

from core.Policies import TensorDict
from core.Rollout import TorchRollout
from core.Train import Train
from core.Logic import FuzzyLogic

class state2action(nn.Module):
    def __init__(self,
                 observation_template: TensorDict,
                 action_template: TensorDict,
                 hidden_sizes: Tuple[int, ...]=(64, 64)) -> None:
        super().__init__()
        if not observation_template:
            raise ValueError('observation_template must contain at least one tensor.')
        if not action_template:
            raise ValueError('action_template must contain at least one tensor.')

        self.observation_specs = self._build_observation_specs(observation_template)
        self.action_specs = self._build_action_specs(action_template)
        self.device = self.observation_specs[0]['device']
        self.dtype = torch.float32

        layers = []
        input_dim = sum(spec['numel'] for spec in self.observation_specs)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        output_dim = sum(spec['numel'] for spec in self.action_specs)
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    @staticmethod
    def _as_tensor(value: Any) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    @classmethod
    def _build_observation_specs(cls, observation_template: TensorDict) -> List[Dict[str, Any]]:
        """
        Builds a list of observation specs from the observation template.
        its get subs and return observation specs for the observation template
        e.g.
        observation_template = {
            'obs1': torch.zeros(2, device='cuda'),
            'obs2': torch.zeros(3, device='cuda'),
        }
        will return
        [
            {'name': 'obs1', 'shape': (2,), 'numel': 2, 'dtype': torch.float32, 'device': device('cuda')},
            {'name': 'obs2', 'shape': (3,), 'numel': 3, 'dtype': torch.float32, 'device': device('cuda')},
        """
        specs: List[Dict[str, Any]] = []
        for (name, template) in observation_template.items():
            tensor = cls._as_tensor(template)
            specs.append({
                'name': name,
                'shape': tuple(tensor.shape),
                'numel': int(tensor.numel()),
                'device': tensor.device,
            })
        return specs

    @classmethod
    def _build_action_specs(cls, action_template: TensorDict) -> List[Dict[str, Any]]:
        """
        
        Builds a list of action specs from the action template.
        its get subs and return action specs for the action template
        e.g.
        action_template = {
            'action1': torch.zeros(2, device='cuda'),
            'action2': torch.zeros(3, device='cuda'),
        }
        will return
        [
            {'name': 'action1', 'shape': (2,), 'numel': 2, 'dtype': torch.float32, 'device': device('cuda')},
            {'name': 'action2', 'shape': (3,), 'numel': 3, 'dtype': torch.float32, 'device': device('cuda')},
        ]
        """
        specs: List[Dict[str, Any]] = []
        for (name, template) in action_template.items():
            tensor = cls._as_tensor(template)
            if not tensor.dtype.is_floating_point:
                raise ValueError(
                    f'state2action supports only floating-point actions, got {name} '
                    f'with dtype {tensor.dtype}.'
                )
            specs.append({
                'name': name,
                'shape': tuple(tensor.shape),
                'numel': int(tensor.numel()),
                'dtype': tensor.dtype,
                'device': tensor.device,
            })
        return specs

    def _flatten_observation(self, observation: TensorDict) -> torch.Tensor:
        """
        Flattens the observation dict into a single tensor by concatenating the tensors in the order of self.observation_specs.
        e.g. if self.observation_specs is
        [
            {'name': 'obs1', 'shape': (2,), 'numel': 2, 'dtype': torch.float32, 'device': device('cuda')},
            {'name': 'obs2', 'shape': (3,), 'numel': 3, 'dtype': torch.float32, 'device': device('cuda')},
        ]
        and the observation is
        {
            'obs1': torch.tensor([1.0, 2.0], device='cuda'),
            'obs2': torch.tensor([3.0, 4.0, 5.0], device='cuda'),
        }
        then it will return
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
        e.g. 
        if we have 3 observation specs and the observation like (temperature, rlevel , sunlight)
        we will flatten them into a single tensor by concatenating them in the order of the observation specs. 
        
        """
        flat_parts: List[torch.Tensor] = []
        for spec in self.observation_specs:
            name = spec['name']
            if name not in observation:
                raise KeyError(f'Missing observation fluent <{name}>.')
            tensor = self._as_tensor(observation[name]).to(device=spec['device'])
            if tuple(tensor.shape) != spec['shape']:
                raise ValueError(
                    f'Observation <{name}> must have shape {spec["shape"]}, '
                    f'got {tuple(tensor.shape)}.'
                )
            flat_parts.append(tensor.to(dtype=self.dtype).reshape(-1))
        return torch.cat(flat_parts, dim=0)

    def _pack_actions(self, flat_action: torch.Tensor) -> TensorDict:
        actions: TensorDict = {}
        start = 0
        for spec in self.action_specs:
            end = start + spec['numel']
            raw_action = flat_action[start:end].reshape(spec['shape'])
            actions[spec['name']] = raw_action.to(dtype=spec['dtype'], device=spec['device'])
            start = end
        return actions

    def forward(self,
                observation: TensorDict,
                step: Optional[int]=None,
                policy_state: Any=None) -> TensorDict:
        del step, policy_state
        flat_observation = self._flatten_observation(observation)
        flat_action = self.network(flat_observation)
        return self._pack_actions(flat_action)


def main() -> None:
    domain = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
    instance = PACKAGE_ROOT / "problems" / "reservoir" / "instance_4.rddl"

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    horizon = int(os.environ.get("NOISE_PLOT_HORIZON", "400"))
    hidden_sizes = (12, 12)
    iterations = int(os.environ.get("NOISE_PLOT_ITERATIONS", "200"))
    num_seeds = int(os.environ.get("NOISE_PLOT_NUM_SEEDS", "20"))
    seed_offset = int(os.environ.get("NOISE_PLOT_SEED_OFFSET", "0"))
    batched_batch_size = int(os.environ.get("NOISE_PLOT_BATCH_SIZE", "10"))
    seeds = [seed_offset + index for index in range(num_seeds)]

    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())
    _, observation_template, _ = template_rollout.reset()
    action_template = template_rollout.noop_actions

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
    
    def init_weights_xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_weights_jax(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def run_experiment(noise_value: float, label: str, batch_size: int=1 , init_weights_fn: str = "jax") -> Dict[str, Any]:
        all_training_returns: List[List[float]] = []
        iteration_axis: List[int] = []
        import torch.nn as nn


        for seed in seeds:
            torch.manual_seed(seed)

            if init_weights_fn == "jax":
                init_weights_fn = init_weights_jax
                policy = state2action(
                    observation_template=observation_template,
                    action_template=action_template,
                    hidden_sizes=hidden_sizes,
                )

                policy.apply(init_weights_jax)
            else:
                init_weights_fn = init_weights_xavier
                policy = state2action(
                    observation_template=observation_template,
                    action_template=action_template,
                    hidden_sizes=hidden_sizes,
                )

                policy.apply(init_weights_xavier)

            trainer = Train(
                horizon=horizon,
                model=env.model,
                action_space=env.action_space,
                policy=policy,
                lr=0.01,
                hidden_sizes=hidden_sizes,
                batch_size=batch_size,
                seed=seed,
                noise_type_dict={'type': 'constant', 'value': noise_value},
            )

            history, trained_policy = trainer.train_trajectory(
                iterations=iterations,
                print_every=0,
                batch_size=batch_size,
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

        print(f"\n=== {label} ===")
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

    def run_jaxplan_experiment(label: str):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                import jax.random as jax_random
                from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxDeepReactivePolicy
                from pyRDDLGym_jax.core.logic import FuzzyLogic as JaxFuzzyLogic
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'JaxPlan comparison requires jax and pyRDDLGym_jax to be installed.'
            ) from exc

        all_training_returns: List[List[float]] = []
        iteration_axis: List[int] = []

        for seed in seeds:
            planner = JaxBackpropPlanner(
                rddl=env.model,
                plan=JaxDeepReactivePolicy(topology=list(hidden_sizes)),
                batch_size_train=1,
                batch_size_test=1,
                rollout_horizon=horizon,
                optimizer_kwargs={'learning_rate': 0.01},
                pgpe=None,
                print_warnings=False,
            )

            seed_returns: List[float] = []
            seed_iterations: List[int] = []
            for callback in planner.optimize_generator(
                key=jax_random.PRNGKey(seed),
                epochs=iterations,
                train_seconds=1.0e9,
                print_summary=False,
                print_progress=False,
            ):
                seed_iterations.append(int(callback['iteration']) + 1)
                seed_returns.append(float(callback['train_return']))

            if len(seed_iterations) != iterations:
                raise ValueError(
                    f'Expected exactly {iterations} JaxPlan history points for seed {seed}, '
                    f'got {len(seed_iterations)}.'
                )
            if not iteration_axis:
                iteration_axis = seed_iterations
            elif iteration_axis != seed_iterations:
                raise ValueError('JaxPlan iterations are inconsistent across seeds.')

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

        print(f"\n=== {label} ===")
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
    ###########################

    #   torch without noise   #
    #      without batch      #

    ###########################
    start = time.perf_counter()
    results_no_noise_no_batch = run_experiment(
        noise_value=0.0,
        label='torch | no noise | batch=1',
        batch_size=1,
    )
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch no noise batch=1")

    ###########################

    #    torch with noise     #
    #      without batch      #

    ###########################
    start = time.perf_counter()
    results_with_noise_no_batch = run_experiment(
        noise_value=3.0,
        label='torch | noise=3.0 | batch=1',
        batch_size=1,
    )
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch noise batch=1")

    ###########################

    #   torch without noise   #
    #       with batch        #

    ###########################
    # start = time.perf_counter()
    # results_no_noise_with_batch = run_experiment(
    #     noise_value=0.0,
    #     label=f'torch | no noise | batch={batched_batch_size}',
    #     batch_size=batched_batch_size,
    # )
    # end = time.perf_counter()
    # elapsed = end - start
    # print(f"Elapsed time: {elapsed:.6f} seconds torch no noise batch={batched_batch_size}")

    ###########################

    #    torch with noise     #
    #       with batch        #

    ###########################
    # start = time.perf_counter()
    # results_with_noise_with_batch = run_experiment(
    #     noise_value=3.0,
    #     label=f'torch | noise=3.0 | batch={batched_batch_size}',
    #     batch_size=batched_batch_size,
    # )
    # end = time.perf_counter()
    # elapsed = end - start
    # print(f"Elapsed time: {elapsed:.6f} seconds torch noise batch={batched_batch_size}")
    ###########################

    #          jax            #

    ###########################
    start = time.perf_counter()
    results_jax = run_jaxplan_experiment(
        label='jax | no noise | batch=1',
    )

    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds jax no noise batch=1")
    ###########################
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

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

    plot_mean_with_std_band(
        axes[0],
        results_no_noise_no_batch,
        color='tab:blue',
        linestyle='-',
    )
    plot_mean_with_std_band(
        axes[0],
        results_with_noise_no_batch,
        color='tab:orange',
        linestyle='-',
    )
    # plot_mean_with_std_band(
    #     axes[0],
    #     results_no_noise_with_batch,
    #     color='tab:blue',
    #     linestyle='--',
    # )
    # plot_mean_with_std_band(
    #     axes[0],
    #     results_with_noise_with_batch,
    #     color='tab:orange',
    #     linestyle='--',
    # )
    plot_mean_with_std_band(
        axes[0],
        results_jax,
        color='tab:green',
        linestyle='-.',
    )
    axes[0].set_ylabel('Training return')
    axes[0].set_title(
        f'Torch and Jax: with/without noise and with/without batch over {len(seeds)} seeds'
    )
    axes[0].grid(True)
    axes[0].legend()

    # plot_mean_with_std_band(
    #     axes[1],
    #     results_no_noise_with_batch,
    #     color='tab:blue',
    #     linestyle='--',
    # )
    # plot_mean_with_std_band(
    #     axes[1],
    #     results_with_noise_with_batch,
    #     color='tab:orange',
    #     linestyle='--',
    # )
    plot_mean_with_std_band(
        axes[1],
        results_jax,
        color='tab:green',
        linestyle='-.',
    )
    axes[1].set_xlabel('Training iteration')
    axes[1].set_ylabel('Training return')
    axes[1].set_title(
        f'Torch with batch vs Jax without batch over {len(seeds)} seeds'
    )
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    output_path = PACKAGE_ROOT / "results_plot.png"
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == '__main__':
    main()
