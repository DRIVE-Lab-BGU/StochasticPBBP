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
    instance = PACKAGE_ROOT / "problems" / "reservoir" / "instance_3.rddl"

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    horizon = int(os.environ.get("NOISE_PLOT_HORIZON", "200"))
    hidden_sizes = (12, 12)
    iterations = int(os.environ.get("NOISE_PLOT_ITERATIONS", "300"))
    num_seeds = int(os.environ.get("NOISE_PLOT_NUM_SEEDS", "10"))
    seed_offset = int(os.environ.get("NOISE_PLOT_SEED_OFFSET", "0"))
    batch_size = 1
    seeds = [seed_offset + index for index in range(num_seeds)]

    if batch_size != 1:
        raise ValueError(
            'This plotting script assumes batch_size=1 so each history point '
            'matches one full training iteration.'
        )

    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())
    _, observation_template, _ = template_rollout.reset()
    action_template = template_rollout.noop_actions

    def run_experiment(noise_value: float, label: str):
        all_training_returns: List[List[float]] = []
        iteration_axis: List[int] = []

        for seed in seeds:
            torch.manual_seed(seed)
            policy = state2action(
                observation_template=observation_template,
                action_template=action_template,
                hidden_sizes=hidden_sizes,
            )

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

            seed_iterations = [int(item['iteration']) for item in history]
            seed_returns = [float(item['return']) for item in history]
            if len(seed_iterations) != iterations:
                raise ValueError(
                    f'Expected exactly {iterations} history points for seed {seed}, '
                    f'got {len(seed_iterations)}.'
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

    #       without noise     #
    
    ###########################
    start = time.perf_counter()
    results_no_noise = run_experiment(
        noise_value=0.0,
        label='torch without exploration noise',
    )
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch without noise")
    
    
    ###########################

    #       with noise        #

    ###########################
    start = time.perf_counter()
    results_with_noise = run_experiment(
        noise_value=3.0,
        label='torch with exploration noise = 3.0',
    )
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch with noise")
    ###########################

    #          jax            #

    ###########################
    start = time.perf_counter()
    results_jax = run_jaxplan_experiment(
        label='jax deep reactive policy (12, 12) without exploration noise',
    )

    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds torch with noise") 
    ###########################
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    def plot_mean_with_std_band(ax, results: Dict[str, Any], color: str) -> None:
        mean_values = results['mean_training_returns']
        std_values = results['std_training_returns']
        lower = [mean - std for (mean, std) in zip(mean_values, std_values)]
        upper = [mean + std for (mean, std) in zip(mean_values, std_values)]

        ax.plot(
            results['iterations'],
            mean_values,
            color=color,
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

    plot_mean_with_std_band(axes[0], results_no_noise, color='tab:blue')
    plot_mean_with_std_band(axes[0], results_with_noise, color='tab:orange')
    axes[0].set_ylabel('Training return')
    axes[0].set_title(
        f'Torch training curves: mean +/- std over {len(seeds)} seeds'
    )
    axes[0].grid(True)
    axes[0].legend()

    plot_mean_with_std_band(axes[1], results_no_noise, color='tab:blue')
    plot_mean_with_std_band(axes[1], results_jax, color='tab:green')
    plot_mean_with_std_band(axes[1], results_with_noise, color='tab:orange')
    axes[1].set_xlabel('Training iteration')
    axes[1].set_ylabel('Training return')
    axes[1].set_title(
        f'Torch no-noise vs Jax DRP (12, 12) no-noise: mean +/- std over {len(seeds)} seeds'
    )
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    output_path = PACKAGE_ROOT / "results_plot.png"
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == '__main__':
    main()
