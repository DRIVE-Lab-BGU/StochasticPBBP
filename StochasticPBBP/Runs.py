from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pyRDDLGym

PACKAGE_ROOT = Path(__file__).resolve().parent
print(f"PACKAGE_ROOT={PACKAGE_ROOT}")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.Policies import TensorDict
from core.Rollout import TorchRollout ,TorchRolloutCell
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
            layers.append(nn.ReLU())
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
        its get subs and retutn observation specs for the observation template
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
    instance = PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    horizon = 113
    hidden_sizes = (12, 12)

    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())
    _, observation_template, _ = template_rollout.reset()
    policy = state2action(
        observation_template=observation_template,
        action_template=template_rollout.noop_actions,
        hidden_sizes=hidden_sizes,
    )

    trainer = Train(
        horizon=horizon,
        model=env.model,
        action_space=env.action_space,
        policy=policy,
        lr=0.01,
        hidden_sizes=hidden_sizes,
        batch_size=5,
        seed=12,
    )
    history, trained_policy = trainer.train_trajectory(
        iterations=2,
        print_every=2,
        batch_size=5,
    )
    final_sub = history[-1]['final_subs'] if history else None

    sample_action_subs = trained_policy(final_sub)
    #print(f"sample action={sample_action_subs} where the observation is {trained_policy._flatten_observation(final_sub)}")
    # or we can ger the ovservevation
    for_obs = TorchRolloutCell(env.model, horizon=1, logic=FuzzyLogic())
    obs = for_obs.observe(final_sub)
    print(f"observation is {obs}")
    sample_action = trained_policy(obs)
    print(f"sample action={sample_action} where the observation is {obs}")
    if history:
        final_metrics = history[-1]
        print(
            f"final chunk return={final_metrics['return']:.4f} "
            f"after iter={int(final_metrics['iteration'])} "
            f"chunk={int(final_metrics['chunk_index'])}/{int(final_metrics['num_chunks'])}"
        )


if __name__ == '__main__':
    main()
