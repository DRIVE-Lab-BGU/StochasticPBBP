from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyRDDLGym
import torch
from torch import nn

from .Simulator import TorchRDDLSimulator

TensorDict = Dict[str, torch.Tensor]


def _as_tensor(value: Any,
               *,
               dtype: Optional[torch.dtype]=None,
               device: Optional[torch.device]=None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None or device is not None:
        tensor = tensor.to(dtype=dtype or tensor.dtype, device=device or tensor.device)
    return tensor

class MPCPolicy:
    pass

class SLPPolicy:
    pass

class random_policy:
    def __init__(self, model, logic, noise=None):
        self.model = model
        self.logic = logic
        self.noise = noise
        self.simulator = TorchRDDLSimulator(
            model,
            logic=logic,
            noise=noise,
            keep_tensors=True,
        )

    def get_action_template(self):
        return TorchRDDLSimulator._clone_structure(self.simulator.noop_actions)

    def get_action(self, obs=None, num_step=None, fill_value=None):
        del obs, num_step
        action = self.get_action_template()
        for (name, value) in action.items():
            action[name] = self._fill_action_value(value, fill_value)
        return action

    @staticmethod
    def _fill_action_value(reference, fill_value):
        if not isinstance(reference, torch.Tensor):
            return reference

        if reference.dtype == torch.bool:
            if fill_value is None:
                return torch.randint(0, 2, reference.shape, device=reference.device).bool()
            return torch.full(
                reference.shape,
                bool(fill_value),
                dtype=torch.bool,
                device=reference.device,
            )

        if reference.dtype.is_floating_point:
            if fill_value is None:
                return torch.rand_like(reference)
            return torch.full_like(reference, float(fill_value))

        if fill_value is None:
            return torch.randint(0, 10, reference.shape, device=reference.device).to(
                dtype=reference.dtype
            )
        return torch.full_like(reference, int(fill_value))


class StationaryMarkov(nn.Module):
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
                    f'StationaryMarkov supports only floating-point actions, got {name} '
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


class GaussianPolicy(nn.Module):
    """State-independent diagonal Gaussian policy over the action vector."""

    def __init__(self,
                 action_template: TensorDict,
                 init_std: float=1.0,
                 min_log_std: float=-5.0,
                 max_log_std: float=2.0) -> None:
        super().__init__()
        if not action_template:
            raise ValueError('action_template must contain at least one tensor.')

        first_action = next(iter(action_template.values()))
        self.device = first_action.device
        self.dtype = first_action.dtype if first_action.dtype.is_floating_point else torch.float32
        self.action_specs = self._build_action_specs(action_template)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        output_dim = sum(spec['numel'] for spec in self.action_specs)
        init_log_std = float(math.log(max(init_std, 1e-6)))
        self.mu = nn.Parameter(
            torch.zeros(output_dim, device=self.device, dtype=self.dtype)
        )
        self.log_std = nn.Parameter(
            torch.full((output_dim,), init_log_std, device=self.device, dtype=self.dtype)
        )

    @staticmethod
    def _build_action_specs(action_template: TensorDict) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for (name, template) in action_template.items():
            template_tensor = _as_tensor(template)
            if not template_tensor.dtype.is_floating_point:
                raise ValueError(
                    f'GaussianPolicy supports only real-valued action tensors, got {name} '
                    f'with dtype {template_tensor.dtype}.'
                )
            specs.append({
                'name': name,
                'shape': tuple(template_tensor.shape),
                'numel': int(template_tensor.numel()),
                'dtype': template_tensor.dtype,
                'device': template_tensor.device,
            })
        return specs

    def distribution(self) -> torch.distributions.Normal:
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).to(dtype=self.mu.dtype, device=self.mu.device)
        return torch.distributions.Normal(self.mu, std)

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
        del observation, step, policy_state
        dist = self.distribution()
        flat_action =  dist.rsample()
        return self._pack_actions(flat_action)
