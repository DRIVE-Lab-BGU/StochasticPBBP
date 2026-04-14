from __future__ import annotations

import math
import shutil
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from pyRDDLGym.core.env import RDDLEnv
from StochasticPBBP.deprecated.Simulator import TorchRDDLSimulator
from StochasticPBBP.utils.seeder import BaseSeeder

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


class MBDPOPolicy(ABC):
    """
    Model-based-Differential Policy Optimization
    """
    use_tensor_obs = True  # uses internal tensor representation of state

    @abstractmethod
    def sample_action(self, observation: Any, training_mode =True, step: Optional[int]=None, policy_state: Any=None) -> Any:
        '''Samples an action from the current policy evaluated at the given state.

        :param observation: the current observation.
        :param training_mode: if includes exploration or not
        :param step: step number in the rollout, for non-stationary policies
        :param policy_state:
        :return: action
        '''
        pass

    def reset(self) -> None:
        '''Resets the policy and prepares it for the next episode/rollout.'''
        pass

    def evaluate(self, env:RDDLEnv, episodes: int=1, verbose: bool=False, render: bool=False,
                 seed_generator: Optional[BaseSeeder]=None) -> List[float]:
        if not env.vectorized:
            raise ValueError(f'RDDLEnv vectorized flag must be turned on for MBDPO type policy')

        gamma = env.discount

        # get terminal width
        if verbose:
            width = shutil.get_terminal_size().columns
            sep_bar = '-' * width

        # start simulation
        history = np.zeros((episodes,))
        for episode in range(episodes):

            # restart episode
            total_reward, cuml_gamma = 0.0, 1.0
            self.reset()
            if seed_generator is not None:
                state, _ = env.reset(seed=next(seed_generator))

            # printing
            if verbose:
                print(f'initial state = \n{self._format(state, width)}')

            # simulate to end of horizon
            for step in range(env.horizon):
                if render:
                    env.render()

                # take a step in the environment
                state = self._unpack_observation(state)
                action = self.sample_action(observation=state, training_mode=False)
                action = self._action_to_env_dict(action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward * cuml_gamma
                cuml_gamma *= gamma
                done = terminated or truncated

                # printing
                if verbose:
                    print(f'{sep_bar}\n'
                          f'step   = {step}\n'
                          f'action = \n{self._format(action, width)}\n'
                          f'state  = \n{self._format(next_state, width)}\n'
                          f'reward = {reward}\n'
                          f'done   = {done}')

                state = next_state
                if done:
                    break

            if verbose:
                print(f'\n'
                      f'episode {episode + 1} ended with return {total_reward}\n'
                      f'{"=" * width}')

            history[episode] = total_reward

            # summary statistics
        return {
            'mean': np.mean(history),
            'median': np.median(history),
            'min': np.min(history),
            'max': np.max(history),
            'std': np.std(history)
        }

    def _format(self, state, width=80, indent=4):
        if len(state) == 0:
            return str(state)
        state = {key: str(value) for (key, value) in state.items()}
        klen = max(map(len, state.keys())) + 1
        vlen = max(map(len, state.values())) + 1
        cols = max(1, (width - indent) // (klen + vlen + 3))
        result = ' ' * indent
        for (count, (key, value)) in enumerate(state.items(), 1):
            result += f'{key.rjust(klen)} = {value.ljust(vlen)}'
            if count % cols == 0:
                result += '\n' + ' ' * indent
        return result

    def _unpack_observation(self, reset_result: Any) -> Dict[str, Any]:
        if isinstance(reset_result, tuple):
            if not reset_result:
                raise ValueError('env.reset() returned an empty tuple.')
            return reset_result[0]
        return reset_result

    def _action_to_env_dict(self, action: Dict[str, Any]) -> Dict[str, Any]:
        env_action: Dict[str, Any] = {}
        for name, value in action.items():
            if isinstance(value, torch.Tensor):
                detached = value.detach().cpu()
                env_action[name] = detached.item() if detached.numel() == 1 else detached.numpy()
            else:
                env_action[name] = value
        return env_action



    @staticmethod
    def _as_tensor(value: Any, *, dtype: Optional[torch.dtype] = None,
                   device: Optional[torch.device] = None) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype or tensor.dtype, device=device or tensor.device)
        return tensor

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
    def _build_action_specs(cls, action_template: TensorDict, action_space: Any) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for (name, template) in action_template.items():
            tensor = cls._as_tensor(template)
            if not tensor.dtype.is_floating_point:
                raise ValueError(
                    f'StationaryMarkov supports only floating-point actions, got {name} '
                    f'with dtype {tensor.dtype}.'
                )
            # lower_bound, upper_bound = cls._resolve_action_bounds(name, action_space, tensor)
            specs.append({
                'name': name,
                'shape': tuple(tensor.shape),
                'numel': int(tensor.numel()),
                'dtype': tensor.dtype,
                'device': tensor.device,
                # 'lower_bound': lower_bound,
                # 'upper_bound': upper_bound,
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
            bounded_action = self._apply_action_constraints(raw_action, spec)
            actions[spec['name']] = bounded_action.to(dtype=spec['dtype'], device=spec['device'])
            start = end
        return actions

    def _apply_action_constraints(self,
                                  raw_action: torch.Tensor,
                                  spec: Dict[str, Any]) -> torch.Tensor:
        bounded_action = raw_action.clone()
        return bounded_action


class NeuralStateFeedbackPolicy(MBDPOPolicy, nn.Module):
    def __init__(self, observation_template: TensorDict, action_template: TensorDict, action_space: Any,
                 hidden_sizes: Tuple[int, ...] = (64, 64), seed: Optional[int]=None) -> None:
        super().__init__()
        if not observation_template:
            raise ValueError('observation_template must contain at least one tensor.')
        if not action_template:
            raise ValueError('action_template must contain at least one tensor.')

        self.observation_specs = self._build_observation_specs(observation_template)
        self.action_specs = self._build_action_specs(action_template, action_space)
        self.device = self.observation_specs[0]['device']
        self.dtype = torch.float32
        self.g = None
        if seed is not None:
            self.g = torch.Generator().manual_seed(seed)

        layers = []
        input_dim = sum(spec['numel'] for spec in self.observation_specs)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        output_dim = sum(spec['numel'] for spec in self.action_specs)
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Default to Xavier init because the network uses Tanh hidden layers.
        self.network.apply(self._init_weights_xavier)

    def forward(self,
                observation: TensorDict) -> TensorDict:
        flat_observation = self._flatten_observation(observation)
        flat_action = self.network(flat_observation)
        return self._pack_actions(flat_action)

    def sample_action(self, observation: Any, training_mode=True, step: Optional[int]=None, policy_state: Any=None) -> Any:
        if not training_mode:
            with torch.no_grad():
                return self.forward(observation)
        return self.forward(observation)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            if self.g is not None:
                nn.init.xavier_uniform_(m.weight, generator=self.g)
            else:
                nn.init.xavier_uniform_(m.weight, generator=self.g)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    # def _apply_action_constraints(self,
    #                               raw_action: torch.Tensor,
    #                               spec: Dict[str, Any]) -> torch.Tensor:
    #     lower_bound = spec['lower_bound'].to(dtype=raw_action.dtype, device=raw_action.device)
    #     upper_bound = spec['upper_bound'].to(dtype=raw_action.dtype, device=raw_action.device)
    #     bounded_action = raw_action.clone()
    #
    #     has_lower = torch.isfinite(lower_bound)
    #     has_upper = torch.isfinite(upper_bound)
    #     both_bounded = has_lower & has_upper
    #     lower_only = has_lower & ~has_upper
    #     upper_only = ~has_lower & has_upper
    #
    #     if torch.any(both_bounded):
    #         normalized_action = torch.sigmoid(raw_action)
    #         scaled_action = lower_bound + (upper_bound - lower_bound) * normalized_action
    #         bounded_action = torch.where(both_bounded, scaled_action, bounded_action)
    #     if torch.any(lower_only):
    #         lower_bounded_action = lower_bound + F.softplus(raw_action)
    #         bounded_action = torch.where(lower_only, lower_bounded_action, bounded_action)
    #     if torch.any(upper_only):
    #         upper_bounded_action = upper_bound - F.softplus(raw_action)
    #         bounded_action = torch.where(upper_only, upper_bounded_action, bounded_action)
    #
    #     return bounded_action



class random_policy_old:
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
                 action_space: Any,
                 hidden_sizes: Tuple[int, ...]=(64, 64),
                 init_weights_fn: str='xavier') -> None:
        super().__init__()
        if not observation_template:
            raise ValueError('observation_template must contain at least one tensor.')
        if not action_template:
            raise ValueError('action_template must contain at least one tensor.')

        self.observation_specs = self._build_observation_specs(observation_template)
        self.action_specs = self._build_action_specs(action_template, action_space)
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
        # Default to Xavier init because the network uses Tanh hidden layers.
        self._initialize_network(init_weights_fn)

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
    def _build_action_specs(
        cls,
        action_template: TensorDict,
        action_space: Any,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for (name, template) in action_template.items():
            tensor = cls._as_tensor(template)
            if not tensor.dtype.is_floating_point:
                raise ValueError(
                    f'StationaryMarkov supports only floating-point actions, got {name} '
                    f'with dtype {tensor.dtype}.'
                )
            lower_bound, upper_bound = cls._resolve_action_bounds(name, action_space, tensor)
            specs.append({
                'name': name,
                'shape': tuple(tensor.shape),
                'numel': int(tensor.numel()),
                'dtype': tensor.dtype,
                'device': tensor.device,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
            })
        return specs

    @staticmethod
    def _resolve_action_bounds(
        action_name: str,
        action_space: Any,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spaces = getattr(action_space, 'spaces', None)
        if spaces is None or action_name not in spaces:
            raise ValueError(
                f'Action space does not contain a Box for action {action_name!r}.'
            )
        box = spaces[action_name]
        if not hasattr(box, 'low') or not hasattr(box, 'high'):
            raise ValueError(
                f'Action space entry for {action_name!r} must define low/high bounds.'
            )
        lower_bound = _as_tensor(box.low, dtype=reference.dtype, device=reference.device)
        upper_bound = _as_tensor(box.high, dtype=reference.dtype, device=reference.device)
        if lower_bound.numel() == 1:
            lower_bound = lower_bound.expand(reference.shape)
        if upper_bound.numel() == 1:
            upper_bound = upper_bound.expand(reference.shape)
        if tuple(lower_bound.shape) != tuple(reference.shape):
            raise ValueError(
                f'Lower bounds for action {action_name!r} must match shape {tuple(reference.shape)}, '
                f'got {tuple(lower_bound.shape)}.'
            )
        if tuple(upper_bound.shape) != tuple(reference.shape):
            raise ValueError(
                f'Upper bounds for action {action_name!r} must match shape {tuple(reference.shape)}, '
                f'got {tuple(upper_bound.shape)}.'
            )
        if torch.any(lower_bound > upper_bound):
            raise ValueError(f'Action space bounds for {action_name!r} contain low > high.')
        return lower_bound.clone(), upper_bound.clone()

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

    def _pack_bounded_actions(self, flat_action: torch.Tensor) -> TensorDict:
        actions: TensorDict = {}
        start = 0
        for spec in self.action_specs:
            end = start + spec['numel']
            raw_action = flat_action[start:end].reshape(spec['shape'])
            bounded_action = self._apply_action_constraints(raw_action, spec)
            actions[spec['name']] = bounded_action.to(dtype=spec['dtype'], device=spec['device'])
            start = end
        return actions

    def _apply_action_constraints(self,
                                  raw_action: torch.Tensor,
                                  spec: Dict[str, Any]) -> torch.Tensor:
        lower_bound = spec['lower_bound'].to(dtype=raw_action.dtype, device=raw_action.device)
        upper_bound = spec['upper_bound'].to(dtype=raw_action.dtype, device=raw_action.device)
        bounded_action = raw_action.clone()

        has_lower = torch.isfinite(lower_bound)
        has_upper = torch.isfinite(upper_bound)
        both_bounded = has_lower & has_upper
        lower_only = has_lower & ~has_upper
        upper_only = ~has_lower & has_upper

        if torch.any(both_bounded):
            normalized_action = torch.sigmoid(raw_action)
            scaled_action = lower_bound + (upper_bound - lower_bound) * normalized_action
            bounded_action = torch.where(both_bounded, scaled_action, bounded_action)
        if torch.any(lower_only):
            lower_bounded_action = lower_bound + F.softplus(raw_action)
            bounded_action = torch.where(lower_only, lower_bounded_action, bounded_action)
        if torch.any(upper_only):
            upper_bounded_action = upper_bound - F.softplus(raw_action)
            bounded_action = torch.where(upper_only, upper_bounded_action, bounded_action)

        return bounded_action

    def _initialize_network(self, init_weights_fn: str) -> None:
        normalized_name = init_weights_fn.strip().lower()
        if normalized_name in {'kaiming', 'jax'}:
            initializer = self._init_weights_kaiming
        elif normalized_name == 'xavier':
            initializer = self._init_weights_xavier
        else:
            raise ValueError(
                f'Unsupported init_weights_fn={init_weights_fn!r}. '
                "Expected 'kaiming', 'jax', or 'xavier'."
            )
        self.network.apply(initializer)

    @staticmethod
    def _init_weights_xavier(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _init_weights_kaiming(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self,
                observation: TensorDict,
                step: Optional[int]=None,
                policy_state: Any=None) -> TensorDict:
        del step, policy_state
        flat_observation = self._flatten_observation(observation)
        flat_action = self.network(flat_observation)
        return self._pack_bounded_actions(flat_action)

    def sample_action(self, observation: TensorDict) -> TensorDict:
        with torch.no_grad():
            return self.forward(observation, step=None, policy_state=None)


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
        flat_action = dist.rsample()
        return self._pack_actions(flat_action)


class state2action(nn.Module):
    def __init__(self,
                 observation_template: TensorDict,
                 action_template: TensorDict,
                 hidden_sizes: Tuple[int, ...] = (12, 12)) -> None:
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
                step: Optional[int] = None,
                policy_state: Any = None) -> TensorDict:
        del step, policy_state
        flat_observation = self._flatten_observation(observation)
        flat_action = self.network(flat_observation)
        return self._pack_actions(flat_action)
