from __future__ import annotations

import inspect
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.debug.logger import Logger
from StochasticPBBP.core.Compiler import TorchRDDLCompiler
from StochasticPBBP.utils.Noise import noise
# from .Compiler import TorchRDDLCompiler
# from .Noise import noise

TensorDict = Dict[str, torch.Tensor]
PolicyOutput = Union[TensorDict, Tuple[TensorDict, Any]]
PolicyFn = Callable[..., PolicyOutput]


@dataclass
class RolloutTrace:
    """Container with the full trajectory produced by a rollout."""

    observations: List[TensorDict]
    actions: List[TensorDict]
    rewards: List[torch.Tensor]
    terminals: List[bool]
    final_observation: TensorDict
    final_subs: Dict[str, Any]
    policy_state: Any
    model_params: Dict[str, Any]

    @property
    def return_(self) -> torch.Tensor:
        if not self.rewards:
            return torch.tensor(0.0)
        return torch.stack(self.rewards, dim=0).sum(dim=0)


class TorchRolloutCell(nn.Module):
    """One transition of the RDDL dynamics.

    The simulator/compiler keeps the full environment state inside `subs`.
    In RNN terms:

    * hidden state  -> `subs`
    * observation   -> projection of `subs`
    * input         -> action
    * next hidden   -> updated `subs`
    """

    def __init__(self,
                 rddl_model: RDDLLiftedModel,
                 horizon: Optional[int]=None,
                 key: Optional[torch.Generator]=None,
                 logger: Optional[Logger]=None,
                 device: Optional[Union[str, torch.device]]=None,
                 **compiler_args) -> None:
        super().__init__()
        if key is None:
            key = torch.Generator()
            key.manual_seed(round(time.time() * 1000))
        self.key = key
        self.rddl = rddl_model
        self.horizon = rddl_model.horizon if horizon is None else horizon
        self.logger = logger
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        # for example - logic 
        self.compiler_args = compiler_args

        compiled = TorchRDDLCompiler(rddl_model, logger=logger, **compiler_args)
        compiled.compile(log_expr=False, heading='ROLLOUT MODEL')

        self.compiler = compiled
        self.step_fn = compiled.compile_transition(cache_path_info=False)
        self.init_values = self._clone_structure(compiled.init_values)
        # parameters for logic
        self.model_params = self._clone_structure(compiled.model_params)
        self.observed_fluents = tuple(
            rddl_model.observ_fluents if rddl_model.observ_fluents else rddl_model.state_fluents
        )
        self.noop_actions = {
            name: self._clone_value(value)
            for (name, value) in self.init_values.items()
            if rddl_model.variable_types[name] == 'action-fluent'
        }

    def reset(self,
              initial_state: Optional[Dict[str, Any]]=None,
              initial_subs: Optional[Dict[str, Any]]=None,
              model_params: Optional[Dict[str, Any]]=None
              ) -> Tuple[Dict[str, Any], TensorDict, Dict[str, Any]]:
        """Create a fresh hidden state for a rollout.

        If you only know the initial observation/state, pass it in `initial_state`.
        The method overlays it on top of the compiler init values and also syncs the
        corresponding next-state fluents, so the first transition starts consistently.
        """
        if initial_state is not None and initial_subs is not None:
            raise ValueError('Pass either initial_state or initial_subs, not both.')

        if initial_subs is None:
            subs = self._clone_structure(self.init_values)
        else:
            subs = self._clone_structure(initial_subs)

        if initial_state is not None:
            for (name, value) in initial_state.items():
                if name not in self.rddl.state_fluents:
                    raise KeyError(f'<{name}> is not a valid state fluent.')
                if name not in subs:
                    raise KeyError(f'Hidden state does not contain state fluent <{name}>.')
                coerced = self._coerce_like(value, subs[name], name)
                subs[name] = coerced
                next_name = self.rddl.next_state.get(name)
                if next_name is not None and next_name in subs:
                    subs[next_name] = self._clone_value(coerced)

        if model_params is None:
            local_model_params = self._clone_structure(self.model_params)
        else:
            local_model_params = self._clone_structure(model_params)

        observation = self.observe(subs)
        return subs, observation, local_model_params
    
    def observe(self, subs: Dict[str, Any]) -> TensorDict:
        """Project the full hidden state to the observation exposed to the policy."""
        return {
            name: self._coerce_obs_value(subs[name], name)
            for name in self.observed_fluents
        }

    def step(self,
             subs: Dict[str, Any],
             actions: Optional[Dict[str, Any]]=None,
             model_params: Optional[Dict[str, Any]]=None , 
             noise_type_dict: Optional[Dict[str, Any]]=None
             ) -> Tuple[Dict[str, Any], TensorDict, torch.Tensor, bool, Dict[str, Any]]:
        """Run one transition starting from an explicit hidden state."""
        local_subs = self._clone_structure(subs)
        local_model_params = self._clone_structure(
            self.model_params if model_params is None else model_params
        )
        prepared_actions = self.prepare_step_actions(
            actions=actions,
            noise_type_dict=noise_type_dict,
        )

        next_subs, log, next_model_params = self.step_fn(
            self.key, prepared_actions, local_subs, local_model_params
        )
        reward = self._ensure_tensor(log['reward'])
        done = self._to_bool(log.get('termination', False))
        next_obs = self.observe(next_subs)
        return next_subs, next_obs, reward, done, next_model_params

    def prepare_step_actions(self,
                             actions: Optional[Dict[str, Any]]=None,
                             noise_type_dict: Optional[Dict[str, Any]]=None
                             ) -> Dict[str, Any]:
        prepared_actions = self.prepare_actions(actions)
        if noise_type_dict is None:
            return prepared_actions
        return self._apply_action_noise(prepared_actions, noise_type_dict)

    def prepare_actions(self, actions: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """ This method prepares the actions to be passed to the step function.
            its:
            - cloning the actions to avoid modifying the original input.
                - validating that the action names are correct and exist in the noop actions template.
            - coercing the action values to the tensor type and shape based on the noop actions template.
            - if no actions are provided, it returns a clone of the noop actions template. 
            """
        if actions is None:
            return {name: self._clone_value(value) for (name, value) in self.noop_actions.items()}

        prepared = {name: self._clone_value(value) for (name, value) in self.noop_actions.items()}
        for (name, value) in actions.items():
            if name not in prepared:
                raise KeyError(
                    f'<{name}> is not a valid lifted action fluent. '
                    'Pass lifted action names, not grounded action names.'
                )
            prepared[name] = self._coerce_like(value, prepared[name], name)
        return prepared

    def _apply_action_noise(self,
                            actions: Dict[str, Any],
                            noise_type_dict: Dict[str, Any]) -> Dict[str, Any]:
        noise_generator = noise(
            horizon=self.horizon,
            action_dim=len(self.noop_actions),
            max_action=1.0,
            model=self.rddl,
        )
        noise_kind = noise_type_dict.get('type')
        if noise_kind == 'constant':
            noise_value = noise_generator.constant_noise(
                noise=noise_type_dict.get('value', 0),
            )
        elif noise_kind in {'get_smaller_1', 'smaller_1'}:
            noise_value = noise_generator.get_smaller_1(
                start_noise=noise_type_dict.get('start_value', 1.0),
                end_noise=noise_type_dict.get('end_value', 0.0),
                step=noise_type_dict.get('step', 0),
            )
        elif noise_kind in {'get_smaller_2', 'smaller_2'}:
            noise_value = noise_generator.get_smaller_2(
                start_noise=noise_type_dict.get('start_value', 1.0),
                end_noise=noise_type_dict.get('end_value', 0.0),
                step=noise_type_dict.get('step', 0),
            )
        else:
            raise ValueError(f'Unsupported noise type: {noise_kind!r}.')

        noised_actions: Dict[str, Any] = {}
        for (name, value) in actions.items():
            noised_actions[name] = self._add_noise_to_value(value, noise_value)
        return noised_actions

    @classmethod
    def _add_noise_to_value(cls, value: Any, noise_value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bool or not value.dtype.is_floating_point:
                return value.clone()
            noise_tensor = noise_value
            if not isinstance(noise_tensor, torch.Tensor):
                noise_tensor = torch.as_tensor(
                    noise_tensor,
                    dtype=value.dtype,
                    device=value.device,
                )
            noise_tensor = noise_tensor.to(dtype=value.dtype, device=value.device)
            if noise_tensor.ndim == 0:
                noise_tensor = torch.full_like(value, noise_tensor)
            else:
                noise_tensor = torch.broadcast_to(noise_tensor, value.shape)
            if torch.any(noise_tensor < 0):
                raise ValueError('Noise standard deviation must be non-negative.')
            if torch.all(noise_tensor == 0):
                return value + torch.zeros_like(value)
            distribution = torch.distributions.Normal(
                loc=torch.zeros_like(value),
                scale=noise_tensor,
            )
            return value + distribution.rsample()
        return cls._clone_value(value)

    def _coerce_obs_value(self, value: Any, name: str) -> torch.Tensor:
        del name
        """
        This method coerces the observation value to a tensor, 
        and also clones it if it's already a tensor to avoid in-place modifications.
        """
        if isinstance(value, torch.Tensor):
            return value.clone()
        return self._ensure_tensor(value)

    def _coerce_like(self, value: Any, reference: Any, name: str) -> Any:
        """
        This method coerces the action value to be like the reference tensor in terms of type and shape.
         - if the value is not a tensor, it converts it to a tensor with the same device as the reference.
         - if the value is already a tensor, it checks that its shape matches the reference and converts its dtype and device to match the reference.
         - if the reference is not a tensor, it returns the value as is (assuming it's a compatible type).
         - this method also raises an error if the provided value has a different shape than the reference tensor.
         """
        if isinstance(reference, torch.Tensor):
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value, device=reference.device)
            if tensor.shape != reference.shape:
                raise ValueError(
                    f'Value for <{name}> must have shape {tuple(reference.shape)}, '
                    f'got {tuple(tensor.shape)}.'
                )
            if reference.dtype == torch.bool:
                return tensor.bool()
            return tensor.to(dtype=reference.dtype, device=reference.device)
        return value

    @staticmethod
    def _ensure_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.all(value.bool()).item())
        if isinstance(value, np.ndarray):
            return bool(np.all(value))
        return bool(value)

    @staticmethod
    def _clone_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if hasattr(value, 'copy'):
            return value.copy()
        return deepcopy(value)

    @classmethod
    def _clone_structure(cls, value: Any) -> Any:
        """ This method recursively clones a nested structure of dicts, lists, tuples, and tensors."""
        if isinstance(value, dict):
            return {k: cls._clone_structure(v) for (k, v) in value.items()}
        if isinstance(value, list):
            return [cls._clone_structure(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_structure(v) for v in value)
        return cls._clone_value(value)


class TorchRollout(nn.Module):
    """This class manages the full rollout process, 
    including maintaining the hidden state across steps and interfacing with the policy."""

    def __init__(self,
                 rddl_model: RDDLLiftedModel,
                 horizon: Optional[int]=None,
                 key: Optional[torch.Generator]=None,
                 logger: Optional[Logger]=None,
                 device: Optional[Union[str, torch.device]]=None,
                 **compiler_args) -> None:
        super().__init__()
        self.rddl = rddl_model
        self.horizon = rddl_model.horizon if horizon is None else horizon
        self.cell = TorchRolloutCell(
            rddl_model=rddl_model,
            horizon=self.horizon,
            key=key,
            logger=logger,
            device=device,
            **compiler_args,
        )

    @property
    def noop_actions(self) -> Dict[str, Any]:
        return self.cell.noop_actions

    def reset(self,
              initial_state: Optional[Dict[str, Any]]=None,
              initial_subs: Optional[Dict[str, Any]]=None,
              model_params: Optional[Dict[str, Any]]=None
              ) -> Tuple[Dict[str, Any], TensorDict, Dict[str, Any]]:
        return self.cell.reset(
            initial_state=initial_state,
            initial_subs=initial_subs,
            model_params=model_params,
        )

    def step(self,
             subs: Dict[str, Any],
             actions: Optional[Dict[str, Any]]=None,
             model_params: Optional[Dict[str, Any]]=None,
             noise_type_dict: Optional[Dict[str, Any]]=None
             ) -> Tuple[Dict[str, Any], TensorDict, torch.Tensor, bool, Dict[str, Any]]:
        return self.cell.step(
            subs=subs,
            actions=actions,
            model_params=model_params,
            noise_type_dict=noise_type_dict,
        )

    def forward(self,
                policy: PolicyFn,
                initial_state: Optional[Dict[str, Any]]=None,
                initial_subs: Optional[Dict[str, Any]]=None,
                model_params: Optional[Dict[str, Any]]=None,
                policy_state: Any=None,
                steps: Optional[int]=None,
                start_step: int=0,
                noise_type_dict: Optional[Dict[str, Any]]=None) -> RolloutTrace:
        # in the rest in the cell is check if exist a subs and if not it will create a new one using the reset function, so we can start with subs = None and let the cell handle it.
        subs, observation, local_model_params = self.reset(
            initial_state=initial_state,
            initial_subs=initial_subs,
            model_params=model_params,
        )

        observations: List[TensorDict] = []
        actions_log: List[TensorDict] = []
        rewards: List[torch.Tensor] = []
        terminals: List[bool] = []

        rollout_steps = self.horizon if steps is None else steps
        if rollout_steps < 0:
            raise ValueError(f'steps must be non-negative, got {rollout_steps}.')

        for local_step in range(rollout_steps):
            step = start_step + local_step
            observations.append(observation)
            raw_action, policy_state = self._call_policy(
                policy, observation, step, policy_state
            )
            step_noise = self._build_step_noise_config(noise_type_dict, step)
            actions = self.cell.prepare_step_actions(raw_action, step_noise)
            subs, observation, reward, done, local_model_params = self.step(
                subs=subs,
                actions=actions,
                model_params=local_model_params,
                noise_type_dict=None,
            )
            actions_log.append(actions)
            rewards.append(reward)
            terminals.append(done)
            if done:
                break

        return RolloutTrace(
            observations=observations,
            actions=actions_log,
            rewards=rewards,
            terminals=terminals,
            final_observation=observation,
            final_subs=subs,
            policy_state=policy_state,
            model_params=local_model_params,
        )

    @staticmethod
    def _call_policy(policy: PolicyFn,
                     observation: TensorDict,
                     step: int,
                     policy_state: Any) -> Tuple[TensorDict, Any]:
        """
        Docstring for _call_policy
        
        :param policy: 
        can be a spesific function or 
        a nn.Module with forward method e.g. a neural network policy.
        The policy is expected to return either just the action, or a tuple of (action, next_policy_state). 
        the function should take at least the observation as input,
        and can also take the step number and policy state if needed.


        :param observation: the spesific state observation at the current step of the rollout, which is passed to the policy to decide on the action.

     
        :param step: if the policy needs the current step number of the rollout, it can use this parameter. This is useful for policies that change their behavior over time, such as epsilon-greedy policies or policies with a learning component.

        :type step: int
        
        :param policy_state: if the policy maintains an internal state across steps (e.g., for RNN-based policies or policies that learn online),
        this parameter can be used to pass that state from one step to the next. 
        The policy is expected to return the next policy state along with the action if it uses this parameter.
        """
    
        target = policy.forward if isinstance(policy, nn.Module) else policy
        # signature inspection to determine how many arguments the policy expects, and call accordingly
        signature = inspect.signature(target)
        #parameters = signature.parameters
        params = list(signature.parameters.values())
        # check if the policy has *args to determine how to call it
        has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params)
        positional = [
            param for param in params
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if has_varargs or len(positional) >= 3:
            output = policy(observation, step, policy_state)
        elif len(positional) == 2:
            output = policy(observation, step)
        else:
            output = policy(observation)

        if isinstance(output, tuple) and len(output) == 2:
            actions, next_policy_state = output
            return actions, next_policy_state
        return output, policy_state

    @staticmethod
    def _build_step_noise_config(noise_type_dict: Optional[Dict[str, Any]],
                                 step: int) -> Optional[Dict[str, Any]]:
        if noise_type_dict is None:
            return None
        step_noise = dict(noise_type_dict)
        step_noise.setdefault('step', step)
        return step_noise
