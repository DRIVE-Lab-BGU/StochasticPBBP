
from __future__ import annotations

from StochasticPBBP.utils.Noise import noise

from ast import Return
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.debug.exception import (
    RDDLActionPreconditionNotSatisfiedError,
    RDDLInvalidActionError,
    RDDLInvalidExpressionError,
    RDDLStateInvariantNotSatisfiedError,
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.parser.expr import Value
from pyRDDLGym.core.simulator import RDDLSimulator
from .Compiler import TorchRDDLCompiler

Args = Dict[str, Union[np.ndarray, torch.Tensor, Value, float, int, bool]]


class TorchRDDLSimulator(RDDLSimulator):
    """Single-step torch simulator used by rollout/training loops."""

    def __init__(self, rddl: RDDLLiftedModel,
                 noise = None,
                 key: Optional[torch.Generator]=None,
                 raise_error: bool=True,
                 logger: Optional[Logger]=None,
                 keep_tensors: bool=False,
                 objects_as_strings: bool=True,
                 device: Optional[Union[str, torch.device]]=None,
                 **compiler_args) -> None:
        if key is None:
            key = torch.Generator()
            key.manual_seed(round(time.time() * 1000))
        self.key = key
        self.raise_error = raise_error
        # example for compiler_args: {'use64bit': True} 
        self.compiler_args = compiler_args
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.compiler: Optional[TorchRDDLCompiler] = None
        self.step_fn = None
        if noise is None:
            self.noise = {"type": "constant", "value": 0}
        else:
            self.noise = noise

        super(TorchRDDLSimulator, self).__init__(
            rddl=rddl, logger=logger,
            keep_tensors=keep_tensors, objects_as_strings=objects_as_strings
        )

    def seed(self, seed: int) -> None:
        super(TorchRDDLSimulator, self).seed(seed)
        self.key.manual_seed(seed)
        torch.manual_seed(seed)

    def _compile(self):
        rddl = self.rddl

        compiled = TorchRDDLCompiler(rddl, logger=self.logger, **self.compiler_args)
        compiled.compile(log_expr=False, heading='SIMULATION MODEL')

        self.compiler = compiled
        self.step_fn = compiled.compile_transition(cache_path_info=False)
        # To make sure we dont change the compiler's internal data structures during simulation,
        # we clone the init_values and model_params before using them in the simulator.
        self.init_values = self._clone_structure(compiled.init_values)
        self.levels = compiled.levels
        self.traced = compiled.traced
        #####
        self.invariants = compiled.invariants
        self.preconds = compiled.preconditions
        self.terminals = compiled.terminations
        ######

        self.reward = compiled.reward
        # To make sure we dont change the compiler's internal data structures during simulation,
        # we clone the init_values and model_params before using them in the simulator.
        self.model_params = self._clone_structure(compiled.model_params)

        self.subs = self._clone_structure(self.init_values)
        self.state = None
        
        # take the fluent actions and their initial values, 
        # and create a dict of noop actions that we can use as a default when no actions 
        # are provided to the step function.
        self.noop_actions = {
            var: self._clone_value(values)
            for (var, values) in self.init_values.items()
            if rddl.variable_types[var] == 'action-fluent'
        }
        #####
        self.grounded_noop_actions = rddl.ground_vars_with_values(self.noop_actions)
        self.grounded_action_ranges = rddl.ground_vars_with_value(rddl.action_ranges)
        self.invariant_names = [f'Invariant {i}' for i in range(len(rddl.invariants))]
        self.precond_names = [f'Precondition {i}' for i in range(len(rddl.preconditions))]
        self.terminal_names = [f'Termination {i}' for i in range(len(rddl.terminations))]

    def handle_error_code(self, error: int, msg: str) -> None:
        if self.raise_error and int(error) != 0:
            raise RDDLInvalidExpressionError(
                f'Internal error in evaluation of {msg}: error code {int(error)}.')

    def check_state_invariants(self, silent: bool=False) -> bool:
        for (i, invariant) in enumerate(self.invariants):
            loc = self.invariant_names[i]
            sample, self.key, error, self.model_params = invariant(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if not self._to_bool(sample):
                if not silent:
                    raise RDDLStateInvariantNotSatisfiedError(
                        f'{loc} is not satisfied.')
                return False
        return True

    def check_action_preconditions(self, actions: Args, silent: bool=False) -> bool:
        sim_actions = self._prepare_actions_for_torch(actions)
        self.subs.update(sim_actions)

        for (i, precond) in enumerate(self.preconds):
            loc = self.precond_names[i]
            sample, self.key, error, self.model_params = precond(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if not self._to_bool(sample):
                if not silent:
                    raise RDDLActionPreconditionNotSatisfiedError(
                        f'{loc} is not satisfied for actions {actions}.')
                return False
        return True

    def check_terminal_states(self) -> bool:
        for (i, terminal) in enumerate(self.terminals):
            loc = self.terminal_names[i]
            sample, self.key, error, self.model_params = terminal(
                self.subs, self.model_params, self.key)
            self.handle_error_code(error, loc)
            if self._to_bool(sample):
                return True
        return False

    def sample_reward(self) -> torch.Tensor:
        """Sample the reward for the current state and action."""
        reward, self.key, error, self.model_params = self.reward(
            self.subs, self.model_params, self.key)
        self.handle_error_code(error, 'reward function')
        return reward

    def reset(self):
        """ 
        Same as in the RDDLSimulator,

        Reset the simulator to the initial state. 
        Returns the initial observation and a boolean indicating whether the initial state is terminal."""
        if self.compiler is None:
            raise RuntimeError('Simulator was not compiled.')

        rddl = self.rddl
        keep_tensors = self.keep_tensors
        self.subs = self._clone_structure(self.init_values)
        self.model_params = self._clone_structure(self.compiler.model_params)

        self.state = {}
        for state in rddl.state_fluents:
            raw_state_values = self.subs[state]
            state_values = raw_state_values
            if self.objects_as_strings:
                ptype = rddl.variable_ranges[state]
                if ptype not in RDDLValueInitializer.NUMPY_TYPES:
                    state_values = rddl.index_to_object_string_array(
                        ptype, self._to_ground_value(raw_state_values))
            if keep_tensors:
                self.state[state] = state_values
            else:
                self.state.update(rddl.ground_var_with_values(
                    state, self._to_ground_value(state_values)))

        obs = self.state

        done = self.check_terminal_states()
        return obs, done

    def step(self, actions: Args , num_step=0):
        if self.step_fn is None:
            raise RuntimeError('Simulator was not compiled.')

        rddl = self.rddl
        keep_tensors = self.keep_tensors

        sim_actions = self._prepare_actions_for_torch(actions)
        
        n = noise(action_dim = len(sim_actions) , 
                  max_action = max(self.grounded_action_ranges.values()) , 
                  horizon = self.rddl.horizon )
        
        if self.noise["type"] == "constant":
            the_constant = self.noise["value"]
            noise4action = n.constant_noise(the_constant)
        
        if self.noise["type"] == "smaller_1":
            start_noise = self.noise["value"][0]
            end_noise = self.noise["value"][1]
            noise4action = n.get_smaller_1( start_noise = start_noise , end_noise = end_noise , step = num_step )
            print(f"noise for action is {noise4action}")
        if self.noise["type"] == "smaller_2":
            start_noise = self.noise["value"][0]
            end_noise = self.noise["value"][1]
            noise4action = n.get_smaller_2( start_noise = start_noise, end_noise = end_noise , step = num_step )

        sim_actions = {k: v +torch.normal(mean=0.0, std=noise4action, size=v.shape) for (k, v) in sim_actions.items()}


        self.subs, log, self.model_params = self.step_fn(
            self.key, sim_actions, self.subs, self.model_params)
        self.handle_error_code(log.get('error', 0), 'transition')
        reward = log['reward']

        self.state = {}

        # Convert internal simulator values to a user-readable state.
        # Internally object-valued fluents are stored as numeric indices for efficient computation.
        # If objects_as_strings=True, these indices are converted back to their object names
        # before returning the state to the user.

        # Convert internal indices to readable object names if needed (e.g. 2 → "room_c").
        for state in rddl.state_fluents:
            raw_state_values = self.subs[state]
            state_values = raw_state_values
            if self.objects_as_strings:
                ptype = rddl.variable_ranges[state]
                if ptype not in RDDLValueInitializer.NUMPY_TYPES:
                    state_values = rddl.index_to_object_string_array(
                        ptype, self._to_ground_value(raw_state_values))
                    
            if keep_tensors:
                self.state[state] = state_values
           
            else:
                #  conver to numpy and ground the state variables to 
                # get a dict of grounded state variable names to their values.
                self.state.update(rddl.ground_var_with_values(
                    state, self._to_ground_value(state_values)))
        
        obs = self.state

        done = self._to_bool(log.get('termination', False))
        
        return obs, reward, done

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _prepare_actions_for_torch(self, actions: Optional[Args]) -> Dict[str, Any]:
        """ make sure the actions are in the right format for the Torch simulator,
          and also clone them."""
        if not actions:
            return {k: self._clone_value(v) for (k, v) in self.noop_actions.items()}

        if all(action in self.noop_actions for action in actions):
            sim_actions = {k: self._clone_value(v) for (k, v) in self.noop_actions.items()}
            for (action, value) in actions.items():
                sim_actions[action] = self._coerce_like(value, self.noop_actions[action], action)
            return sim_actions

        numpy_actions = {
            k: self._to_numpy(v) if isinstance(v, torch.Tensor) else v
            for (k, v) in actions.items()
        }
        prepared = super(TorchRDDLSimulator, self).prepare_actions_for_sim(numpy_actions)
        sim_actions = {}
        for (action, value) in prepared.items():
            sim_actions[action] = self._coerce_like(value, self.noop_actions[action], action)
        return sim_actions

    def _coerce_like(self, value: Any, reference: Any, action_name: str) -> Any:
        if isinstance(reference, torch.Tensor):
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value, device=self.device)
            tensor = tensor.to(device=reference.device)
            if tensor.shape != reference.shape:
                raise RDDLInvalidActionError(
                    f'Value for action-fluent <{action_name}> must be of shape '
                    f'{tuple(reference.shape)}, got {tuple(tensor.shape)}.')
            if reference.dtype == torch.bool:
                return tensor.bool()
            if reference.dtype.is_floating_point:
                return tensor.to(dtype=reference.dtype)
            return tensor.to(dtype=reference.dtype)
        return value

    @staticmethod
    def _to_numpy(value: torch.Tensor):
        #  probelm: if the tensor requires grad,the dethac make problems.
        tensor = value.detach()
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return tensor.numpy()

    @classmethod
    def _to_ground_value(cls, value: Any):
        # Convert Torch tensors to NumPy before grounding variables.
        # pyRDDLGym grounding utilities expect NumPy arrays and may fail
        # with autograd-enabled tensors. This also detaches values from
        # the computation graph since observations do not require gradients.
        if isinstance(value, torch.Tensor):
            return cls._to_numpy(value)
        return value

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.all(value.bool()).item())
        if isinstance(value, np.ndarray):
            return bool(np.all(value))
        return bool(value)

    @staticmethod
    def _clone_value(value: Any) -> Any:
        """Create a safe copy of a value before using it inside the simulator.

                Return a safe copy of a value before using it in the simulator.
                The simulator updates its internal state in-place. If the state shares
                memory with the compiler's initial values, those values would be modified.

                Example (reservoir domain):
                subs["rlevel"] = compiler.init_values["rlevel"]
                subs["rlevel"][0] -= release[0]   # water released
                # now compiler.init_values["rlevel"] also changed!

            Cloning/copying ensures the simulator modifies only its own state.
        """
        if isinstance(value, torch.Tensor):
            return value.clone()
        #if isinstance(value, np.ndarray):
            return value.copy()
        return deepcopy(value)

    @classmethod
    def _clone_structure(cls, value: Any) -> Any:
        """ this function is used to clone the structure 
        of the init_values and model_params, 
        which can be nested dicts/lists/tuples of tensors/arrays/values."""
        if isinstance(value, dict):
            return {k: cls._clone_structure(v) for (k, v) in value.items()}
        if isinstance(value, list):
            return [cls._clone_structure(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_structure(v) for v in value)
        return cls._clone_value(value)


# Backward-compatible alias used in earlier local experiments.
Simulator = TorchRDDLSimulator
