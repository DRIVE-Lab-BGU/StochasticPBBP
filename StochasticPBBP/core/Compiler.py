"""Torch-native RDDL compiler producing differentiable PyTorch callables."""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

import torch

from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.debug.exception import (
    print_stack_trace,
    RDDLNotImplementedError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.debug.logger import Logger

from StochasticPBBP.core.Initializer import RDDLValueInitializer as TorchRDDLValueInitializer
from StochasticPBBP.core.Logic import ExactLogic, FuzzyLogic
from StochasticPBBP.utils.device import (
    as_tensor_on_device,
    infer_torch_device,
    make_generator,
    resolve_torch_device,
)
# from .Initializer import RDDLValueInitializer as TorchRDDLValueInitializer
# from .Logic import ExactLogic, FuzzyLogic


# # domain.rdlll + instance.rddl 
# |
# v
# reader + parser 
# |
# v
# model = RDDLLiftedModel(rddl) 
# lifted mean that we have the objects and types and the 
# expressions are still in their original form, not compiled to any specific backend
# e.g. 
#  for lifted : 
#               rddl.variable_ranges['rlevel'] = 'real'
#               rddl.pvariables['rlevel'] = ('reservoir',)
# for grounded : 
#               rlevel = tensor([rlevel_R1,rlevel_R2,rlevel_R3])
# 
# model - > sorted levels
# sorter = RDDLLevelAnalysis(self.rddl, allow_synchronous_state=True,logger=self.logger)
# 
Args = Dict[str, Any]
# explanation: Callable that takes (subs, params, key) and returns (value, key, error_code, params)
CallableExpr = Callable[[Args, Dict[str, Any], Optional[torch.Generator]],
                        Tuple[torch.Tensor, Optional[torch.Generator], int, Dict[str, Any]]]


class TorchRDDLCompiler:
    """Compiles RDDL expressions into eager PyTorch callables."""

    ERROR_CODES = {
        'NORMAL': 0,
        'INVALID_CAST': 2 ** 0,
        'INVALID_PARAM_UNIFORM': 2 ** 1,
        'INVALID_PARAM_NORMAL': 2 ** 2,
        'INVALID_PARAM_EXPONENTIAL': 2 ** 3,
        'INVALID_PARAM_WEIBULL': 2 ** 4,
        'INVALID_PARAM_BERNOULLI': 2 ** 5,
        'INVALID_PARAM_POISSON': 2 ** 6,
        'INVALID_PARAM_GAMMA': 2 ** 7,
        'INVALID_PARAM_BETA': 2 ** 8,
        'INVALID_PARAM_GEOMETRIC': 2 ** 9,
        'INVALID_PARAM_PARETO': 2 ** 10,
        'INVALID_PARAM_STUDENT': 2 ** 11,
        'INVALID_PARAM_GUMBEL': 2 ** 12,
        'INVALID_PARAM_LAPLACE': 2 ** 13,
        'INVALID_PARAM_CAUCHY': 2 ** 14,
        'INVALID_PARAM_GOMPERTZ': 2 ** 15,
        'INVALID_PARAM_CHISQUARE': 2 ** 16,
        'INVALID_PARAM_KUMARASWAMY': 2 ** 17,
        'INVALID_PARAM_DISCRETE': 2 ** 18,
        'INVALID_PARAM_KRON_DELTA': 2 ** 19,
        'INVALID_PARAM_DIRICHLET': 2 ** 20,
        'INVALID_PARAM_MULTIVARIATE_STUDENT': 2 ** 21,
        'INVALID_PARAM_MULTINOMIAL': 2 ** 22,
        'INVALID_PARAM_BINOMIAL': 2 ** 23,
        'INVALID_PARAM_NEGATIVE_BINOMIAL': 2 ** 24
    }

    INVERSE_ERROR_CODES = {
        0: 'Casting occurred that could result in loss of precision.',
        1: 'Found Uniform(a, b) distribution where a > b.',
        2: 'Found Normal(m, v^2) distribution where v < 0.',
        3: 'Found Exponential(s) distribution where s <= 0.',
        4: 'Found Weibull(k, l) distribution where either k <= 0 or l <= 0.',
        5: 'Found Bernoulli(p) distribution where either p < 0 or p > 1.',
        6: 'Found Poisson(l) distribution where l < 0.',
        7: 'Found Gamma(k, l) distribution where either k <= 0 or l <= 0.',
        8: 'Found Beta(a, b) distribution where either a <= 0 or b <= 0.',
        9: 'Found Geometric(p) distribution where either p < 0 or p > 1.',
        10: 'Found Pareto(k, l) distribution where either k <= 0 or l <= 0.',
        11: 'Found Student(df) distribution where df <= 0.',
        12: 'Found Gumbel(m, s) distribution where s <= 0.',
        13: 'Found Laplace(m, s) distribution where s <= 0.',
        14: 'Found Cauchy(m, s) distribution where s <= 0.',
        15: 'Found Gompertz(k, l) distribution where either k <= 0 or l <= 0.',
        16: 'Found ChiSquare(df) distribution where df <= 0.',
        17: 'Found Kumaraswamy(a, b) distribution where either a <= 0 or b <= 0.',
        18: 'Found Discrete(p) distribution where either p < 0 or p does not sum to 1.',
        19: 'Found KronDelta(x) distribution where x is not int nor bool.',
        20: 'Found Dirichlet(alpha) distribution where alpha < 0.',
        21: 'Found MultivariateStudent(mean, cov, df) distribution where df <= 0.',
        22: 'Found Multinomial(n, p) distribution where either p < 0, p does not sum to 1, or n <= 0.',
        23: 'Found Binomial(n, p) distribution where either p < 0, p > 1, or n < 0.',
        24: 'Found NegativeBinomial(n, p) distribution where either p < 0, p > 1, or n <= 0.'
    }

    def __init__(self, rddl: RDDLLiftedModel,
   
                 logger: Optional[Logger]=None,
                 python_functions: Optional[Dict[str, Callable]]=None,
                 use64bit: bool=False,
                 logic: Optional[object]=None,
                 fuzzy_logic: Optional[object]=None,
                 device: Optional[Union[str, torch.device]]=None,
                 **_) -> None:
        """Prepare a compiler that mirrors the pyRDDLGym expression DAG.

        Args:
            rddl: Lifted model describing the domain.
            logger: Optional logger to capture compilation traces.
            python_functions: Mapping of external python functions.
            use64bit: Whether tensors should default to 64-bit precision.
            logic: Optional logic backend overriding the default.
            fuzzy_logic: Alternate logic backend (legacy API).
            **_: Ignored keyword arguments for compatibility.

        Example:
            >>> compiler = TorchRDDLCompiler(rddl_model, use64bit=True)
            >>> compiler.compile()
        """
        if not isinstance(rddl, RDDLLiftedModel):
            raise ValueError("rddl must be an instance of RDDLLiftedModel.")
        self.rddl = rddl

        self.logger = logger
        self.python_functions = python_functions or {}
        self.device = resolve_torch_device(device)

        backend = logic or fuzzy_logic
        if backend is None:
            backend = ExactLogic(use64bit=use64bit)
        elif hasattr(backend, 'set_use64bit'):
            backend.set_use64bit(use64bit)
        self.logic = backend
        # exact and fuzzy backends expose the same operator table, 
        # so expression compilation can bind whichever
        # semantics were selected once up-front instead of guessing at runtime.
        self.EXACT_OPS = ExactLogic(use64bit=use64bit).get_operator_dicts()
        self.OPS = (self.logic.get_operator_dicts()
                    if hasattr(self.logic, 'get_operator_dicts')
                    else self.EXACT_OPS)
        # Import aliasing in local scripts/tests can produce multiple ExactLogic
        # class identities (`core.Logic` vs `StochasticPBBP.core.Logic`). Treat
        # backends by their semantic role rather than strict class identity.
        self.uses_relaxed_logic = not self._is_exact_logic_backend(self.logic)
        self.use64bit = use64bit

        self.INT = torch.int64 if use64bit else torch.int32
        self.REAL = torch.float64 if use64bit else torch.float32
        self.TORCH_TYPES = {
            'int': self.INT,
            'real': self.REAL,
            'bool': torch.bool
        }

        # compile initial values in torch tensors
        if self.device.type == 'mps' and use64bit:
            raise ValueError(
                'TorchRDDLCompiler does not support use64bit=True on MPS because '
                'the backend does not support float64 tensors.'
            )

        initializer = TorchRDDLValueInitializer(
            rddl,
            device=self.device,
            use64bit=use64bit,
        )
        self.init_values: Dict[str, torch.Tensor] = initializer.initialize()
        self.cpfs: Dict[str, CallableExpr] = {}
        self.reward: CallableExpr | None = None
        self.invariants: List[CallableExpr] = []
        self.preconditions: List[CallableExpr] = []
        self.terminations: List[CallableExpr] = []
        self.model_params: Dict[str, Any] = {}
        self.levels = None
        self.traced = None

    @staticmethod
    def _is_exact_logic_backend(backend: Any) -> bool:
        return isinstance(backend, ExactLogic) or type(backend).__name__ == 'ExactLogic'

    @staticmethod
    def get_error_codes(error: int) -> List[int]:
        binary = reversed(bin(error)[2:])
        return [i for (i, c) in enumerate(binary) if c == '1']

    @staticmethod
    def get_error_messages(error: int) -> List[str]:
        return [
            TorchRDDLCompiler.INVERSE_ERROR_CODES[i]
            for i in TorchRDDLCompiler.get_error_codes(error)
            if i in TorchRDDLCompiler.INVERSE_ERROR_CODES
        ]

    def _error_if_invalid(self, invalid: Any, code_name: str) -> int:
        tensor = self._ensure_tensor(invalid)
        if bool(torch.any(tensor).item()):
            return self.ERROR_CODES[code_name]
        return self.ERROR_CODES['NORMAL']

    def _distribution_param_error(self, name: str, values: List[torch.Tensor]) -> int:
        if name == 'KronDelta':
            tensor = values[0]
            invalid = not (tensor.dtype == torch.bool or not torch.is_floating_point(tensor))
            return self._error_if_invalid(invalid, 'INVALID_PARAM_KRON_DELTA')

        if name == 'Uniform':
            low, high = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid(low > high, 'INVALID_PARAM_UNIFORM')

        if name == 'Normal':
            _, var = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid(var < 0, 'INVALID_PARAM_NORMAL')

        if name == 'Exponential':
            scale = values[0].to(self.REAL)
            return self._error_if_invalid(scale <= 0, 'INVALID_PARAM_EXPONENTIAL')

        if name == 'Weibull':
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid((shape <= 0) | (scale <= 0), 'INVALID_PARAM_WEIBULL')

        if name == 'Gamma':
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid((shape <= 0) | (scale <= 0), 'INVALID_PARAM_GAMMA')

        if name == 'Beta':
            a, b = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid((a <= 0) | (b <= 0), 'INVALID_PARAM_BETA')

        if name == 'Poisson':
            rate = values[0].to(self.REAL)
            return self._error_if_invalid(rate < 0, 'INVALID_PARAM_POISSON')

        if name == 'Bernoulli':
            prob = values[0].to(self.REAL)
            return self._error_if_invalid((prob < 0) | (prob > 1), 'INVALID_PARAM_BERNOULLI')

        if name == 'Binomial':
            trials, prob = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            invalid = (prob < 0) | (prob > 1) | (trials < 0)
            return self._error_if_invalid(invalid, 'INVALID_PARAM_BINOMIAL')

        if name == 'NegativeBinomial':
            trials, prob = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            invalid = (prob < 0) | (prob > 1) | (trials <= 0)
            return self._error_if_invalid(invalid, 'INVALID_PARAM_NEGATIVE_BINOMIAL')

        if name == 'Geometric':
            prob = values[0].to(self.REAL)
            return self._error_if_invalid((prob < 0) | (prob > 1), 'INVALID_PARAM_GEOMETRIC')

        if name == 'Pareto':
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid((shape <= 0) | (scale <= 0), 'INVALID_PARAM_PARETO')

        if name == 'Student':
            df = values[0].to(self.REAL)
            return self._error_if_invalid(df <= 0, 'INVALID_PARAM_STUDENT')

        if name == 'Gumbel':
            _, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid(scale <= 0, 'INVALID_PARAM_GUMBEL')

        if name == 'Laplace':
            _, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid(scale <= 0, 'INVALID_PARAM_LAPLACE')

        if name == 'Cauchy':
            _, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid(scale <= 0, 'INVALID_PARAM_CAUCHY')

        if name == 'Gompertz':
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid((shape <= 0) | (scale <= 0), 'INVALID_PARAM_GOMPERTZ')

        if name == 'ChiSquare':
            df = values[0].to(self.REAL)
            return self._error_if_invalid(df <= 0, 'INVALID_PARAM_CHISQUARE')

        if name == 'Kumaraswamy':
            a, b = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            return self._error_if_invalid((a <= 0) | (b <= 0), 'INVALID_PARAM_KUMARASWAMY')

        if name in {'Discrete', 'UnnormDiscrete'}:
            prob = torch.stack([val.to(self.REAL) for val in values], dim=-1)
            if name == 'UnnormDiscrete':
                prob = prob / torch.sum(prob, dim=-1, keepdim=True)
            invalid = torch.any(prob < 0)
            prob_sum = torch.sum(prob, dim=-1)
            invalid = invalid | torch.logical_not(
                torch.all(torch.isclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4, rtol=1e-4))
            )
            return self._error_if_invalid(invalid, 'INVALID_PARAM_DISCRETE')

        if name in {'Discrete(p)', 'UnnormDiscrete(p)'}:
            prob = values[0].to(self.REAL)
            if name == 'UnnormDiscrete(p)':
                prob = prob / torch.sum(prob, dim=-1, keepdim=True)
            invalid = torch.any(prob < 0)
            prob_sum = torch.sum(prob, dim=-1)
            invalid = invalid | torch.logical_not(
                torch.all(torch.isclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4, rtol=1e-4))
            )
            return self._error_if_invalid(invalid, 'INVALID_PARAM_DISCRETE')

        return self.ERROR_CODES['NORMAL']


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, log_expr: bool=False, log_jax_expr: bool=False,
                heading: str='') -> None:
        """Initialize tensors, analyze dependencies, and build callables.

        Args:
            log_expr: Whether to log symbolic expressions.
            log_jax_expr: Kept for API parity (unused in torch backend).
            heading: Heading passed to the logger when printing expressions.

        Example:
            >>> compiler.compile(log_expr=True, heading='SIM')
            >>> compiler.cpfs['next_state']  # torch callable
        """

        # initializer = TorchRDDLValueInitializer(self.rddl, logger=self.logger)
        # init_values_np = initializer.initialize()
        # self.init_values = self._tensorize_structure(init_values_np)

        
        # RDDLLevlesAnalysis computes a topological sort of the CPFs to determine a 
        # safe evaluation order, and also detects any cycles in the CPF dependencies.
        
        # The allow_synchronous_state=True flag allows it to handle cases where 
        # next-state variables depend on each other,which is common in RDDL models. 
        
        # The resulting levels are used to ensure that when we compile the CPFs into callables, 
        # we can evaluate them in an order that respects their dependencies.
        
        sorter = RDDLLevelAnalysis(self.rddl, allow_synchronous_state=True,
                                   logger=self.logger) # not in numpy
        self.levels = sorter.compute_levels() # not in numpy

        # Although the Torch backend executes expressions eagerly (step-by-step),
        # we still use the tracer to analyze the RDDL AST once and cache structural
        # information (such as tensor slices, object indices, and aggregation axes),
        # so the simulator can evaluate expressions efficiently without repeatedly
        # interpreting the symbolic RDDL structure.
        # Example (Reservoir domain):
            #
            # RDDL expression:
            #     sum_{?r : reservoir} rlevel(?r)
            #
            # Meaning:
            #     compute the total water level across all reservoirs.
            #
            # The tracer analyzes this expression once and determines:
            #     axis = 0   # the tensor dimension corresponding to ?r (reservoir objects)
            #
            # Then during simulation we can directly execute:
            # total_level = torch.sum(subs['rlevel'], dim=0)
                    
        tracer = RDDLObjectsTracer(self.rddl, logger=self.logger,
                                   cpf_levels=self.levels) # not in numpy
        self.traced = tracer.trace() # not in numpy
        
       
        init_params: Dict[str, Any] = {}
        self.model_params = init_params


        constraint_dtype = None if self.uses_relaxed_logic else torch.bool
        # Compile invariants meaning the conditions that must hold in every state, 
        

        # and terminations meaning the conditions that determine whether a state is terminal.
        self.invariants = [self._torch(expr, init_params, dtype=constraint_dtype)
                           for expr in self.rddl.invariants]
        
        # preconditions meaning the conditions that must hold for an action to be applicable.
        self.preconditions = [self._torch(expr, init_params, dtype=constraint_dtype) for expr in self.rddl.preconditions]
        
        # and terminations meaning the conditions that determine whether a state is terminal.        
        self.terminations = [self._torch(expr, init_params, dtype=constraint_dtype) for expr in self.rddl.terminations]
        
        self.cpfs = self._compile_cpfs(init_params)

        self.reward = self._torch(self.rddl.reward, init_params, dtype=self.REAL)
       
        self.model_params = init_params

    # ------------------------------------------------------------------
    def compile_transition(self, cache_path_info: bool=False) -> CallableExpr:
        rddl = self.rddl
        reward_fn = self.reward
        cpfs = self.cpfs
        preconds = self.preconditions
        invariants = self.invariants
        terminals = self.terminations

        if reward_fn is None:
            raise RuntimeError('compile() must be called before compile_transition().')
        
        # helper to coerce tensors to bool for precondition/invariant/termination checks
        def _to_bool(value: Any) -> bool:
            tensor = self._ensure_tensor(value)
            if isinstance(tensor, torch.Tensor):
                return bool(torch.all(tensor.bool()).item())
            return bool(tensor)

        def _clone_log_value(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, dict):
                return {k: _clone_log_value(v) for (k, v) in value.items()}
            if isinstance(value, list):
                return [_clone_log_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_clone_log_value(v) for v in value)
            return deepcopy(value)

        def _torch_wrapped_single_step(key, actions, subs, model_params):
            errors = self.ERROR_CODES['NORMAL']

        
            #####################################################
            ########## main idea of the step function ###########
            #####################################################


            # subs is the current state and action values, which we update in-place as we compute CPFs and reward. 
            # The final subs returned at the end of the step will have the next state values.
            subs.update(actions)
            # subs is a dictionary mapping variable names to
            # their current values (as tensors).
            # this dictionary come from the tracer, 
            # and is updated in-place as we compute the CPFs and reward for the current step.
            # subs look like :
            # {
            #     'rlevel_R1': tensor(...),
            #     'rlevel_R2': tensor(...),
            #     'rlevel_R3': tensor(...),
            #     'action': tensor(...),
            #     ...
            # }
            precondition_check = True
            for precond in preconds:
                sample, key, err, model_params = precond(subs, model_params, key)
                precondition_check = precondition_check and _to_bool(sample)
                errors |= err

            # calculate CPFs in topological order
            for (name, cpf) in cpfs.items():
                value, key, err, model_params = cpf(subs, model_params, key)
                subs[name] = value
                errors |= err
                # the cpf come from:
                #
                #    RDDL file
                #       ↓
                #    Parser (pyRDDLGym)
                #       ↓
                #    RDDLLiftedModel
                #       ↓
                #    self.rddl.cpfs
                #       ↓
                #    expr  (AST)
                #       ↓
                #    _torch(expr)
                #       ↓
                #    cpf function


            # calculate the immediate reward
            reward, key, err, model_params = reward_fn(subs, model_params, key)
            errors |= err

            # Cache a pre-commit snapshot so state fluents reflect s_t while
            # next-state fluents still expose s_{t+1}, matching the JAX backend.
            if cache_path_info:
                fluents = {
                    name: _clone_log_value(values) for (name, values) in subs.items()
                    if name not in rddl.non_fluents
                }
            else:
                fluents = {}

            # set the next state to the current state
            for (state, next_state) in rddl.next_state.items():
                # here the state update to the next state
                subs[state] = subs[next_state]

            
            #####################################################



            invariant_check = True
            for invariant in invariants:
                sample, key, err, model_params = invariant(subs, model_params, key)
                invariant_check = invariant_check and _to_bool(sample)
                errors |= err

            terminated_check = False
            for terminal in terminals:
                sample, key, err, model_params = terminal(subs, model_params, key)
                terminated_check = terminated_check or _to_bool(sample)
                errors |= err

            log = {
                'fluents': fluents,
                'reward': reward,
                'error': errors,
                'precondition': precondition_check,
                'invariant': invariant_check,
                'termination': terminated_check
            }
            return subs, log, model_params

        return _torch_wrapped_single_step
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Expression dispatch
    # ------------------------------------------------------------------

    def _torch(self, expr, init_params, dtype=None) -> CallableExpr:
        """Recursively dispatch expressions to specialized Torch builders.

        Args:
            expr: Parsed pyRDDLGym expression node.
            init_params: Dict of compiler hyperparameters.
            dtype: Optional override for the callable output dtype.

        Returns:
            CallableExpr: Torch-ready callable evaluating the expression.

        Example:
            >>> fn = compiler._torch(expr, {})
            >>> value, key, err, params = fn(subs, {}, None)
        """
        etype, _ = expr.etype
        if etype == 'constant':
            fn = self._torch_constant(expr) # check
        elif etype == 'pvar':
            fn = self._torch_pvar(expr, init_params) # check
        elif etype == 'arithmetic':
            fn = self._torch_arithmetic(expr, init_params) # helf check - need to check unary and
        elif etype == 'relational':
            fn = self._torch_relational(expr, init_params)
        elif etype == 'boolean':
            fn = self._torch_logical(expr, init_params)
        elif etype == 'aggregation':
            fn = self._torch_aggregation(expr, init_params)
        elif etype == 'func':
            fn = self._torch_functional(expr, init_params)
        elif etype == 'pyfunc':
            fn = self._torch_pyfunc(expr, init_params)
        elif etype == 'control':
            fn = self._torch_control(expr, init_params)
        elif etype == 'randomvar':
            fn = self._torch_random(expr, init_params)
        elif etype == 'randomvector':
            fn = self._torch_random_vector(expr, init_params)
        elif etype == 'matrix':
            fn = self._torch_matrix(expr, init_params)
        else:
            raise RDDLNotImplementedError(
                f'Expression type {etype} is not supported.\n' + print_stack_trace(expr))

        if dtype is not None:
            def _cast(subs, params, key):
                value, key, err, params = fn(subs, params, key)
                tensor = self._ensure_tensor(value)


                # If we get here with a non-tensor (e.g., a Python function),
                # casting will crash. Raise an informative error instead.
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"Expected torch.Tensor from compiled expr (etype={expr.etype}) "
                        f"but got {type(tensor).__name__}: {tensor!r}"
                    )

                return tensor.to(dtype=dtype), key, err, params
            return _cast
        return fn

    # ------------------------------------------------------------------
    # Leaves
    # ------------------------------------------------------------------

    def _torch_constant(self, expr) -> CallableExpr:
        """Return a callable that always emits the cached constant value.

        Args:
            expr: Constant expression node.

        Returns:
            CallableExpr: Callable ignoring substitutions and yielding tensor.

        Example:
            >>> fn = compiler._torch_constant(expr)
            >>> tensor, _, _, _ = fn({}, {}, None)
        """
        cached_value = self.traced.cached_sim_info(expr)
        tensor = self._ensure_tensor(cached_value)

        def _fn(subs, params, key):
            return tensor, key, self.ERROR_CODES['NORMAL'], params

        return _fn

    def _torch_pvar(self, expr, init_params) -> CallableExpr:
        """Compile state/action parameterized variables, honoring slices.

        Args:
            expr: Parameterized variable expression.
            init_params: Compilation hyperparameters.

        Returns:
            CallableExpr: Callable retrieving the right slice/value.

        Example:
            >>> fn = compiler._torch_pvar(expr, {})
            >>> value, _, _, _ = fn(subs, {}, None)
        """
        var, pvars = expr.args
        is_value, cached_info = self.traced.cached_sim_info(expr)

        if is_value:
            tensor = self._ensure_tensor(cached_info)

            def _fn(subs, params, key):
                return tensor, key, self.ERROR_CODES['NORMAL'], params

            return _fn

        if cached_info is None:
            def _scalar(subs, params, key):
                value = self._ensure_tensor(subs[var])
                return value, key, self.ERROR_CODES['NORMAL'], params

            return _scalar

        slices, axis, shape, op_code, op_args = cached_info
        # its not numpy, but we use the same tracer op codes for slicing/reshaping/etc. so we can reuse the same cached info
        tracer = RDDLObjectsTracer.NUMPY_OP_CODE
        # to make sure the slice objects are properly converted to callables that return tensors/indices, 
        # we wrap them using _torch_slice
        if slices and op_code == tracer.NESTED_SLICE:
            compiled_slices = [
                self._torch(arg, init_params) if _slice is None
                else self._torch_slice(_slice) for (arg, _slice) in zip(pvars, slices)
                              ]

            def _nested(subs, params, key):
                value = self._ensure_tensor(subs[var])
                new_slices = []
                error = self.ERROR_CODES['NORMAL']
                for inner in compiled_slices:
                    idx, key, err, params = inner(subs, params, key)
                    if isinstance(idx, torch.Tensor):
                        new_slices.append(idx.to(dtype=torch.long))
                    else:
                        new_slices.append(idx)
                    error |= err
                tuple_slices = tuple(new_slices)
                sample = value[tuple_slices]
                return sample, key, error, params

            return _nested

        def _non_nested(subs, params, key):
            sample = self._ensure_tensor(subs[var])
            if not isinstance(sample, torch.Tensor):
                return sample, key, self.ERROR_CODES['NORMAL'], params
            if slices:
                sample = sample[slices]

            
            if axis:
                current = sample
                for ax in sorted(axis):
                    # unsqueeze to add singleton dimensions at the specified axes, then expand to the target shape
                    # the same idea of expand_dims of jnp 
                    current = torch.unsqueeze(current, dim = ax)
                # the shape from the tracer is the target shape after broadcasting, so we need to expand the unsqueezed tensor to that shape
                target_shape = shape
                if not isinstance(target_shape, tuple):
                    target_shape = (target_shape,)

                if len(target_shape) < current.dim():
                    # preserve trailing dims from current if target is shorter
                    trailing = tuple(current.shape[len(target_shape):])
                    target_shape = tuple(target_shape) + trailing
                try:
                    sample = current.expand(target_shape) # duplicates the values along the new axes without actually copying data, line the jax version
                except Exception:
                    sample = current.expand(*target_shape)


            # apply tensor contraction when duplicated logical variables appear in the RDDL expression
            # example: fluent(?x, ?x)
            # after slicing/broadcasting the tensor may contain repeated axes that represent
            # the same logical variable; in this case the tracer marks the operation as EINSUM.
            # op_args contains:
            #   op_args[0] -> the einsum equation (e.g. 'ij->i', 'ij,jk->ik')
            #   op_args[1:] -> optional additional tensors used in the contraction
            # torch.einsum then performs the required reduction / contraction to merge the
            # duplicated dimensions and produce the correct tensor shape.
            if op_code == tracer.EINSUM:
                equation = op_args[0] # the einsum equation string (e.g. 'ij,jk->ik')
                operands = op_args[1:] if len(op_args) > 1 else () # any additional operand tensors needed for the einsum (e.g. the 'jk' matrix in a matmul)
                sample = torch.einsum(equation, sample, *operands)
            
            
            elif op_code == tracer.TRANSPOSE:
                sample = sample.permute(*op_args)
            return sample, key, self.ERROR_CODES['NORMAL'], params

        return _non_nested

    def _torch_slice(self, slice_value):
        """Wrap literal slices into callables returning tensors/indices.

        Args:
            slice_value: Slice, Ellipsis, None, or tensor-literal index.

        Returns:
            CallableExpr: Callable ignoring inputs and returning the slice.

        Example:
            >>> slice_fn = compiler._torch_slice(slice(0, 2))
            >>> slice_fn({}, {}, None)[0]
            slice(0, 2, None)
        """
        # check if slice_value is a native indexing object:
        # slice(...)  -> standard Python slice (e.g. slice(0,2), slice(None,None,-1))
        # Ellipsis (...) -> shorthand meaning "all remaining dimensions"
        # None -> adds a new axis (equivalent to numpy.newaxis)
        if isinstance(slice_value, (slice, type(Ellipsis), type(None))):
            stored = slice_value
        else:
            stored = self._ensure_tensor(slice_value)

        def _fn(subs, params, key):
            return stored, key, self.ERROR_CODES['NORMAL'], params

        return _fn

    # ------------------------------------------------------------------
    # Helpers for operations
    # ------------------------------------------------------------------

    def _bind_logic_operator(self, category: str, expr_id: Any,
                             init_params: Dict[str, Any],
                             op_name: Optional[str]=None) -> Callable:
        """Bind a backend operator once during compilation.

        The torch compiler used to probe the logic backend dynamically at
        runtime. Fuzzy backends expose factory-style APIs instead, so we resolve
        the callable here and reuse it for every execution of the compiled node.
        """
        factory = self.OPS[category]
        if op_name is not None:
            factory = factory[op_name]
        return factory(expr_id, init_params)

    def _coerce_logic_tensor(self, value: Any, *, promote_real: bool=False) -> torch.Tensor:
        """Convert values before dispatching them into a logic backend."""
        tensor = self._ensure_tensor(value)
        if not isinstance(tensor, torch.Tensor):
            dtype = self.REAL if promote_real else None
            tensor = as_tensor_on_device(value, dtype=dtype, device=self.device)
        elif promote_real and not tensor.is_floating_point():
            tensor = tensor.to(dtype=self.REAL)
        return tensor

    def _apply_unary(self, name: str, value: torch.Tensor):
        """Apply exact unary ops.

        Relaxed/fuzzy unary operators are already intercepted earlier in the
        compiler and bound through `self.OPS`. By the time execution reaches
        this helper, the remaining path should be plain torch semantics.

        Args:
            name: Identifier of the unary operation.
            value: Input tensor.

        Returns:
            torch.Tensor: Result after applying the operator.

        Example:
            >>> compiler._apply_unary('neg', torch.tensor(1.))
            tensor(-1.)
        """
        if name == 'neg':
            return -value
        if name == 'logical_not':
            return torch.logical_not(value)
        raise ValueError(f'Unsupported unary op {name}.')

    def _apply_binary(self, name: str, lhs: torch.Tensor, rhs: torch.Tensor):
        """Apply exact binary operations.

        Relaxed comparisons and logical operators are compiled via bound logic
        callables before this helper is ever used. This helper therefore keeps
        only the exact torch fallback behavior.

        Args:
            name: Operation identifier.
            lhs: Left-hand tensor.
            rhs: Right-hand tensor.

        Returns:
            torch.Tensor: Result tensor.

        Example:
            >>> compiler._apply_binary('add', torch.tensor(1.), torch.tensor(2.))
            tensor(3.)
        """
        if not isinstance(lhs, torch.Tensor):
            reference_device = infer_torch_device(rhs, default=self.device)
            try:
                lhs = as_tensor_on_device(lhs, dtype=self.REAL, device=reference_device)
            except Exception:
                lhs = torch.tensor(0.0, dtype=self.REAL, device=reference_device)
        if not isinstance(rhs, torch.Tensor):
            reference_device = infer_torch_device(lhs, default=self.device)
            try:
                rhs = as_tensor_on_device(rhs, dtype=lhs.dtype, device=reference_device)
            except Exception:
                rhs = torch.tensor(0.0, dtype=lhs.dtype, device=reference_device)

        # align ranks
        while lhs.dim() < rhs.dim():
            lhs = lhs.unsqueeze(0)
        while rhs.dim() < lhs.dim():
            rhs = rhs.unsqueeze(0)

        # try to align leading batch dimension if mismatched
        if lhs.dim() > 0 and rhs.dim() > 0 and lhs.shape[0] != rhs.shape[0]:
            target = max(lhs.shape[0], rhs.shape[0])
            if lhs.shape[0] != target:
                if lhs.shape[0] != 1:
                    lhs = lhs.unsqueeze(0)
                lhs = lhs.expand((target,) + lhs.shape[1:])
            if rhs.shape[0] != target:
                if rhs.shape[0] != 1:
                    rhs = rhs.unsqueeze(0)
                rhs = rhs.expand((target,) + rhs.shape[1:])
        if name == 'add':
            return torch.add(lhs, rhs)
        if name == 'sub':
            return torch.sub(lhs, rhs)
        if name == 'mul':
            try:
                return torch.mul(lhs, rhs)
            except RuntimeError:
                # heuristic broadcasting to align batch/object dims
                if lhs.dim() == 1 and rhs.dim() == 1 and lhs.shape[0] != rhs.shape[0]:
                    lhs = lhs.unsqueeze(-1)
                    rhs = rhs.unsqueeze(0)
                # pad smaller rank tensor with trailing singleton dims
                while lhs.dim() < rhs.dim():
                    lhs = lhs.unsqueeze(-1)
                while rhs.dim() < lhs.dim():
                    rhs = rhs.unsqueeze(-1)
                try:
                    lhs_b, rhs_b = torch.broadcast_tensors(lhs, rhs)
                    return torch.mul(lhs_b, rhs_b)
                except Exception:
                    if lhs.dim() == 2 and rhs.dim() == 2:
                        lhs = lhs.unsqueeze(-1)  # (b,n)->(b,n,1) or (n,n)->(n,n,1)
                        rhs = rhs.unsqueeze(0)   # (n,n)->(1,n,n)
                        lhs_b, rhs_b = torch.broadcast_tensors(lhs, rhs)
                        return torch.mul(lhs_b, rhs_b)
                    return torch.mul(lhs, rhs)
        if name == 'div':
            return torch.div(lhs, rhs)
        if name == 'pow':
            return torch.pow(lhs, rhs)
        if name == 'lt':
            return torch.lt(lhs, rhs)
        if name == 'le':
            return torch.le(lhs, rhs)
        if name == 'gt':
            return torch.gt(lhs, rhs)
        if name == 'ge':
            return torch.ge(lhs, rhs)
        if name == 'eq':
            return torch.eq(lhs, rhs)
        if name == 'ne':
            return torch.ne(lhs, rhs)
        if name == 'logical_and':
            return torch.logical_and(lhs, rhs)
        if name == 'logical_or':
            return torch.logical_or(lhs, rhs)
        if name == 'implies':
            return torch.logical_or(torch.logical_not(lhs.bool()), rhs.bool())
        if name == 'equiv':
            return torch.eq(lhs.bool(), rhs.bool())
        raise ValueError(f'Unsupported binary op {name}.')

    def _apply_control_if(self, pred: torch.Tensor,
                          then_value: torch.Tensor,
                          else_value: torch.Tensor):
        """Evaluate an exact `if` control expression.

        Relaxed control flow is handled in `_torch_control` by binding the
        backend-specific operator during compilation. This helper is only for
        the exact branch.

        Args:
            pred: Predicate tensor.
            then_value: Tensor when predicate evaluates to true.
            else_value: Tensor when predicate evaluates to false.

        Returns:
            torch.Tensor: Branch result.

        Example:
            >>> pred = torch.tensor(True)
            >>> compiler._apply_control_if(pred, torch.tensor(1.), torch.tensor(0.))
            tensor(1.)
        """
        # coerce non-tensor predicates to a boolean
        if not isinstance(pred, torch.Tensor):
            pred_device = infer_torch_device(then_value, else_value, default=self.device)
            try:
                pred_val = bool(pred)
            except Exception:
                pred_val = False
            pred = torch.tensor(pred_val, dtype=self.REAL, device=pred_device)
        return torch.where(pred.to(dtype=self.REAL) > 0.5, then_value, else_value)

    def _apply_control_switch(self, pred: torch.Tensor, cases: torch.Tensor):
        """Select a case tensor according to the predicate index.

        The fuzzy `switch` path is resolved earlier by `_torch_control`. This
        helper preserves the exact gather-style semantics only.

        Args:
            pred: Tensor encoding the case index.
            cases: Tensor stack of possible values.

        Returns:
            torch.Tensor: Selected case tensor.

        Example:
            >>> compiler._apply_control_switch(torch.tensor(1), torch.tensor([[0.], [2.]]))
            tensor(2.)
        """
        pred_long = pred.to(dtype=torch.long).unsqueeze(0)
        reference = cases[:1]
        expanded_index = pred_long.expand_as(reference)
        gathered = torch.gather(cases, 0, expanded_index)
        return gathered.squeeze(0)

    def _aggregate(self, op: str, tensor: torch.Tensor, axes: Optional[Sequence[int]]):
        """Run reduction ops across specified axes.

        Args:
            op: Aggregation name.
            tensor: Input tensor.
            axes: Axes to reduce, or `None` for all.

        Returns:
            torch.Tensor: Reduced tensor.

        Example:
            >>> compiler._aggregate('sum', torch.ones(2, 3), axes=(0,))
            tensor([2., 2., 2.])
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = self._ensure_tensor(tensor)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(bool(tensor), device=self.device)
        if axes is None:
            axes = tuple(range(tensor.dim()))
        axes = tuple(axes) if isinstance(axes, (list, tuple)) else (axes,)
        normalized_axes = []
        for axis in axes:
            if axis is None:
                continue
            normalized_axis = axis if axis >= 0 else tensor.dim() + axis
            if normalized_axis not in normalized_axes:
                normalized_axes.append(normalized_axis)
        axes = tuple(normalized_axes)
        if not axes:
            return tensor
        if op in {'argmin', 'argmax'}:
            reducer = torch.argmin if op == 'argmin' else torch.argmax
            if tensor.dim() == 0:
                return torch.zeros((), dtype=self.INT, device=tensor.device)
            if len(axes) == 1:
                return reducer(tensor, dim=axes[0]).to(dtype=self.INT)
            keep_axes = tuple(axis for axis in range(tensor.dim()) if axis not in axes)
            permuted = tensor.permute(keep_axes + axes)
            kept_shape = [tensor.shape[axis] for axis in keep_axes]
            reduced_size = 1
            for axis in axes:
                reduced_size *= tensor.shape[axis]
            if kept_shape:
                reshaped = permuted.reshape(*kept_shape, reduced_size)
            else:
                reshaped = permuted.reshape(reduced_size)
            return reducer(reshaped, dim=-1).to(dtype=self.INT)
        result = tensor
        for axis in sorted(axes, reverse=True):
            if op == 'sum':
                result = torch.sum(result, dim=axis)
            elif op == 'avg':
                reduced_input = result
                if not reduced_input.is_floating_point() and not torch.is_complex(reduced_input):
                    reduced_input = reduced_input.to(dtype=self.REAL)
                result = torch.mean(reduced_input, dim=axis)
            elif op == 'prod':
                result = torch.prod(result, dim=axis)
            elif op == 'minimum':
                result = torch.min(result, dim=axis).values
            elif op == 'maximum':
                result = torch.max(result, dim=axis).values
            elif op == 'forall':
                result = torch.all(result.bool(), dim=axis)
            elif op == 'exists':
                result = torch.any(result.bool(), dim=axis)
            else:
                raise ValueError(f'Unsupported aggregation op {op}.')
        return result

    # ------------------------------------------------------------------
    # Arithmetic / Logical / Relational
    # ------------------------------------------------------------------

    def _torch_arithmetic(self, expr, init_params) -> CallableExpr:
        """Compile arithmetic expressions into torch callables.

        Args:
            expr: Arithmetic AST node.
            init_params: Compilation parameters.

        Returns:
            CallableExpr: Callable executing +,-,*,/ chains.

        Example:
            >>> fn = compiler._torch_arithmetic(expr, {})
            >>> value, key, err, params = fn(subs, {}, None)
        """
        _, op = expr.etype
        #its happen in the elif in jax  its means 
        args = [self._torch(arg, init_params) for arg in expr.args]

        if len(args) == 1 and op == '-':
            def _neg(subs, params, key):
                value, key, err, params = args[0](subs, params, key)
                return self._apply_unary('neg', value), key, err, params
            return _neg

        if len(args) < 2:
            raise RDDLNotImplementedError(
                f'Arithmetic operator {op} requires at least two arguments.\n' +
                print_stack_trace(expr))

        def _fn(subs, params, key):
            value, key, err, params = args[0](subs, params, key)
            for arg_fn in args[1:]:
                rhs, key, err_rhs, params = arg_fn(subs, params, key)
                err |= err_rhs
                if op == '+':
                    value = self._apply_binary('add', value, rhs)
                elif op == '-':
                    value = self._apply_binary('sub', value, rhs)
                elif op == '*':
                    value = self._apply_binary('mul', value, rhs)
                elif op == '/':
                    value = self._apply_binary('div', value, rhs)
                else:
                    raise RDDLNotImplementedError(
                        f'Arithmetic operator {op} is not supported.\n' +
                        print_stack_trace(expr))
            return value, key, err, params

        return _fn

    def _torch_relational(self, expr, init_params) -> CallableExpr:
        """Compile <, <=, ==, etc. comparisons.

        Args:
            expr: Relational AST node.
            init_params: Compilation parameters.

        Returns:
            CallableExpr: Callable returning boolean tensors.

        Example:
            >>> fn = compiler._torch_relational(expr, {})
            >>> fn(subs, {}, None)
        """
        _, op = expr.etype
        lhs, rhs = expr.args
        lhs_fn = self._torch(lhs, init_params)
        rhs_fn = self._torch(rhs, init_params)

        if self.uses_relaxed_logic:
            logic_op = self._bind_logic_operator('relational', expr.id, init_params, op)

            def _fn(subs, params, key):
                left, key, err1, params = lhs_fn(subs, params, key)
                right, key, err2, params = rhs_fn(subs, params, key)
                left = self._coerce_logic_tensor(left, promote_real=True)
                right = self._coerce_logic_tensor(right, promote_real=True)
                value, params = logic_op(left, right, params)
                return value, key, err1 | err2, params

            return _fn

        def _fn(subs, params, key):
            left, key, err1, params = lhs_fn(subs, params, key)
            right, key, err2, params = rhs_fn(subs, params, key)
            if op == '<':
                value = self._apply_binary('lt', left, right)
            elif op == '<=':
                value = self._apply_binary('le', left, right)
            elif op == '>':
                value = self._apply_binary('gt', left, right)
            elif op == '>=':
                value = self._apply_binary('ge', left, right)
            elif op == '==':
                value = self._apply_binary('eq', left, right)
            elif op == '~=':
                value = self._apply_binary('ne', left, right)
            else:
                raise RDDLNotImplementedError(
                    f'Relational operator {op} is not supported.\n' +
                    print_stack_trace(expr))
            return value, key, err1 | err2, params

        return _fn

    def _torch_logical(self, expr, init_params) -> CallableExpr:
        """Compile logical operators (~, &, |, ^, =>, <=>).

        Args:
            expr: Boolean AST node.
            init_params: Compilation parameters.

        Returns:
            CallableExpr: Callable producing boolean tensors.

        Example:
            >>> fn = compiler._torch_logical(expr, {})
            >>> fn(subs, {}, None)
        """
        _, op = expr.etype
        args = [self._torch(arg, init_params) for arg in expr.args]

        if self.uses_relaxed_logic:
            if len(args) == 1 and op == '~':
                logic_not = self._bind_logic_operator('logical_not', expr.id, init_params)

                def _not(subs, params, key):
                    value, key, err, params = args[0](subs, params, key)
                    tensor = self._coerce_logic_tensor(value, promote_real=True)
                    result, params = logic_not(tensor, params)
                    return result, key, err, params

                return _not

            if len(args) < 2:
                raise RDDLNotImplementedError(
                    f'Logical operator {op} requires at least two arguments.\n' +
                    print_stack_trace(expr))
            if op in {'=>', '<=>'} and len(args) != 2:
                raise RDDLNotImplementedError(
                    f'Logical operator {op} requires exactly two arguments.\n' +
                    print_stack_trace(expr))

            if op in {'^', '&', '|'}:
                logic_ops = [
                    self._bind_logic_operator('logical', f'{expr.id}_{op}{i}', init_params, op)
                    for i in range(len(args) - 1)
                ]
            else:
                logic_ops = [self._bind_logic_operator('logical', expr.id, init_params, op)]

            def _fn(subs, params, key):
                value, key, err, params = args[0](subs, params, key)
                value = self._coerce_logic_tensor(value, promote_real=True)
                for i, arg_fn in enumerate(args[1:]):
                    rhs, key, err_rhs, params = arg_fn(subs, params, key)
                    rhs = self._coerce_logic_tensor(rhs, promote_real=True)
                    err |= err_rhs
                    logic_op = logic_ops[i if op in {'^', '&', '|'} else 0]
                    value, params = logic_op(value, rhs, params)
                return value, key, err, params

            return _fn

        if len(args) == 1 and op == '~':
            def _not(subs, params, key):
                value, key, err, params = args[0](subs, params, key)
                tensor = self._ensure_tensor(value)
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(bool(tensor), device=self.device)
                return self._apply_unary('logical_not', tensor.bool()), key, err, params
            return _not

        if len(args) < 2:
            raise RDDLNotImplementedError(
                f'Logical operator {op} requires at least two arguments.\n' +
                print_stack_trace(expr))
        if op in {'=>', '<=>'} and len(args) != 2:
            raise RDDLNotImplementedError(
                f'Logical operator {op} requires exactly two arguments.\n' +
                print_stack_trace(expr))

        def _fn(subs, params, key):
            value, key, err, params = args[0](subs, params, key)
            tensor = self._ensure_tensor(value)
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(bool(tensor), device=self.device)
            value = tensor.bool()
            for arg_fn in args[1:]:
                rhs, key, err_rhs, params = arg_fn(subs, params, key)
                rhs_tensor = self._ensure_tensor(rhs)
                if not isinstance(rhs_tensor, torch.Tensor):
                    rhs_tensor = torch.tensor(bool(rhs_tensor), device=value.device)
                rhs = rhs_tensor.bool()
                err |= err_rhs
                if op in {'^', '&'}:
                    value = self._apply_binary('logical_and', value, rhs)
                elif op == '|':
                    value = self._apply_binary('logical_or', value, rhs)
                elif op == '=>':
                    value = self._apply_binary('implies', value, rhs)
                elif op == '<=>':
                    value = self._apply_binary('equiv', value, rhs)
                else:
                    raise RDDLNotImplementedError(
                        f'Logical operator {op} is not supported.\n' +
                        print_stack_trace(expr))
            return value, key, err, params

        return _fn

    # ------------------------------------------------------------------
    # Aggregation / Functional
    # ------------------------------------------------------------------

    def _torch_aggregation(self, expr, init_params) -> CallableExpr:
        """Compile aggregation expressions.

        Args:
            expr: Aggregation AST node.
            init_params: Compilation parameters.

        Returns:
            CallableExpr: Callable reducing tensors along axes.

        Example:
            >>> fn = compiler._torch_aggregation(expr, {})
            >>> fn(subs, {}, None)
        """
        _, op = expr.etype
        *_, arg = expr.args
        _, axes = self.traced.cached_sim_info(expr)
        arg_fn = self._torch(arg, init_params)

        if self.uses_relaxed_logic and op in {'forall', 'exists', 'argmin', 'argmax'}:
            logic_op = self._bind_logic_operator('aggregation', expr.id, init_params, op)

            def _fn(subs, params, key):
                value, key, err, params = arg_fn(subs, params, key)
                promote_real = op in {'argmin', 'argmax'}
                tensor = self._coerce_logic_tensor(value, promote_real=promote_real)
                reduced, params = logic_op(tensor, axis=axes, params=params)
                return reduced, key, err, params

            return _fn

        def _fn(subs, params, key):
            value, key, err, params = arg_fn(subs, params, key)
            if op == 'sum':
                reduced = self._aggregate('sum', value, axes)
            elif op == 'avg':
                reduced = self._aggregate('avg', value, axes)
            elif op == 'prod':
                reduced = self._aggregate('prod', value, axes)
            elif op == 'minimum':
                reduced = self._aggregate('minimum', value, axes)
            elif op == 'maximum':
                reduced = self._aggregate('maximum', value, axes)
            elif op == 'forall':
                reduced = self._aggregate('forall', value, axes)
            elif op == 'exists':
                reduced = self._aggregate('exists', value, axes)
            elif op == 'argmin':
                reduced = self._aggregate('argmin', value, axes)
            elif op == 'argmax':
                reduced = self._aggregate('argmax', value, axes)
            else:
                raise RDDLNotImplementedError(
                    f'Aggregation {op} not supported.\n' + print_stack_trace(expr))
            return reduced, key, err, params

        return _fn

    def _torch_functional(self, expr, init_params) -> CallableExpr:
        """Compile unary/binary function calls like sin, pow, min, etc.

        Args:
            expr: Functional AST node.
            init_params: Compilation parameters.

        Returns:
            CallableExpr: Callable dispatching to torch math.

        Example:
            >>> fn = compiler._torch_functional(expr, {})
            >>> fn(subs, {}, None)
        """
        _, op = expr.etype
        if len(expr.args) == 1:
            arg_fn = self._torch(expr.args[0], init_params)

            if self.uses_relaxed_logic and op in {'sgn', 'round', 'floor', 'ceil', 'sqrt'}:
                logic_op = self._bind_logic_operator('unary', expr.id, init_params, op)

                def _unary_relaxed(subs, params, key):
                    val, key, err, params = arg_fn(subs, params, key)
                    tensor = self._coerce_logic_tensor(val, promote_real=True)
                    value, params = logic_op(tensor, params)
                    return value, key, err, params

                return _unary_relaxed

            def _unary(subs, params, key):
                val, key, err, params = arg_fn(subs, params, key)
                value = self._apply_function_unary(op, val)
                return value, key, err, params

            return _unary

        elif len(expr.args) == 2:
            lhs_fn = self._torch(expr.args[0], init_params)
            rhs_fn = self._torch(expr.args[1], init_params)

            if self.uses_relaxed_logic and op in {'div', 'mod', 'fmod'}:
                logic_op = self._bind_logic_operator('binary', expr.id, init_params, op)

                def _binary_relaxed(subs, params, key):
                    lhs, key, err1, params = lhs_fn(subs, params, key)
                    rhs, key, err2, params = rhs_fn(subs, params, key)
                    lhs = self._coerce_logic_tensor(lhs, promote_real=True)
                    rhs = self._coerce_logic_tensor(rhs, promote_real=True)
                    value, params = logic_op(lhs, rhs, params)
                    return value, key, err1 | err2, params

                return _binary_relaxed

            def _binary(subs, params, key):
                lhs, key, err1, params = lhs_fn(subs, params, key)
                rhs, key, err2, params = rhs_fn(subs, params, key)
                value = self._apply_function_binary(op, lhs, rhs)
                return value, key, err1 | err2, params

            return _binary

        raise RDDLNotImplementedError(
            f'Functional operator {op} is not supported.\n' + print_stack_trace(expr))

    def _apply_function_unary(self, op: str, value: torch.Tensor) -> torch.Tensor:
        """Apply exact unary math ops.

        Relaxed variants such as `sgn`, `floor`, `round`, `ceil`, and `sqrt`
        are intercepted in `_torch_functional` before reaching this helper.

        Args:
            op: Name of unary function (e.g., `sin`, `abs`).
            value: Input tensor.

        Returns:
            torch.Tensor: Result after applying the unary function.

        Example:
            >>> compiler._apply_function_unary('sqrt', torch.tensor(4.))
            tensor(2.)
        """
        # simple override for ops that in logic expect init_params (e.g., sgn)
        if op == 'sgn':
            tensor = self._ensure_tensor(value)
            if not isinstance(tensor, torch.Tensor):
                tensor = as_tensor_on_device(tensor, dtype=self.REAL, device=self.device)
            return torch.sign(tensor)
        funcs = {
            'abs': torch.abs,
            'exp': torch.exp,
            'ln': torch.log,
            'floor': torch.floor,
            'ceil': torch.ceil,
            'round': torch.round,
            'sqrt': torch.sqrt,
            'sin': torch.sin,
            'cos': torch.cos,
            'sigmoid': torch.sigmoid,
            'tan': torch.tan,
            'asin': torch.asin,
            'acos': torch.acos,
            'atan': torch.atan,
            'sinh': torch.sinh,
            'cosh': torch.cosh,
            'tanh': torch.tanh,
            'lngamma': torch.lgamma,
            'gamma': torch.special.gamma if hasattr(torch.special, 'gamma')
            else lambda x: torch.exp(torch.lgamma(x)),
            'sgn': torch.sign
        }
        if op not in funcs:
            raise RDDLNotImplementedError(f'Unary function {op} is not supported.')
        return funcs[op](value)

    def _apply_function_binary(self, op: str, lhs: torch.Tensor,
                               rhs: torch.Tensor) -> torch.Tensor:
        """Apply exact binary math ops.

        Relaxed variants such as `div` and `mod` are intercepted in
        `_torch_functional` and therefore never reach this helper.

        Args:
            op: Name of binary function (e.g., `min`, `pow`).
            lhs: Left operand.
            rhs: Right operand.

        Returns:
            torch.Tensor: Result tensor.

        Example:
            >>> compiler._apply_function_binary('min', torch.tensor(1.), torch.tensor(2.))
            tensor(1.)
        """
        funcs = {
            'min': torch.minimum,
            'max': torch.maximum,
            'pow': torch.pow,
            'div': torch.floor_divide,
            'mod': torch.remainder,
            'fmod': torch.remainder,
            'hypot': lambda a, b: torch.sqrt(a * a + b * b),
            'log': lambda a, b: torch.log(a) / torch.log(b)
        }
        if op not in funcs:
            raise RDDLNotImplementedError(f'Binary function {op} is not supported.')
        return funcs[op](lhs, rhs)

    # ------------------------------------------------------------------
    # External python functions
    # ------------------------------------------------------------------

    def _torch_pyfunc(self, expr, init_params) -> CallableExpr:
        """Compile external python function calls used inside RDDL.

        Args:
            expr: pyfunc AST node.
            init_params: Compilation parameters.

        Returns:
            CallableExpr: Callable invoking the python function and tensorizing outputs.

        Example:
            >>> fn = compiler._torch_pyfunc(expr, {})
            >>> fn(subs, {}, None)
        """
        _, pyfunc_name = expr.etype
        pyfunc = self.python_functions.get(pyfunc_name)
        if pyfunc is None:
            raise RDDLUndefinedVariableError(
                f'Undefined Python function <{pyfunc_name}>.\n' + print_stack_trace(expr))
        captured_vars, args = expr.args
        compiled_args = [self._torch(arg, init_params) for arg in args]

        def _fn(subs, params, key):
            values: List[torch.Tensor] = []
            error = self.ERROR_CODES['NORMAL']
            for arg in compiled_args:
                val, key, err, params = arg(subs, params, key)
                error |= err
                values.append(val)
            result = pyfunc(*values)
            tensor = self._ensure_tensor(result)
            return tensor, key, error, params

        return _fn

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _torch_control(self, expr, init_params) -> CallableExpr:
        """Compile control structures (`if`, `switch`).

        Args:
            expr: Control AST node.
            init_params: Compilation parameters.

        Returns:
            CallableExpr: Callable evaluating the control flow.

        Example:
            >>> fn = compiler._torch_control(expr, {})
            >>> fn(subs, {}, None)
        """
        _, op = expr.etype
        if op == 'if':
            pred, then_expr, else_expr = expr.args
            pred_fn = self._torch(pred, init_params)
            then_fn = self._torch(then_expr, init_params)
            else_fn = self._torch(else_expr, init_params)

            if self.uses_relaxed_logic:
                logic_if = self._bind_logic_operator('control', expr.id, init_params, 'if')

                def _fn(subs, params, key):
                    pred_val, key, err_pred, params = pred_fn(subs, params, key)
                    then_val, key, err_then, params = then_fn(subs, params, key)
                    else_val, key, err_else, params = else_fn(subs, params, key)
                    pred_tensor = self._coerce_logic_tensor(pred_val, promote_real=True)
                    then_tensor = self._coerce_logic_tensor(then_val, promote_real=True)
                    else_tensor = self._coerce_logic_tensor(else_val, promote_real=True)
                    result, params = logic_if(pred_tensor, then_tensor, else_tensor, params)
                    return result, key, err_pred | err_then | err_else, params

                return _fn

            def _fn(subs, params, key):
                pred_val, key, err_pred, params = pred_fn(subs, params, key)
                then_val, key, err_then, params = then_fn(subs, params, key)
                else_val, key, err_else, params = else_fn(subs, params, key)
                result = self._apply_control_if(pred_val, then_val, else_val)
                return result, key, err_pred | err_then | err_else, params

            return _fn

        elif op == 'switch':
            pred, *_ = expr.args
            cases, default = self.traced.cached_sim_info(expr)
            pred_fn = self._torch(pred, init_params)
            case_fns = [None if arg is None else self._torch(arg, init_params)
                        for arg in cases]
            default_fn = None if default is None else self._torch(default, init_params)

            if self.uses_relaxed_logic:
                logic_switch = self._bind_logic_operator('control', expr.id, init_params, 'switch')

                def _fn(subs, params, key):
                    pred_val, key, err, params = pred_fn(subs, params, key)
                    evaluated_cases = []
                    total_err = err
                    default_value = None
                    if default_fn is not None:
                        default_value, key, err_def, params = default_fn(subs, params, key)
                        total_err |= err_def
                    for case_fn in case_fns:
                        if case_fn is None:
                            if default_value is None:
                                raise RDDLNotImplementedError(
                                    'Switch case missing value and default is None.\n' +
                                    print_stack_trace(expr))
                            evaluated_cases.append(default_value)
                        else:
                            val, key, err_case, params = case_fn(subs, params, key)
                            total_err |= err_case
                            evaluated_cases.append(val)
                    pred_tensor = self._coerce_logic_tensor(pred_val, promote_real=True)
                    stacked = torch.stack([
                        self._coerce_logic_tensor(case_value, promote_real=True)
                        for case_value in evaluated_cases
                    ], dim=0)
                    result, params = logic_switch(pred_tensor, stacked, params)
                    return result, key, total_err, params

                return _fn

            def _fn(subs, params, key):
                pred_val, key, err, params = pred_fn(subs, params, key)
                evaluated_cases = []
                total_err = err
                default_value = None
                if default_fn is not None:
                    default_value, key, err_def, params = default_fn(subs, params, key)
                    total_err |= err_def
                for case_fn in case_fns:
                    if case_fn is None:
                        if default_value is None:
                            raise RDDLNotImplementedError(
                                'Switch case missing value and default is None.\n' +
                                print_stack_trace(expr))
                        evaluated_cases.append(default_value)
                    else:
                        val, key, err_case, params = case_fn(subs, params, key)
                        total_err |= err_case
                        evaluated_cases.append(val)
                stacked = torch.stack(evaluated_cases, dim=0)
                result = self._apply_control_switch(pred_val, stacked)
                return result, key, total_err, params

            return _fn

        raise RDDLNotImplementedError(
            f'Control operator {op} is not supported.\n' + print_stack_trace(expr))

    # ------------------------------------------------------------------
    # Random variables
    # ------------------------------------------------------------------

    def _torch_random(self, expr, init_params) -> CallableExpr:
        """Compile random variable AST nodes into Torch samplers."""
        _, name = expr.etype

        if name in {'Discrete', 'UnnormDiscrete'}:
            ordered_args = self.traced.cached_sim_info(expr)
            arg_fns = [self._torch(arg, init_params) for arg in ordered_args]
        elif name in {'Discrete(p)', 'UnnormDiscrete(p)'}:
            _, args = expr.args
            arg, = args
            arg_fns = [self._torch(arg, init_params)]
        else:
            arg_fns = [self._torch(arg, init_params) for arg in expr.args]
        # this flag is used to determine whether to bind relaxed sampling(e.g., normal) or exact sampling
        # operators for this random variable, which is needed for relaxed logic compilation
        relaxed_sampling_name = None
        if self.uses_relaxed_logic:
            if name in {'Discrete', 'UnnormDiscrete', 'Discrete(p)', 'UnnormDiscrete(p)'}:
                relaxed_sampling_name = 'Discrete'
            elif name in self.OPS['sampling']:
                relaxed_sampling_name = name

        if relaxed_sampling_name is not None:
            logic_sampler = self._bind_logic_operator(
                'sampling', expr.id, init_params, relaxed_sampling_name
            )

            def _fn(subs, params, key):
                values = []
                total_err = self.ERROR_CODES['NORMAL']
                for arg_fn in arg_fns:
                    val, key, err, params = arg_fn(subs, params, key)
                    values.append(self._coerce_logic_tensor(val, promote_real=True))
                    total_err |= err
                generator = self._ensure_generator(
                    key,
                    values[0].device if values else self.device,
                )

                if name in {'Discrete', 'UnnormDiscrete'}:
                    prob = torch.stack(values, dim=-1)
                    if name == 'UnnormDiscrete':
                        normalizer = torch.sum(prob, dim=-1, keepdim=True)
                        prob = prob / torch.clamp(normalizer, min=1e-12)
                    sample, params = logic_sampler(generator, prob, params)
                elif name in {'Discrete(p)', 'UnnormDiscrete(p)'}:
                    prob = values[0]
                    if name == 'UnnormDiscrete(p)':
                        normalizer = torch.sum(prob, dim=-1, keepdim=True)
                        prob = prob / torch.clamp(normalizer, min=1e-12)
                    sample, params = logic_sampler(generator, prob, params)
                elif name in {'Bernoulli', 'Poisson', 'Geometric'}:
                    sample, params = logic_sampler(generator, values[0], params)
                elif name in {'Binomial', 'NegativeBinomial'}:
                    sample, params = logic_sampler(generator, values[0], values[1], params)
                else:
                    raise RDDLNotImplementedError(
                        f'Relaxed random variable {name} is not supported.\n' +
                        print_stack_trace(expr))

                total_err |= self._distribution_param_error(name, values)
                return sample, generator, total_err, params

            return _fn

        def _fn(subs, params, key):
            values = []
            total_err = self.ERROR_CODES['NORMAL']
            for arg_fn in arg_fns:
                val, key, err, params = arg_fn(subs, params, key)
                values.append(self._ensure_tensor(val))
                total_err |= err
            device = values[0].device if values else self.device
            generator = self._ensure_generator(key, device)
            sample, sample_err = self._sample_random_variable(name, values, generator, expr)
            total_err |= sample_err
            return sample, generator, total_err, params

        return _fn

    def _sample_random_variable(self, name: str, values: List[torch.Tensor],
                                generator: torch.Generator, expr) -> Tuple[torch.Tensor, int]:
        error = self._distribution_param_error(name, values)

        if name == 'KronDelta': # checked
            sample = values[0].to(dtype=self.INT)

        elif name == 'DiracDelta': # checked
            sample = values[0].to(dtype=self.REAL)
        
        # reparameterization trick U(a, b) = a + (b - a) * U(0, 1)
        elif name == 'Uniform': # checked
            low, high = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(high.shape, generator=generator, device=high.device, dtype=self.REAL)
            sample = low + (high - low) * rand

        # reparameterization trick N(m, s^2) = m + s * N(0, 1)
        elif name == 'Normal': # checked
            # reparametrization trick to allow backprop through the sampling process, 
            # following the convention of mean and variance as parameters
            mean, var = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            std = torch.sqrt(var)
            eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=self.REAL)
            sample = mean + std * eps
            

         # reparameterization trick Exp(s) = s * Exp(1)
        elif name == 'Exponential': #checked
            scale = values[0].to(self.REAL)
            exp1 = torch.empty_like(scale).exponential_(1.0, generator=generator)  # Exp(rate=1)
            sample = scale * exp1

         # reparameterization trick W(s, r) = r * (-ln(1 - U(0, 1))) ** (1 / s)
        
        elif name == 'Weibull': #checked
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(scale.shape, generator=generator, device=scale.device, dtype=self.REAL)
            # log1p means lod(1+x)
            sample = scale * torch.pow(-torch.log1p(-torch.clamp(rand, max=1.0 - 1e-8)), 1.0 / shape)




        elif name == 'Gamma': # checked
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            gamma = torch._standard_gamma(shape, generator=generator).to(dtype=self.REAL)
            sample = scale * gamma
        
        
        
        elif name == 'Beta': # checked
            a, b = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            gamma_a = torch._standard_gamma(a, generator=generator).to(dtype=self.REAL)
            gamma_b = torch._standard_gamma(b, generator=generator).to(dtype=self.REAL)
            sample = gamma_a / torch.clamp(gamma_a + gamma_b, min=1e-12)
        

        # TBD
        
        elif name == 'Poisson':
            rate = values[0].to(self.REAL)
            safe_rate = torch.where(rate >= 0, rate, torch.zeros_like(rate))
            sample = torch.poisson(safe_rate, generator=generator).to(dtype=self.INT)

        elif name == 'Bernoulli':
            probs = values[0].to(self.REAL)
            rand = torch.rand(probs.shape, generator=generator, device=probs.device, dtype=self.REAL)
            sample = (rand <= probs).to(dtype=self.REAL)

        elif name == 'Binomial':
            trials, prob = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            safe_trials = torch.where(trials >= 0, trials, torch.zeros_like(trials))
            safe_prob = torch.where((prob >= 0) & (prob <= 1), prob, torch.zeros_like(prob))
            dist = torch.distributions.Binomial(total_count=safe_trials, probs=safe_prob)
            sample = dist.sample().to(dtype=self.INT)
        
        elif name == 'NegativeBinomial':
            trials, prob = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            safe_trials = torch.where(trials > 0, trials, torch.ones_like(trials))
            safe_prob = torch.where((prob >= 0) & (prob <= 1), prob, torch.zeros_like(prob))
            # keep pyRDDLGym_jax convention: failures before `trials` successes
            dist = torch.distributions.NegativeBinomial(total_count=safe_trials, probs=1.0 - safe_prob)
            sample = dist.sample().to(dtype=self.INT)
        
        elif name == 'Geometric':
            prob = values[0].to(self.REAL)
            safe_prob = torch.clamp(prob, min=1e-8, max=1.0 - 1e-8)
            rand = torch.rand(prob.shape, generator=generator, device=prob.device, dtype=self.REAL)
            sample = torch.floor(torch.log1p(-rand) / torch.log1p(-safe_prob)) + 1
            sample = sample.to(dtype=self.INT)
        
        
        elif name == 'Pareto':
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(scale.shape, generator=generator, device=scale.device, dtype=self.REAL)
            sample = scale * torch.pow(torch.clamp(1.0 - rand, min=1e-8), -1.0 / shape)
        
        
        elif name == 'Student':
            df = values[0].to(self.REAL)
            z = torch.randn(df.shape, generator=generator, device=df.device, dtype=self.REAL)
            chi2 = 2.0 * torch._standard_gamma(df * 0.5, generator=generator).to(dtype=self.REAL)
            sample = z / torch.sqrt(chi2 / df)
        
        elif name == 'Gumbel':
            mean, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(mean.shape, generator=generator, device=mean.device, dtype=self.REAL)
            rand = rand.clamp(1e-8, 1.0 - 1e-8)
            sample = mean - scale * torch.log(-torch.log(rand))
        
        
        elif name == 'Laplace':
            mean, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(mean.shape, generator=generator, device=mean.device, dtype=self.REAL)
            centered = rand - 0.5
            centered = centered.clamp(-0.5 + 1e-8, 0.5 - 1e-8)
            sample = mean - scale * torch.sign(centered) * torch.log1p(-2.0 * torch.abs(centered))
        
        
        elif name == 'Cauchy':
            mean, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(mean.shape, generator=generator, device=mean.device, dtype=self.REAL)
            centered = rand - 0.5
            centered = centered.clamp(-0.5 + 1e-8, 0.5 - 1e-8)
            sample = mean + scale * torch.tan(torch.pi * centered)
        
        
        elif name == 'Gompertz':
            shape, scale = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(scale.shape, generator=generator, device=scale.device, dtype=self.REAL)
            inner = 1.0 - torch.log1p(-torch.clamp(rand, max=1.0 - 1e-8)) / shape
            sample = torch.log(inner) / scale
        
        
        elif name == 'ChiSquare':
            df = values[0].to(self.REAL)
            sample = 2.0 * torch._standard_gamma(df * 0.5, generator=generator).to(dtype=self.REAL)
        
        
        elif name == 'Kumaraswamy':
            a, b = torch.broadcast_tensors(values[0].to(self.REAL), values[1].to(self.REAL))
            rand = torch.rand(a.shape, generator=generator, device=a.device, dtype=self.REAL)
            sample = torch.pow(1.0 - torch.pow(rand, 1.0 / b), 1.0 / a)
        
        
        elif name in {'Discrete', 'UnnormDiscrete'}:
            prob = torch.stack([val.to(self.REAL) for val in values], dim=-1)
            if name == 'UnnormDiscrete':
                normalizer = torch.sum(prob, dim=-1, keepdim=True)
                prob = prob / torch.clamp(normalizer, min=1e-12)
            sample = self._sample_discrete(prob, generator)
        
        
        elif name in {'Discrete(p)', 'UnnormDiscrete(p)'}:
            prob = values[0].to(self.REAL)
            if name == 'UnnormDiscrete(p)':
                normalizer = torch.sum(prob, dim=-1, keepdim=True)
                prob = prob / torch.clamp(normalizer, min=1e-12)
            sample = self._sample_discrete(prob, generator)
        
        else:
            raise RDDLNotImplementedError(
                f'Random variable {name} is not supported.\n' + print_stack_trace(expr))
        return sample, error

    def _sample_discrete(self, prob: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        prob = torch.clamp(prob, min=0.0).to(dtype=self.REAL)
        if prob.dim() == 0:
            return torch.zeros((), dtype=self.INT, device=prob.device)
        if prob.shape[-1] == 0:
            return torch.zeros(prob.shape[:-1], dtype=self.INT, device=prob.device)
        normalizer = torch.sum(prob, dim=-1, keepdim=True)
        uniform = torch.full_like(prob, 1.0 / float(prob.shape[-1]))
        prob = torch.where(normalizer > 0, prob / torch.clamp(normalizer, min=1e-12), uniform)
        flat = prob.reshape(-1, prob.shape[-1])
        sampled = torch.multinomial(flat, num_samples=1, replacement=True, generator=generator)
        sampled = sampled.squeeze(-1).reshape(prob.shape[:-1])
        return sampled.to(dtype=self.INT)

    # ------------------------------------------------------------------
    # Random vectors
    # ------------------------------------------------------------------

    def _torch_random_vector(self, expr, init_params) -> CallableExpr:
        _, name = expr.etype
        if name == 'MultivariateNormal':
            return self._torch_multivariate_normal(expr, init_params)
        if name == 'MultivariateStudent':
            return self._torch_multivariate_student(expr, init_params)
        if name == 'Dirichlet':
            return self._torch_dirichlet(expr, init_params)
        if name == 'Multinomial':
            return self._torch_multinomial(expr, init_params)
        raise RDDLNotImplementedError(
            f'Distribution {name} is not supported.\n' + print_stack_trace(expr))

    def _torch_multivariate_normal(self, expr, init_params) -> CallableExpr:
        _, args = expr.args
        mean, cov = args
        mean_fn = self._torch(mean, init_params)
        cov_fn = self._torch(cov, init_params)
        index, = self.traced.cached_sim_info(expr)

        def _fn(subs, params, key):
            mean_val, key, err1, params = mean_fn(subs, params, key)
            cov_val, key, err2, params = cov_fn(subs, params, key)
            mean_t = self._ensure_tensor(mean_val).to(self.REAL)
            cov_t = self._ensure_tensor(cov_val).to(self.REAL)
            generator = self._ensure_generator(key, mean_t.device)
            z = torch.randn(mean_t.shape + (1,), generator=generator,
                            device=mean_t.device, dtype=self.REAL)
            chol = torch.linalg.cholesky(cov_t)
            sample = torch.matmul(chol, z)[..., 0] + mean_t
            sample = torch.movedim(sample, source=-1, destination=index)
            return sample, generator, err1 | err2, params

        return _fn

    def _torch_multivariate_student(self, expr, init_params) -> CallableExpr:
        _, args = expr.args
        mean, cov, df = args
        mean_fn = self._torch(mean, init_params)
        cov_fn = self._torch(cov, init_params)
        df_fn = self._torch(df, init_params)
        index, = self.traced.cached_sim_info(expr)

        def _fn(subs, params, key):
            mean_val, key, err1, params = mean_fn(subs, params, key)
            cov_val, key, err2, params = cov_fn(subs, params, key)
            df_val, key, err3, params = df_fn(subs, params, key)
            mean_t = self._ensure_tensor(mean_val).to(self.REAL)
            cov_t = self._ensure_tensor(cov_val).to(self.REAL)
            df_t = self._ensure_tensor(df_val).to(self.REAL)
            err = err1 | err2 | err3 | self._error_if_invalid(
                df_t <= 0, 'INVALID_PARAM_MULTIVARIATE_STUDENT'
            )
            generator = self._ensure_generator(key, mean_t.device)
            df_expand = df_t[..., None, None].expand(mean_t.shape + (1,))
            z = torch.randn(mean_t.shape + (1,), generator=generator,
                            device=mean_t.device, dtype=self.REAL)
            chi2 = 2.0 * torch._standard_gamma(df_expand * 0.5, generator=generator).to(dtype=self.REAL)
            z = z / torch.sqrt(chi2 / df_expand)
            chol = torch.linalg.cholesky(cov_t)
            sample = torch.matmul(chol, z)[..., 0] + mean_t
            sample = torch.movedim(sample, source=-1, destination=index)
            return sample, generator, err, params

        return _fn

    def _torch_dirichlet(self, expr, init_params) -> CallableExpr:
        _, args = expr.args
        alpha, = args
        alpha_fn = self._torch(alpha, init_params)
        index, = self.traced.cached_sim_info(expr)

        def _fn(subs, params, key):
            alpha_val, key, err, params = alpha_fn(subs, params, key)
            alpha_t = self._ensure_tensor(alpha_val).to(self.REAL)
            generator = self._ensure_generator(key, alpha_t.device)
            gamma = torch._standard_gamma(alpha_t, generator=generator).to(dtype=self.REAL)
            sample = gamma / torch.clamp(torch.sum(gamma, dim=-1, keepdim=True), min=1e-12)
            sample = torch.movedim(sample, source=-1, destination=index)
            err |= self._error_if_invalid(alpha_t <= 0, 'INVALID_PARAM_DIRICHLET')
            return sample, generator, err, params

        return _fn

    def _torch_multinomial(self, expr, init_params) -> CallableExpr:
        _, args = expr.args
        trials, prob = args
        trials_fn = self._torch(trials, init_params)
        prob_fn = self._torch(prob, init_params)
        index, = self.traced.cached_sim_info(expr)

        def _fn(subs, params, key):
            trials_val, key, err1, params = trials_fn(subs, params, key)
            prob_val, key, err2, params = prob_fn(subs, params, key)
            total_count = self._ensure_tensor(trials_val).to(self.REAL)
            prob_t = self._ensure_tensor(prob_val).to(self.REAL)
            err = err1 | err2
            prob_sum = torch.sum(prob_t, dim=-1)
            invalid = (
                torch.any(prob_t < 0)
                | torch.logical_not(
                    torch.all(torch.isclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4, rtol=1e-4))
                )
                | torch.any(total_count < 0)
            )
            err |= self._error_if_invalid(invalid, 'INVALID_PARAM_MULTINOMIAL')
            safe_total_count = torch.where(total_count >= 0, total_count, torch.zeros_like(total_count))
            safe_prob = torch.clamp(prob_t, min=0.0)
            normalizer = torch.sum(safe_prob, dim=-1, keepdim=True)
            safe_prob = safe_prob / torch.clamp(normalizer, min=1e-12)
            generator = self._ensure_generator(key, safe_prob.device)
            try:
                sample = torch.distributions.Multinomial(total_count=safe_total_count, probs=safe_prob).sample()
            except Exception:
                flat_prob = safe_prob.reshape(-1, safe_prob.shape[-1])
                flat_count = safe_total_count.reshape(-1)
                rows = []
                for i in range(flat_prob.shape[0]):
                    count = max(0, int(round(float(flat_count[i].item()))))
                    if count == 0:
                        row = torch.zeros_like(flat_prob[i])
                    else:
                        idx = torch.multinomial(flat_prob[i], count, replacement=True,
                                                generator=generator)
                        row = torch.bincount(idx, minlength=flat_prob.shape[-1]).to(self.REAL)
                    rows.append(row)
                sample = torch.stack(rows, dim=0).reshape(safe_prob.shape)
            sample = sample.to(dtype=self.INT)
            sample = torch.movedim(sample, source=-1, destination=index)
            return sample, generator, err, params

        return _fn

    # ------------------------------------------------------------------
    # Matrix algebra
    # ------------------------------------------------------------------

    def _torch_matrix(self, expr, init_params) -> CallableExpr:
        _, op = expr.etype
        if op == 'det':
            return self._torch_matrix_det(expr, init_params)
        if op == 'inverse':
            return self._torch_matrix_inv(expr, init_params, pseudo=False)
        if op == 'pinverse':
            return self._torch_matrix_inv(expr, init_params, pseudo=True)
        if op == 'cholesky':
            return self._torch_matrix_cholesky(expr, init_params)
        raise RDDLNotImplementedError(
            f'Matrix operation {op} is not supported.\n' + print_stack_trace(expr))

    def _torch_matrix_det(self, expr, init_params) -> CallableExpr:
        *_, arg = expr.args
        arg_fn = self._torch(arg, init_params)

        def _fn(subs, params, key):
            sample_arg, key, error, params = arg_fn(subs, params, key)
            sample = torch.linalg.det(sample_arg.to(self.REAL))
            return sample, key, error, params

        return _fn

    def _torch_matrix_inv(self, expr, init_params, pseudo: bool) -> CallableExpr:
        _, arg = expr.args
        arg_fn = self._torch(arg, init_params)
        indices = self.traced.cached_sim_info(expr)
        op = torch.linalg.pinv if pseudo else torch.linalg.inv

        def _fn(subs, params, key):
            sample_arg, key, error, params = arg_fn(subs, params, key)
            sample = op(sample_arg.to(self.REAL))
            sample = torch.movedim(sample, source=(-2, -1), destination=indices)
            return sample, key, error, params

        return _fn

    def _torch_matrix_cholesky(self, expr, init_params) -> CallableExpr:
        _, arg = expr.args
        arg_fn = self._torch(arg, init_params)
        indices = self.traced.cached_sim_info(expr)

        def _fn(subs, params, key):
            sample_arg, key, error, params = arg_fn(subs, params, key)
            sample = torch.linalg.cholesky(sample_arg.to(self.REAL))
            sample = torch.movedim(sample, source=(-2, -1), destination=indices)
            return sample, key, error, params

        return _fn




    # ------------------------------------------------------------------
    # CPF / Reward compilation
    # ------------------------------------------------------------------

    def _compile_cpfs(self, init_params) -> Dict[str, CallableExpr]:
        """Topologically compile every CPF using the traced dependency order.

        Args:
            init_params: Compilation parameters shared across CPFs.

        Returns:
            Dict[str, CallableExpr]: CPF name to torch callable.

        Example:
            >>> cpfs = compiler._compile_cpfs({})
            >>> list(cpfs.keys())
        """
        # here we use the traced dependency order to ensure 
        # CPFs are compiled after their dependencies
        torch_cpfs = {}
        for level in sorted(self.levels.keys()):
            for cpf in self.levels[level]:
                _, expr = self.rddl.cpfs[cpf]
                prange = self.rddl.variable_ranges[cpf]
                dtype = self.TORCH_TYPES.get(prange, self.INT)
                torch_cpfs[cpf] = self._torch(expr, init_params, dtype=dtype)
        return torch_cpfs



    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    
    #. i dont think i need it because thr initial is in torch but lets keep it for now
    def _ensure_tensor(self, value: Any) -> torch.Tensor:
        """Convert arbitrary python/numpy values into Torch tensors.

        Args:
            value: Python / numpy / torch input.

        Returns:
            torch.Tensor: Tensor using compiler precision defaults.

        Example:
            >>> compiler._ensure_tensor(1.0)
            tensor(1.)
        """
        if isinstance(value, torch.Tensor):
            return value.to(
                dtype=self.REAL if value.dtype.is_floating_point else value.dtype,
                device=self.device,
            )
        if isinstance(value, np.ndarray):
            if value.dtype == np.object_:
                raise TypeError('Object arrays are not supported in torch compilation.')
            if np.issubdtype(value.dtype, np.bool_):
                dtype = torch.bool
            elif np.issubdtype(value.dtype, np.integer):
                dtype = self.INT
            elif np.issubdtype(value.dtype, np.floating):
                dtype = self.REAL
            else:
                dtype = self.REAL
            return as_tensor_on_device(value, dtype=dtype, device=self.device)
        if isinstance(value, (bool, np.bool_)):
            return torch.tensor(bool(value), dtype=torch.bool, device=self.device)
        if isinstance(value, (int, np.integer)):
            return torch.tensor(int(value), dtype=self.INT, device=self.device)
        if isinstance(value, (float, np.floating)):
            return torch.tensor(float(value), dtype=self.REAL, device=self.device)
        if isinstance(value, (list, tuple)):
            return as_tensor_on_device(value, dtype=self.REAL, device=self.device)
        try:
            return torch.tensor(value, device=self.device)
        except Exception:
            return value

    def _tensorize_structure(self, data: Any):
        """Recursively map python containers into tensors of matching shape.

        Args:
            data: Arbitrary nested structure.

        Returns:
            Same structure with tensors replacing scalars/arrays.

        Example:
            >>> compiler._tensorize_structure({'a': [1, 2]})
            {'a': tensor([1., 2.])}
        """
        if isinstance(data, dict):
            return {k: self._tensorize_structure(v) for (k, v) in data.items()}
        if isinstance(data, list):
            return [self._tensorize_structure(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._tensorize_structure(v) for v in data)
        return self._ensure_tensor(data)

    def _ensure_generator(self, key: Optional[torch.Generator], device: torch.device):
        """Return an RNG, creating a seeded generator tied to the tensor device.

        Args:
            key: Optional existing generator.
            device: Device hosting the tensors being sampled.

        Returns:
            torch.Generator: Torch RNG ready for sampling.

        Example:
            >>> gen = compiler._ensure_generator(None, torch.device('cpu'))
            >>> isinstance(gen, torch.Generator)
            True
        """
        if key is not None:
            requested_device = device if isinstance(device, torch.device) else resolve_torch_device(device)
            key_device = torch.device(str(key.device))
            if key_device.type != requested_device.type or (
                requested_device.index is not None and key_device.index not in {None, requested_device.index}
            ):
                raise ValueError(
                    f'Random generator device {key_device} does not match requested execution device '
                    f'{requested_device}.'
                )
            return key
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item()
        return make_generator(seed=int(seed), device=device)

class TorchRDDLCompilerWithGrad(TorchRDDLCompiler):
    """Gradient-aware compiler placeholder (shares implementation)."""
    pass
