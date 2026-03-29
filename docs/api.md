# API Reference

## `TorchRDDLCompiler`

Location: `StochasticPBBP/core/Compiler.py`

Purpose:

- accepts a `pyRDDLGym.core.compiler.model.RDDLLiftedModel`
- compiles CPFs, reward, invariants, preconditions, and terminal conditions into
  eager PyTorch callables
- stores initialized Torch values for all pvariables

Main constructor shape:

```python
TorchRDDLCompiler(
    rddl,
    logger=None,
    python_functions=None,
    use64bit=False,
    logic=None,
    fuzzy_logic=None,
)
```

Key methods:

- `compile()`: analyzes dependencies and builds compiled callables
- `compile_transition()`: returns the step function used by the simulator and
  rollout code

Important attributes after `compile()`:

- `init_values`: initial Torch tensors for state, action, and non-fluent values
- `cpfs`: compiled CPFs in topological order
- `reward`: compiled reward callable
- `preconditions`, `invariants`, `terminations`
- `model_params`: mutable parameters used by some relaxed operators

## `TorchRDDLSimulator`

Location: `StochasticPBBP/core/Simulator.py`

Purpose:

- exact, stateful step-by-step execution on top of the compiled transition
- optional action noise injection
- compatible with Torch tensors while still supporting grounded outputs

Typical usage:

```python
from pathlib import Path

import pyRDDLGym

from StochasticPBBP.core.Logic import ExactLogic
from StochasticPBBP.core.Simulator import TorchRDDLSimulator

root = Path("StochasticPBBP/problems/reservoir")
env = pyRDDLGym.make(
    domain=root / "domain.rddl",
    instance=root / "instance_1.rddl",
    vectorized=True,
)

sim = TorchRDDLSimulator(
    env.model,
    logic=ExactLogic(),
    keep_tensors=True,
    noise={"type": "constant", "value": 0},
)

sim.seed(0)
obs, done = sim.reset()
obs, reward, done = sim.step(sim.noop_actions, num_step=1)
```

Important behavior:

- `reset()` restores the compiler's initial values
- `step(actions, num_step=0)` executes one transition and returns
  `(observation, reward, done)`
- with `keep_tensors=True`, returned observations stay as Torch tensors
- if you pass lifted action names, they are used directly
- if you pass grounded action names, the simulator falls back to
  `pyRDDLGym` action preparation

Noise modes currently implemented:

- `{"type": "constant", "value": ...}`
- `{"type": "smaller_1", "value": [start, end]}`
- `{"type": "smaller_2", "value": [start, end]}`

## `TorchRollout`

Location: `StochasticPBBP/core/Rollout.py`

Purpose:

- wraps the compiled transition as a multi-step rollout inside Torch
- keeps the hidden simulator state in `subs`
- returns a structured trajectory object

Typical usage:

```python
from pathlib import Path

import pyRDDLGym

from StochasticPBBP.core.Logic import ExactLogic
from StochasticPBBP.core.Rollout import TorchRollout

root = Path("StochasticPBBP/problems/reservoir")
env = pyRDDLGym.make(
    domain=root / "domain.rddl",
    instance=root / "instance_1.rddl",
    vectorized=True,
)

rollout = TorchRollout(env.model, horizon=2, logic=ExactLogic())
rollout.cell.key.manual_seed(0)


def policy(observation, step):
    del observation, step
    return rollout.noop_actions


trace = rollout(policy=policy)
print(trace.final_observation)
print(float(trace.return_))
```

Key methods:

- `reset(initial_state=None, initial_subs=None, model_params=None)`
- `step(subs, actions=None, model_params=None)`
- `forward(policy, initial_state=None, initial_subs=None, model_params=None,
  policy_state=None)`

Policy signatures supported by `forward()`:

- `policy(observation)`
- `policy(observation, step)`
- `policy(observation, step, policy_state)`

The policy may return either:

- `actions`
- `(actions, next_policy_state)`

Important constraint:

- rollout actions must use lifted action fluent names, not grounded names

## `RolloutTrace`

Location: `StochasticPBBP/core/Rollout.py`

Fields captured during a rollout:

- `observations`
- `actions`
- `rewards`
- `terminals`
- `final_observation`
- `final_subs`
- `policy_state`
- `model_params`

Derived property:

- `return_`: sum of all reward tensors in the trajectory

## Logic Backends

Location: `StochasticPBBP/core/Logic.py`

### `ExactLogic`

Use this when you want faithful discrete semantics:

- exact boolean logic
- exact comparisons
- exact sampling primitives where applicable
- best fit for simulation and debugging

### `FuzzyLogic`

Use this when you want differentiable relaxations:

- soft comparisons through sigmoid-style approximations
- soft rounding and control flow
- relaxed sampling helpers for non-reparameterizable choices
- best fit for gradient-based optimization through rollouts

`Runs.py` uses `FuzzyLogic()` in its training loop for exactly this reason.

## Policies And Training

Two policy-related entry points appear in the repository:

- `StochasticPBBP/core/Policies.py`: simple random / neural policy helpers
- `StochasticPBBP/Runs.py`: a runnable `GaussianPolicy` and `Train` example

The most complete training example today is the one in `Runs.py`:

- `GaussianPolicy`: state-independent diagonal Gaussian over the action vector
- `Train`: wraps a `TorchRollout` and optimizes policy parameters with
  `torch.optim.RMSprop`

That script is the best reference if you want to add your own differentiable
policy.
