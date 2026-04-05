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

## `TorchRDDLSimulator` (Deprecated)

Location: `StochasticPBBP/deprecated/Simulator.py`

Purpose:

- exact, stateful step-by-step execution on top of the compiled transition
- optional action noise injection
- compatible with Torch tensors while still supporting grounded outputs

Typical usage:

```python
from pathlib import Path

import pyRDDLGym

from StochasticPBBP.core.Logic import ExactLogic
from StochasticPBBP.deprecated.Simulator import TorchRDDLSimulator

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
  policy_state=None, steps=None, start_step=0)`

Policy signatures supported by `forward()`:

- `policy(observation)`
- `policy(observation, step)`
- `policy(observation, step, policy_state)`

The policy may return either:

- `actions`
- `(actions, next_policy_state)`

Important constraint:

- rollout actions must use lifted action fluent names, not grounded names
- if `initial_subs` is provided, the rollout starts from that hidden state
- `steps` lets you run only part of the horizon
- `start_step` lets the policy see global step numbers even when the horizon is
  split into chunks

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

The training code is now split across three files:

- `StochasticPBBP/core/Policies.py`: simple random / neural helpers plus
  `GaussianPolicy`
- `StochasticPBBP/core/Train.py`: horizon batch sampling training loop
- `StochasticPBBP/Runs.py`: runnable CLI entrypoint

### `GaussianPolicy`

`GaussianPolicy` is a state-independent diagonal Gaussian over the lifted action
vector. It now lives in `core/Policies.py`.

### `Train`

`Train` wraps a `TorchRollout` and optimizes policy parameters with
`torch.optim.RMSprop`.

Important batching behavior:

- `batch_size` is the number of rollout steps used for one optimizer update
- the horizon is partitioned into contiguous batches of at most `batch_size`
- `batch_num` is the number of partitions sampled per training iteration
- each sampled batch produces one optimizer step
- if a sampled batch starts after step 0, the trainer replays the prefix to
  recover the simulator state at that partition start

Example:

- `horizon=113`, `batch_size=113`, `batch_num=1` -> one full-horizon batch
- `horizon=113`, `batch_size=23`, `batch_num=5` -> partition sizes `[23, 23, 23, 23, 21]`

Typical usage:

```python
from pathlib import Path

import pyRDDLGym

from StochasticPBBP.utils.Policies import GaussianPolicy
from StochasticPBBP.deprecated.Simulator import TorchRDDLSimulator
from StochasticPBBP.core.Train import Train

root = Path("StochasticPBBP/problems/reservoir")
env = pyRDDLGym.make(
    domain=root / "domain.rddl",
    instance=root / "instance_1.rddl",
    vectorized=True,
)

simulator = TorchRDDLSimulator(env.model)
policy = GaussianPolicy(action_template=simulator.noop_actions)
trainer = Train(
    model=env.model,
    policy=policy,
    horizon=113,
    batch_size=23,
    batch_num=5,
)

history = trainer.train_trajectory(iterations=1, print_every=1)
```
