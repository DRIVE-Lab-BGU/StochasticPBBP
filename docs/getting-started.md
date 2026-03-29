# Getting Started

## Prerequisites

This repository does not yet pin runtime dependencies in a project-level
`requirements.txt` or `pyproject.toml`, so install the main dependencies
manually.

Validated in this workspace with:

- Python `3.13.5`
- `torch==2.9.1`
- `pyRDDLGym==2.6`
- `pyRDDLGym-jax==2.6` for JAX comparison scripts

## Installation

Core runtime:

```bash
python -m pip install torch pyRDDLGym
```

Optional JAX stack for comparison scripts:

```bash
python -m pip install jax pyRDDLGym-jax
```

If Matplotlib complains about an unwritable cache directory during imports, set
`MPLCONFIGDIR` to a writable path before running commands.

## Repository Layout

```text
StochasticPBBP/
├── core/
│   ├── Compiler.py
│   ├── Initializer.py
│   ├── Logic.py
│   ├── Policies.py
│   ├── Rollout.py
│   ├── Simulator.py
│   └── Train.py
├── problems/
│   ├── hvac/
│   ├── race_car/
│   └── reservoir/
├── tests/
└── Runs.py
```

## First Successful Run

Training example:

```bash
python StochasticPBBP/Runs.py --iterations 5 --print-every 1
```

What this does:

- loads the default reservoir domain
- builds a `TorchRollout` with `FuzzyLogic()`
- initializes a state-independent Gaussian policy
- performs gradient ascent on cumulative return

Important limitation:

- `Runs.py` parses `--domain` and `--instance`, but `main()` currently ignores
  those arguments and hardcodes the reservoir example paths

Simulator smoke test:

```bash
python StochasticPBBP/tests/simulator_test.py
```

This is the cleanest script in `tests/` for confirming that the Torch simulator
can reset and step through a problem instance.

## First Python Session

The simplest way to use the project from code is to build a `pyRDDLGym`
environment and pass the lifted model into one of the Torch wrappers.

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
observation, done = sim.reset()
observation, reward, done = sim.step(sim.noop_actions, num_step=1)

print(observation)
print(float(reward), done)
```

## When To Use Which Entry Point

- Use `TorchRDDLSimulator` when you want exact step-by-step execution.
- Use `TorchRollout` when you want to evaluate or optimize a policy over a full
  horizon.
- Use `TorchRDDLCompiler` when you need direct access to compiled CPFs and the
  transition function.
