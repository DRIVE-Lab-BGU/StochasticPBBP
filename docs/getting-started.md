# Getting Started

## Prerequisites

This repository includes a lightweight runtime `requirements.txt`, but it still
does not have full packaging metadata such as `pyproject.toml`.

Validated in this workspace with:

- Python `3.13.5`
- `torch==2.9.1`
- `pyRDDLGym==2.6`
- `pyRDDLGym-jax==2.6` for JAX comparison scripts

## Installation

Core runtime:

```bash
python -m pip install -r requirements.txt
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

Chunked horizon training example:

```bash
python StochasticPBBP/Runs.py --iterations 1 --horizon 113 --batch-size 5 --print-every 1
```

What this does:

- loads the default reservoir domain unless you pass `--domain` and `--instance`
- initializes `GaussianPolicy` from `core/Policies.py`
- trains through `Train` from `core/Train.py`
- if `batch_size > 1`, splits the horizon into nearly equal chunks
- updates parameters after each chunk
- resumes the next chunk from the previous chunk's final simulator state

Chunking example:

- `horizon=113`, `batch_size=5` -> `[23, 23, 23, 22, 22]`

Useful CLI arguments:

- `--domain`
- `--instance`
- `--horizon`
- `--batch-size`
- `--iterations`
- `--print-every`

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
- Use `Train` when you want chunked training with optimizer updates between
  horizon segments.
