# StochasticPBBP

StochasticPBBP is a Torch-native backend for RDDL planning models. It compiles
`pyRDDLGym` lifted models into eager PyTorch callables, exposes a single-step
simulator, supports differentiable multi-step rollouts, and includes a simple
training loop for gradient-based policy optimization.

The repository is currently best understood as research code and experimental
infrastructure for:

- compiling RDDL dynamics into Torch
- simulating stochastic planning domains
- relaxing discrete logic with fuzzy operators so gradients can flow through a
  rollout

## What Is In The Repository

- `StochasticPBBP/core/Compiler.py`: builds Torch callables from a
  `pyRDDLGym` `RDDLLiftedModel`
- `StochasticPBBP/core/Simulator.py`: exact single-step simulator on top of the
  compiled transition function
- `StochasticPBBP/core/Rollout.py`: differentiable rollout wrapper that returns
  a `RolloutTrace`
- `StochasticPBBP/core/Logic.py`: exact and fuzzy logic backends used during
  compilation
- `StochasticPBBP/Runs.py`: runnable training example with a Gaussian policy
- `StochasticPBBP/problems/`: example RDDL domains (`reservoir`, `race_car`,
  `hvac`)

## Quick Start

Install the core runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional comparison / experimentation dependencies:

```bash
python -m pip install jax pyRDDLGym-jax
```

Verified locally in this workspace with:

- Python `3.13.5`
- `torch==2.9.1`
- `pyRDDLGym==2.6`
- `pyRDDLGym-jax==2.6`

Run the built-in training example:

```bash
python StochasticPBBP/Runs.py --iterations 5 --print-every 1
```

Chunked horizon training example:

```bash
python StochasticPBBP/Runs.py --iterations 1 --horizon 113 --batch-size 5 --print-every 1
```

If you do not pass explicit paths, `Runs.py` uses the default reservoir domain:

- `StochasticPBBP/problems/reservoir/domain.rddl`
- `StochasticPBBP/problems/reservoir/instance_1.rddl`

The CLI now accepts:

- `--domain`
- `--instance`
- `--horizon`
- `--batch-size`

When `batch_size > 1`, the trainer splits the horizon into nearly equal chunks,
updates the policy after each chunk, and continues from the previous chunk's
final `subs` / `model_params` / `policy_state`.

Example:

- `horizon=113`, `batch_size=5` -> chunk sizes `[23, 23, 23, 22, 22]`

## Python Example

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


def noop_policy(observation, step):
    del observation, step
    return rollout.noop_actions


trace = rollout(policy=noop_policy)
print(float(trace.return_))
```

## Documentation

This repository now includes a small Read the Docs / MkDocs documentation
scaffold:

- [Docs home](docs/index.md)
- [Getting started](docs/getting-started.md)
- [API reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Development notes](docs/development.md)

To preview the docs locally:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

## Current Status

The core ideas are implemented and usable, but the project is still rough
around the edges:

- there is a lightweight `requirements.txt`, but no full packaging metadata yet
- there is no packaging metadata (`pyproject.toml` / `setup.py`) yet
- some files under `StochasticPBBP/tests/` are exploratory scripts rather than a
  polished automated test suite

## License

MIT, see [LICENSE](LICENSE).
