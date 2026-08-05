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
- `StochasticPBBP/deprecated/Simulator.py`: deprecated exact single-step simulator on top of the
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

Partitioned horizon training example:

```bash
python StochasticPBBP/Runs.py --iterations 1 --horizon 113 --batch-size 23 --batch-num 5 --print-every 1
```

If you do not pass explicit paths, `Runs.py` uses the default reservoir domain:

- `StochasticPBBP/problems/reservoir/domain.rddl`
- `StochasticPBBP/problems/reservoir/instance_1.rddl`

The CLI now accepts:

- `--domain`
- `--instance`
- `--horizon`
- `--batch-size`

`batch_size` now means the number of horizon steps used for one gradient
update. The horizon is partitioned into contiguous batches of at most
`batch_size` steps, and `batch_num` controls how many of those partitions are
sampled per training iteration.

Example:

- `horizon=113`, `batch_size=113`, `batch_num=1` -> one full-horizon batch
- `horizon=113`, `batch_size=23`, `batch_num=5` -> partitions `[23, 23, 23, 23, 21]`

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

## Direct-Action Trajectory Optimization

`TO` is an open-loop policy whose trainable parameters are the action values
for every timestep. The existing `Train` loop optimizes this sequence with
RMSProp:

```python
from StochasticPBBP.core.Logic import FuzzyLogic
from StochasticPBBP.core.Train import Train
from StochasticPBBP.utils.Policies import TO

horizon = 50
template_rollout = TorchRollout(env.model, horizon=horizon)
policy = TO(
    action_template=template_rollout.noop_actions,
    horizon=horizon,
)
trainer = Train(
    model=env.model,
    policy=policy,
    horizon=horizon,
    lr=0.01,
    logic=FuzzyLogic(),
    batch_size=horizon,
    batch_num=1,
    seed=42,
)
history, policy = trainer.train_trajectory(iterations=100)
optimized_actions = policy.action_sequence()
```

The policy returns raw action parameters: it does not apply Gym bounds,
sigmoid/softplus transforms, or projection. A compatible RDDL domain must clip
actions in its CPFs when clipping is required. For example, the reservoir
domain maps `release` to the effective `released_water` with
`max[0, min[rlevel, release]]`. Action preconditions alone are only evaluated
and logged by the current Torch transition; they do not clip or reject an
action.

Use `batch_size == horizon` and `batch_num == 1` for TO. These settings are
validated by `Train`.

The experiment CLI exposes the policy with:

```bash
python StochasticPBBP/run.py \
  --policy to \
  --domain reservoir \
  --instance 1 \
  --horizon 50 \
  --iterations 100 \
  --noisestd 0
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
