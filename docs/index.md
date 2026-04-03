# StochasticPBBP

StochasticPBBP is a Torch-native toolkit for working with RDDL planning models.
It sits on top of `pyRDDLGym` and focuses on one specific goal: make RDDL
dynamics available as PyTorch computations that can be simulated, rolled out,
and optimized with gradients.

## Why This Project Exists

`pyRDDLGym` already parses RDDL and builds lifted planning models. This
repository adds a PyTorch backend around that model so you can:

- compile RDDL expressions into eager Torch callables
- simulate one environment step at a time
- unroll full trajectories in Torch
- switch between exact logic and fuzzy relaxations
- train simple policies with gradient-based optimization

## Main Components

| Component | Location | Purpose |
| --- | --- | --- |
| Torch compiler | `StochasticPBBP/core/Compiler.py` | Compiles `RDDLLiftedModel` expressions into PyTorch callables |
| Simulator | `StochasticPBBP/core/Simulator.py` | Executes one compiled transition at a time |
| Rollout wrapper | `StochasticPBBP/core/Rollout.py` | Unrolls a policy over a horizon and returns a `RolloutTrace` |
| Logic backends | `StochasticPBBP/core/Logic.py` | Provides exact and fuzzy semantics for logical/comparison operators |
| Policy and trainer | `StochasticPBBP/core/Policies.py`, `StochasticPBBP/core/Train.py` | Defines `GaussianPolicy` and horizon batch-sampling training |
| Training entrypoint | `StochasticPBBP/Runs.py` | Wires the policy, trainer, CLI arguments, and example domain together |

## Example Domains Included

The repository ships with a few RDDL domains under `StochasticPBBP/problems/`:

- `reservoir`
- `race_car`
- `hvac`

The current runnable training example uses the `reservoir` domain by default.

## Typical Workflow

1. Load a domain with `pyRDDLGym.make(..., vectorized=True)`.
2. Extract `env.model`.
3. Choose one of the Torch wrappers:
   - `TorchRDDLCompiler` if you want raw compiled callables
   - `TorchRDDLSimulator` if you want exact single-step execution
   - `TorchRollout` if you want multi-step rollouts inside Torch
4. Choose `ExactLogic()` for faithful discrete execution or `FuzzyLogic()` for
   soft relaxations.
5. Plug in a policy and optimize it if needed.
6. Optionally partition the horizon into batches and sample training updates from those partitions.

## Verified Entry Points

The following commands were checked in this workspace:

```bash
python StochasticPBBP/Runs.py --iterations 1 --print-every 1
python StochasticPBBP/Runs.py --iterations 1 --horizon 113 --batch-size 5 --print-every 1
python StochasticPBBP/tests/simulator_test.py
```

## Next Reading

- [Getting started](getting-started.md)
- [API reference](api.md)
- [Architecture](architecture.md)
- [Development notes](development.md)
