# Architecture

## End-To-End Flow

The package builds on top of `pyRDDLGym` rather than replacing it.

```text
RDDL domain + instance
        |
        v
pyRDDLGym parser / lifted model
        |
        v
TorchRDDLCompiler
        |
        +--> compiled CPFs
        +--> compiled reward
        +--> compiled invariants / preconditions / terminations
        |
        v
TorchRDDLSimulator or TorchRollout
        |
        v
policy evaluation / optimization in Torch
```

## The Compiler Layer

`TorchRDDLCompiler` is the core of the repository.

Its job is to:

- initialize all pvariables as Torch values
- analyze CPF dependency order using `RDDLLevelAnalysis`
- trace object structure with `RDDLObjectsTracer`
- translate expressions into eager Torch callables

The compiler caches:

- `init_values`
- `cpfs`
- `reward`
- `invariants`
- `preconditions`
- `terminations`
- `model_params`

Those compiled objects are then reused by the simulator and rollout wrappers.

## Hidden State Representation

The simulator and rollout code keep the full RDDL hidden state in a dictionary
called `subs`.

Conceptually:

- hidden state: `subs`
- observation: selected projection of `subs`
- action: lifted action-fluent tensors
- next hidden state: updated `subs`

That design is especially explicit in `TorchRolloutCell`, which behaves a bit
like an RNN cell over RDDL dynamics.

## Exact vs Fuzzy Execution

The repository separates environment dynamics from logical semantics.

`ExactLogic()` provides:

- exact boolean operators
- exact comparisons
- exact reductions such as `all`, `any`, `argmax`, `argmin`

`FuzzyLogic()` swaps those pieces for soft relaxations:

- sigmoid-based comparisons
- soft rounding
- soft control flow
- relaxed sampling helpers

This is what makes it possible to optimize policies through a rollout in
`Runs.py`.

## Simulator vs Rollout

### `TorchRDDLSimulator`

Use the simulator when:

- you want explicit step-by-step control
- you care about exact environment execution
- you want optional action noise during stepping
- you want grounded observation dictionaries similar to `pyRDDLGym`

### `TorchRollout`

Use the rollout wrapper when:

- you want to unroll a full trajectory under a policy
- you want a structured `RolloutTrace`
- you want to keep everything in Torch for optimization

The rollout wrapper delegates single-step dynamics to `TorchRolloutCell`, which
internally uses the same compiler transition function as the simulator.

## Batch Training

`Train` now performs batch sampling over the rollout horizon.

With the current API:

- `batch_size` is the number of rollout steps per optimizer update
- the horizon is partitioned into contiguous batches of at most `batch_size`
- `batch_num` controls how many partitions are sampled per training iteration
- each sampled partition produces one policy update

This is implemented by:

- replaying a no-grad prefix when a sampled batch starts after step 0
- running `TorchRollout.forward(..., initial_subs=..., steps=..., start_step=...)`
  for the sampled batch
- updating the policy from that batch return only

Example:

- `horizon=113`, `batch_size=23`, `batch_num=5` -> `[23, 23, 23, 23, 21]`

## Action Handling

There is one important API difference between the two execution layers:

- `TorchRDDLSimulator` can accept either lifted or grounded actions because it
  can fall back to `pyRDDLGym` action preparation
- `TorchRollout` expects lifted action names and validates them against
  `noop_actions`

If you are building policies for rollout optimization, define them in terms of
lifted action fluents.
