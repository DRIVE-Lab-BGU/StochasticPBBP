# TorchRollout

# TorchRolloutCell

## Role

`TorchRolloutCell` executes one RDDL transition.

It receives the current simulator state and action, runs the compiled RDDL transition, and returns the next state, next observation, reward, termination status, and updated model parameters.

## Important Attributes

| Attribute | Meaning |
|---|---|
| `rddl` | The RDDL model |
| `compiler` | `TorchRDDLCompiler` used to compile the model |
| `step_fn` | Compiled function that executes one transition |
| `init_values` | Initial values of the RDDL variables |
| `model_params` | Parameters used by the compiled model |
| `observed_fluents` | Variables exposed to the policy |
| `noop_actions` | Default action values |

## Main Methods

| Method | What it does |
|---|---|
| `__init__()` | Compiles the RDDL model and creates `step_fn` |
| `reset()` | Creates the initial simulator state and observation |
| `observe()` | Extracts the variables visible to the policy |
| `prepare_actions()` | Completes and validates the action dictionary |
| `step()` | Executes one RDDL transition |

## One Transition Flow

```mermaid
flowchart TD
    Current["Current State (subs)"]
    Action["Noisy Action"]
    Prepare["prepare_actions()"]
    Transition["Compiled step_fn()"]
    CPFs["Evaluate CPFs"]
    Reward["Calculate Reward"]
    Next["Commit Next State"]
    Done["Check Termination"]
    Observation["Create Next Observation"]

    Current --> Prepare
    Action --> Prepare
    Prepare --> Transition
    Transition --> CPFs
    CPFs --> Reward
    Reward --> Next
    Next --> Done
    Done --> Observation