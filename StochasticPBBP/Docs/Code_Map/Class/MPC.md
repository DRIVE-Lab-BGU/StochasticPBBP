# MPC

## Role

`MPC` is a Model Predictive Control wrapper around a `TO` planner.

At every control step, it re-optimizes the `TO` action sequence from the current observation, then executes only the first planned action.

This creates a receding-horizon control process:

current state
→ optimize a short action plan
→ execute first action
→ observe new state
→ optimize again

## Important Attributes

| Attribute | Meaning |
|---|---|
| `planner` | The `TO` policy whose action sequence is optimized |
| `trainer` | The `Train` object used to optimize the planner |
| `planning_steps` | Number of future steps considered during each MPC optimization |
| `optimization_iterations` | Number of optimization iterations performed before selecting an action |
| `additive_noise` | Noise used during the internal planning optimization |

## Main Methods

| Method | What it does |
|---|---|
| `__init__()` | Validates and stores the MPC components and configuration |
| `sample_action()` | Re-optimizes the TO planner from the current observation and returns the first planned action |
| `reset()` | Resets the underlying TO planner |

## MPC Flow

```mermaid
flowchart TD
    Observation["Current Observation"]
    Optimize["trainer.optimize_from_state()"]
    TO["TO Planner"]
    Plan["Optimized Action Sequence"]
    First["Select Action at Step 0"]
    Action["Return First Action"]
    Environment["Environment"]
    Next["New Observation"]

    Observation --> Optimize
    Optimize --> TO
    TO --> Plan
    Plan --> First
    First --> Action
    Action --> Environment
    Environment --> Next
    Next --> Observation