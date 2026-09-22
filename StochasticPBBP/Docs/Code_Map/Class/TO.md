# TO

## Role

`TO` is an open-loop trajectory-optimization policy.

It stores a trainable action sequence for the entire horizon and selects the action for the current timestep without using the current observation.

## Important Attributes

| Attribute | Meaning |
|---|---|
| `requires_full_horizon` | Requires full-horizon training |
| `horizon` | Number of planned timesteps |
| `_cursor` | Internal timestep when no explicit step is given |
| `action_specs` | Defines action names, shapes, and types |
| `action_parameters` | Trainable action values for every timestep |

## Main Methods

| Method | What it does |
|---|---|
| `__init__()` | Creates the trainable action trajectory |
| `forward()` | Returns the action values for the current timestep |
| `_validate_step()` | Checks that the timestep is valid |
| `sample_action()` | Selects an action using the timestep |
| `reset()` | Resets the internal timestep |
| `action_sequence()` | Returns the complete planned trajectory |

## Policy Flow

```mermaid
flowchart LR
    Step["Current Timestep"]
    TO["TO.forward()"]
    Parameters["Trainable Action Sequence"]
    Select["Select Action at Step"]
    Action["Action Dictionary"]

    Step --> TO
    Parameters --> TO
    TO --> Select
    Select --> Action