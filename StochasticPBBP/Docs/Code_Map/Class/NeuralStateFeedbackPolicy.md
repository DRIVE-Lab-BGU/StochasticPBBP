# NeuralStateFeedbackPolicy

## Role

`NeuralStateFeedbackPolicy` is a trainable neural policy that converts the current RDDL observation into a dictionary of continuous actions.

The same neural network is used at every timestep, so the chosen action depends on the current observation.

## Important Attributes

| Attribute | Meaning |
|---|---|
| `observation_specs` | Defines the observation names, shapes, and order |
| `action_specs` | Defines the action names, shapes, and order |
| `network` | Feed-forward neural network |
| `dtype` | Network input type (`float32`) |
| `g` | Random generator used for deterministic initialization |

## Main Methods

| Method | What it does |
|---|---|
| `__init__()` | Builds the observation/action mappings and neural network |
| `forward()` | Converts an observation into an action dictionary |
| `_flatten_observation()` | Flattens and concatenates observation values |
| `_pack_actions()` | Converts the network output back into RDDL actions |
| `sample_action()` | Calls the policy, without gradients during evaluation |

## Policy Flow

```mermaid
flowchart LR
    Observation["Observation Dictionary"]
    Flatten["Flatten + Convert to float32"]
    Network["Neural Network"]
    FlatAction["Flat Action Vector"]
    Pack["Split + Restore Shapes"]
    Actions["Action Dictionary"]

    Observation --> Flatten
    Flatten --> Network
    Network --> FlatAction
    FlatAction --> Pack
    Pack --> Actions