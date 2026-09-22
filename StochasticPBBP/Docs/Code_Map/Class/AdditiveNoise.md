# AdditiveNoise

## Role

The additive-noise system perturbs floating-point policy actions before they are passed to the RDDL model.

It changes the action used during the rollout, but does not directly modify the policy itself.

## Main Classes

| Class | What it does |
|---|---|
| `NoiseContext` | Stores information about the current timestep and training iteration |
| `AdditiveNoise` | Base class that applies noise to action values |
| `NoAdditiveNoise` | Adds zero noise |
| `ConstantAdditiveNoise` | Adds Gaussian noise with a fixed sigma |
| `LinearDecayAdditiveNoise` | Changes sigma linearly during training |
| `AdditiveNoiseFactory` | Creates the correct noise implementation |

## Important Methods

| Method | What it does |
|---|---|
| `AdditiveNoise.__call__()` | Applies the noise process to the action dictionary |
| `apply_to_value()` | Adds noise to one floating-point action tensor |
| `sample_like()` | Generates a noise tensor with the same shape as the action |
| `AdditiveNoiseFactory.create()` | Selects which noise class to create |

## Noise Flow

```mermaid
flowchart LR
    Action["Prepared Policy Action"]
    Noise["AdditiveNoise"]
    Sample["Sample Noise"]
    Add["Action + Noise"]
    Noisy["Noisy Action"]
    Cell["TorchRolloutCell.step()"]

    Action --> Noise
    Noise --> Sample
    Sample --> Add
    Add --> Noisy
    Noisy --> Cell