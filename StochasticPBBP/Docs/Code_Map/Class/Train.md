# Train

## Role

`Train` optimizes a policy using differentiable RDDL rollouts.

It runs the current policy through the environment model, collects rewards, converts the return into a loss, computes gradients, and updates the policy parameters using RMSprop.

## Important Attributes

| Attribute | Meaning |
|---|---|
| `policy` | The policy being trained |
| `rollout` | `TorchRollout` used to simulate the policy |
| `logic` | Logic backend used during the rollout |
| `default_additive_noise` | Noise applied to actions |
| `optimizer` | RMSprop optimizer used to update the policy |

## Main Methods

| Method | What it does |
|---|---|
| `__init__()` | Creates the rollout, training settings, and optimizer |
| `train_trajectory()` | Runs the training iterations |
| `_run_training_batch()` | Runs one rollout, computes loss, and updates the policy |
| `_advance_to_batch_start()` | Reconstructs the starting state for partial batches |
| `_build_partitions()` | Divides the horizon into batches |
| `_sample_partition_indices()` | Selects which batch to train on |

## Training Flow

```mermaid
flowchart TD
    Train["Train.train_trajectory()"]
    Rollout["TorchRollout.forward()"]
    Policy["Policy"]
    Noise["Additive Noise"]
    Model["RDDL Model"]
    Return["Sum Rewards"]
    Loss["Loss = -Return"]
    Backward["Backward"]
    Update["RMSprop Update"]

    Train --> Rollout
    Rollout --> Policy
    Policy --> Noise
    Noise --> Model
    Model --> Return
    Return --> Loss
    Loss --> Backward
    Backward --> Update