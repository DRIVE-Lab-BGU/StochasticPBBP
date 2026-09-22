# MBDPOPolicy

## Role

`MBDPOPolicy` provides the common evaluation interface used by both `NeuralStateFeedbackPolicy` and `TO`.

It evaluates a policy directly inside the pyRDDLGym environment and returns statistics about the policy's performance.

## Main Methods

| Method | What it does |
|---|---|
| `evaluate()` | Runs evaluation episodes in pyRDDLGym |
| `sample_action()` | Interface used by each policy to generate an action |
| `reset()` | Resets policy state before an episode |
| `_normalize_external_observation()` | Converts environment observations to policy-compatible format |
| `_action_to_env_dict()` | Converts tensor actions to values accepted by pyRDDLGym |

## Evaluation Flow

```mermaid
flowchart TD
    Reset["Reset Policy + Environment"]
    Obs["Observation"]
    Policy["Policy.sample_action()"]
    Action["Action"]
    Env["pyRDDLGym env.step()"]
    Reward["Reward + Next Observation"]
    Done{"Done?"}
    Return["Evaluation Return"]

    Reset --> Obs
    Obs --> Policy
    Policy --> Action
    Action --> Env
    Env --> Reward
    Reward --> Done
    Done -- No --> Obs
    Done -- Yes --> Return