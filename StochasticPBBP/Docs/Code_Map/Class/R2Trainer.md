# R2Trainer

## Role

`R2Trainer` is the special trainer used when the noise type is `gradient2noise`.

Unlike standard `Train`, each iteration contains:
1. A policy update rollout.
2. A separate analysis rollout.
3. A sigma-profile update based on action sensitivity.

## Important Attributes

| Attribute | Meaning |
|---|---|
| `policy` | Policy being trained |
| `rollout` | `TorchRollout` used for both update and analysis |
| `optimizer` | RMSprop optimizer |
| `default_additive_noise` | R2 noise used during the update rollout |
| `analysis_additive_noise` | Noise used during analysis, normally zero |
| `r2_profile` | Most recent sigma profile |
| `current_phase` | Current R2 phase: update, analysis, or profile refresh |

## Main Methods

| Method | What it does |
|---|---|
| `train_trajectory()` | Repeats R2 training iterations |
| `train_iteration()` | Runs update, analysis, and sigma refresh |
| `_run_update_phase()` | Updates the policy |
| `_run_analysis_phase()` | Runs the updated policy for sensitivity analysis |
| `refresh_noise_profile()` | Creates a new sigma profile |
| `_profile_to_sigma_vector()` | Converts the profile to one sigma value per timestep |

## One Iteration Flow

```mermaid
flowchart TD
    Update["Update Rollout"]
    Loss["Loss = -Return"]
    Policy["Update Policy"]
    Analysis["Analysis Rollout"]
    Gradient["Gradient of Return w.r.t. Actions"]
    Sensitivity["Sensitivity per Timestep"]
    Sigma["New Sigma Profile"]
    Next["Used in Next Iteration"]

    Update --> Loss
    Loss --> Policy
    Policy --> Analysis
    Analysis --> Gradient
    Gradient --> Sensitivity
    Sensitivity --> Sigma
    Sigma --> Next