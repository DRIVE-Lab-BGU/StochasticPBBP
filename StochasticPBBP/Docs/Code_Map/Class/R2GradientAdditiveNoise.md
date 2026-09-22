# R2GradientAdditiveNoise

## Role

`R2GradientAdditiveNoise` creates a timestep-specific noise profile from gradients measured during the R2 analysis rollout.

It uses the sensitivity of the analysis return to actions in order to determine the sigma used in future update rollouts.

## Important Attributes

| Attribute | Meaning |
|---|---|
| `min_std` | Minimum allowed sigma |
| `max_std` | Maximum allowed sigma |
| `alpha` | Controls the sensitivity-to-sigma mapping |
| `eps` | Prevents division by zero during normalization |
| `normalization_quantile` | Quantile used to normalize sensitivity scores |
| `step_score_aggregate` | How action sensitivities are combined per timestep |
| `fallback_noise` | Noise used when no profile exists |
| `_std_profile` | Stored sigma profile by timestep and action |
| `last_gradients` | Most recent action gradients |
| `last_action_scores` | Sensitivity per action |
| `last_step_scores` | Sensitivity per timestep |

## Main Methods

| Method | What it does |
|---|---|
| `refresh_from_analysis_trace()` | Creates a new sigma profile from the analysis rollout |
| `_collect_action_gradients()` | Calculates gradient of analysis return with respect to actions |
| `_build_step_scores()` | Converts gradients into timestep sensitivities |
| `_sigma_from_step_score()` | Converts normalized sensitivity into sigma |
| `_build_std_profile()` | Creates sigma tensors for each timestep/action |
| `set_profile()` | Stores the new sigma profile |
| `_resolve_std_tensor()` | Finds sigma for the current timestep/action |
| `apply_to_value()` | Adds Gaussian noise scaled by sigma |

## Sigma Flow

```mermaid
flowchart LR
    Return["Analysis Return"]
    Gradient["Gradient dJ/da"]
    ActionSensitivity["Action Sensitivity"]
    StepSensitivity["Timestep Sensitivity"]
    Normalize["Normalize"]
    Sigma["Sigma Profile"]
    Next["Next Update Rollout"]

    Return --> Gradient
    Gradient --> ActionSensitivity
    ActionSensitivity --> StepSensitivity
    StepSensitivity --> Normalize
    Normalize --> Sigma
    Sigma --> Next