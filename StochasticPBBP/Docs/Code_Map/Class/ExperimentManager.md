# ExperimentManager

## Role

`ExperimentManager` coordinates the experiment setup, policy and trainer selection, training, evaluation, aggregation of results, logging, and sigma artifact generation.

## Important Attributes

| Attribute | Meaning |
|---|---|
| `env` | pyRDDLGym environment |
| `horizon` | Number of steps in each rollout |
| `policy_type` | Selects Neural policy or TO policy |
| `arch` | Neural-network architecture |
| `lr` | Learning rate |
| `noise` | Noise configuration |
| `train_seeder` | Generates training seeds |
| `eval_seeder` | Generates evaluation seeds |
| `template_rollout` | Rollout used to obtain observation/action templates |
| `logic` | Fuzzy logic used during differentiable training |

## Main Methods

| Method | What it does |
|---|---|
| `__init__()` | Creates the environment and experiment configuration |
| `run_experiment()` | Runs all configured experiments and aggregates the results |
| `_run_single_experiment()` | Builds a policy and trainer and performs one experiment |
| `_build_policy()` | Creates `NeuralStateFeedbackPolicy` or `TO` |
| `_build_trainer()` | Creates `Train` or `R2Trainer` |
| `_build_additive_noise()` | Creates the configured noise object |
| `log()` | Saves experiment results |
| `save_sigma_artifacts()` | Saves sigma-related results and plots |

## Main Flow

```mermaid
flowchart TD
    Manager["ExperimentManager"]
    Policy["Build Policy"]
    Trainer["Build Trainer"]
    Training["Train Policy"]
    Evaluation["Evaluate Policy"]
    Results["Aggregate Results"]

    Manager --> Policy
    Policy --> Trainer
    Trainer --> Training
    Training --> Evaluation
    Evaluation --> Results