# Main Flow

This file describes the main runtime flow of the project, from the entry point to training, evaluation, and result generation.

---

## 1. Main System Flow

The program starts in `run.py`.

`run.py` builds the experiment configuration and creates an `ExperimentManager`.

The manager then creates the policy and trainer, runs the training process, evaluates the policy when required, and stores the experiment results.

```mermaid
flowchart TD
    Run["run.py"]
    Config["Build Experiment Configuration"]
    Manager["ExperimentManager"]
    Policy["Build Policy"]
    Trainer["Build Trainer"]
    Training["Training"]
    Evaluation["Evaluation"]
    Results["Aggregate Results"]
    Save["Log / Save Results"]

    Run --> Config --> Manager --> Policy --> Trainer --> Training --> Evaluation --> Results --> Save
```

---

## 2. Policy Selection

`ExperimentManager` creates one of two policy types:

- `NeuralStateFeedbackPolicy`
- `TO`

The selected policy determines how an action is generated at each timestep.

```mermaid
flowchart TD
    Manager["ExperimentManager"]
    Choice{"Policy Type?"}
    Neural["NeuralStateFeedbackPolicy"]
    TO["TO"]

    Manager --> Choice
    Choice -- neural --> Neural
    Choice -- to --> TO
```
---

## 3. Trainer Selection

`ExperimentManager` selects the trainer according to the configured noise type.

```mermaid
flowchart TD
    Manager["ExperimentManager"]
    NoiseType{"Noise Type?"}
    Train["Train"]
    R2["R2Trainer"]

    Manager --> NoiseType
    NoiseType -- "gradient2noise" --> R2
    NoiseType -- "other noise types" --> Train
```

### Trainer Behavior

```mermaid
flowchart LR
    Train["Train"]
    Standard["Standard Policy Update"]

    R2["R2Trainer"]
    Update["Policy Update"]
    Analysis["Analysis Pass"]
    Sigma["Sigma Profile"]

    Train --> Standard

    R2 --> Update --> Analysis --> Sigma
```

### Main Difference

- `Train` performs the standard differentiable policy-training process.
- `R2Trainer` is used for `gradient2noise` and adds an analysis phase that calculates a new sigma profile for future training iterations.

---

## 4. Standard Training Flow

The standard training path is used when the trainer is `Train`.

The policy is executed through a differentiable rollout, rewards are accumulated into a return, and the negative return is used as the loss for policy optimization.

```mermaid
flowchart TD
    Train["Train"]
    Rollout["TorchRollout"]
    Policy["Policy"]
    Prepare["Prepare Action"]
    Noise["Additive Noise"]
    Cell["TorchRolloutCell"]
    RDDL["RDDL Transition"]
    Reward["Reward + Next State"]
    Return["Rollout Return"]
    Loss["Loss = -Return"]
    Backward["Backpropagation"]
    Update["RMSprop Policy Update"]

    Train --> Rollout
    Rollout --> Policy
    Policy --> Prepare
    Prepare --> Noise
    Noise --> Cell
    Cell --> RDDL
    RDDL --> Reward
    Reward --> Rollout
    Rollout --> Return
    Return --> Loss
    Loss --> Backward
    Backward --> Update
```

### One Rollout Timestep

```mermaid
flowchart LR
    Observation["Current Observation"]
    Policy["Policy"]
    Action["Raw Action"]
    Prepare["Prepare Action"]
    Noise["Add Noise"]
    Transition["RDDL Transition"]
    Result["Reward + Next Observation"]

    Observation --> Policy --> Action --> Prepare --> Noise --> Transition --> Result
```

### Main Idea

- `Train` starts the optimization step.
- `TorchRollout` runs the policy over multiple timesteps.
- At each timestep, the policy creates an action.
- The action is prepared and optionally perturbed by additive noise.
- `TorchRolloutCell` executes one differentiable RDDL transition.
- Rewards from the rollout are combined into the return.
- The loss is the negative return.
- Backpropagation calculates gradients.
- RMSprop updates the policy parameters.

---

## 5. Noise Flow

The noise system receives a prepared policy action and optionally perturbs its floating-point values before the action enters the RDDL transition.

```mermaid
flowchart TD
    Action["Prepared Policy Action"]
    NoiseType{"Noise Type?"}

    NoNoise["NoAdditiveNoise"]
    Constant["ConstantAdditiveNoise"]
    Linear["LinearDecayAdditiveNoise"]
    R2["R2GradientAdditiveNoise"]

    Sigma["Select Sigma"]
    Sample["Sample Gaussian Noise"]
    Add["Action + Noise"]
    Cell["TorchRolloutCell"]

    Action --> NoiseType

    NoiseType -- "no noise" --> NoNoise --> Cell
    NoiseType -- "constant" --> Constant --> Sigma
    NoiseType -- "linear decay" --> Linear --> Sigma
    NoiseType -- "gradient2noise" --> R2 --> Sigma

    Sigma --> Sample --> Add --> Cell
```

### How Sigma Is Chosen

```mermaid
flowchart LR
    Constant["Constant Noise"]
    ConstantSigma["Fixed Sigma"]

    Linear["Linear Decay"]
    Iteration["Training Iteration"]
    LinearSigma["Scheduled Sigma"]

    R2["Gradient2Noise"]
    Sensitivity["Timestep Sensitivity"]
    R2Sigma["Timestep-Specific Sigma"]

    Constant --> ConstantSigma

    Linear --> Iteration --> LinearSigma

    R2 --> Sensitivity --> R2Sigma
```

### Main Idea

For floating-point actions, the general form is:

`noisy_action = action + GaussianNoise * sigma`

The difference between the noise types is mainly how `sigma` is selected:

- `NoAdditiveNoise` uses zero noise.
- `ConstantAdditiveNoise` uses a fixed sigma.
- `LinearDecayAdditiveNoise` changes sigma according to the training iteration.
- `R2GradientAdditiveNoise` uses a sigma profile calculated from action sensitivity.

---

## 6. R2 Training Flow

The R2 training path is used when the noise type is `gradient2noise`.

Each iteration contains:
1. A policy update rollout.
2. A post-update analysis rollout.
3. A sigma-profile refresh.

```mermaid
flowchart TD
    Start["Current Sigma Profile"]
    UpdateRollout["Update Rollout"]
    PolicyUpdate["Policy Update"]
    AnalysisRollout["Analysis Rollout"]
    AnalysisReturn["Analysis Return"]
    Gradient["d(Return) / d(Action)"]
    ActionSensitivity["Action Sensitivity"]
    StepSensitivity["Timestep Sensitivity"]
    Normalize["Normalize Sensitivity"]
    Sigma["New Sigma Profile"]
    Noise["R2GradientAdditiveNoise"]
    Next["Next Training Iteration"]

    Start --> UpdateRollout --> PolicyUpdate --> AnalysisRollout --> AnalysisReturn --> Gradient --> ActionSensitivity --> 
    StepSensitivity --> Normalize --> Sigma --> Noise --> Next --> UpdateRollout
```

### Update Phase

```mermaid
flowchart LR
    Policy["Policy"]
    Noise["Current R2 Noise"]
    Rollout["TorchRollout"]
    Return["Return"]
    Loss["Loss = -Return"]
    Backward["Backpropagation"]
    Update["RMSprop Update"]

    Policy --> Rollout
    Noise --> Rollout
    Rollout --> Return --> Loss --> Backward --> Update
```

### Analysis Phase

```mermaid
flowchart LR
    UpdatedPolicy["Updated Policy"]
    Analysis["Analysis Rollout with Zero Noise"]
    Return["Analysis Return"]
    Gradient["Gradient w.r.t. Actions"]
    Sensitivity["Sensitivity per Timestep"]
    Sigma["Sigma Profile"]

    UpdatedPolicy --> Analysis --> Return --> Gradient --> Sensitivity --> Sigma
```

### Main Idea

The update phase changes the policy.

The analysis phase does not update the policy. It measures how sensitive the analysis return is to the actions executed at each timestep.

The resulting sensitivity values are converted into a sigma profile.

That sigma profile is used during the next update iteration.

---

## 7. Exact Evaluation Flow

Exact evaluation measures the current policy directly inside the pyRDDLGym environment.

No policy optimization is performed during this phase.

```mermaid
flowchart TD
    Manager["ExperimentManager"]
    Evaluate["MBDPOPolicy.evaluate()"]
    Reset["Reset Policy + Environment"]
    Observation["Current Observation"]
    Policy["Policy.sample_action()"]
    Action["Environment-Compatible Action"]
    Env["pyRDDLGym env.step()"]
    Reward["Reward + Next Observation"]
    Done{"Done?"}
    Return["Episode Return"]
    Stats["Evaluation Statistics"]

    Manager --> Evaluate --> Reset --> Observation --> Policy --> Action --> Env--> Reward --> Done
    Done -- No --> Observation
    Done -- Yes --> Return --> Stats
```

### Main Idea

During exact evaluation:

- The policy is not updated.
- Actions are selected without gradient-based training.
- The differentiable `TorchRollout` is not used.
- The policy interacts directly with the pyRDDLGym environment.
- Training additive noise is not applied.
- Rewards are accumulated into an evaluation return.
- Statistics such as mean and standard deviation are returned to `ExperimentManager`.