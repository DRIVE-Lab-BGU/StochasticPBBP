from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import pyRDDLGym

import torch
from core.Train import Train
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Policies import NeuralStateFeedbackPolicy
from StochasticPBBP.utils.helper import collapse_history_to_iterations
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory, NoiseInfo
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow



class ExperimentManager:
    def __init__(self,domain: str,
                 instance: str,
                 seed: int=42,
                 horizon: int=100,
                 arch: Tuple[int, ...]=(12, 12),
                 fuzzy_weight: float=50.0,
                 learning_rate: float=0.01,
                 noise: Optional[NoiseInfo]=None) -> None:
        self.env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
        self.horizon = horizon
        self.arch = arch
        self.lr = learning_rate
        self.seed = seed
        if noise is None:
            noise = {"type": "constant", "value":0.0}
        torch.manual_seed(seed)

        template_rollout = TorchRollout(self.env.model, horizon=self.horizon)
        _, observation_template, _ = template_rollout.reset()

        self.policy = NeuralStateFeedbackPolicy(
            observation_template=observation_template,
            action_template=template_rollout.noop_actions,
            hidden_sizes=self.arch,
            action_space=self.env.action_space,
        )

        self.logic = FuzzyLogic(
            tnorm=ProductTNorm(),
            comparison=SigmoidComparison(weight=fuzzy_weight),
            rounding=SoftRounding(weight=fuzzy_weight),
            control=SoftControlFlow(weight=fuzzy_weight),
            sampling=SoftRandomSampling(
                poisson_max_bins=100,
                binomial_max_bins=100,
                bernoulli_gumbel_softmax=False
            )
        )

        self.trainer = Train(
            horizon=self.horizon,
            model=self.env.model,
            action_space=self.env.action_space,
            policy=self.policy,
            logic=self.logic,
            lr=self.lr,
            hidden_sizes=self.arch,  # why train need the network arch?!?
            batch_size=self.horizon,
            seed=self.seed,
            additive_noise=AdditiveNoiseFactory.create(
                noise_type=noise["type"],
                std=noise["value"],
                source=template_rollout,
            ),
        )

    def run_experiment(self, iterations: int=100, log_frequency: int=10) -> None:
        history, trained_policy = self.trainer.train_trajectory(
            iterations=iterations,
            print_every=log_frequency,
            batch_size=self.horizon,  # why again?
            additive_noise=self.trainer.default_additive_noise,  # why again if already in trainer?
        )

        label = "test"
        iterations, returns = collapse_history_to_iterations(
            history,
            label=label,
            seed=self.seed,
        )
        return iterations, returns