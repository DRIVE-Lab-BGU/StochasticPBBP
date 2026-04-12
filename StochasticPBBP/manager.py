from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict
import time
import numpy as np

import pyRDDLGym

import torch
from core.Train import Train
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Policies import NeuralStateFeedbackPolicy
from StochasticPBBP.utils.helper import collapse_history_to_iterations
from StochasticPBBP.utils.seeder import FibonacciSeeder
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory, NoiseInfo
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow



class ExperimentManager:
    def __init__(self,domain: str,
                 instance: str,
                 seed: int=42,
                 seeds: int=1,
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
        self.seeds = seeds

        seed_start = int(str(time.time_ns())[10:13])
        self.train_seeder = FibonacciSeeder(seed_start)
        self.noise = noise
        if self.noise is None:
            self.noise = {"type": "constant", "value":0.0}
        torch.manual_seed(seed)

        self.template_rollout = TorchRollout(self.env.model, horizon=self.horizon)
        _, self.observation_template, _ = self.template_rollout.reset()

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



    def run_experiment(self, iterations: int=100, log_frequency: int=10) -> None:
        iterations_axis: List[int] = []
        all_returns: List[List[float]] = []
        i = 1
        for seed in range(self.seeds):
            print("[INFO] Starting experiment {}, running {} iterations, with noise {}".format(i, iterations,
                                                                                               self.noise["value"]))
            iterations_i, returns_i = self._run_single_experiment(iterations=iterations, log_frequency=log_frequency)
            all_returns.append(returns_i)
            i = i + 1
        iterations_axis = iterations_i
        mean, std = self._average_over_returns(all_returns)
        return iterations_axis, mean, std

    def _average_over_returns(self, returns: List[List[float]]) -> List[float]:
        rows_avg = np.mean(returns, axis=0)
        rows_std = np.std(returns, axis=0)
        return rows_avg, rows_std

    def _run_single_experiment(self, iterations: int=100, log_frequency: int=10) -> None:
        policy = NeuralStateFeedbackPolicy(
            observation_template=self.observation_template,
            action_template=self.template_rollout.noop_actions,
            hidden_sizes=self.arch,
            action_space=self.env.action_space,
            seed=next(self.train_seeder)
        )

        trainer = Train(
            horizon=self.horizon,
            model=self.env.model,
            action_space=self.env.action_space,
            policy=policy,
            logic=self.logic,
            lr=self.lr,
            hidden_sizes=self.arch,  # why train need the network arch?!?
            batch_size=self.horizon,
            seed=self.seed,
            additive_noise=AdditiveNoiseFactory.create(
                noise_type=self.noise["type"],
                std=self.noise["value"],
                source=self.template_rollout,
            ),
        )

        history, trained_policy = trainer.train_trajectory(
            iterations=iterations,
            print_every=log_frequency,
            batch_size=self.horizon,  # why again?
        )

        label = "test"
        iterations, returns = collapse_history_to_iterations(
            history,
            label=label,
            seed=self.seed,
        )
        return iterations, returns