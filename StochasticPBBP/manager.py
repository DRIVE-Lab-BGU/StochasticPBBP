from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict
import time
import numpy as np

import pyRDDLGym

import torch
from core.Train import Train
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Policies import NeuralStateFeedbackPolicy, MBDPOPolicy
from StochasticPBBP.utils.helper import collapse_history_to_iterations
from StochasticPBBP.utils.seeder import FibonacciSeeder
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory, NoiseInfo
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow
from StochasticPBBP.utils.logger import CSVLogger



class ExperimentManager:
    def __init__(self,domain: str,
                 instance: str,
                 seed: int=42,
                 seeds: int=1,
                 eval_seed: int=42,
                 eval_seeds: int=1,
                 horizon: int=100,
                 arch: Tuple[int, ...]=(12, 12),
                 fuzzy_weight: float=50.0,
                 learning_rate: float=0.01,
                 noise: Optional[NoiseInfo]=None,
                 exact_eval_mode=False,
                 output_folder=None) -> None:
        self.env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
        self.env.horizon = horizon
        self.horizon = horizon
        self.arch = arch
        self.lr = learning_rate
        self.seed = seed
        self.seeds = seeds
        self.eval_seed = eval_seed
        self.eval_seeds = eval_seeds
        self.exact_eval_mode = exact_eval_mode
        self.logger = None
        self.output_folder = output_folder

        seed_start = int(str(time.time_ns())[10:13])
        self.train_seeder = FibonacciSeeder(seed_start)
        self.eval_seeder = FibonacciSeeder(self.eval_seed)
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
        # iterations_axis: List[int] = []
        all_returns: List[List[float]] = []
        all_eval_returns: List[List[float]] = []
        i = 1
        for seed in range(self.seeds):
            print("[INFO] Starting experiment {}, running {} iterations, with noise {}".format(i, iterations,
                                                                                               self.noise["value"]))
            iterations_i, returns_i, eval_iterations_i, eval_returns_i, policy = self._run_single_experiment(
                iterations=iterations, log_frequency=log_frequency)
            all_returns.append(returns_i)
            all_eval_returns.append(eval_returns_i)
            i = i + 1
        if self.exact_eval_mode:
            iterations_axis = eval_iterations_i
            mean, std = self._average_over_returns(all_eval_returns)
        else:
            iterations_axis = iterations_i
            mean, std = self._average_over_returns(all_returns)
        return iterations_axis, mean, std

    def log(self, file_name, iterations, returns, stds):
        data = {}
        headers = ["iteration", "mean", "std"]
        data[headers[0]] = iterations
        data[headers[1]] = returns
        data[headers[2]] = stds

        if self.logger is None:
            csv_output_file = os.path.join(self.output_folder, "run_logs", file_name)
            self.logger = CSVLogger(csv_file_name=csv_output_file)
        self.logger.write_CSV(data=data, headers=headers)
        pass

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

        eval_returns = []
        eval_iterations = []
        all_train_iterations = []
        all_train_returns = []
        chunks = -(-iterations // log_frequency)
        to_go = iterations
        print_iter = 0

        # log = log_frequency
        # evaluate policy at beginning
        if self.exact_eval_mode:
            # log = 0
            self.eval_seeder.reset()
            result = policy.evaluate(self.env, episodes=self.eval_seeds, seed_generator=self.eval_seeder)
            eval_returns.append(result['mean'])
            eval_iterations.append(print_iter)
            print('[INFO] iter={:4d}, steps={:3d}, discounted return={:.2f}, std={:.2f}'.format(print_iter, self.horizon, result['mean'],
                                                                                  result['std']))

        all_train_iterations.append(0)

        # execute training with evaluation on pyrddlgym
        for i in range(chunks):
            to_run = min(to_go, log_frequency)
            history, trained_policy = trainer.train_trajectory(
                iterations=to_run,
                print_every=0,
                batch_size=self.horizon,  # why again?
            )
            to_go = to_go - log_frequency
            train_iterations, train_returns = collapse_history_to_iterations(
                history,
                label="",
                seed=self.seed,
            )

            # evaluate policy
            if self.exact_eval_mode:
                self.eval_seeder.reset()
                result = policy.evaluate(self.env, episodes=self.eval_seeds, seed_generator=self.eval_seeder)
                eval_returns.append(result['mean'])

                print_iter = print_iter + to_run
                eval_iterations.append(print_iter)
                print('[INFO] iter={:4d}, steps={:3d}, discounted return={:.2f}, std={:.2f}'.format(print_iter, self.horizon, result['mean'], result['std']))
            else:
                print('[INFO] iter={:4d}, steps={:3d}, discounted return={:.2f}'.format(all_train_iterations[-1]+train_iterations[-1], self.horizon,
                                                                                      train_returns[0]))

            all_train_returns.extend(train_returns)
            all_train_iterations.extend(list(map(lambda x: x + all_train_iterations[-1], train_iterations)))

        return all_train_iterations[1:], all_train_returns, eval_iterations, eval_returns, policy