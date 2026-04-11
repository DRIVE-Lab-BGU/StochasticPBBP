from __future__ import annotations

from pathlib import Path
import os

import pyRDDLGym

from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.manager2 import MultiSeedExperimentManager
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.utils.helper import plot_output_folder_summary
from StochasticPBBP.utils.seeder import FibonacciSeeder
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow


PACKAGE_ROOT = Path(__file__).resolve().parent


def main() -> None:
    problem = "reservoir"
    instance_number = 3
    num_random_policies = 4
    seed_start = 42
    num_eval_seeds = 1
    eval_seeder = FibonacciSeeder(seed_start)
    evaluation_seeds = tuple(next(eval_seeder) for _ in range(num_eval_seeds))
    horizon = 200
    net_arc = (12, 12)
    lr = 1e-2
    iterations = 200
    log_every = 5
    exact_evaluation = False
    tnorm_weight = 50.0
    grad_clip = 1000000.0

    domain = os.path.join(PACKAGE_ROOT, 'problems', problem, 'domain.rddl')
    instance = os.path.join(PACKAGE_ROOT, 'problems', problem, 'instance_'+str(instance_number)+'.rddl')
    output_dir = os.path.join(PACKAGE_ROOT, 'outputs', problem+'_'+str(instance_number))

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)

    #################
    # No noise
    #################
    logic = FuzzyLogic(
        tnorm=ProductTNorm(),
        comparison=SigmoidComparison(weight=tnorm_weight),
        rounding=SoftRounding(weight=tnorm_weight),
        control=SoftControlFlow(weight=tnorm_weight),
        sampling=SoftRandomSampling(
            poisson_max_bins=100,
            binomial_max_bins=100,
            bernoulli_gumbel_softmax=False,
        ),
    )

    no_noise_manager = MultiSeedExperimentManager(
        domain=domain,
        instance=instance,
        num_random_policies=num_random_policies,
        evaluation_seeds=evaluation_seeds,
        exact_evaluation=exact_evaluation,
        horizon=horizon,
        hidden_sizes=net_arc,
        lr=lr,
        output_dir=output_dir,
        debug_logging=True,
        grad_clip_norm=grad_clip,
        logic=logic,
    )
    no_noise_manager.Train(
        iterations,
        csv_name=problem+'_no_noise.csv',
        log_every=log_every,
    )

    #################
    # Noise 1.0
    #################
    logic = FuzzyLogic(
        tnorm=ProductTNorm(),
        comparison=SigmoidComparison(weight=tnorm_weight),
        rounding=SoftRounding(weight=tnorm_weight),
        control=SoftControlFlow(weight=tnorm_weight),
        sampling=SoftRandomSampling(
            poisson_max_bins=100,
            binomial_max_bins=100,
            bernoulli_gumbel_softmax=False,
        ),
    )
    template_rollout = TorchRollout(env.model, horizon=horizon, logic=logic)
    constant_noise = AdditiveNoiseFactory.create(
        noise_type='constant',
        std=1.0,
        source=template_rollout,
    )
    noise_manager = MultiSeedExperimentManager(
        domain=domain,
        instance=instance,
        num_random_policies=num_random_policies,
        evaluation_seeds=evaluation_seeds,
        exact_evaluation=exact_evaluation,
        horizon=horizon,
        hidden_sizes=net_arc,
        lr=lr,
        grad_clip_norm=grad_clip,
        additive_noise=constant_noise,
        debug_logging=True,
        logic=logic,
        output_dir=output_dir,
    )
    noise_manager.Train(
        iterations,
        csv_name=problem+'_noise_1.csv',
        log_every=log_every,
    )

    #################
    # Noise 3.0
    #################
    logic = FuzzyLogic(
        tnorm=ProductTNorm(),
        comparison=SigmoidComparison(weight=tnorm_weight),
        rounding=SoftRounding(weight=tnorm_weight),
        control=SoftControlFlow(weight=tnorm_weight),
        sampling=SoftRandomSampling(
            poisson_max_bins=100,
            binomial_max_bins=100,
            bernoulli_gumbel_softmax=False,
        ),
    )
    template_rollout = TorchRollout(env.model, horizon=horizon, logic=logic)
    constant_noise = AdditiveNoiseFactory.create(
        noise_type='constant',
        std=3.0,
        source=template_rollout,
    )
    noise_manager = MultiSeedExperimentManager(
        domain=domain,
        instance=instance,
        num_random_policies=num_random_policies,
        evaluation_seeds=evaluation_seeds,
        exact_evaluation=exact_evaluation,
        horizon=horizon,
        hidden_sizes=net_arc,
        lr=lr,
        grad_clip_norm=grad_clip,
        additive_noise=constant_noise,
        debug_logging=True,
        logic=logic,
        output_dir=output_dir,
    )
    noise_manager.Train(
        iterations,
        csv_name=problem + '_noise_3.csv',
        log_every=log_every,
    )

    #################
    # Plot
    #################
    plot_output_folder_summary(output_dir)


if __name__ == '__main__':
    main()
