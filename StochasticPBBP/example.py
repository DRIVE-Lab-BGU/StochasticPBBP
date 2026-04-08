from __future__ import annotations

from pathlib import Path
import os

import pyRDDLGym

from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.manager import MultiSeedExperimentManager
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.utils.helper import plot_output_folder_summary
from StochasticPBBP.utils.seeder import FibonacciSeeder
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow


PACKAGE_ROOT = Path(__file__).resolve().parent


def main() -> None:
    problem = "reservoir"
    instance_number = 3
    num_random_policies = 5
    seed_start = 42
    num_eval_seeds = 10
    eval_seeder = FibonacciSeeder(seed_start)
    evaluation_seeds = tuple(next(eval_seeder) for _ in range(num_eval_seeds))
    horizon = 200
    net_arc = (128, 64)
    lr = 1e-2
    iterations = 1000
    log_every = 50

    domain = os.path.join(PACKAGE_ROOT, 'problems', problem, 'domain.rddl')
    instance = os.path.join(PACKAGE_ROOT, 'problems', problem, 'instance_'+str(instance_number)+'.rddl')
    output_dir = os.path.join(PACKAGE_ROOT, 'outputs', problem+'_'+str(instance_number))

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    logic = FuzzyLogic(
        tnorm=ProductTNorm(),
        comparison=SigmoidComparison(weight=100.0),
        rounding=SoftRounding(weight=100.0),
        control=SoftControlFlow(weight=100.0),
        sampling=SoftRandomSampling(
            poisson_max_bins=100,
            binomial_max_bins=100,
            bernoulli_gumbel_softmax=False,
        ),
    )
    template_rollout = TorchRollout(env.model, horizon=horizon, logic=logic)

    no_noise_manager = MultiSeedExperimentManager(
        domain=domain,
        instance=instance,
        num_random_policies=num_random_policies,
        evaluation_seeds=evaluation_seeds,
        horizon=horizon,
        hidden_sizes=net_arc,
        lr=lr,
        output_dir=output_dir,
        logic=logic,
    )
    no_noise_manager.Train(
        iterations,
        csv_name=problem+'_no_noise.csv',
        log_every=log_every,
    )

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
        horizon=horizon,
        hidden_sizes=net_arc,
        lr=lr,
        additive_noise=constant_noise,
        logic=logic,
        output_dir=output_dir,
    )
    noise_manager.Train(
        iterations,
        csv_name=problem+'_noise_1.csv',
        log_every=log_every,
    )
    plot_output_folder_summary(output_dir)


if __name__ == '__main__':
    main()
