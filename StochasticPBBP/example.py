from __future__ import annotations

from pathlib import Path
import os

import pyRDDLGym

from StochasticPBBP.core.Logic import FuzzyLogic
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.manager import MultiSeedExperimentManager
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.utils.helper import plot_output_folder_summary
from StochasticPBBP.utils.seeder import FibonacciSeeder

PACKAGE_ROOT = Path(__file__).resolve().parent


def main() -> None:
    problem = "reservoir"
    instance_number = 3
    num_random_policies = 10
    seed_start = 42
    num_eval_seeds = 10
    eval_seeder = FibonacciSeeder(seed_start)
    evaluation_seeds = tuple(next(eval_seeder) for _ in range(num_eval_seeds))
    horizon = 250
    net_arc = (12, 12)
    lr = 1e-2
    iterations = 600
    log_every = 10

    domain = os.path.join(PACKAGE_ROOT, 'problems', problem, 'domain.rddl')
    instance = os.path.join(PACKAGE_ROOT, 'problems', problem, 'instance_'+str(instance_number)+'.rddl')
    output_dir = os.path.join(PACKAGE_ROOT, 'outputs', 'exptest3')

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())

    no_noise_manager = MultiSeedExperimentManager(
        domain=domain,
        instance=instance,
        num_random_policies=num_random_policies,
        evaluation_seeds=evaluation_seeds,
        horizon=horizon,
        hidden_sizes=net_arc,
        lr=lr,
        output_dir=output_dir,
    )
    no_noise_manager.Train(
        iterations,
        csv_name='exptest3_no_noise.csv',
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
        output_dir=output_dir,
    )
    noise_manager.Train(
        iterations,
        csv_name='exptest3_noise_1.csv',
        log_every=log_every,
    )
    plot_output_folder_summary(output_dir)


if __name__ == '__main__':
    main()
