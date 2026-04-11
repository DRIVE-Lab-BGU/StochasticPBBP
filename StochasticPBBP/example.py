from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys

import pyRDDLGym
from StochasticPBBP.manager import ExperimentManager

import torch
from core.Train import Train
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Policies import NeuralStateFeedbackPolicy
from StochasticPBBP.utils.helper import collapse_history_to_iterations
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow


parser = argparse.ArgumentParser()
parser.add_argument("--instance", type=int, default=1, help="instance number")
parser.add_argument("--domain", type=str, default='reservoir', help="domain name")
parser.add_argument("--seeds", type=int, default=1, help="number of seeds for training")
parser.add_argument("--eval", type=int, default=1, help="number of averaging evaluations")
parser.add_argument("--trainkey", type=int, default=112, help="start seed for the training seeds")
parser.add_argument("--evalkey", type=int, default=42, help="start seed for the eval seeds")
parser.add_argument("--horizon", type=int, default=200, help="number of steps in a rollout")
parser.add_argument("--lr", type=float, default=0.01, help="RMSProp learning rate")
parser.add_argument("--iterations", type=int, default=100, help="number of training iterations")
parser.add_argument('--arch', nargs='+', type=int, default=(12, 12))
parser.add_argument("--logfreq", type=int, default=20, help="log iteration frequency")
parser.add_argument("--weight", type=float, default=50.0, help="t-norms approximation weight")
parser.add_argument("-e", "--exact", action="store_true", help="Exact evaluation mode - evaluate on a"
                                                                " separate pyRDDLGym instance")
args = parser.parse_args()





# eval_seeder = FibonacciSeeder(seed_start)
# evaluation_seeds = tuple(next(eval_seeder) for _ in range(num_eval_seeds))


PACKAGE_ROOT = Path(__file__).resolve().parent

def main(args) -> None:
    domain = os.path.join(PACKAGE_ROOT, 'problems', args.domain, 'domain.rddl')
    instance = os.path.join(PACKAGE_ROOT, 'problems', args.domain, 'instance_' + str(args.instance) + '.rddl')
    output_dir = os.path.join(PACKAGE_ROOT, 'outputs', args.domain + '_' + str(args.instance))

    # noise = {"type": "constant", "value":0.0}
    # manager = ExperimentManager(domain=domain, instance=instance,seed=args.trainkey, horizon=args.horizon,
    #                             fuzzy_weight=args.weight, learning_rate=args.lr)
    #
    # iterations, returns = manager.run_experiment(iterations=args.iterations, log_frequency=args.logfreq)

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    torch.manual_seed(args.trainkey)

    template_rollout = TorchRollout(env.model, horizon=args.horizon)
    _, observation_template, _ = template_rollout.reset()

    policy = NeuralStateFeedbackPolicy(
        observation_template=observation_template,
        action_template=template_rollout.noop_actions,
        hidden_sizes=args.arch,
        action_space=env.action_space,
    )

    logic = FuzzyLogic(
        tnorm=ProductTNorm(),
        comparison=SigmoidComparison(weight=args.weight),
        rounding=SoftRounding(weight=args.weight),
        control=SoftControlFlow(weight=args.weight),
        sampling=SoftRandomSampling(
            poisson_max_bins=100,
            binomial_max_bins=100,
            bernoulli_gumbel_softmax=False
        )
    )

    trainer = Train(
        horizon=args.horizon,
        model=env.model,
        action_space=env.action_space,
        policy=policy,
        logic=logic,
        lr=args.lr,
        hidden_sizes=args.arch,                         # why train need the network arch?!?
        batch_size=args.horizon,
        seed=args.trainkey,
        additive_noise=AdditiveNoiseFactory.create(
            noise_type='constant',
            std=0.0,
            source=template_rollout,
        ),
    )

    history, trained_policy = trainer.train_trajectory(
        iterations=args.iterations,
        print_every=args.logfreq,
        batch_size=args.horizon,                            # why again?
        additive_noise=trainer.default_additive_noise,      # why again if already in trainer?
    )

    label = "test"
    iterations, returns = collapse_history_to_iterations(
        history,
        label=label,
        seed=args.trainkey,
    )

    print(
        f"{label}: seed={args.trainkey:3d} "
        f"start_return={returns[0]:10.4f} "
        f"final_return={returns[-1]:10.4f}"
    )







if __name__ == '__main__':
    main(args)
