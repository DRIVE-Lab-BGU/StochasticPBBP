from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys

import pyRDDLGym
from StochasticPBBP.manager import ExperimentManager
import matplotlib.pyplot as plt

import torch
from core.Train import Train
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Policies import NeuralStateFeedbackPolicy
from StochasticPBBP.utils.helper import collapse_history_to_iterations
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.core.Logic import FuzzyLogic, SoftRounding, ProductTNorm, SigmoidComparison, SoftRandomSampling, SoftControlFlow


parser = argparse.ArgumentParser()
parser.add_argument("--instance", type=int, default=5, help="instance number")
parser.add_argument("--domain", type=str, default='reservoir', help="domain name")
parser.add_argument("--seeds", type=int, default=5, help="number of seeds for training")
parser.add_argument("--eval", type=int, default=1, help="number of averaging evaluations")
parser.add_argument("--trainkey", type=int, default=112, help="start seed for the training seeds")
parser.add_argument("--evalkey", type=int, default=42, help="start seed for the eval seeds")
parser.add_argument("--horizon", type=int, default=200, help="number of steps in a rollout")
parser.add_argument("--lr", type=float, default=0.01, help="RMSProp learning rate")
parser.add_argument("--iterations", type=int, default=600, help="number of training iterations")
parser.add_argument('--arch', nargs='+', type=int, default=(12, 12))
parser.add_argument("--logfreq", type=int, default=20, help="log iteration frequency")
parser.add_argument("--weight", type=float, default=50.0, help="t-norms approximation weight")
parser.add_argument("-e", "--exact", action="store_true", help="Exact evaluation mode - evaluate on a"
                                                                " separate pyRDDLGym instance")
args = parser.parse_args()
PACKAGE_ROOT = Path(__file__).resolve().parent



def main(args) -> None:
    domain = os.path.join(PACKAGE_ROOT, 'problems', args.domain, 'domain.rddl')
    instance = os.path.join(PACKAGE_ROOT, 'problems', args.domain, 'instance_' + str(args.instance) + '.rddl')
    output_dir = os.path.join(PACKAGE_ROOT, 'outputs', args.domain + '_' + str(args.instance))

    returns = []
    stds = []
    colors = []
    labels = []

    noise = {"type": "constant", "value":0.0}
    manager = ExperimentManager(domain=domain, instance=instance,seed=args.trainkey, horizon=args.horizon,
                                seeds=args.seeds, fuzzy_weight=args.weight, learning_rate=args.lr, noise=noise)
    iterations0, returns0, stds0 = manager.run_experiment(iterations=args.iterations, log_frequency=args.logfreq)
    returns.append(returns0)
    stds.append(stds0)
    colors.append("green")
    labels.append("w/o noise")

    noise = {"type": "constant", "value": 3.0}
    manager = ExperimentManager(domain=domain, instance=instance, seed=args.trainkey, horizon=args.horizon,
                                seeds=args.seeds, fuzzy_weight=args.weight, learning_rate=args.lr, noise=noise)
    _, returns1, stds1 = manager.run_experiment(iterations=args.iterations, log_frequency=args.logfreq)
    returns.append(returns1)
    stds.append(stds1)
    colors.append("red")
    labels.append("noise=1.0")

    noise = {"type": "constant", "value": 3.0}
    manager = ExperimentManager(domain=domain, instance=instance, seed=args.trainkey, horizon=args.horizon,
                                seeds=args.seeds, fuzzy_weight=args.weight, learning_rate=args.lr, noise=noise)
    _, returns3, stds3 = manager.run_experiment(iterations=args.iterations, log_frequency=args.logfreq)
    returns.append(returns3)
    stds.append(stds3)
    colors.append("blue")
    labels.append("noise=3.0")


    plt.switch_backend("Agg")
    fig, axis = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    axis.set_ylabel('Training return')
    axis.set_title(
        f'Noise={noise["value"]} vs no noise'
    )
    for mean, std, col, label in zip(returns, stds, colors, labels):
        lower = [m - s for (m, s) in zip(mean, std)]
        upper = [m + s for (m, s) in zip(mean, std)]
        axis.plot(
            iterations0,
            mean,
            color=col,
            linestyle='-',
            linewidth=2.0,
            label=label,
        )
        axis.fill_between(
            iterations0,
            lower,
            upper,
            color=col,
            alpha=0.18,
        )
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    output_path = PACKAGE_ROOT / "results_plot2.png"
    fig.savefig(output_path)





if __name__ == '__main__':
    main(args)
