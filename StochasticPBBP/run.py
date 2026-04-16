from __future__ import annotations

from pathlib import Path
import argparse
import os

from StochasticPBBP.manager import ExperimentManager


parser = argparse.ArgumentParser()
parser.add_argument("--instance", type=int, default=2, help="instance number")
parser.add_argument("--domain", type=str, default='hvac', help="domain name")
parser.add_argument("--seeds", type=int, default=5, help="number of seeds for training")
parser.add_argument("--eval", type=int, default=5, help="number of averaging evaluations")
parser.add_argument("--trainkey", type=int, default=112, help="start seed for the training seeds")
parser.add_argument("--evalkey", type=int, default=42, help="start seed for the eval seeds")
parser.add_argument("--horizon", type=int, default=120, help="number of steps in a rollout")
parser.add_argument("--lr", type=float, default=0.01, help="RMSProp learning rate")
parser.add_argument("--iterations", type=int, default=300, help="number of training iterations")
parser.add_argument('--arch', nargs='+', type=int, default=(128, 64))
parser.add_argument("--logfreq", type=int, default=10, help="log iteration frequency")
parser.add_argument("--weight", type=float, default=100.0, help="t-norms approximation weight")
parser.add_argument("--output", type=str, default="", help="the output directory, default is the output subfolder")
parser.add_argument("--noisetype", type=str, default="constant", help="type of exploration noise")
parser.add_argument("--noisestd", type=float, default=0.0, help="initial std of noise")
parser.add_argument("--noisestdend", type=float, default=0.0, help="final std of noise")
parser.add_argument("-e", "--exact", action="store_true", help="Exact evaluation mode - evaluate on a"
                                                                " separate pyRDDLGym instance")
args = parser.parse_args()
PACKAGE_ROOT = Path(__file__).resolve().parent



def main(args) -> None:
    domain = os.path.join(PACKAGE_ROOT, 'problems', args.domain, 'domain.rddl')
    instance = os.path.join(PACKAGE_ROOT, 'problems', args.domain, 'instance_' + str(args.instance) + '.rddl')
    if args.output == "":
        output_dir = os.path.join(PACKAGE_ROOT, 'outputs', args.domain + '_' + str(args.instance))
    else:
        output_dir = args.output
    noise = {"type": args.noisetype, "value":args.noisestd, "final":args.noisestdend}
    manager = ExperimentManager(domain=domain, instance=instance,seed=args.trainkey, horizon=args.horizon,
                                seeds=args.seeds, fuzzy_weight=args.weight, learning_rate=args.lr, noise=noise,
                                eval_seed=args.evalkey, eval_seeds=args.eval, exact_eval_mode=args.exact,
                                output_folder=output_dir)

    iterations, returns, stds = manager.run_experiment(iterations=args.iterations, log_frequency=args.logfreq)
    if args.exact:
        eval = "exact"
    else:
        eval = "train"
    logname = "returns_horizon" + str(args.horizon) + "_weight" + str(args.weight) + "_noise_" + str(
        args.noisetype) + str(args.noisestd) + "_"+ eval+".csv"
    manager.log(file_name=logname, iterations=iterations, returns=returns, stds=stds)


if __name__ == '__main__':
    main(args)