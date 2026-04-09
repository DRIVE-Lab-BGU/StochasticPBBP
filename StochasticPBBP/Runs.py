from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import torch

# ###########
# if __package__ in {None, ""}:
#     # Let "Run Python File" work from VS Code by exposing the repo root.
#     project_root = Path(__file__).resolve().parents[1]
#     project_root_str = str(project_root)
#     if project_root_str not in sys.path:
#         sys.path.insert(0, project_root_str)
# ###########

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "stochasticpbbp-matplotlib"),
)

import pyRDDLGym

from StochasticPBBP.core.Logic import FuzzyLogic, ProductTNorm
from StochasticPBBP.core.Rollout import TorchRolloutCell, TorchRollout
from StochasticPBBP.core.Train import Train
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.utils.Policies import StationaryMarkov
# from core.Logic import FuzzyLogic
# from core.Rollout import TorchRollout, TorchRolloutCell
# from core.Train import Train
# from utils.Policies import StationaryMarkov

def main() -> None:
    package_root = Path(__file__).resolve().parent
    domain = os.path.join(package_root, "problems", "reservoir", "domain.rddl")
    instance = os.path.join(package_root, "problems", "reservoir", "instance_4.rddl")
    instance_case = instance.split("/")[-1].split(".")[0]

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    # horizon = env.horizon
    horizon = 100
    hidden_sizes = (12, 12)
    # One full-horizon batch per iteration. Set batch_size smaller than horizon
    # to partition the horizon, and increase batch_num to draw more batches.
    batch_size = horizon
    batch_num = 1
    partitions = 0      # fix
    iterations = 100
    times = 2
    # i think to add the batch um and size we need to change size of zeros
    mean_returns = torch.zeros(iterations)
    for seeds in range(times):
        template_rollout = TorchRollout(env.model, horizon=horizon)
        _, observation_template, _ = template_rollout.reset()




        additive_noise = AdditiveNoiseFactory.create(
            noise_type='constant',
            std=0.0,
            source=template_rollout,
        )
        ############
        from StochasticPBBP.core.Logic import (
            FuzzyLogic,
            SigmoidComparison,
            SoftRounding,
            SoftControlFlow,
            SoftRandomSampling,
            ProductTNorm,
        )
        logic = FuzzyLogic(
            tnorm=ProductTNorm(),
            comparison=SigmoidComparison(weight=50.0),
            rounding=SoftRounding(weight=50.0),
            control=SoftControlFlow(weight=50.0),
            sampling=SoftRandomSampling(
                poisson_max_bins=100,
                binomial_max_bins=100,
                bernoulli_gumbel_softmax=False,),)

        policy = StationaryMarkov(
            observation_template=observation_template,
            action_template=template_rollout.noop_actions,
            action_space=env.action_space,
            hidden_sizes=hidden_sizes,
        )


        

        trainer = Train(
                horizon=horizon,
                model=env.model,
                action_space=env.action_space,
                policy=policy,
                lr=0.01,
                hidden_sizes=hidden_sizes,
                batch_size=batch_size,
                batch_num=batch_num,
                seed=seeds+20,
                additive_noise=additive_noise ,logic=logic)
            
        history, trained_policy = trainer.train_trajectory(
                iterations=iterations,
                print_every=500,
                batch_size=batch_size,
                batch_num=batch_num,
                additive_noise=additive_noise,
            )
        all_rewards = [per_inter['return'] for per_inter in history]
        tensor_rewards = torch.tensor(all_rewards)

        mean_returns += tensor_rewards

    mean_returns /= times

    final_sub = history[-1]['final_subs'] if history else None

    for_obs = TorchRolloutCell(env.model, horizon=1, logic=FuzzyLogic())
    obs = for_obs.observe(final_sub)
    #print(f"observation is {obs}")
    sample_action = trained_policy(obs)
    #print(f"sample action={sample_action} where the observation is {obs}")
    if history:
        start = history[0]
        print(
            f"first batch return={start['return']:.4f} "
        )
        final_metrics = history[-1]
        print(
            f"final batch return={final_metrics['return']:.4f} "
            f"after iter={int(final_metrics['iteration'])} "
            f"batch={int(final_metrics['batch_index'])}/{int(final_metrics['batch_num'])} "
            f"partition={int(final_metrics['partition_index'])}/"
            f"{int(final_metrics['num_partitions'])}"
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        all_rewards = [per_inter['return'] for per_inter in history]
        iteration_axis = list(range(1, len(all_rewards) + 1))
        output_path = package_root / "runs_stationary_markov_returns.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iteration_axis, mean_returns, color="tab:blue", marker="o", linewidth=2.0)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Return")
        ax.set_title("StationaryMarkov _ {instance_case} :case, Iterations:{iterations}  , horizon:{horizon},different seeds:{times}, batch_size:{batch_size}, batch_num:{batch_num}".format(**locals()))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"plot saved to {output_path}")

if __name__ == '__main__':
    main()
