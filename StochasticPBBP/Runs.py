from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "stochasticpbbp-matplotlib"),
)

import pyRDDLGym

from StochasticPBBP.core.Logic import FuzzyLogic
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
    instance = os.path.join(package_root, "problems", "reservoir", "instance_3.rddl")

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

    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())
    _, observation_template, _ = template_rollout.reset()

    policy = StationaryMarkov(
        observation_template=observation_template,
        action_template=template_rollout.noop_actions,
        hidden_sizes=hidden_sizes,
    )

    additive_noise = AdditiveNoiseFactory.create(
        noise_type='constant',
        std=0.0,
        source=template_rollout,
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
        seed=112,
        additive_noise=additive_noise,
    )
    history, trained_policy = trainer.train_trajectory(
        iterations=iterations,
        print_every=100,
        batch_size=batch_size,
        batch_num=batch_num,
        additive_noise=additive_noise,
    )
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

        all_rewards = [entry['return'] for entry in history]
        iteration_axis = list(range(1, len(all_rewards) + 1))
        output_path = package_root / "runs_stationary_markov_returns.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iteration_axis, all_rewards, color="tab:blue", marker="o", linewidth=2.0)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Return")
        ax.set_title("StationaryMarkov {iterations} Iterations , horizon={horizon}, batch_size={batch_size}, batch_num={batch_num}".format(**locals()))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"plot saved to {output_path}")

if __name__ == '__main__':
    main()
