from __future__ import annotations

import os
from pathlib import Path

import pyRDDLGym

from core.Policies import StationaryMarkov
from core.Logic import FuzzyLogic
from core.Rollout import TorchRollout, TorchRolloutCell
from core.Train import Train

def main() -> None:
    package_root = Path(__file__).resolve().parent
    domain = os.path.join(package_root, "problems", "reservoir", "domain.rddl")
    instance = os.path.join(package_root, "problems", "reservoir", "instance_3.rddl")

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    # horizon = env.horizon
    horizon = 120
    hidden_sizes = (12, 12)
    # One full-horizon batch per iteration. Set batch_size smaller than horizon
    # to partition the horizon, and increase batch_num to draw more batches.
    batch_size = horizon
    batch_num = 1

    template_rollout = TorchRollout(env.model, horizon=horizon, logic=FuzzyLogic())
    _, observation_template, _ = template_rollout.reset()
    policy = StationaryMarkov(
        observation_template=observation_template,
        action_template=template_rollout.noop_actions,
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
        seed=12,
        noise_type_dict={'type': 'constant', 'value': 0.0},
    )
    history, trained_policy = trainer.train_trajectory(
        iterations=200,
        print_every=2,
        batch_size=batch_size,
        batch_num=batch_num,
    )
    final_sub = history[-1]['final_subs'] if history else None

    for_obs = TorchRolloutCell(env.model, horizon=1, logic=FuzzyLogic())
    obs = for_obs.observe(final_sub)
    print(f"observation is {obs}")
    sample_action = trained_policy(obs)
    print(f"sample action={sample_action} where the observation is {obs}")
    if history:
        final_metrics = history[-1]
        print(
            f"final batch return={final_metrics['return']:.4f} "
            f"after iter={int(final_metrics['iteration'])} "
            f"batch={int(final_metrics['batch_index'])}/{int(final_metrics['batch_num'])} "
            f"partition={int(final_metrics['partition_index'])}/"
            f"{int(final_metrics['num_partitions'])}"
        )


if __name__ == '__main__':
    main()
