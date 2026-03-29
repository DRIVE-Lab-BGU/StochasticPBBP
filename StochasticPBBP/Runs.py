from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyRDDLGym

PACKAGE_ROOT = Path(__file__).resolve().parent
print(f"PACKAGE_ROOT={PACKAGE_ROOT}")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.Policies import GaussianPolicy
from core.Simulator import TorchRDDLSimulator
from core.Train import Train


def main() -> None:


    domain = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
    instance =  PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    simulator = TorchRDDLSimulator(env.model)
    policy = GaussianPolicy(action_template=simulator.noop_actions)
    trainer = Train(
        horizon= 113,
        model=env.model,
        action_space=env.action_space,
        policy=policy,
        lr=0.01,
        hidden_sizes=[12,12],
        batch_size=5,
        seed=12,
        simulator=simulator,
    )
    history = trainer.train_trajectory(
        iterations=20,
        print_every=5,
        batch_size=5,
    )

    if history:
        final_metrics = history[-1]
        print(
            f"final chunk return={final_metrics['return']:.4f} "
            f"after iter={int(final_metrics['iteration'])} "
            f"chunk={int(final_metrics['chunk_index'])}/{int(final_metrics['num_chunks'])}"
        )


if __name__ == '__main__':
    main()
