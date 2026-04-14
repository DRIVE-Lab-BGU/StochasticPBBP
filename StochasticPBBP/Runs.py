from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "stochasticpbbp-matplotlib"),
)

import jax
import pyRDDLGym
import torch

from StochasticPBBP.core.Logic import (
    FuzzyLogic,
    ProductTNorm,
    SigmoidComparison,
    SoftControlFlow,
    SoftRandomSampling,
    SoftRounding,
)
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.core.Train import Train
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.utils.Policies import StationaryMarkov

PROBLEM = "reservoir"
INSTANCE = "instance_4.rddl"
HORIZON = 200
HIDDEN_SIZES = (12, 12)
BATCH_SIZE = HORIZON
BATCH_NUM = 1
ITERATIONS = 1
NUM_SEEDS = 10
SEED_OFFSET = 2
LEARNING_RATE = 0.01
NOISE_STD = 0
PRINT_EVERY = 10
end_std = 0.01
start_std = 3.0
type_noise = "constant"


def build_logic() -> FuzzyLogic:
    return FuzzyLogic(
        tnorm=ProductTNorm(),
        comparison=SigmoidComparison(weight=10.0),
        rounding=SoftRounding(weight=10.0),
        control=SoftControlFlow(weight=10.0),
        sampling=SoftRandomSampling(
            poisson_max_bins=100,
            binomial_max_bins=100,
            bernoulli_gumbel_softmax=True,
        ),
    )


def history_returns_by_iteration(
    history: list[dict[str, float]],
    iterations: int,
) -> torch.Tensor:
    per_iteration: list[list[float]] = [[] for _ in range(iterations)]
    for entry in history:
        iteration = int(entry["iteration"]) - 1
        if 0 <= iteration < iterations:
            per_iteration[iteration].append(float(entry["return"]))
    return torch.tensor(
        [
            sum(values) / len(values) if values else float("nan")
            for values in per_iteration
        ],
        dtype=torch.float32,
    )


def jax_returns_by_iteration(
    returns: list[float],
    iterations: int,
) -> torch.Tensor:
    curve = torch.full((iterations,), float("nan"), dtype=torch.float32)
    if not returns:
        return curve
    upto = min(len(returns), iterations)
    curve[:upto] = torch.tensor(returns[:upto], dtype=torch.float32)
    return curve


def train_one_seed(env, seed: int) -> list[dict[str, float]]:
    template_rollout = TorchRollout(env.model, horizon=HORIZON)
    _, observation_template, _ = template_rollout.reset()

    additive_noise = AdditiveNoiseFactory.create(
        noise_type="constant",
        std=NOISE_STD,
        source=template_rollout,
    )

    policy = StationaryMarkov(
        observation_template=observation_template,
        action_template=template_rollout.noop_actions,
        action_space=env.action_space,
        hidden_sizes=HIDDEN_SIZES,
    )

    trainer = Train(
        horizon=HORIZON,
        model=env.model,
        action_space=env.action_space,
        policy=policy,
        lr=LEARNING_RATE,
        hidden_sizes=HIDDEN_SIZES,
        batch_size=BATCH_SIZE,
        batch_num=BATCH_NUM,
        seed=seed,
        additive_noise=additive_noise,
        logic=build_logic(),
    )

    history, _ = trainer.train_trajectory(
        iterations=ITERATIONS,
        print_every=PRINT_EVERY,
        batch_size=BATCH_SIZE,
        batch_num=BATCH_NUM,
        additive_noise=additive_noise,
    )
    return history


def run_jax(env, seed: int) -> torch.Tensor:
    from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxDeepReactivePolicy

    planner = JaxBackpropPlanner(
        rddl=env.model,
        plan=JaxDeepReactivePolicy(topology=list(HIDDEN_SIZES)),
        batch_size_train=1,
        batch_size_test=1,
        rollout_horizon=HORIZON,
        optimizer_kwargs={"learning_rate": LEARNING_RATE},
        pgpe=None,
        print_warnings=False,
    )
    key = jax.random.PRNGKey(seed)
    returns = [
        float(callback["train_return"])
        for callback in planner.optimize_generator(
            key=key,
            epochs=ITERATIONS,
            train_seconds=1.0e9,
            print_summary=False,
            print_progress=False,
        )
    ]
    return jax_returns_by_iteration(returns, ITERATIONS)


def save_plot(
    output_path: Path,
    returns_by_iteration: torch.Tensor,
    instance_name: str,
    jax_returns: torch.Tensor | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iteration_axis = list(range(1, len(returns_by_iteration) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iteration_axis, returns_by_iteration, color="tab:blue", linewidth=2.0, label="torch")
    if jax_returns is not None:
        ax.plot(list(range(1, len(jax_returns) + 1)), jax_returns, color="tab:orange", linewidth=2.0, label="jax")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.set_title(
        f"StationaryMarkov | {instance_name} | "
        f"iterations={ITERATIONS}, horizon={HORIZON}, seeds={NUM_SEEDS}, "
        f"batch_size={BATCH_SIZE}, batch_num={BATCH_NUM}"
    )
    ax.grid(True, alpha=0.3)
    if jax_returns is not None:
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:

    package_root = Path(__file__).resolve().parent
    domain = package_root / "problems" / PROBLEM / "domain.rddl"
    instance = package_root / "problems" / PROBLEM / INSTANCE

    print(f"DOMAIN={domain}")
    print(f"INSTANCE={instance}")

    env = pyRDDLGym.make(domain=str(domain), instance=str(instance), vectorized=True)

    seed_curves = []
    jax_seed_curves = []
    last_history: list[dict[str, float]] = []
    num_train_seed = 0
    for seed_offset in range(NUM_SEEDS):
        seed = SEED_OFFSET + seed_offset
        num_train_seed += 1
        print(f"############### now {num_train_seed} number  , left {NUM_SEEDS - num_train_seed} ###############")
        #history = train_one_seed(env, seed=seed)
        #seed_curves.append(history_returns_by_iteration(history, ITERATIONS))
        jax_seed_curves.append(run_jax(env, seed=seed))
        #last_history = history

    #if not seed_curves:
    #    raise ValueError("No training history was produced.")
    if not jax_seed_curves:
        raise ValueError("No JAX training history was produced.")

    mean_returns = torch.stack(seed_curves, dim=0).mean(dim=0)
    print(f"mean_returns for torch={mean_returns}")
    mean_jax_returns = torch.stack(jax_seed_curves, dim=0).mean(dim=0)
    print(f"mean_returns for jax={mean_jax_returns}")
    output_path = package_root / "runs_stationary_markov_returns.png"
    save_plot(output_path, mean_returns, instance.stem, jax_returns=mean_jax_returns)

    print(f"first iteration return={float(mean_returns[0]):.4f}")
    print(f"final iteration return={float(mean_returns[-1]):.4f}")
    print(f"JAX planner final return={float(mean_jax_returns[-1]):.4f}")
    print(f"plot saved to {output_path}")

    if last_history:
        final_metrics = last_history[-1]
        print(
            f"last batch: iter={int(final_metrics['iteration'])} "
            f"batch={int(final_metrics['batch_index'])}/{int(final_metrics['batch_num'])} "
            f"return={float(final_metrics['return']):.4f}"
        )


if __name__ == "__main__":
    main()
