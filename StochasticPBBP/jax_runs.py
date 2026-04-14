from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "stochasticpbbp-matplotlib"),
)

import jax
import numpy as np
import pyRDDLGym

PROBLEM = "reservoir"
INSTANCE = "instance_1.rddl"
HORIZON = 200
HIDDEN_SIZES = (12, 12)
ITERATIONS = 200
NUM_SEEDS = 1
SEED_OFFSET = 2
LEARNING_RATE = 0.01
PRINT_EVERY = 100


def curve_from_history(
    history: list[dict[str, float]],
    key: str,
    iterations: int,
) -> np.ndarray:
    curve = np.full((iterations,), np.nan, dtype=np.float32)
    for metrics in history:
        iteration = int(metrics["iteration"]) - 1
        if 0 <= iteration < iterations:
            curve[iteration] = float(metrics[key])
    return curve


def run_jax(env: Any, seed: int) -> list[dict[str, float]]:
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
    history: list[dict[str, float]] = []
    for callback in planner.optimize_generator(
        key=key,
        epochs=ITERATIONS,
        train_seconds=1.0e9,
        print_summary=False,
        print_progress=False,):

        iteration = int(callback["iteration"]) + 1
        train_return = float(callback["train_return"])

        metrics = {
            "iteration": float(iteration),
            "train_return": train_return,
        }
        history.append(metrics)

        if PRINT_EVERY > 0 and (
            iteration == 1 or iteration % PRINT_EVERY == 0 or iteration == ITERATIONS
        ):
            print(
                f"iter={iteration:4d} "
                f"train_return={train_return:12.4f}"
            )

    return history


def save_plot(
    output_path: Path,
    train_returns: np.ndarray,
    instance_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iteration_axis = np.arange(1, len(train_returns) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iteration_axis, train_returns, color="tab:blue", linewidth=2.0, label="train return")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.set_title(
        f"JAX Planner | {instance_name} | "
        f"iterations={ITERATIONS}, horizon={HORIZON}, seeds={NUM_SEEDS}"
    )
    ax.grid(True, alpha=0.3)
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

    histories: list[list[dict[str, float]]] = []
    for seed_offset in range(NUM_SEEDS):
        seed = SEED_OFFSET + seed_offset
        print(
            f"############### jax run {seed_offset + 1} / {NUM_SEEDS} "
            f"(seed={seed}) ###############"
        )
        histories.append(run_jax(env, seed=seed))

    if not histories:
        raise ValueError("No JAX training history was produced.")

    train_curves = np.stack(
        [curve_from_history(history, "train_return", ITERATIONS) for history in histories],
        axis=0,
    )
    mean_train_returns = np.nanmean(train_curves, axis=0)


    output_path = package_root / "jax_runs_returns.png"
    save_plot(
        output_path=output_path,
        train_returns=mean_train_returns,
        instance_name=instance.stem,
    )

    print(f"first iteration mean train return={float(mean_train_returns[0]):.4f}")
    print(f"final iteration mean train return={float(mean_train_returns[-1]):.4f}")
    print(f"plot saved to {output_path}")

    last_history = histories[-1]
    if last_history:
        final_metrics = last_history[-1]
        print(
            f"last jax run: iter={int(final_metrics['iteration'])} "
            f"train_return={float(final_metrics['train_return']):.4f}"
        )

if __name__ == "__main__":
    main()
