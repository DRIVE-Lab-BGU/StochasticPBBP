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
INSTANCE = "instance_4.rddl"
HORIZON = 350
HIDDEN_SIZES = (12, 12)
ITERATIONS = 400
NUM_SEEDS = 5
SEED_OFFSET = 123
LEARNING_RATE = 0.01
PRINT_EVERY = 100
EVAL_EPISODES = 20 # importent!!! if u cahnge the horizon u must change also in the instace file!!!! 
EVAL_SEED_OFFSET = 10_000


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


def evaluate_jax_policy(
    env: Any,
    planner: Any,
    best_params: Any,
    policy_hyperparams: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    from pyRDDLGym_jax.core.planner import JaxOfflineController

    policy = JaxOfflineController(
        planner=planner,
        key=jax.random.PRNGKey(seed),
        params=best_params,
        eval_hyperparams=policy_hyperparams,
        train_on_reset=False,
    )
    eval_stats = policy.evaluate(env, episodes=EVAL_EPISODES, seed=seed)
    return {name: float(value) for name, value in eval_stats.items()}


def format_eval_stats(eval_stats: dict[str, float]) -> str:
    return (
        f"eval_mean={eval_stats['mean']:12.4f} "
        f"eval_median={eval_stats['median']:12.4f} "
        f"eval_std={eval_stats['std']:12.4f} "
        f"eval_min={eval_stats['min']:12.4f} "
        f"eval_max={eval_stats['max']:12.4f}"
    )


def run_jax(env: Any, seed: int) -> tuple[list[dict[str, float]], dict[str, float]]:
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
    last_callback: dict[str, Any] | None = None
    for callback in planner.optimize_generator(
        key=key,
        epochs=ITERATIONS,
        train_seconds=1.0e9,
        print_summary=False,
        print_progress=False,
    ):
        last_callback = callback

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

    if last_callback is None:
        raise ValueError("JAX planner did not produce any callbacks.")

    eval_stats = evaluate_jax_policy(
        env=env,
        planner=planner,
        best_params=last_callback["best_params"],
        policy_hyperparams=last_callback["policy_hyperparams"],
        seed=seed + EVAL_SEED_OFFSET,
    )
    print(
        f"seed={seed} eval over {EVAL_EPISODES} episodes | "
        f"{format_eval_stats(eval_stats)}"
    )

    return history, eval_stats


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
    ax.plot(
        iteration_axis,
        train_returns,
        color="tab:blue",
        linewidth=2.0,
        label="train return",
    )
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
    eval_results: list[dict[str, float]] = []
    for seed_offset in range(NUM_SEEDS):
        seed = SEED_OFFSET + seed_offset
        print(
            f"############### jax run {seed_offset + 1} / {NUM_SEEDS} "
            f"(seed={seed}) ###############"
        )
        history, eval_stats = run_jax(env, seed=seed)
        histories.append(history)
        eval_results.append(eval_stats)

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
    if eval_results:
        eval_means = np.asarray(
            [result["mean"] for result in eval_results],
            dtype=np.float32,
        )
        mean_eval_return = float(np.mean(eval_means))
        std_eval_return = float(np.std(eval_means))
        print(
            f"mean eval return across learned policies={mean_eval_return:.4f} "
            f"(std across seeds={std_eval_return:.4f}, episodes={EVAL_EPISODES})"
        )
    print(f"plot saved to {output_path}")

    last_history = histories[-1]
    if last_history:
        final_metrics = last_history[-1]
        print(
            f"last jax run: iter={int(final_metrics['iteration'])} "
            f"train_return={float(final_metrics['train_return']):.4f}"
        )
    if eval_results:
        print(f"last jax run eval | {format_eval_stats(eval_results[-1])}")


if __name__ == "__main__":
    main()
