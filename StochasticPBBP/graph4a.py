from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Some pyRDDLGym imports touch matplotlib internals; keep their cache writable.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
from StochasticPBBP.core.Logic import (
    FuzzyLogic,
    SigmoidComparison,
    SoftRounding,
    SoftControlFlow,
    ProductTNorm,
    SoftRandomSampling,
)
import matplotlib
import numpy as np
import pyRDDLGym
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from StochasticPBBP.core.Logic import ExactLogic
from StochasticPBBP.core.Rollout import TorchRollout

DOMAIN_PATH = PACKAGE_ROOT / "problems" / "powergen" / "domain.rddl"
INSTANCE_PATH = PACKAGE_ROOT / "problems" / "powergen" / "instance_1.rddl"
INSTANCE_NAME = "inst_power_gen_1c"
PLANT_NAMES = ("p1", "p2")


@dataclass(frozen=True)
class PowerGenSpec:
    prod_units_min: np.ndarray
    prod_units_max: np.ndarray
    prod_change_penalty: np.ndarray
    turn_on_cost: np.ndarray
    cost_per_unit: np.ndarray
    income_per_unit: float
    temp_min: float
    temp_max: float
    demand_exp_coef: float
    min_demand_temp: float
    min_consumption: float
    unfulfilled_demand_penalty: float
    prod_mean_noise: np.ndarray
    discount: float


@dataclass(frozen=True)
class PowerGenState:
    prev_prod: np.ndarray
    prev_on: np.ndarray
    temperature: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV for a 3D action-value surface on power_gen "
            f"instance {INSTANCE_NAME}."
        )
    )
    parser.add_argument(
        "--horizon",
        default=50,
        type=int,
        help="Rollout horizon. The same action pair is repeated at every step.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50,
        help="Number of linspace points per action axis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed used by TorchRollout. The same seed is reapplied to every "
            "grid point for a reproducible surface."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False ,

        help=(
            "Use mean/noise-free dynamics computed directly from the RDDL CPF "
            "instead of stochastic rollout sampling."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the CSV will be written.",
    )
    parser.add_argument(
        "--plot",
        default=True,
        action="store_true",
        help="After generating the CSV, also save a 3D plot as PNG.",
    )
    parser.add_argument(
        "--plot-csv",
        type=Path,
        help="Read an existing CSV and build a 3D plot from it instead of running rollouts.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Explicit PNG output path for the 3D plot.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        help="Optional plot title. Defaults to the CSV stem.",
    )
    args = parser.parse_args()

    if args.plot_csv is None:
        if args.horizon is None:
            parser.error("--horizon is required unless --plot-csv is used.")
        if args.horizon <= 0:
            parser.error("--horizon must be positive.")
    elif args.horizon is not None:
        parser.error("Use either --plot-csv or rollout generation arguments, not both.")

    if args.num_points <= 0:
        parser.error("--num-points must be positive.")
    return args


def _vectorize(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(PLANT_NAMES),):
        raise ValueError(
            f"{name} must have shape {(len(PLANT_NAMES),)}, got {tuple(vector.shape)}."
        )
    return vector


def load_model():
    env = pyRDDLGym.make(domain=DOMAIN_PATH, instance=INSTANCE_PATH, vectorized=True)
    model = env.model

    plants = tuple(model.type_to_objects.get("plant", []))
    if plants != PLANT_NAMES:
        raise ValueError(
            f"Expected plant ordering {PLANT_NAMES} in {INSTANCE_NAME}, got {plants}."
        )
    if "curProd" not in model.action_fluents:
        raise ValueError("Expected lifted action fluent 'curProd' in the power_gen model.")
    return model


def build_spec(model) -> PowerGenSpec:
    ''' this function make the necessary computations to build the PowerGenSpec dataclass from the model's non-fluents, 
    including computing the mean noise for each plant based on the Weibull distribution parameters.
    '''
    non_fluents = model.non_fluents
    prod_shape = _vectorize(non_fluents["PROD-SHAPE"], "PROD-SHAPE")
    prod_scale = _vectorize(non_fluents["PROD-SCALE"], "PROD-SCALE")
    if np.any(prod_shape <= 0.0):
        raise ValueError("PROD-SHAPE must be strictly positive for Weibull mean evaluation.")

    prod_mean_noise = prod_scale * np.asarray(
        [math.gamma(1.0 + (1.0 / shape)) for shape in prod_shape],
        dtype=np.float64,
    )

    return PowerGenSpec(
        prod_units_min=_vectorize(non_fluents["PROD-UNITS-MIN"], "PROD-UNITS-MIN"),
        prod_units_max=_vectorize(non_fluents["PROD-UNITS-MAX"], "PROD-UNITS-MAX"),
        prod_change_penalty=_vectorize(
            non_fluents["PROD-CHANGE-PENALTY"], "PROD-CHANGE-PENALTY"
        ),
        turn_on_cost=_vectorize(non_fluents["TURN-ON-COST"], "TURN-ON-COST"),
        cost_per_unit=_vectorize(non_fluents["COST-PER-UNIT"], "COST-PER-UNIT"),
        income_per_unit=float(non_fluents["INCOME-PER-UNIT"]),
        temp_min=float(non_fluents["TEMP-MIN"]),
        temp_max=float(non_fluents["TEMP-MAX"]),
        demand_exp_coef=float(non_fluents["DEMAND-EXP-COEF"]),
        min_demand_temp=float(non_fluents["MIN-DEMAND-TEMP"]),
        min_consumption=float(non_fluents["MIN-CONSUMPTION"]),
        unfulfilled_demand_penalty=float(non_fluents["UNFULFILLED-DEMAND-PENALTY"]),
        prod_mean_noise=prod_mean_noise,
        discount=float(getattr(model, "discount", 1.0)),
    )


def build_initial_state(model) -> PowerGenState:
    state_fluents = model.state_fluents
    return PowerGenState(
        prev_prod=_vectorize(state_fluents["prevProd"], "prevProd"),
        prev_on=np.asarray(state_fluents["prevOn"], dtype=bool),
        temperature=float(state_fluents["temperature"]),
    )


def make_output_path(args: argparse.Namespace) -> Path:
    suffix = "deterministic" if args.deterministic else f"seed_{args.seed}"
    filename = (
        f"powergen_instance1_3d_data_horizon_{args.horizon}"
        f"_points_{args.num_points}_{suffix}.csv"
    )
    return args.output_dir / filename


def make_plot_output_path(csv_path: Path, plot_output: Path | None) -> Path:
    if plot_output is not None:
        return plot_output
    return csv_path.with_suffix(".png")


def cumulative_reward_with_rollout(
    rollout: TorchRollout,
    action_vector: np.ndarray,
    horizon: int,
    discount: float,
    seed: int | None,
) -> float:
    action_template = rollout.noop_actions["curProd"]
    action_tensor = torch.as_tensor(
        action_vector,
        dtype=action_template.dtype,
        device=action_template.device,
    )
    action_dict = {"curProd": action_tensor}

    if seed is not None:
        rollout.cell.key.manual_seed(seed)

    subs, _, model_params = rollout.reset()
    cumulative_reward = 0.0
    discount_factor = 1.0

    for _ in range(horizon):
        subs, _, reward, done, model_params = rollout.step(
            subs=subs,
            actions=action_dict,
            model_params=model_params,
        )
        cumulative_reward += discount_factor * float(reward.detach().cpu().item())
        discount_factor *= discount
        if done:
            break

    return cumulative_reward


def cumulative_reward_deterministic(
    action_vector: np.ndarray,
    horizon: int,
    spec: PowerGenSpec,
    initial_state: PowerGenState,
) -> float:
    prev_prod = initial_state.prev_prod.copy()
    prev_on = initial_state.prev_on.copy()
    temperature = float(initial_state.temperature)
    cumulative_reward = 0.0
    discount_factor = 1.0

    for _ in range(horizon):
        cur_on = action_vector >= spec.prod_units_min
        unclipped_prod = action_vector + spec.prod_mean_noise
        effective_prod = np.where(
            cur_on,
            np.maximum(np.minimum(unclipped_prod, spec.prod_units_max), spec.prod_units_min),
            0.0,
        )

        demand = spec.min_consumption + spec.demand_exp_coef * (
            temperature - spec.min_demand_temp
        ) ** 2
        fulfilled_demand = min(demand, float(np.sum(effective_prod)))

        reward = (
            -float(np.sum(effective_prod * spec.cost_per_unit))
            + (fulfilled_demand * spec.income_per_unit)
            - (
                spec.unfulfilled_demand_penalty
                if demand > fulfilled_demand
                else 0.0
            )
            + float(np.sum(np.abs(effective_prod - prev_prod) * spec.prod_change_penalty))
            - float(np.sum((~prev_on & cur_on) * spec.turn_on_cost))
        )

        cumulative_reward += discount_factor * reward
        discount_factor *= spec.discount

        prev_prod = effective_prod.copy()
        prev_on = cur_on.copy()
        temperature = float(np.clip(temperature, spec.temp_min, spec.temp_max))

    return cumulative_reward


def generate_csv(args: argparse.Namespace) -> Path:
    model = load_model()
    spec = build_spec(model)
    initial_state = build_initial_state(model)
    output_path = make_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    action_grid_p1 = np.linspace(spec.prod_units_min[0], spec.prod_units_max[0], args.num_points, dtype=np.float64)
    action_grid_p2 = np.linspace(spec.prod_units_min[1], spec.prod_units_max[1], args.num_points, dtype=np.float64)

    total_pairs = len(action_grid_p1) * len(action_grid_p2)
    progress_interval = max(1, total_pairs // 10)
    processed = 0

    rollout = None
    if not args.deterministic:
        logic = FuzzyLogic(
            tnorm=ProductTNorm(),
            comparison=SigmoidComparison(weight=100.0),
            rounding=SoftRounding(weight=100.0),
            control=SoftControlFlow(weight=100.0),
            sampling=SoftRandomSampling(),
        )
        rollout = TorchRollout(model, horizon=args.horizon, logic=logic)
        rollout.cell.key.manual_seed(args.seed)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_p1", "action_p2", "cumulative_reward"])

        with torch.no_grad():
            for action_p1 in action_grid_p1:
                for action_p2 in action_grid_p2:
                    action_vector = np.asarray([action_p1, action_p2], dtype=np.float64)
                    if args.deterministic:
                        cumulative_reward = cumulative_reward_deterministic(
                            action_vector=action_vector,
                            horizon=args.horizon,
                            spec=spec,
                            initial_state=initial_state,
                        )
                    else:
                        cumulative_reward = cumulative_reward_with_rollout(
                            rollout=rollout,
                            action_vector=action_vector,
                            horizon=args.horizon,
                            discount=spec.discount,
                            seed=args.seed,
                        )

                    writer.writerow([action_p1, action_p2, cumulative_reward])

                    processed += 1
                    if processed == 1 or processed % progress_interval == 0 or processed == total_pairs:
                        print(
                            f"Processed {processed}/{total_pairs} action pairs...",
                            flush=True,
                        )

    return output_path


def load_csv_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    action_p1: list[float] = []
    action_p2: list[float] = []
    rewards: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"action_p1", "action_p2", "cumulative_reward"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain the columns {sorted(required_columns)}."
            )

        for row in reader:
            action_p1.append(float(row["action_p1"]))
            action_p2.append(float(row["action_p2"]))
            rewards.append(float(row["cumulative_reward"]))

    if not action_p1:
        raise ValueError(f"{csv_path} does not contain any data rows.")

    return (
        np.asarray(action_p1, dtype=np.float64),
        np.asarray(action_p2, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
    )


def plot_csv(csv_path: Path, plot_output: Path | None = None, title: str | None = None) -> Path:
    action_p1, action_p2, rewards = load_csv_points(csv_path)
    output_path = make_plot_output_path(csv_path, plot_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_x = np.unique(action_p1)
    unique_y = np.unique(action_p2)
    expected_points = unique_x.size * unique_y.size
    observed_pairs = {(float(x), float(y)) for x, y in zip(action_p1, action_p2)}
    complete_grid = len(observed_pairs) == expected_points == len(action_p1)

    fig = plt.figure(figsize=(11, 8))
    axis = fig.add_subplot(111, projection="3d")

    if complete_grid:
        x_mesh, y_mesh = np.meshgrid(unique_x, unique_y, indexing="xy")
        z_mesh = np.empty_like(x_mesh, dtype=np.float64)
        reward_map = {
            (float(x), float(y)): float(z)
            for x, y, z in zip(action_p1, action_p2, rewards)
        }
        for row_index, y_value in enumerate(unique_y):
            for col_index, x_value in enumerate(unique_x):
                z_mesh[row_index, col_index] = reward_map[(float(x_value), float(y_value))]

        artist = axis.plot_surface(
            x_mesh,
            y_mesh,
            z_mesh,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
    else:
        artist = axis.scatter(
            action_p1,
            action_p2,
            rewards,
            c=rewards,
            cmap="viridis",
            s=24,
            depthshade=True,
        )

    axis.set_xlabel("action_p1")
    axis.set_ylabel("action_p2")
    axis.set_zlabel("cumulative_reward")
    axis.set_title(title or csv_path.stem)
    fig.colorbar(artist, ax=axis, shrink=0.65, pad=0.08, label="cumulative_reward")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.plot_csv is not None:
        plot_path = plot_csv(
            csv_path=args.plot_csv,
            plot_output=args.plot_output,
            title=args.plot_title,
        )
        print(f"Saved 3D plot to {plot_path}")
        return

    output_path = generate_csv(args)
    mode = "deterministic mean CPF rollout" if args.deterministic else f"seeded TorchRollout (seed={args.seed})"
    print(f"Saved {mode} data to {output_path}")
    if args.plot:
        plot_path = plot_csv(
            csv_path=output_path,
            plot_output=args.plot_output,
            title=args.plot_title,
        )
        print(f"Saved 3D plot to {plot_path}")


if __name__ == "__main__":
    main()
