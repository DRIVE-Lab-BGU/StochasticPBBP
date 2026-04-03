"""Minimal rollout examples using the torch rollout wrapper and simulator."""

import copy
from email import policy
import sys
from pathlib import Path

import pyRDDLGym
import torch


ROOT = Path(__file__).resolve().parents[1]
print(f"ROOT={ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOMAIN = ROOT / "instances" / "reservoir" / "domain.rddl"
INSTANCE = ROOT / "instances" / "reservoir" / "instance_1.rddl"
print(f"DOMAIN={DOMAIN}")
print(f"INSTANCE={INSTANCE}")

from StochasticPBBP.tests.simulator_test import PACKAGE_ROOT
from core.Logic import ExactLogic  # noqa: E402
from core.Rollout import TorchRollout  # noqa: E402
from core.Simulator import TorchRDDLSimulator  # noqa: E402
from utils.Policies import random_policy




actions = [
    {"release": [12, 68]},
    {"release": [12, 3]},
    {"release": [3, 3]},
    {"release": [4, 3]},
    {"release": [5, 3]},
    {"release": [6, 3]},
]


def rollout_step_by_step(model) -> None:
    print(f"=== TorchRollout.step | {len(actions)} steps ===")
    # i horizon is none its take from the rddl file
    rollout = TorchRollout(model, horizon=len(actions), logic=ExactLogic())
    rollout.cell.key.manual_seed(0)
    policy = random_policy(model, logic=None)
    subs, obs, model_params = rollout.reset()
    print(f"initial observation = {obs}")

    for step_number in range(len(actions)):
        action = policy.get_action()
        subs, obs, reward, done, model_params = rollout.step(
            subs=subs,
            actions=action,
            model_params=model_params,
        )
        print(f"step {step_number} observation = {obs}")
        print(f"step {step_number} reward = {float(reward)} done = {done}")
        if done:
            break


def rollout_forward(model) -> None:
    print(f"=== TorchRollout.forward | {len(actions)} steps ===")
    rollout = TorchRollout(model, horizon=None, logic=ExactLogic())
    rollout.cell.key.manual_seed(0)

    scripted_actions = copy.deepcopy(actions)

    def scripted_policy(obs, step):
        del obs
        if step < len(scripted_actions):
            return scripted_actions[step]
        return rollout.noop_actions
    print(f"scripted policy actions = {scripted_policy}")
    rp = random_policy(model, logic=None)
    def random_policy_wrapper(obs, step,state=None):
        return rp.get_action(obs=obs, num_step=step) , state

    trace = rollout(policy=random_policy_wrapper)
    print(f"num observations = {len(trace.observations)}")
    #print(f"observations = {trace.observations}")
    print(f"cumulative reward = {sum(trace.rewards)}")
    print(f"final observation = {trace.final_observation}")
    exit(0)  # we exit here to avoid running the simulator, which is not the focus of this test.

def compare_rollout_to_simulator(model) -> None:
    print(f"=== Rollout vs Simulator | {len(actions)} steps ===")
    rollout = TorchRollout(model, horizon=len(actions), logic=ExactLogic())
    rollout.cell.key.manual_seed(0)

    sim = TorchRDDLSimulator(
        model,
        logic=ExactLogic(),
        keep_tensors=True,
        noise={"type": "constant", "value": 0},
    )
    sim.seed(0)

    subs, _, model_params = rollout.reset()
    sim.reset()

    action_list = copy.deepcopy(actions)
    for step_number, action in enumerate(action_list, start=1):
        subs, rollout_obs, rollout_reward, rollout_done, model_params = rollout.step(
            subs=subs,
            actions=action,
            model_params=model_params,
        )
        sim_obs, sim_reward, sim_done = sim.step(action, step_number)

        print(f"step {step_number} rollout reward = {float(rollout_reward)}")
        print(f"step {step_number} simulator reward = {float(sim_reward)}")

        for state_name in model.state_fluents:
            if not torch.allclose(rollout_obs[state_name], sim_obs[state_name], atol=1e-5):
                print(f"state mismatch in {state_name}")
                print(f"rollout:   {rollout_obs[state_name]}")
                print(f"simulator: {sim_obs[state_name]}")

        if not torch.allclose(rollout_reward, sim_reward, atol=1e-5):
            print("reward mismatch")
            print(f"rollout:   {rollout_reward}")
            print(f"simulator: {sim_reward}")

        print(f"done rollout={rollout_done} simulator={sim_done}")
        if rollout_done or sim_done:
            break


def main() -> None:
    env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
    model = env.model

    #rollout_step_by_step(model)
    rollout_forward(model)

    #compare_rollout_to_simulator(model)
    import torch
import torch.nn as nn


if __name__ == "__main__":
    main()
