import torch
import pyRDDLGym
from pathlib import Path
from core.Simulator import TorchRDDLSimulator

class neural_policy():

    def __init__(self, model, logic, noise=None):
        self.model = model
        self.logic = logic
        self.noise = noise

    def neural_policy(self, network, obs, num_step):
        neural = network
        action = neural(obs)
        return action


class slp_policy():

    def __init__(self, model, logic, noise=None):
        self.model = model
        self.logic = logic
        self.noise = noise

    def get_action(self, network, obs, num_step):
        neural = network
        action = neural(obs)
        return action


class random_policy():

    def __init__(self, model, logic, noise=None):
        self.model = model
        self.logic = logic
        self.noise = noise
        self.simulator = TorchRDDLSimulator(
            model,
            logic=logic,
            noise=noise,
            keep_tensors=True,
        )

    def get_action_template(self):
        return TorchRDDLSimulator._clone_structure(self.simulator.noop_actions)

    def get_action(self, obs=None, num_step=None, fill_value=None):
        action = self.get_action_template()
        for (name, value) in action.items():
            action[name] = self._fill_action_value(value, fill_value)
        return action

    @staticmethod
    def _fill_action_value(reference, fill_value):
        if not isinstance(reference, torch.Tensor):
            return reference

        if reference.dtype == torch.bool:
            if fill_value is None:
                return torch.randint(0, 2, reference.shape, device=reference.device).bool()
            return torch.full(
                reference.shape,
                bool(fill_value),
                dtype=torch.bool,
                device=reference.device,
            )

        if reference.dtype.is_floating_point:
            if fill_value is None:
                return torch.rand_like(reference)
            return torch.full_like(reference, float(fill_value))

        if fill_value is None:
            return torch.randint(0, 10, reference.shape, device=reference.device).to(dtype=reference.dtype)
        return torch.full_like(reference, int(fill_value))


def main():
    root = Path(__file__).resolve().parents[1]

    domain = root / "instances" / "reservoir" / "domain.rddl"
    instance = root / "instances" / "reservoir" / "instance_1.rddl"
    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    model = env.model



    policy = random_policy(model, logic=None)

    action_template = policy.get_action_template()
    print(f"action template is {action_template}")

    action = policy.get_action() # if empty, it will be filled with random values
    print(f"constant action is {action}")




if __name__ == "__main__":
    main()
