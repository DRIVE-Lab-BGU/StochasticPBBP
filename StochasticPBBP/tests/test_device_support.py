from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyRDDLGym


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.Logic import ExactLogic  # noqa: E402
from core.Rollout import TorchRollout  # noqa: E402
from core.Train import Train  # noqa: E402
from utils.Policies import NeuralStateFeedbackPolicy  # noqa: E402
from utils.device import resolve_torch_device  # noqa: E402


DOMAIN = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
INSTANCE = PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"


class DeviceSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
        cls.model = env.model
        cls.action_space = env.action_space

    def test_explicit_cpu_rollout_keeps_templates_on_cpu(self) -> None:
        rollout = TorchRollout(self.model, horizon=1, logic=ExactLogic(), device='cpu')
        _, observation, _ = rollout.reset()

        self.assertEqual(rollout.cell.device.type, 'cpu')
        self.assertTrue(all(value.device.type == 'cpu' for value in observation.values()))
        self.assertTrue(all(value.device.type == 'cpu' for value in rollout.noop_actions.values()))

    def test_explicit_cpu_policy_and_trainer_stay_on_cpu(self) -> None:
        rollout = TorchRollout(self.model, horizon=1, logic=ExactLogic(), device='cpu')
        _, observation, _ = rollout.reset()
        policy = NeuralStateFeedbackPolicy(
            observation_template=observation,
            action_template=rollout.noop_actions,
            action_space=self.action_space,
            hidden_sizes=(8,),
            seed=0,
            device='cpu',
        )
        trainer = Train(
            model=self.model,
            action_space=self.action_space,
            policy=policy,
            horizon=1,
            seed=0,
            device='cpu',
        )

        self.assertEqual(trainer.device.type, 'cpu')
        self.assertEqual(trainer.rollout.cell.device.type, 'cpu')
        self.assertEqual({parameter.device.type for parameter in trainer.policy.parameters()}, {'cpu'})

    def test_auto_device_resolves_to_known_backend(self) -> None:
        resolved = resolve_torch_device('auto')
        self.assertIn(resolved.type, {'cpu', 'cuda', 'mps'})


if __name__ == '__main__':
    unittest.main()
