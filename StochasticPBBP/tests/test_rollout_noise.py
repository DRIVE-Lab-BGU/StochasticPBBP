from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyRDDLGym
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.Logic import ExactLogic  # noqa: E402
from core.Rollout import TorchRollout, TorchRolloutCell  # noqa: E402
from utils.Noise import (  # noqa: E402
    ConstantAdditiveNoise,
    JacobianBasedAdditiveNoise,
    LinearDecayAdditiveNoise,
    NoiseContext,
)
DOMAIN = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
INSTANCE = PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"


class RolloutNoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
        cls.model = env.model

    @staticmethod
    def _clone_actions(actions):
        cloned = {}
        for (name, value) in actions.items():
            cloned[name] = value.clone() if isinstance(value, torch.Tensor) else value
        return cloned

    def _install_capture_step_fn(self, rollout: TorchRollout):
        captured_actions = []

        def fake_step_fn(key, actions, subs, model_params):
            del key
            captured_actions.append(self._clone_actions(actions))
            return subs, {'reward': torch.tensor(0.0), 'termination': False}, model_params

        rollout.cell.step_fn = fake_step_fn
        return captured_actions

    def test_constant_noise_is_applied_without_mutating_input(self) -> None:
        rollout = TorchRollout(self.model, horizon=5, logic=ExactLogic())
        captured_actions = self._install_capture_step_fn(rollout)

        subs, _, model_params = rollout.reset()
        original_action = {'release': torch.zeros_like(rollout.noop_actions['release'])}
        original_snapshot = original_action['release'].clone()
        noise_std = 0.25

        torch.manual_seed(1234)
        expected_noise = torch.distributions.Normal(
            loc=torch.zeros_like(original_snapshot),
            scale=torch.full_like(original_snapshot, noise_std),
        ).rsample()

        torch.manual_seed(1234)
        rollout.step(
            subs=subs,
            actions=ConstantAdditiveNoise(std=noise_std)(original_action, context=NoiseContext()),
            model_params=model_params,
        )

        self.assertEqual(len(captured_actions), 1)
        self.assertTrue(
            torch.allclose(
                captured_actions[0]['release'],
                original_snapshot + expected_noise,
            )
        )
        self.assertTrue(torch.allclose(original_action['release'], original_snapshot))
        self.assertFalse(
            torch.allclose(
                captured_actions[0]['release'],
                torch.full_like(original_snapshot, captured_actions[0]['release'][0].item()),
            )
        )

    def test_forward_uses_training_iteration_for_linear_decay_noise(self) -> None:
        rollout = TorchRollout(self.model, horizon=3, logic=ExactLogic())
        captured_actions = self._install_capture_step_fn(rollout)

        def zero_policy(observation, step):
            del observation, step
            return {'release': torch.zeros_like(rollout.noop_actions['release'])}

        expected_std = 0.5
        base = torch.zeros_like(rollout.noop_actions['release'])
        additive_noise = LinearDecayAdditiveNoise(
            start_std=1.0,
            end_std=0.0,
            num_iterations=3,
        )
        torch.manual_seed(4321)
        expected_actions = []
        for _ in range(3):
            expected_actions.append(
                base + torch.distributions.Normal(
                    loc=torch.zeros_like(base),
                    scale=torch.full_like(base, expected_std),
                ).rsample()
            )

        torch.manual_seed(4321)
        trace = rollout(
            policy=zero_policy,
            steps=3,
            iteration=1,
            additive_noise=additive_noise,
        )

        self.assertEqual(len(captured_actions), 3)
        self.assertEqual(len(trace.actions), 3)
        for index, expected in enumerate(expected_actions):
            self.assertTrue(torch.allclose(captured_actions[index]['release'], expected))
            self.assertTrue(torch.allclose(trace.actions[index]['release'], expected))

    def test_constant_noise_keeps_gradient_through_actions(self) -> None:
        value = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        additive_noise = ConstantAdditiveNoise(std=0.5)

        torch.manual_seed(99)
        noised_value = additive_noise.apply_to_value(
            name='action',
            value=value,
            context=NoiseContext(),
        )
        loss = noised_value.sum()
        loss.backward()

        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.allclose(value.grad, torch.ones_like(value)))

    def test_jacobian_noise_uses_context_subs(self) -> None:
        rollout = TorchRollout(self.model, horizon=1, logic=ExactLogic())
        captured_actions = self._install_capture_step_fn(rollout)

        def zero_policy(observation, step):
            del observation, step
            return {'release': torch.zeros_like(rollout.noop_actions['release'])}

        additive_noise = JacobianBasedAdditiveNoise(
            cpf_name='rlevel',
            model=self.model,
        )

        def fake_norm(*, cpf_name, subs, wrt=None, create_graph=False, context=None):
            del cpf_name, wrt, create_graph
            self.assertIsNotNone(context)
            self.assertIs(context.model, self.model)
            self.assertEqual(subs.keys(), context.subs.keys())
            return torch.tensor(0.25)

        additive_noise.cpf_jacobian_norm = fake_norm  # type: ignore[method-assign]

        torch.manual_seed(7)
        expected = torch.distributions.Normal(
            loc=torch.zeros_like(rollout.noop_actions['release']),
            scale=torch.full_like(rollout.noop_actions['release'], 0.25),
        ).rsample()

        torch.manual_seed(7)
        trace = rollout(
            policy=zero_policy,
            steps=1,
            iteration=3,
            additive_noise=additive_noise,
        )

        self.assertEqual(len(captured_actions), 1)
        self.assertTrue(torch.allclose(captured_actions[0]['release'], expected))
        self.assertTrue(torch.allclose(trace.actions[0]['release'], expected))

if __name__ == '__main__':
    unittest.main()
