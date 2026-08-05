from __future__ import annotations

import unittest
from pathlib import Path

import pyRDDLGym
import torch

from StochasticPBBP.core.Logic import ExactLogic
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.core.Train import Train
from StochasticPBBP.manager import ExperimentManager
from StochasticPBBP.utils.Policies import TO


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _RecordingEnv:
    vectorized = True
    discount = 1.0
    horizon = 3

    def __init__(self) -> None:
        self.actions = []

    def reset(self, seed=None):
        del seed
        self.actions = []
        return {}, {}

    def step(self, action):
        self.actions.append(action)
        return {}, 0.0, False, False, {}


class TOPolicyUnitTest(unittest.TestCase):
    def test_initializes_registered_noop_sequences(self) -> None:
        template = {
            'scalar': torch.tensor(1.5, dtype=torch.float32),
            'vector': torch.tensor([2.0, -3.0], dtype=torch.float64),
        }

        policy = TO(action_template=template, horizon=4)

        self.assertEqual(len(policy.action_parameters), 2)
        self.assertEqual(tuple(policy.action_parameters[0].shape), (4,))
        self.assertEqual(tuple(policy.action_parameters[1].shape), (4, 2))
        registered_parameters = list(policy.parameters())
        self.assertEqual(len(registered_parameters), 2)
        for registered, action_parameter in zip(
            registered_parameters,
            policy.action_parameters,
        ):
            self.assertIs(registered, action_parameter)
        for action in policy.action_sequence():
            self.assertTrue(torch.equal(action['scalar'], template['scalar']))
            self.assertTrue(torch.equal(action['vector'], template['vector']))

    def test_forward_returns_raw_parameter_row_with_gradients(self) -> None:
        policy = TO(
            action_template={'action': torch.zeros(2, dtype=torch.float64)},
            horizon=3,
        )
        with torch.no_grad():
            policy.action_parameters[0][1].copy_(
                torch.tensor([20.0, -5.0], dtype=torch.float64)
            )

        action = policy({}, step=1)

        self.assertTrue(
            torch.equal(
                action['action'],
                torch.tensor([20.0, -5.0], dtype=torch.float64),
            )
        )
        self.assertIsNotNone(action['action'].grad_fn)

        objective = action['action'].sum()
        objective.backward()
        expected_gradient = torch.zeros_like(policy.action_parameters[0])
        expected_gradient[1] = 1.0
        self.assertTrue(
            torch.equal(policy.action_parameters[0].grad, expected_gradient)
        )

    def test_sampling_cursor_and_explicit_steps(self) -> None:
        policy = TO(
            action_template={'action': torch.tensor(0.0)},
            horizon=3,
        )
        with torch.no_grad():
            policy.action_parameters[0].copy_(torch.tensor([1.0, 2.0, 3.0]))

        explicit = policy.sample_action({}, training_mode=False, step=2)
        first = policy.sample_action({}, training_mode=False)
        second = policy.sample_action({}, training_mode=False)
        policy.reset()
        reset_first = policy.sample_action({}, training_mode=False)

        self.assertEqual(float(explicit['action'].detach()), 3.0)
        self.assertEqual(float(first['action'].detach()), 1.0)
        self.assertEqual(float(second['action'].detach()), 2.0)
        self.assertEqual(float(reset_first['action'].detach()), 1.0)
        self.assertIsNone(explicit['action'].grad_fn)

    def test_evaluate_passes_timestep_to_open_loop_policy(self) -> None:
        policy = TO(
            action_template={'action': torch.tensor(0.0)},
            horizon=3,
        )
        with torch.no_grad():
            policy.action_parameters[0].copy_(torch.tensor([1.0, 2.0, 3.0]))
        env = _RecordingEnv()

        policy.evaluate(env, episodes=1)

        self.assertEqual(
            [float(action['action']) for action in env.actions],
            [1.0, 2.0, 3.0],
        )

    def test_rejects_invalid_inputs_and_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            TO({'action': torch.tensor(0.0)}, horizon=0)
        with self.assertRaisesRegex(ValueError, 'at least one tensor'):
            TO({}, horizon=2)
        with self.assertRaisesRegex(ValueError, 'floating-point'):
            TO({'action': torch.tensor(0, dtype=torch.int64)}, horizon=2)

        policy = TO({'action': torch.tensor(0.0)}, horizon=2)
        with self.assertRaisesRegex(TypeError, 'step must be an integer'):
            policy({}, step=1.5)
        with self.assertRaisesRegex(IndexError, r'must be in \[0, 1\]'):
            policy({}, step=2)


class TOIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        vtol_root = PACKAGE_ROOT / 'problems' / 'vtol'
        cls.vtol_env = pyRDDLGym.make(
            domain=vtol_root / 'domain.rddl',
            instance=vtol_root / 'instance_1.rddl',
            vectorized=True,
        )

    def test_rmsprop_training_improves_deterministic_return(self) -> None:
        horizon = 5
        baseline_rollout = TorchRollout(
            self.vtol_env.model,
            horizon=horizon,
            logic=ExactLogic(),
        )
        policy = TO(
            action_template=baseline_rollout.noop_actions,
            horizon=horizon,
        )
        before = float(baseline_rollout(policy=policy).return_.detach())

        trainer = Train(
            model=self.vtol_env.model,
            policy=policy,
            horizon=horizon,
            lr=0.01,
            logic=ExactLogic(),
            batch_size=horizon,
            batch_num=1,
            seed=0,
        )
        trainer.train_trajectory(iterations=20, print_every=0)
        after = float(trainer.rollout(policy=policy).return_.detach())

        self.assertIsInstance(trainer.optimizer, torch.optim.RMSprop)
        self.assertGreater(after, before)
        self.assertFalse(
            torch.equal(
                policy.action_parameters[0].detach(),
                torch.zeros_like(policy.action_parameters[0]),
            )
        )

    def test_to_rejects_partial_or_repeated_batches(self) -> None:
        horizon = 5
        rollout = TorchRollout(
            self.vtol_env.model,
            horizon=horizon,
            logic=ExactLogic(),
        )
        policy = TO(rollout.noop_actions, horizon=horizon)

        with self.assertRaisesRegex(ValueError, 'full-horizon'):
            Train(
                model=self.vtol_env.model,
                policy=policy,
                horizon=horizon,
                batch_size=horizon - 1,
            )
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            Train(
                model=self.vtol_env.model,
                policy=policy,
                horizon=horizon,
                batch_size=horizon,
                batch_num=2,
            )
        with self.assertRaisesRegex(ValueError, 'must match'):
            Train(
                model=self.vtol_env.model,
                policy=TO(rollout.noop_actions, horizon=horizon - 1),
                horizon=horizon,
                batch_size=horizon,
            )

        trainer = Train(
            model=self.vtol_env.model,
            policy=policy,
            horizon=horizon,
            batch_size=horizon,
        )
        with self.assertRaisesRegex(ValueError, 'full-horizon'):
            trainer.train_trajectory(
                iterations=1,
                batch_size=horizon - 1,
                print_every=0,
            )

    def test_reservoir_domain_clips_raw_release(self) -> None:
        reservoir_root = PACKAGE_ROOT / 'problems' / 'reservoir'
        env = pyRDDLGym.make(
            domain=reservoir_root / 'domain.rddl',
            instance=reservoir_root / 'instance_1.rddl',
            vectorized=True,
        )
        rollout = TorchRollout(env.model, horizon=1, logic=ExactLogic())
        policy = TO(rollout.noop_actions, horizon=1)
        raw_release = torch.tensor(
            [-10.0, 1000.0],
            dtype=policy.action_parameters[0].dtype,
        )
        with torch.no_grad():
            policy.action_parameters[0][0].copy_(raw_release)

        trace = rollout(policy=policy)
        initial_level = trace.observations[0]['rlevel']

        self.assertTrue(torch.equal(trace.actions[0]['release'], raw_release))
        self.assertTrue(
            torch.allclose(
                trace.final_subs['released_water'],
                torch.tensor(
                    [0.0, float(initial_level[1])],
                    dtype=trace.final_subs['released_water'].dtype,
                ),
            )
        )

    def test_manager_builds_selected_to_policy(self) -> None:
        reservoir_root = PACKAGE_ROOT / 'problems' / 'reservoir'
        manager = ExperimentManager(
            domain=str(reservoir_root / 'domain.rddl'),
            instance=str(reservoir_root / 'instance_1.rddl'),
            horizon=3,
            policy_type='to',
        )

        policy = manager._build_policy()

        self.assertIsInstance(policy, TO)
        self.assertEqual(policy.horizon, 3)


if __name__ == '__main__':
    unittest.main()
