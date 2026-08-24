from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.R2Trainer import R2Trainer, R2TrainingPhase  # noqa: E402
from core.Train import Train  # noqa: E402
from manager import ActionTableCSVWriter, ExperimentManager  # noqa: E402


class ActionTableCSVWriterTest(unittest.TestCase):
    def test_writes_compact_json_and_null_for_unfinished_timesteps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / 'actions.csv'
            writer = ActionTableCSVWriter(csv_path, horizon=3)
            writer.write_iteration(
                1,
                [
                    {
                        'fan-in': torch.tensor([1.1, 0.95], dtype=torch.float64),
                        'heat-input': torch.tensor([8.2, 7.6, 9.1], dtype=torch.float64),
                    },
                    {
                        'fan-in': torch.tensor([1.2, 1.0], dtype=torch.float64),
                        'heat-input': torch.tensor([8.0, 7.5, 9.0], dtype=torch.float64),
                    },
                ],
            )

            # write_iteration flushes, so completed rows are readable before close().
            with csv_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.reader(handle))
            writer.close()

        self.assertEqual(
            rows[0],
            ['iteration', 'timestep_1', 'timestep_2', 'timestep_3'],
        )
        self.assertEqual(rows[1][0], '1')
        self.assertEqual(
            json.loads(rows[1][1]),
            {
                'fan-in': [1.1, 0.95],
                'heat-input': [8.2, 7.6, 9.1],
            },
        )
        self.assertEqual(json.loads(rows[1][3]), None)

    def test_serializes_expected_domain_action_shapes(self) -> None:
        powergen = json.loads(ActionTableCSVWriter._serialize_actions({
            'curProd': torch.arange(11, dtype=torch.float64),
        }))
        reservoir = json.loads(ActionTableCSVWriter._serialize_actions({
            'release': torch.tensor([4.25, 1.8], dtype=torch.float64),
        }))
        hvac = json.loads(ActionTableCSVWriter._serialize_actions({
            'fan-in': torch.tensor([1.1, 0.95], dtype=torch.float64),
            'heat-input': torch.tensor([8.2, 7.6, 9.1], dtype=torch.float64),
        }))

        self.assertEqual(len(powergen['curProd']), 11)
        self.assertEqual(reservoir['release'], [4.25, 1.8])
        self.assertEqual(list(hvac), ['fan-in', 'heat-input'])
        self.assertEqual(len(hvac['fan-in']), 2)
        self.assertEqual(len(hvac['heat-input']), 3)


class R2AnalysisActionCallbackTest(unittest.TestCase):
    def test_callback_receives_analysis_actions_before_profile_refresh(self) -> None:
        trainer = object.__new__(R2Trainer)
        trainer.default_additive_noise = object()
        trainer.analysis_additive_noise = object()
        trainer.r2_profile = None
        trainer.current_phase = R2TrainingPhase.UPDATE

        events: List[str] = []
        update_actions = [{'release': torch.tensor([99.0, 99.0])}]
        analysis_actions = [{'release': torch.tensor([4.25, 1.8])}]

        def run_update_phase(**_: Any) -> Dict[str, Any]:
            events.append('update')
            return {
                'objective': torch.tensor(1.0),
                'loss': torch.tensor(-1.0),
                'trace': SimpleNamespace(actions=update_actions, rewards=[torch.tensor(1.0)]),
            }

        def run_analysis_phase(**_: Any) -> Dict[str, Any]:
            events.append('analysis')
            return {
                'objective': torch.tensor(2.0),
                'trace': SimpleNamespace(actions=analysis_actions, rewards=[torch.tensor(2.0)]),
            }

        def refresh_noise_profile(**_: Any) -> str:
            events.append('refresh')
            trainer.current_phase = R2TrainingPhase.PROFILE_REFRESH
            return 'profile'

        trainer._run_update_phase = run_update_phase
        trainer._run_analysis_phase = run_analysis_phase
        trainer.refresh_noise_profile = refresh_noise_profile

        received: List[Sequence[Dict[str, Any]]] = []

        def callback(iteration: int, actions: Sequence[Dict[str, Any]]) -> None:
            self.assertEqual(iteration, 1)
            events.append('callback')
            received.append(actions)

        trainer.train_iteration(
            iteration=1,
            analysis_action_callback=callback,
        )

        self.assertEqual(events, ['update', 'analysis', 'callback', 'refresh'])
        self.assertIs(received[0], analysis_actions)
        self.assertIsNot(received[0], update_actions)


class ConstantUpdateActionCallbackTest(unittest.TestCase):
    def test_callback_reuses_existing_noisy_update_trace(self) -> None:
        trainer = object.__new__(Train)
        trainer.default_batch_size = 2
        trainer.default_batch_num = 1
        trainer.default_additive_noise = object()
        trainer.rollout = SimpleNamespace(horizon=2)
        trainer.policy = SimpleNamespace(train=lambda mode=True: None)

        events: List[str] = []
        trainer.optimizer = SimpleNamespace(
            zero_grad=lambda **_: events.append('zero_grad')
        )
        noisy_update_actions = [
            {'release': torch.tensor([4.25, 1.8])},
            {'release': torch.tensor([4.4, 1.75])},
        ]

        def run_training_batch(**_: Any) -> Dict[str, Any]:
            events.append('update')
            return {
                'objective': torch.tensor(1.0),
                'loss': torch.tensor(-1.0),
                'trace': SimpleNamespace(
                    actions=noisy_update_actions,
                    rewards=[torch.tensor(0.5), torch.tensor(0.5)],
                    final_subs={},
                ),
            }

        trainer._run_training_batch = run_training_batch
        received: List[Sequence[Dict[str, Any]]] = []

        def callback(iteration: int, actions: Sequence[Dict[str, Any]]) -> None:
            self.assertEqual(iteration, 1)
            events.append('callback')
            received.append(actions)

        history, _ = trainer.train_trajectory(
            iterations=1,
            print_every=0,
            update_action_callback=callback,
        )

        self.assertEqual(events, ['zero_grad', 'update', 'callback'])
        self.assertEqual(len(history), 1)
        self.assertIs(received[0], noisy_update_actions)

    def test_callback_requires_one_full_horizon_update(self) -> None:
        trainer = object.__new__(Train)
        trainer.default_batch_size = 2
        trainer.default_batch_num = 1
        trainer.default_additive_noise = object()
        trainer.rollout = SimpleNamespace(horizon=2)
        trainer.policy = SimpleNamespace(train=lambda mode=True: None)

        with self.assertRaisesRegex(ValueError, 'one full-horizon update'):
            trainer.train_trajectory(
                iterations=1,
                batch_size=1,
                update_action_callback=lambda iteration, actions: None,
            )


class ExperimentManagerActionTableTest(unittest.TestCase):
    def test_streams_global_iterations_across_training_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = object.__new__(ExperimentManager)
            manager.horizon = 3
            manager.noise = {'type': 'gradient2noise', 'value': 3.0, 'alpha': 0.1}
            manager.output_folder = temporary_directory
            manager.domain_name = 'reservoir'
            manager.instance_number = 1
            manager.save_actions_table = True
            manager.action_table_paths = []
            manager.current_policy_seed = None
            manager.exact_eval_mode = False

            policy = object()
            trainer = object.__new__(R2Trainer)
            trainer_call_count = 0

            def build_policy() -> object:
                manager.current_policy_seed = 112
                return policy

            def train_trajectory(
                *,
                iterations: int,
                print_every: int,
                batch_size: int,
                analysis_action_callback,
            ):
                nonlocal trainer_call_count
                del print_every, batch_size
                history = []
                for local_iteration in range(1, iterations + 1):
                    trainer_call_count += 1
                    analysis_action_callback(
                        local_iteration,
                        [
                            {'release': torch.tensor([trainer_call_count, 1.0])},
                            {'release': torch.tensor([trainer_call_count, 2.0])},
                        ],
                    )
                    history.append({
                        'iteration': float(local_iteration),
                        'analysis_return': float(trainer_call_count),
                    })
                return history, policy

            manager._build_policy = build_policy
            manager._build_trainer = lambda *, policy, iterations: trainer
            trainer.train_trajectory = train_trajectory

            manager._run_single_experiment(iterations=2, log_frequency=1)

            self.assertEqual(trainer_call_count, 2)
            self.assertEqual(len(manager.action_table_paths), 1)
            action_table_path = Path(manager.action_table_paths[0])
            self.assertEqual(
                action_table_path.name,
                'actions_table_reservoir_instance1_policyseed112_h3_i2_'
                'gradient2noise_std3.0_alpha0.1.csv',
            )
            self.assertEqual(action_table_path.parent.name, 'actions_table')

            with action_table_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(len(rows), 3)
        self.assertEqual([row[0] for row in rows[1:]], ['1', '2'])
        self.assertEqual(json.loads(rows[1][1]), {'release': [1, 1.0]})
        self.assertEqual(json.loads(rows[2][1]), {'release': [2, 1.0]})
        self.assertIsNone(json.loads(rows[1][3]))

    def test_routes_constant_update_actions_and_uses_constant_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = object.__new__(ExperimentManager)
            manager.horizon = 2
            manager.seed = 112
            manager.noise = {'type': 'constant', 'value': 3.0, 'alpha': 0.1}
            manager.output_folder = temporary_directory
            manager.domain_name = 'reservoir'
            manager.instance_number = 1
            manager.save_actions_table = True
            manager.action_table_paths = []
            manager.current_policy_seed = None
            manager.exact_eval_mode = False

            policy = object()
            trainer = SimpleNamespace()
            trainer_call_count = 0

            def build_policy() -> object:
                manager.current_policy_seed = 112
                return policy

            def train_trajectory(
                *,
                iterations: int,
                print_every: int,
                batch_size: int,
                update_action_callback,
            ):
                nonlocal trainer_call_count
                del print_every, batch_size
                history = []
                for local_iteration in range(1, iterations + 1):
                    trainer_call_count += 1
                    update_action_callback(
                        local_iteration,
                        [
                            {'release': torch.tensor([trainer_call_count, 3.0])},
                            {'release': torch.tensor([trainer_call_count, 4.0])},
                        ],
                    )
                    history.append({
                        'iteration': float(local_iteration),
                        'return': float(trainer_call_count),
                        'num_chunks': 1.0,
                    })
                return history, policy

            manager._build_policy = build_policy
            manager._build_trainer = lambda *, policy, iterations: trainer
            trainer.train_trajectory = train_trajectory

            manager._run_single_experiment(iterations=2, log_frequency=1)

            self.assertEqual(trainer_call_count, 2)
            action_table_path = Path(manager.action_table_paths[0])
            self.assertEqual(
                action_table_path.name,
                'actions_table_reservoir_instance1_policyseed112_h2_i2_'
                'constant_std3.0_noisy-update.csv',
            )
            with action_table_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.reader(handle))

        self.assertEqual([row[0] for row in rows[1:]], ['1', '2'])
        self.assertEqual(json.loads(rows[1][1]), {'release': [1, 3.0]})
        self.assertEqual(json.loads(rows[2][1]), {'release': [2, 3.0]})

    def test_accepts_constant_and_rejects_unknown_noise(self) -> None:
        ExperimentManager._validate_action_table_configuration(
            save_actions_table=True,
            noise_type='constant',
            output_folder='/tmp/output',
        )
        with self.assertRaisesRegex(ValueError, 'gradient2noise or constant'):
            ExperimentManager._validate_action_table_configuration(
                save_actions_table=True,
                noise_type='unsupported',
                output_folder='/tmp/output',
            )


if __name__ == '__main__':
    unittest.main()
