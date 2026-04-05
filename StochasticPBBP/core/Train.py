from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from StochasticPBBP.core.Logic import FuzzyLogic
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Noise import AdditiveNoise, AdditiveNoiseFactory
from StochasticPBBP.utils.Policies import GaussianPolicy

# from .Logic import FuzzyLogic
# from .Policies import GaussianPolicy
# from .Rollout import TorchRollout


class Train:
    """Policy optimization over contiguous horizon batches.

    `batch_size` now means the number of rollout steps used for one gradient
    update. The rollout horizon is partitioned into contiguous batches of at
    most `batch_size` steps.

    `batch_num` controls how many batches are sampled per training iteration:

    * `batch_size == horizon`, `batch_num == 1`:
      one full-horizon batch and one optimizer step per iteration
    * `batch_size < horizon`:
      partition the horizon and sample `batch_num` partitions uniformly
      (with replacement); each sampled partition produces one optimizer step

    Example:
        horizon=113, batch_size=23 -> partition sizes [23, 23, 23, 23, 21]
    """

    def __init__(self,
                 model: Any,
                 action_space: Optional[Any]=None,
                 policy: Optional[nn.Module]=None,
                 horizon: Optional[int]=None,
                 lr: float=1e-2,
                 hidden_sizes: Sequence[int]=(64, 64),
                 seed: int=0,
                 simulator: Optional[Any]=None,
                 additive_noise: Optional[AdditiveNoise]=None,
                 batch_size: Optional[int]=None,
                 batch_num: int=1) -> None:
        self.action_space = action_space
        self.hidden_sizes = tuple(hidden_sizes)
        self.seed = seed
        torch.manual_seed(seed)

        self.rollout = TorchRollout(model, horizon=horizon, logic=FuzzyLogic())
        self.rollout.cell.key.manual_seed(seed)
        self.batch_key = torch.Generator()
        self.batch_key.manual_seed(seed)
        self.simulator = simulator
        self.rollout.reset()
        self.default_additive_noise = self._resolve_additive_noise(additive_noise)
        self.default_batch_size = self._resolve_batch_size(batch_size)
        self.default_batch_num = self._validate_batch_num(batch_num)

        if policy is None:
            policy = GaussianPolicy(action_template=self.rollout.noop_actions)
        self.policy = policy
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    def _resolve_batch_size(self, batch_size: Optional[int]) -> int:
        # Default to one full-horizon batch per iteration.
        if batch_size is None:
            return int(self.rollout.horizon)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f'batch_size must be a positive integer, got {batch_size!r}.')
        horizon = int(self.rollout.horizon)
        if batch_size > horizon:
            raise ValueError(
                f'batch_size={batch_size} cannot be larger than horizon={horizon}.'
            )
        return batch_size

    def _resolve_additive_noise(self,
                                additive_noise: Optional[AdditiveNoise]) -> AdditiveNoise:
        if additive_noise is None:
            return AdditiveNoiseFactory.create(
                noise_type='constant',
                std=0.0,
                source=self.rollout,
            )
        return additive_noise

    @staticmethod
    def _validate_batch_num(batch_num: int) -> int:
        if not isinstance(batch_num, int) or batch_num < 1:
            raise ValueError(f'batch_num must be a positive integer, got {batch_num!r}.')
        return batch_num

    @staticmethod
    def _reduce_objective(objective: torch.Tensor) -> torch.Tensor:
        return objective.mean() if objective.ndim > 0 else objective

    @classmethod
    def _detach_structure(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: cls._detach_structure(v) for (k, v) in value.items()}
        if isinstance(value, list):
            return [cls._detach_structure(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._detach_structure(v) for v in value)
        return deepcopy(value)

    def _build_partitions(self, batch_size: int) -> List[Dict[str, int]]:
        horizon = int(self.rollout.horizon)
        partitions: List[Dict[str, int]] = []
        start_step = 0
        while start_step < horizon:
            steps = min(batch_size, horizon - start_step)
            partitions.append({
                'start_step': start_step,
                'steps': steps,
            })
            start_step += steps
        return partitions

    def _sample_partition_indices(self,
                                  *,
                                  num_partitions: int,
                                  batch_num: int) -> List[int]:
        if num_partitions == 1:
            return [0] * batch_num
        draws = torch.randint(
            low=0,
            high=num_partitions,
            size=(batch_num,),
            generator=self.batch_key,
        )
        return [int(index) for index in draws.tolist()]

    def _advance_to_batch_start(self,
                                *,
                                start_step: int,
                                iteration: int,
                                additive_noise: AdditiveNoise
                                ) -> Dict[str, Any]:
        if start_step == 0:
            return {
                'initial_subs': None,
                'model_params': None,
                'policy_state': None,
            }

        # Each sampled batch is anchored at a partition of a fresh horizon rollout.
        # We replay the prefix without gradients to recover the simulator state at
        # the sampled batch start under the current policy.
        with torch.no_grad():
            prefix_trace = self.rollout(
                policy=self.policy,
                steps=start_step,
                start_step=0,
                iteration=iteration,
                additive_noise=additive_noise,
            )
        return {
            'initial_subs': self._detach_structure(prefix_trace.final_subs),
            'model_params': self._detach_structure(prefix_trace.model_params),
            'policy_state': self._detach_structure(prefix_trace.policy_state),
        }

    def _run_training_batch(self,
                            *,  # enforce keyword arguments for clarity
                            initial_subs: Optional[Dict[str, Any]],
                            model_params: Optional[Dict[str, Any]],
                            policy_state: Any,
                            batch_steps: int,
                            start_step: int,
                            iteration: int,
                            additive_noise: AdditiveNoise) -> Dict[str, Any]:
        trace = self.rollout(
            policy=self.policy,
            initial_subs=initial_subs,
            model_params=model_params,
            policy_state=policy_state,
            steps=batch_steps,
            start_step=start_step,
            iteration=iteration,
            additive_noise=additive_noise,
        )
        objective = self._reduce_objective(trace.return_)
        loss = -objective

        loss.backward()
        self.optimizer.step()

        return {
            'objective': objective,
            'loss': loss,
            'trace': trace,
        }

    def train_trajectory(self,
                         iterations: int=10,
                         print_every: int=1,
                         batch_size: Optional[int]=None,
                         batch_num: Optional[int]=None,
                         batch: Optional[bool]=None,
                         additive_noise: Optional[AdditiveNoise]=None
                         ) -> Tuple[List[Dict[str, float]], nn.Module]:
        """Train the policy with sampled horizon batches.

        Args:
            iterations: Number of outer training iterations.
            print_every: Print metrics on iteration 1 and then every N iterations.
            batch_size: Number of rollout steps in one training batch. If omitted,
                defaults to the full horizon, which gives one full-horizon batch.
            batch_num: Number of partition batches to draw per iteration. Sampling
                is uniform over the horizon partitions and uses replacement.
            batch: Legacy compatibility argument; ignored.
            additive_noise: Action-noise object applied during rollout. Defaults
                to `NoAdditiveNoise` via the factory.
        """
        del batch
        history: List[Dict[str, float]] = []
        effective_batch_size = self.default_batch_size if batch_size is None else (
            self._resolve_batch_size(batch_size)
        )
        effective_batch_num = self.default_batch_num if batch_num is None else (
            self._validate_batch_num(batch_num)
        )
        effective_additive_noise = self.default_additive_noise if additive_noise is None else (
            self._resolve_additive_noise(additive_noise)
        )
        partitions = self._build_partitions(effective_batch_size)

        self.policy.train()

        for iteration in range(1, iterations + 1):
            sampled_indices = self._sample_partition_indices(
                num_partitions=len(partitions),
                batch_num=effective_batch_num,
            )

            for batch_index, partition_index in enumerate(sampled_indices, start=1):
                partition = partitions[partition_index]
                start_step = int(partition['start_step'])
                batch_steps = int(partition['steps'])
                batch_start = self._advance_to_batch_start(
                    start_step=start_step,
                    iteration=iteration,
                    additive_noise=effective_additive_noise,
                )
                self.optimizer.zero_grad(set_to_none=True)
                result = self._run_training_batch(
                    initial_subs=batch_start['initial_subs'],
                    model_params=batch_start['model_params'],
                    policy_state=batch_start['policy_state'],
                    batch_steps=batch_steps,
                    start_step=start_step,
                    iteration=iteration,
                    additive_noise=effective_additive_noise,
                )
                trace = result['trace']

                end_step = start_step + len(trace.rewards)
                display_step_start = start_step + 1 if len(trace.rewards) > 0 else start_step
                display_step_end = end_step

                metrics = {
                    'iteration': float(iteration),
                    'batch_index': float(batch_index),
                    'batch_num': float(effective_batch_num),
                    'partition_index': float(partition_index + 1),
                    'num_partitions': float(len(partitions)),
                    'batch_steps': float(batch_steps),
                    'step_start': float(display_step_start),
                    'step_end': float(display_step_end),
                    'return': float(result['objective'].detach()),
                    'loss': float(result['loss'].detach()),
                    'steps': float(len(trace.rewards)),
                    'final_subs': self._detach_structure(trace.final_subs),
                    # Legacy aliases kept for downstream scripts that still read chunk fields.
                    'chunk_index': float(batch_index),
                    'num_chunks': float(effective_batch_num),
                    'chunk_steps': float(batch_steps),
                }
                history.append(metrics)

                if print_every > 0 and (
                    iteration == 1 or iteration % print_every == 0 or iteration == iterations
                ):
                    print(
                        f"iter={iteration:4d} "
                        f"batch={batch_index:2d}/{effective_batch_num:2d} "
                        f"partition={partition_index + 1:2d}/{len(partitions):2d} "
                        f"range=[{display_step_start:3.0f},{display_step_end:3.0f}] "
                        f"return={metrics['return']:10.4f} "
                        f"loss={metrics['loss']:10.4f} "
                        f"steps={int(metrics['steps'])}"
                    )

        if isinstance(self.policy, GaussianPolicy):
            dist = self.policy.distribution()
            print(f'self.policy.mu: {dist.mean.detach()}')
            print(f'self.policy.std: {dist.stddev.detach()}')

        return history, self.policy

    def train_batch(self,
                    batch_size: int,
                    iterations: int=10,
                    print_every: int=1,
                    batch_num: int=1,
                    additive_noise: Optional[AdditiveNoise]=None
                    ) -> Tuple[List[Dict[str, float]], nn.Module]:
        return self.train_trajectory(
            iterations=iterations,
            print_every=print_every,
            batch_size=batch_size,
            batch_num=batch_num,
            additive_noise=additive_noise,
        )
