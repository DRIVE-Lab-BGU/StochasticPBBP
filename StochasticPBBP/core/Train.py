from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from .Logic import FuzzyLogic
from .Policies import GaussianPolicy
from .Rollout import TorchRollout


class Train:
    """Policy optimization with optional horizon chunking.

    If `batch_size > 1`, the rollout horizon is split into `batch_size` equal
    chunks as evenly as possible. Extra steps from the remainder are assigned to
    the earliest chunks. The trainer updates parameters after each chunk and
    continues from the last simulator state of the previous chunk.

    Example:
        horizon=113, batch_size=5 -> chunk sizes [23, 23, 23, 22, 22]
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
                 batch_size: int=1) -> None:
        self.action_space = action_space
        self.hidden_sizes = tuple(hidden_sizes)
        self.default_batch_size = self._validate_batch_size(batch_size)
        self.seed = seed
        torch.manual_seed(seed)

        self.rollout = TorchRollout(model, horizon=horizon, logic=FuzzyLogic())
        self.rollout.cell.key.manual_seed(seed)
        self.simulator = simulator
        self.rollout.reset()

        if policy is None:
            policy = GaussianPolicy(action_template=self.rollout.noop_actions)
        self.policy = policy
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f'batch_size must be a positive integer, got {batch_size!r}.')
        return batch_size

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

    def _build_chunk_sizes(self, batch_size: int) -> List[int]:
        horizon = int(self.rollout.horizon)
        if batch_size == 1:
            return [horizon]
        if batch_size > horizon:
            raise ValueError(
                f'batch_size={batch_size} cannot be larger than horizon={horizon} '
                'for chunked training.'
            )

        base_chunk = horizon // batch_size
        remainder = horizon % batch_size
        chunk_sizes = [base_chunk] * batch_size
        if remainder:
            for i in range(remainder):
                chunk_sizes[i] += 1
        return chunk_sizes

    def _run_training_chunk(self,
                            *, # enforce keyword arguments for clarity
                            initial_subs: Optional[Dict[str, Any]],
                            model_params: Optional[Dict[str, Any]],
                            policy_state: Any,
                            chunk_steps: int,
                            start_step: int) -> Dict[str, Any]:
        trace = self.rollout(
            policy=self.policy,
            initial_subs=initial_subs,
            model_params=model_params,
            policy_state=policy_state,
            steps=chunk_steps,
            start_step=start_step,
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
                         batch: Optional[bool]=None) -> List[Dict[str, float]]:
        del batch
        history: List[Dict[str, float]] = []
        effective_batch_size = self.default_batch_size if batch_size is None else (
            self._validate_batch_size(batch_size)
        )
        chunk_sizes = self._build_chunk_sizes(effective_batch_size)

        self.policy.train()

        for iteration in range(1, iterations + 1):
            current_subs: Optional[Dict[str, Any]] = None
            current_model_params: Optional[Dict[str, Any]] = None
            current_policy_state: Any = None
            # start_step is used for display purposes and passed to the rollout for potential use in time-based policies.
            start_step = 0

            for chunk_index, chunk_steps in enumerate(chunk_sizes, start=1):
                self.optimizer.zero_grad(set_to_none=True)
                result = self._run_training_chunk(
                    initial_subs=current_subs,
                    model_params=current_model_params,
                    policy_state=current_policy_state,
                    chunk_steps=chunk_steps,
                    start_step=start_step,
                )
                trace = result['trace']

                current_subs = self._detach_structure(trace.final_subs)
                current_model_params = self._detach_structure(trace.model_params)
                current_policy_state = self._detach_structure(trace.policy_state)
                end_step = start_step + len(trace.rewards)
                display_step_start = start_step + 1 if len(trace.rewards) > 0 else start_step
                display_step_end = end_step

                metrics = {
                    'iteration': float(iteration),
                    'chunk_index': float(chunk_index),
                    'num_chunks': float(len(chunk_sizes)),
                    'chunk_steps': float(chunk_steps),
                    'step_start': float(display_step_start),
                    'step_end': float(display_step_end),
                    'return': float(result['objective'].detach()),
                    'loss': float(result['loss'].detach()),
                    'steps': float(len(trace.rewards)),
                    'final_subs': current_subs,
                }
                history.append(metrics)

                if print_every > 0 and (
                    iteration == 1 or iteration % print_every == 0 or iteration == iterations
                ):
                    print(
                        f"iter={iteration:4d} "
                        f"chunk={chunk_index:2d}/{len(chunk_sizes):2d} "
                        f"range=[{display_step_start:3.0f},{display_step_end:3.0f}] "
                        f"return={metrics['return']:10.4f} "
                        f"loss={metrics['loss']:10.4f} "
                        f"steps={int(metrics['steps'])}"
                    )

                start_step = end_step

        if isinstance(self.policy, GaussianPolicy):
            dist = self.policy.distribution()
            print(f'self.policy.mu: {dist.mean.detach()}')
            print(f'self.policy.std: {dist.stddev.detach()}')

        return history, self.policy

    def train_batch(self,
                    batch_size: int,
                    iterations: int=10,
                    print_every: int=1) -> List[Dict[str, float]]:
        return self.train_trajectory(
            iterations=iterations,
            print_every=print_every,
            batch_size=batch_size,
        )
