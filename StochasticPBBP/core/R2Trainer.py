from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn

from StochasticPBBP.core.Train import Train
from StochasticPBBP.utils.Noise import AdditiveNoise, AdditiveNoiseFactory


class R2TrainingPhase(str, Enum):
    BOOTSTRAP = 'bootstrap' # initial phase with no updates, for profiling the initial policy
    UPDATE = 'update' # standard policy-update phase with gradient steps and optimizer updates
    ANALYSIS = 'analysis' # post-update analysis phase with a fresh rollout and no gradient steps
    PROFILE_REFRESH = 'profile_refresh' # dedicated phase for refreshing the R2 noise profile from the analysis rollout, if applicable


class R2Trainer(Train):
    """Dedicated trainer scaffold for two-pass R2-style training.

    The current implementation keeps the standard policy-update pass separate
    from the post-update analysis pass and reserves a dedicated hook for future
    R2 noise-profile computation.
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
                 analysis_additive_noise: Optional[AdditiveNoise]=None,
                 logic: Optional[object]=None,
                 batch_size: Optional[int]=None,
                 batch_num: int=1) -> None:
        super().__init__(
            model=model,
            action_space=action_space,
            policy=policy,
            horizon=horizon,
            lr=lr,
            hidden_sizes=hidden_sizes,
            seed=seed,
            simulator=simulator,
            additive_noise=additive_noise,
            logic=logic,
            batch_size=batch_size,
            batch_num=batch_num,
        )
        self._validate_r2_batching(
            batch_size=self.default_batch_size,
            batch_num=self.default_batch_num,
        )
        # the noise to check during the analysis phase.
        self.analysis_additive_noise = self._resolve_analysis_additive_noise(
            analysis_additive_noise
        )
        # By default, start with a no-noise profile for the update phase, which can be refreshed later based on the analysis rollout.
        self.current_phase = R2TrainingPhase.BOOTSTRAP
        self.r2_profile: Optional[Any] = None

    def _validate_r2_batching(self, *, batch_size: int, batch_num: int) -> None:
        horizon = int(self.rollout.horizon)
        if batch_size != horizon:
            raise ValueError(
                'R2Trainer currently requires full-horizon updates: '
                f'batch_size must equal horizon={horizon}, got {batch_size}.'
            )
        if batch_num != 1:
            raise ValueError(
                'R2Trainer currently supports exactly one optimizer update per '
                f'iteration, got batch_num={batch_num}.'
            )

    def _resolve_analysis_additive_noise(
        self,
        additive_noise: Optional[AdditiveNoise],
    ) -> AdditiveNoise:
        if additive_noise is None:
            return AdditiveNoiseFactory.create(
                noise_type='constant',
                std=0.0,
                source=self.rollout,
            )
        return self._resolve_additive_noise(additive_noise)

    def _set_phase(self, phase: R2TrainingPhase) -> None:
        self.current_phase = phase

    def _run_update_phase(
        self,
        *,
        iteration: int,
        additive_noise: AdditiveNoise,
    ) -> Dict[str, Any]:
        """ is the update phase result supposed to be from the same trajectory used for the update,
          or from a fresh trajectory with the updated policy """
        self._set_phase(R2TrainingPhase.UPDATE)
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        result = self._run_training_batch(
            initial_subs=None,
            model_params=None,
            policy_state=None,
            batch_steps=int(self.rollout.horizon),
            start_step=0,
            iteration=iteration,
            additive_noise=additive_noise,
        )
        return {
            'phase': self.current_phase.value,
            'objective': result['objective'],
            'loss': result['loss'],
            'trace': result['trace'],
        }

    def _run_analysis_phase(
        self,
        *,
        iteration: int,
        additive_noise: AdditiveNoise,
    ) -> Dict[str, Any]:
        self._set_phase(R2TrainingPhase.ANALYSIS)
        was_training = self.policy.training
        self.policy.eval()
        try:
            trace = self.rollout(
                policy=self.policy,
                steps=int(self.rollout.horizon),
                start_step=0,
                iteration=iteration,
                additive_noise=additive_noise,
            )
        finally:
            self.policy.train(was_training)
        objective = self._reduce_objective(trace.return_)
        return {
            'phase': self.current_phase.value,
            'objective': objective,
            'trace': trace,
        }

    def refresh_noise_profile(
        self,
        *,
        iteration: int,
        update_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        additive_noise: Optional[AdditiveNoise]=None,
    ) -> Optional[Any]:
        del iteration, update_result
        self._set_phase(R2TrainingPhase.PROFILE_REFRESH)
        if additive_noise is None:
            return None
        refresh = getattr(additive_noise, 'refresh_from_analysis_trace', None)
        if not callable(refresh):
            return None
        trace = analysis_result.get('trace')
        if trace is None:
            return None
        return refresh(
            trace=trace,
            objective=analysis_result.get('objective'),
        )

    def train_iteration(
        self,
        *,
        iteration: int,
        additive_noise: Optional[AdditiveNoise]=None,
        analysis_additive_noise: Optional[AdditiveNoise]=None,
    ) -> Dict[str, Any]:
        effective_additive_noise = self.default_additive_noise if additive_noise is None else (
            self._resolve_additive_noise(additive_noise)
        )
        effective_analysis_noise = (
            self.analysis_additive_noise
            if analysis_additive_noise is None else
            self._resolve_analysis_additive_noise(analysis_additive_noise)
        )

        if self.r2_profile is None:
            self._set_phase(R2TrainingPhase.BOOTSTRAP)

        update_result = self._run_update_phase(
            iteration=iteration,
            additive_noise=effective_additive_noise,
        )
        analysis_result = self._run_analysis_phase(
            iteration=iteration,
            additive_noise=effective_analysis_noise,
        )
        self.r2_profile = self.refresh_noise_profile(
            iteration=iteration,
            update_result=update_result,
            analysis_result=analysis_result,
            additive_noise=effective_additive_noise,
        )
        return {
            'iteration': iteration,
            'update': update_result,
            'analysis': analysis_result,
            'profile': self.r2_profile,
            'phase': self.current_phase.value,
        }

    def train_trajectory(self,
                         iterations: int=10,
                         print_every: int=1,
                         batch_size: Optional[int]=None,
                         batch_num: Optional[int]=None,
                         batch: Optional[bool]=None,
                         additive_noise: Optional[AdditiveNoise]=None,
                         analysis_additive_noise: Optional[AdditiveNoise]=None
                         ) -> Tuple[List[Dict[str, float]], nn.Module]:
        del batch
        effective_batch_size = self.default_batch_size if batch_size is None else (
            self._resolve_batch_size(batch_size)
        )
        effective_batch_num = self.default_batch_num if batch_num is None else (
            self._validate_batch_num(batch_num)
        )
        self._validate_r2_batching(
            batch_size=effective_batch_size,
            batch_num=effective_batch_num,
        )

        history: List[Dict[str, float]] = []
        for iteration in range(1, iterations + 1):
            result = self.train_iteration(
                iteration=iteration,
                additive_noise=additive_noise,
                analysis_additive_noise=analysis_additive_noise,
            )
            update_result = result['update']
            analysis_result = result['analysis']
            metrics = {
                'iteration': float(iteration),
                'update_return': float(update_result['objective'].detach()),
                'update_loss': float(update_result['loss'].detach()),
                'update_steps': float(len(update_result['trace'].rewards)),
                'analysis_return': float(analysis_result['objective'].detach()),
                'analysis_steps': float(len(analysis_result['trace'].rewards)),
            }
            history.append(metrics)

            if print_every > 0 and (
                iteration == 1 or iteration % print_every == 0 or iteration == iterations
            ):
                print(
                    f"iter={iteration:4d} "
                    f"update_return={metrics['update_return']:10.4f} "
                    f"update_loss={metrics['update_loss']:10.4f} "
                    f"analysis_return={metrics['analysis_return']:10.4f} "
                    f"steps={int(metrics['update_steps'])}"
                )

        return history, self.policy

    def train_batch(self,
                    batch_size: int,
                    iterations: int=10,
                    print_every: int=1,
                    batch_num: int=1,
                    additive_noise: Optional[AdditiveNoise]=None,
                    analysis_additive_noise: Optional[AdditiveNoise]=None
                    ) -> Tuple[List[Dict[str, float]], nn.Module]:
        return self.train_trajectory(
            iterations=iterations,
            print_every=print_every,
            batch_size=batch_size,
            batch_num=batch_num,
            additive_noise=additive_noise,
            analysis_additive_noise=analysis_additive_noise,
        )
