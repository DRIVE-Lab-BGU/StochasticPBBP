from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, TypedDict

import torch

from StochasticPBBP.core.Compiler import TorchRDDLCompiler

if TYPE_CHECKING:
    from StochasticPBBP.core.Rollout import RolloutTrace


class NoiseInfo(TypedDict):
    type: str    # noise type
    value: float # std value
    final: float # final std value

@dataclass(frozen=True)
class NoiseContext:
    """Execution context passed into additive-noise objects.

    This keeps the noise API extensible for scheduled noise, model-based noise,
    and rollout-aware perturbations without repeatedly changing method
    signatures.
    """

    step: Optional[int]=None
    iteration: Optional[int]=None
    subs: Optional[Dict[str, Any]]=None
    observation: Optional[Dict[str, Any]]=None
    model: Optional[Any]=None
    model_params: Optional[Dict[str, Any]]=None
    policy_state: Optional[Any]=None


class AdditiveNoise(ABC):
    """Base class for additive action-noise processes.

    Subclasses receive the prepared lifted action dictionary and return a new
    action dictionary with additive perturbations applied to floating-point
    tensors. Non-tensor and non-floating tensors are passed through unchanged.
    """

    def __init__(self, seed: Optional[int]=None) -> None:
        if seed is None:
            seed = time.time_ns()
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self,
                 actions: Optional[Dict[str, Any]],
                 *,
                 context: Optional[NoiseContext]=None) -> Optional[Dict[str, Any]]:
        if actions is None:
            return None
        local_context = NoiseContext() if context is None else context

        noisy_actions: Dict[str, Any] = {}
        for (name, value) in actions.items():
            noisy_actions[name] = self.apply_to_value(
                name=name,
                value=value,
                context=local_context,
            )
        return noisy_actions

    def apply_to_value(self,
                       *,
                       name: str,
                       value: Any,
                       context: NoiseContext) -> Any:
        del name
        if isinstance(value, torch.Tensor):
            cloned = value.clone()
            if cloned.dtype.is_floating_point:
                noise = self.sample_like(cloned, context=context)
                return cloned + noise
            return cloned
        if hasattr(value, 'copy'):
            return value.copy()
        return deepcopy(value)

    @abstractmethod
    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        """Return an additive noise tensor compatible with `reference`."""


class NoAdditiveNoise(AdditiveNoise):
    """Additive-noise implementation that leaves floating actions unchanged."""

    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        del context
        return torch.zeros_like(reference)


class ConstantAdditiveNoise(AdditiveNoise):
    """Additive Gaussian noise with a constant standard deviation."""

    def __init__(self, std: float, seed: Optional[int]=None) -> None:
        if std < 0:
            raise ValueError(f'std must be non-negative, got {std!r}.')
        self.std = float(std)
        if seed is None:
            seed = time.time_ns()
        self.g = torch.Generator().manual_seed(seed)

    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        del context
        if self.std == 0.0:
            return torch.zeros_like(reference)
        return torch.empty_like(reference).normal_(generator=self.g) * self.std


class LinearDecayAdditiveNoise(ConstantAdditiveNoise):
    """Additive Gaussian noise with linearly decaying standard deviation."""

    def __init__(self,
                 start_std: float,
                 end_std: float,
                 num_iterations: int) -> None:
        if start_std < 0:
            raise ValueError(f'start_std must be non-negative, got {start_std!r}.')
        if end_std < 0:
            raise ValueError(f'end_std must be non-negative, got {end_std!r}.')
        if num_iterations < 1:
            raise ValueError(
                f'num_iterations must be a positive integer, got {num_iterations!r}.'
            )
        self.start_std = float(start_std)
        self.end_std = float(end_std)
        self.num_iterations = int(num_iterations)
        super().__init__(std=start_std)

    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        self.std = self._std_at(context.iteration)
        return super().sample_like(reference, context=context)

    def _std_at(self, step: Optional[int]) -> float:
        if step is None:
            return self.start_std
        if self.num_iterations == 1:
            return self.end_std
        clamped_step = min(max(int(step), 0), self.num_iterations - 1)
        progress = clamped_step / float(self.num_iterations - 1)
        return self.start_std + progress * (self.end_std - self.start_std)


TensorProfile = List[Dict[str, torch.Tensor]]
ScalarProfile = List[Dict[str, float]]
ActionLocation = Tuple[int, str]
ActionReference = Tuple[int, str, torch.Tensor]


class R2GradientAdditiveNoise(AdditiveNoise):
    """Additive Gaussian noise driven by trajectory-level action gradients.

    After an analysis rollout, the class computes a first-order sensitivity
    score for every floating action tensor:

    1. collect ``dJ / da_t`` for all differentiable action tensors
    2. reduce each action tensor to a scalar score via its gradient norm
    3. aggregate action scores into one score per timestep
    4. normalize timestep scores with a trajectory quantile
    5. map the normalized score to a bounded ``sigma_t``

    The scalar ``sigma_t`` is then broadcast over every floating action tensor
    at timestep ``t`` during future rollout calls.
    """

    def __init__(self,
                 *,
                 scale: float=1.0,
                 eps: float=1e-6,
                 min_std: float=0.0,
                 max_std: Optional[float] = 5.0,
                 alpha: float=1.0,
                 norm_scope: str='global',
                 step_score_aggregate: str='mean',
                 normalization_quantile: float=0.95,
                 fallback_noise: Optional[AdditiveNoise]=None) -> None:
        super().__init__()
        if scale < 0:
            raise ValueError(f'scale must be non-negative, got {scale!r}.')
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps!r}.')
        if min_std < 0:
            raise ValueError(f'min_std must be non-negative, got {min_std!r}.')
        if max_std is None or not torch.isfinite(torch.tensor(max_std)):
            raise ValueError(
                'R2GradientAdditiveNoise requires a finite max_std for the '
                'bounded timestep-noise rule.'
            )
        if max_std < min_std:
            raise ValueError(
                f'max_std must be greater than or equal to min_std, got {max_std!r} < {min_std!r}.'
            )
        if alpha <= 0:
            raise ValueError(f'alpha must be positive, got {alpha!r}.')
        normalized_scope = norm_scope.strip().lower()
        if normalized_scope not in {'global', 'step'}:
            raise ValueError(
                f'norm_scope must be "global" or "step", got {norm_scope!r}.'
            )
        normalized_step_aggregate = step_score_aggregate.strip().lower()
        if normalized_step_aggregate not in {'mean', 'sum'}:
            raise ValueError(
                'step_score_aggregate must be "mean" or "sum", '
                f'got {step_score_aggregate!r}.'
            )
        if not 0.0 < normalization_quantile <= 1.0:
            raise ValueError(
                'normalization_quantile must lie in (0, 1], '
                f'got {normalization_quantile!r}.'
            )

        self.scale = float(scale)
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.alpha = float(alpha)
        # Kept for API compatibility with existing call sites. The current
        # quantile-based normalization operates on timestep scores across the
        # full trajectory regardless of this flag.
        self.norm_scope = normalized_scope
        # ignored by the gradient-only implementation.
        self.step_score_aggregate = normalized_step_aggregate
        self.normalization_quantile = float(normalization_quantile)
        self.fallback_noise = NoAdditiveNoise() if fallback_noise is None else fallback_noise
        self._std_profile: TensorProfile = []
        self.last_gradients: TensorProfile = []
        self.last_action_scores: ScalarProfile = []
        self.last_step_scores: List[float] = []
        self.last_gradient_inf_norm: float = 0.0
        self.last_score_quantile: float = 0.0

    @property
    def has_profile(self) -> bool:
        return bool(self._std_profile)

    @property
    def profile(self) -> TensorProfile:
        return self._clone_profile(self._std_profile)

    def clear_profile(self) -> None:
        self._std_profile = []
        self.last_gradients = []
        self.last_action_scores = []
        self.last_step_scores = []
        self.last_gradient_inf_norm = 0.0
        self.last_score_quantile = 0.0

    def set_profile(self, profile: Sequence[Dict[str, torch.Tensor]]) -> None:
        self._std_profile = self._clone_profile(profile)

    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        return self.fallback_noise.sample_like(reference, context=context)

    def apply_to_value(self,
                       *,
                       name: str,
                       value: Any,
                       context: NoiseContext) -> Any:
        if not isinstance(value, torch.Tensor):
            if hasattr(value, 'copy'):
                return value.copy()
            return deepcopy(value)

        cloned = value.clone()
        if not cloned.dtype.is_floating_point:
            return cloned

        std_tensor = self._resolve_std_tensor(
            name=name,
            reference=cloned,
            context=context,
        )
        if std_tensor is None:
            return self.fallback_noise.apply_to_value(
                name=name,
                value=cloned,
                context=context,
            )
        if torch.count_nonzero(std_tensor).item() == 0:
            return cloned
        noise = torch.empty_like(cloned).normal_(generator=self.g)
        return cloned + noise * std_tensor

    def refresh_from_analysis_trace(self,
                                    *,
                                    trace: RolloutTrace,
                                    objective: Optional[torch.Tensor]=None) -> TensorProfile:
        objective_tensor = trace.return_ if objective is None else objective
        reduced_objective = objective_tensor.mean() if objective_tensor.ndim > 0 else objective_tensor

        references, differentiable_inputs, differentiable_locations = self._collect_action_references(
            trace=trace,
        )
        grad_map = self._collect_action_gradients(
            references=references,
            differentiable_inputs=differentiable_inputs,
            differentiable_locations=differentiable_locations,
            objective=reduced_objective,
        )
        # calculate the sensitivity scores for each action and step -> s_t in our method
        action_score_map, step_scores = self._build_step_scores(
            references=references,
            grad_map=grad_map,
        ) 
        profile = self._build_std_profile(
            references=references,
            step_scores=step_scores,
        ) # map the step scores to std values and broadcast to action shapes -> sigma_t in our method
        self.set_profile(profile)
        self.last_gradients = self._materialize_gradient_profile(
            references=references,
            grad_map=grad_map,
        )

        self.last_action_scores = self._materialize_scalar_profile(
            references=references,
            scalar_map=action_score_map,
        )
        self.last_step_scores = list(step_scores)
        self.last_gradient_inf_norm = self._gradient_inf_norm(list(grad_map.values()))
        self.last_score_quantile = self._score_quantile(step_scores)
        return self.profile

    def _resolve_std_tensor(self,
                            *,
                            name: str,
                            reference: torch.Tensor,
                            context: NoiseContext) -> Optional[torch.Tensor]:
        if context.step is None:
            return None
        step_index = int(context.step)
        if step_index < 0 or step_index >= len(self._std_profile):
            return None
        step_profile = self._std_profile[step_index]
        if name not in step_profile:
            return None
        std_tensor = step_profile[name]
        if tuple(std_tensor.shape) != tuple(reference.shape):
            raise ValueError(
                f'R2 std profile for <{name}> must have shape {tuple(reference.shape)}, '
                f'got {tuple(std_tensor.shape)}.'
            )
        return std_tensor.to(dtype=reference.dtype, device=reference.device)

    def _collect_action_references(
        self,
        *,
        trace: RolloutTrace,
    ) -> Tuple[List[ActionReference], List[torch.Tensor], List[ActionLocation]]:
        references: List[ActionReference] = []
        differentiable_inputs: List[torch.Tensor] = []
        differentiable_locations: List[ActionLocation] = []

        for step_index, step_actions in enumerate(trace.actions):
            for name, value in step_actions.items():
                if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                    continue
                references.append((step_index, name, value))
                if value.requires_grad:
                    differentiable_inputs.append(value)
                    differentiable_locations.append((step_index, name))

        return references, differentiable_inputs, differentiable_locations

    def _collect_action_gradients(
        self,
        *,
        references: Sequence[ActionReference],
        differentiable_inputs: Sequence[torch.Tensor],
        differentiable_locations: Sequence[ActionLocation],
        objective: torch.Tensor,
    ) -> Dict[ActionLocation, torch.Tensor]:
        grad_map: Dict[ActionLocation, torch.Tensor] = {}

        if differentiable_inputs and (objective.requires_grad or objective.grad_fn is not None):
            gradients = torch.autograd.grad(
                outputs=objective,
                inputs=list(differentiable_inputs),
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
            for location, reference, gradient in zip(
                differentiable_locations,
                differentiable_inputs,
                gradients,
            ):
                grad_map[location] = (
                    torch.zeros_like(reference) if gradient is None else gradient.detach().clone()
                )

        for step_index, name, value in references:
            location = (step_index, name)
            if location not in grad_map:
                grad_map[location] = torch.zeros_like(value)

        return grad_map

    def _build_step_scores(
        self,
        *,
        references: Sequence[ActionReference],
        grad_map: Dict[ActionLocation, torch.Tensor],
    ) -> Tuple[Dict[ActionLocation, float], List[float]]:
        ''' Compute a scalar sensitivity score for each action(time step)tensor and aggregate(in case of multiple actions) into step scores.'''
        if not references:
            return {}, []

        num_steps = max(step_index for step_index, _, _ in references) + 1
        per_step_action_scores: List[List[float]] = [[] for _ in range(num_steps)]
        action_score_map: Dict[ActionLocation, float] = {}

        for step_index, name, _ in references:
            location = (step_index, name)
            score = self._gradient_tensor_norm(grad_map[location])
            action_score_map[location] = score
            per_step_action_scores[step_index].append(score)

        step_scores = [
            self._aggregate_step_score(action_scores)
            for action_scores in per_step_action_scores
        ]
        return action_score_map, step_scores

    def _aggregate_step_score(self, action_scores: Sequence[float]) -> float:
        '''Aggregate multiple action scores into a single step score for normalization and noise mapping.'''
        if not action_scores:
            return 0.0
        if self.step_score_aggregate == 'sum':
            return float(sum(action_scores))
        return float(sum(action_scores) / len(action_scores))

    def _build_std_profile(
        self,
        *,
        references: Sequence[ActionReference],
        step_scores: Sequence[float],
    ) -> TensorProfile:
        ''' create a noise profile by mapping step scores to std values and broadcasting to action shapes.'''
        if not references:
            return []

        num_steps = max(step_index for step_index, _, _ in references) + 1
        profile: TensorProfile = [{} for _ in range(num_steps)]
        score_quantile = self._score_quantile(step_scores)

        for step_index, name, reference in references:
            sigma = self._sigma_from_step_score(
                step_score=step_scores[step_index],
                score_quantile=score_quantile,
            )
            profile[step_index][name] = torch.full_like(reference, fill_value=sigma)

        return profile

    def _sigma_from_step_score(self,
                               *,
                               step_score: float,
                               score_quantile: float) -> float:
        normalized_score = step_score / (score_quantile + self.eps)
        normalized_score = max(0.0, min(1.0, normalized_score))
        complement = (1.0 - normalized_score) ** self.alpha
        #sigma = self.min_std + (self.max_std - self.min_std) * complement
        sigma = self.max_std + (self.min_std - self.max_std) * complement
        if self.scale != 1.0:
            sigma = self.min_std + self.scale * (sigma - self.min_std)
        return max(self.min_std, min(self.max_std, float(sigma)))

    def _score_quantile(self, step_scores: Sequence[float]) -> float:
        ''' return the specified quantile of the step scores for normalization, treating NaNs and Infs as zeros.'''
        if not step_scores:
            return 0.0
        scores = torch.as_tensor(step_scores, dtype=torch.float64)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return float(torch.quantile(scores, self.normalization_quantile).item())

    def _materialize_gradient_profile(
        self,
        *,
        references: Sequence[ActionReference],
        grad_map: Dict[ActionLocation, torch.Tensor],
    ) -> TensorProfile:
        if not references:
            return []
        num_steps = max(step_index for step_index, _, _ in references) + 1
        profile: TensorProfile = [{} for _ in range(num_steps)]
        for step_index, name, _ in references:
            profile[step_index][name] = grad_map[(step_index, name)].clone()
        return profile

    def _materialize_scalar_profile(
        self,
        *,
        references: Sequence[ActionReference],
        scalar_map: Dict[ActionLocation, float],
    ) -> ScalarProfile:
        if not references:
            return []
        num_steps = max(step_index for step_index, _, _ in references) + 1
        profile: ScalarProfile = [{} for _ in range(num_steps)]
        for step_index, name, _ in references:
            profile[step_index][name] = float(scalar_map[(step_index, name)])
        return profile

    @staticmethod
    def _gradient_tensor_norm(gradient: torch.Tensor) -> float:
        ''''Compute the L2 norm of a gradient tensor, treating NaNs and Infs as zeros.'''
        if gradient.numel() == 0:
            return 0.0
        clean_gradient = torch.nan_to_num(
            gradient.detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return float(torch.linalg.vector_norm(clean_gradient.reshape(-1)).item())

    @staticmethod
    def _gradient_inf_norm(gradients: Sequence[torch.Tensor]) -> float:
        if not gradients:
            return 0.0
        max_values = [
            torch.nan_to_num(
                gradient.detach().abs(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).max()
            for gradient in gradients
            if gradient.numel() > 0
        ]
        if not max_values:
            return 0.0
        return float(torch.stack(max_values).max().item())

    @staticmethod
    def _clone_profile(profile: Sequence[Dict[str, torch.Tensor]]) -> TensorProfile:
        return [
            {
                name: tensor.clone()
                for name, tensor in step_profile.items()
            }
            for step_profile in profile
        ]


class AdditiveNoiseFactory:
    """Factory for additive-noise helpers derived from rollout or model metadata."""

    @classmethod
    def create(cls,
               *,
               noise_type: str='constant',
               std: float=0.0,
               start_std: Optional[float]=None,
               end_std: Optional[float]=None,
               num_iterations: Optional[int]=None,
               cpf_name: Optional[str]=None,
               model: Optional[Any]=None,
               wrt: Optional[list[str]]=None,
               noise_scale: float=1.0,
               source: Any,
               alpha: float=1.0) -> AdditiveNoise:
        normalized_type = noise_type.strip().lower()
        resolved_model = cls._extract_rddl_model(source) if model is None else model

        if normalized_type == 'constant':
            if std == 0.0:
                return NoAdditiveNoise()
            return ConstantAdditiveNoise(std=std)
        ### here #### 
        if normalized_type == "gradient2noise":
            r2_noise = R2GradientAdditiveNoise(
                scale=1.0,
                eps=1e-6,
                min_std=1,
                max_std=std,
                alpha=alpha,# for linear 1.0
                norm_scope='global',
                step_score_aggregate='mean',
                normalization_quantile=0.95,
            )
            return r2_noise

        if normalized_type in {'linear_decay', 'linear-decay', 'decay'}:
            if start_std is None or end_std is None or num_iterations is None:
                raise ValueError(
                    'start_std, end_std, and num_iterations must be provided for '
                    'linear decay noise.'
                )
            if start_std == 0.0 and end_std == 0.0:
                return NoAdditiveNoise()
            return LinearDecayAdditiveNoise(
                start_std=start_std,
                end_std=end_std,
                num_iterations=num_iterations,
            )

        else:
            raise ValueError(
                f'Unsupported noise_type={noise_type!r}. '
                'Supported types are "constant", "linear_decay", "jacobian", '
                'and "gradient_norm".'
            )

    @staticmethod
    def _require_cpf_name(cpf_name: Optional[str]) -> str:
        if not cpf_name:
            raise ValueError('cpf_name must be provided for Jacobian-based noise.')
        return cpf_name

    @staticmethod
    def _extract_rddl_model(source: Any) -> Any:
        if hasattr(source, 'variable_types') and hasattr(source, 'action_ranges'):
            return source
        if hasattr(source, 'rddl'):
            return getattr(source, 'rddl')
        if hasattr(source, 'cell') and hasattr(source.cell, 'rddl'):
            return source.cell.rddl
        raise ValueError(
            'Could not extract RDDL model from source. '
            'Pass a rollout or model object with an rddl attribute.'
        )
