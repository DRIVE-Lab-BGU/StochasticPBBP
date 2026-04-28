from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from StochasticPBBP.core.Rollout import RolloutTrace
from StochasticPBBP.utils.Noise import AdditiveNoise, NoiseContext, NoAdditiveNoise

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
                 max_std: Optional[float]=5.0,
                 alpha: float=1.0,
                 norm_scope: str='global',
                 curvature_weight: float=0.0,
                 curvature_reduce: str='mean_abs',
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
        if curvature_weight < 0:
            raise ValueError(
                f'curvature_weight must be non-negative, got {curvature_weight!r}.'
            )
        normalized_scope = norm_scope.strip().lower()
        if normalized_scope not in {'global', 'step'}:
            raise ValueError(
                f'norm_scope must be "global" or "step", got {norm_scope!r}.'
            )
        normalized_curvature_reduce = curvature_reduce.strip().lower()
        if normalized_curvature_reduce not in {'mean_abs', 'norm'}:
            raise ValueError(
                'curvature_reduce must be "mean_abs" or "norm", '
                f'got {curvature_reduce!r}.'
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
        # Curvature arguments are accepted for backward compatibility but are
        # ignored by the gradient-only implementation.
        self.curvature_weight = float(curvature_weight)
        self.curvature_reduce = normalized_curvature_reduce
        self.step_score_aggregate = normalized_step_aggregate
        self.normalization_quantile = float(normalization_quantile)
        self.fallback_noise = NoAdditiveNoise() if fallback_noise is None else fallback_noise
        self._std_profile: TensorProfile = []
        self.last_gradients: TensorProfile = []
        self.last_curvatures: ScalarProfile = []
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
        self.last_curvatures = []
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
        generator = self._get_generator(cloned.device)
        noise = torch.empty_like(cloned).normal_(generator=generator)
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
        action_score_map, step_scores = self._build_step_scores(
            references=references,
            grad_map=grad_map,
        )
        profile = self._build_std_profile(
            references=references,
            step_scores=step_scores,
        )
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
        sigma = self.min_std + (self.max_std - self.min_std) * complement
        if self.scale != 1.0:
            sigma = self.min_std + self.scale * (sigma - self.min_std)
        return max(self.min_std, min(self.max_std, float(sigma)))

    def _score_quantile(self, step_scores: Sequence[float]) -> float:
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
