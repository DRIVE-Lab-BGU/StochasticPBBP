from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from StochasticPBBP.core.Rollout import RolloutTrace
from StochasticPBBP.utils.Noise import AdditiveNoise, NoiseContext, NoAdditiveNoise

TensorProfile = List[Dict[str, torch.Tensor]]


class R2GradientAdditiveNoise(AdditiveNoise):
    """Additive Gaussian noise driven by action sensitivities dJ/da.

    After an analysis rollout, the class builds a per-step, per-action standard
    deviation profile from the cumulative-return gradient with respect to the
    executed action tensors:

        r_i = |g_i| / (||g||_inf + eps)
        sigma_i = min_std + (max_std - min_std) * (1 - r_i)^alpha

    where `g` is the chosen action-gradient collection. Smaller absolute
    gradients therefore receive larger noise, while the resulting std tensors
    remain bounded between `min_std` and `max_std`. Those std tensors are then
    used in subsequent rollout calls through `context.step`.
    """

    def __init__(self,
                 *,
                 scale: float=1.0,
                 eps: float=1e-6,
                 min_std: float=0.0,
                 max_std: Optional[float]=None,
                 alpha: float=1.0,
                 norm_scope: str='global',
                 fallback_noise: Optional[AdditiveNoise]=None) -> None:
        if scale < 0:
            raise ValueError(f'scale must be non-negative, got {scale!r}.')
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps!r}.')
        if min_std < 0:
            raise ValueError(f'min_std must be non-negative, got {min_std!r}.')
        if max_std is None or not math.isfinite(max_std):
            raise ValueError(
                'R2GradientAdditiveNoise requires a finite max_std for the '
                'bounded complement-of-normalized-gradient noise rule.'
            )
        if max_std < min_std:
            raise ValueError(
                f'max_std must be greater than or equal to min_std, got {max_std!r} < {min_std!r}.'
            )
        if alpha <= 0:
            raise ValueError(f'alpha must be positive, got {alpha!r}.')
        #
        normalized_scope = norm_scope.strip().lower()
        if normalized_scope not in {'global', 'step'}:
            raise ValueError(
                f'norm_scope must be "global" or "step", got {norm_scope!r}.'
            )

        self.scale = float(scale)
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.alpha = float(alpha)
        self.norm_scope = normalized_scope
        self.fallback_noise = NoAdditiveNoise() if fallback_noise is None else fallback_noise
        self._std_profile: TensorProfile = []
        self.last_gradients: TensorProfile = []
        self.last_gradient_inf_norm: float = 0.0

    @property
    def has_profile(self) -> bool:
        return bool(self._std_profile)

    @property
    def profile(self) -> TensorProfile:
        return self._clone_profile(self._std_profile)

    def clear_profile(self) -> None:
        self._std_profile = []
        self.last_gradients = []
        self.last_gradient_inf_norm = 0.0

    def set_profile(self, profile: Sequence[Dict[str, torch.Tensor]]) -> None:
        self._std_profile = self._clone_profile(profile)

    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        del context
        return self.fallback_noise.sample_like(reference, context=NoiseContext())

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
            fallback = self.fallback_noise.apply_to_value(
                name=name,
                value=cloned,
                context=context,
            )
            return fallback
        if torch.count_nonzero(std_tensor).item() == 0:
            return cloned
        return cloned + torch.randn_like(cloned) * std_tensor

    def refresh_from_analysis_trace(self,
                                    *,
                                    trace: RolloutTrace,
                                    objective: Optional[torch.Tensor]=None) -> TensorProfile:
        objective_tensor = trace.return_ if objective is None else objective
        reduced_objective = objective_tensor.mean() if objective_tensor.ndim > 0 else objective_tensor

        references, grad_map = self._collect_action_gradients(
            trace=trace,
            objective=reduced_objective,
        )
        profile = self._build_std_profile(
            references=references,
            grad_map=grad_map,
        )
        self.set_profile(profile)
        self.last_gradients = self._materialize_gradient_profile(
            references=references,
            grad_map=grad_map,
        )
        self.last_gradient_inf_norm = self._gradient_inf_norm(list(grad_map.values()))
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

    def _collect_action_gradients(
        self,
        *,
        trace: RolloutTrace,
        objective: torch.Tensor,
    ) -> Tuple[List[Tuple[int, str, torch.Tensor]], Dict[Tuple[int, str], torch.Tensor]]:
        references: List[Tuple[int, str, torch.Tensor]] = []
        differentiable_inputs: List[torch.Tensor] = []
        differentiable_locations: List[Tuple[int, str]] = []

        for step_index, step_actions in enumerate(trace.actions):
            for name, value in step_actions.items():
                if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                    continue
                references.append((step_index, name, value))
                if value.requires_grad:
                    differentiable_inputs.append(value)
                    differentiable_locations.append((step_index, name))

        grad_map: Dict[Tuple[int, str], torch.Tensor] = {}
        if differentiable_inputs and (objective.requires_grad or objective.grad_fn is not None):
            gradients = torch.autograd.grad(
                outputs=objective,
                inputs=differentiable_inputs,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
            for location, reference, grad in zip(
                differentiable_locations,
                differentiable_inputs,
                gradients,
            ):
                grad_map[location] = (
                    torch.zeros_like(reference) if grad is None else grad.detach().clone()
                )

        for step_index, name, value in references:
            location = (step_index, name)
            if location not in grad_map:
                grad_map[location] = torch.zeros_like(value)

        return references, grad_map

    def _build_std_profile(
        self,
        *,
        references: Sequence[Tuple[int, str, torch.Tensor]],
        grad_map: Dict[Tuple[int, str], torch.Tensor],
    ) -> TensorProfile:
        if not references:
            return []

        num_steps = max(step_index for step_index, _, _ in references) + 1
        profile: TensorProfile = [{} for _ in range(num_steps)]

        if self.norm_scope == 'global':
            global_inf_norm = self._gradient_inf_norm(list(grad_map.values()))
            for step_index, name, _ in references:
                gradient = grad_map[(step_index, name)]
                profile[step_index][name] = self._std_from_gradient(
                    gradient=gradient,
                    inf_norm=global_inf_norm,
                )
            return profile

        for step_index in range(num_steps):
            step_gradients = [
                gradient
                for (current_step, _), gradient in grad_map.items()
                if current_step == step_index
            ]
            step_inf_norm = self._gradient_inf_norm(step_gradients)
            for current_step, name, _ in references:
                if current_step != step_index:
                    continue
                profile[current_step][name] = self._std_from_gradient(
                    gradient=grad_map[(current_step, name)],
                    inf_norm=step_inf_norm,
                )
        return profile

    def _materialize_gradient_profile(
        self,
        *,
        references: Sequence[Tuple[int, str, torch.Tensor]],
        grad_map: Dict[Tuple[int, str], torch.Tensor],
    ) -> TensorProfile:
        if not references:
            return []
        num_steps = max(step_index for step_index, _, _ in references) + 1
        profile: TensorProfile = [{} for _ in range(num_steps)]
        for step_index, name, _ in references:
            profile[step_index][name] = grad_map[(step_index, name)].clone()
        return profile

    def _std_from_gradient(self,
                           *,
                           gradient: torch.Tensor,
                           inf_norm: float) -> torch.Tensor:
        abs_gradient = gradient.detach().abs()
        inf_norm_tensor = abs_gradient.new_tensor(float(inf_norm))
        normalizer = inf_norm_tensor + self.eps
        # Normalize by the selected inf-norm so each action gets a stable score in [0, 1].
        normalized_gradient = abs_gradient / normalizer
        if inf_norm > 0.0:
            normalized_gradient = torch.where(
                abs_gradient == inf_norm_tensor,
                torch.ones_like(normalized_gradient),
                normalized_gradient,
            )
        normalized_gradient = normalized_gradient.clamp(0.0, 1.0)
        # Map the complement so smaller gradients receive larger bounded noise.
        complement = (1.0 - normalized_gradient).pow(self.alpha)
        min_std_tensor = abs_gradient.new_tensor(self.min_std)
        max_std_tensor = abs_gradient.new_tensor(self.max_std)
        return min_std_tensor + (max_std_tensor - min_std_tensor) * complement

    @staticmethod
    def _gradient_inf_norm(gradients: Sequence[torch.Tensor]) -> float:
        if not gradients:
            return 0.0
        max_values = [
            gradient.detach().abs().max()
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
