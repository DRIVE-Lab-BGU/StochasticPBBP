from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

from StochasticPBBP.core.Compiler import TorchRDDLCompiler

ActionBound = Union[float, torch.Tensor]
ActionBounds = Union[ActionBound, Dict[str, ActionBound]]


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

    def __init__(self,
                 std: float,
                 *,
                 bounded: bool=False,
                 action_dim: Optional[int]=None,
                 max_action: Optional[ActionBounds]=None) -> None:
        if std < 0:
            raise ValueError(f'std must be non-negative, got {std!r}.')
        self.std = float(std)
        self.bounded = bounded
        self.action_dim = action_dim
        self.max_action = max_action
        if self.bounded:
            if self.action_dim is None:
                raise ValueError('action_dim must be provided when bounded=True.')
            if self.max_action is None:
                raise ValueError('max_action must be provided when bounded=True.')

    def __call__(self,
                 actions: Optional[Dict[str, Any]],
                 *,
                 context: Optional[NoiseContext]=None) -> Optional[Dict[str, Any]]:
        if actions is None:
            return None
        self._validate_action_dim(actions)
        return super().__call__(actions, context=context)

    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        del context
        if self.std == 0.0:
            return torch.zeros_like(reference)
        return torch.randn_like(reference) * self.std

    def apply_to_value(self,
                       *,
                       name: str,
                       value: Any,
                       context: NoiseContext) -> Any:
        result = super().apply_to_value(name=name, value=value, context=context)
        if not self.bounded or not isinstance(result, torch.Tensor) or not result.dtype.is_floating_point:
            return result
        max_action = self._coerce_max_action(name=name, reference=result)
        min_action = -max_action
        return torch.clamp(result, min=min_action, max=max_action)

    def _validate_action_dim(self, actions: Dict[str, Any]) -> None:
        if self.action_dim is None:
            return
        actual_dim = sum(
            int(value.numel()) if isinstance(value, torch.Tensor) else 1
            for value in actions.values()
        )
        if actual_dim != int(self.action_dim):
            raise ValueError(
                f'Expected action tensor with numel={self.action_dim}, '
                f'got numel={actual_dim}.'
            )

    def _coerce_max_action(self, *, name: str, reference: torch.Tensor) -> torch.Tensor:
        bound_value: ActionBound
        if isinstance(self.max_action, dict):
            if name not in self.max_action:
                raise KeyError(f'Missing max_action bound for action <{name}>.')
            bound_value = self.max_action[name]
        else:
            bound_value = self.max_action

        if isinstance(bound_value, torch.Tensor):
            max_action = bound_value.to(dtype=reference.dtype, device=reference.device)
            if max_action.numel() == 1:
                return torch.full_like(reference, float(max_action.item()))
            if int(max_action.numel()) != int(reference.numel()):
                raise ValueError(
                    f'max_action tensor must have numel=1 or numel={int(reference.numel())}, '
                    f'got numel={int(max_action.numel())}.'
                )
            return max_action.reshape(reference.shape)
        return torch.full_like(reference, float(bound_value))


class LinearDecayAdditiveNoise(ConstantAdditiveNoise):
    """Additive Gaussian noise with linearly decaying standard deviation."""

    def __init__(self,
                 start_std: float,
                 end_std: float,
                 num_iterations: int,
                 *,
                 bounded: bool=False,
                 action_dim: Optional[int]=None,
                 max_action: Optional[ActionBounds]=None) -> None:
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
        super().__init__(
            std=start_std,
            bounded=bounded,
            action_dim=action_dim,
            max_action=max_action,
        )

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


class _BaseModelBasedAdditiveNoise(ConstantAdditiveNoise):
    """Shared utilities for model-based additive-noise processes."""

    def __init__(self,
                 *,
                 model: Optional[Any]=None,
                 wrt: Optional[list[str]]=None,
                 noise_scale: float=1.0,
                 bounded: bool=False,
                 action_dim: Optional[int]=None,
                 max_action: Optional[ActionBounds]=None) -> None:
        super().__init__(
            std=0.0,
            bounded=bounded,
            action_dim=action_dim,
            max_action=max_action,
        )
        if noise_scale < 0:
            raise ValueError(f'noise_scale must be non-negative, got {noise_scale!r}.')
        self.model = model
        self.wrt = None if wrt is None else list(wrt)
        self.noise_scale = float(noise_scale)
        self._compiler: Optional[TorchRDDLCompiler] = None
        self._compiler_model: Optional[Any] = None

    def sample_like(self,
                    reference: torch.Tensor,
                    *,
                    context: NoiseContext) -> torch.Tensor:
        std = self._noise_std_from_context(context)
        self.std = std
        return super().sample_like(reference, context=context)

    def _require_subs(self, context: NoiseContext) -> Dict[str, Any]:
        if context.subs is None:
            raise ValueError(f'{type(self).__name__} requires context.subs.')
        return context.subs

    def _get_model(self, context: Optional[NoiseContext]=None) -> Any:
        model = self.model
        if model is None and context is not None:
            model = context.model
        if model is None:
            raise ValueError('ModelBasedAdditiveNoise requires a model or context.model.')
        return model

    def _get_compiler(self, context: Optional[NoiseContext]=None) -> TorchRDDLCompiler:
        model = self._get_model(context)
        if self._compiler is None or self._compiler_model is not model:
            self._compiler = TorchRDDLCompiler(model)
            self._compiler.compile(log_expr=False)
            self._compiler_model = model
        return self._compiler

    @staticmethod
    def _clone_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if hasattr(value, 'copy'):
            return value.copy()
        return deepcopy(value)

    @classmethod
    def _clone_structure(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: cls._clone_structure(v) for (k, v) in value.items()}
        if isinstance(value, list):
            return [cls._clone_structure(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_structure(v) for v in value)
        return cls._clone_value(value)

    @staticmethod
    def _coerce_like(value: Any, reference: Any, name: str) -> Any:
        if isinstance(reference, torch.Tensor):
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(
                value,
                device=reference.device,
            )
            if tensor.shape != reference.shape:
                raise ValueError(
                    f'Value for <{name}> must have shape {tuple(reference.shape)}, '
                    f'got {tuple(tensor.shape)}.'
                )
            if reference.dtype == torch.bool:
                return tensor.bool()
            return tensor.to(dtype=reference.dtype, device=reference.device)
        return value

    def cpf_gradients(self,
                      subs: Dict[str, Any],
                      wrt: Optional[list[str]]=None,
                      create_graph: bool=False,
                      context: Optional[NoiseContext]=None
                      ) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        if not isinstance(subs, dict):
            raise TypeError('subs must be a dict mapping fluent names to values.')

        compiler = self._get_compiler(context)
        local_subs = self._clone_structure(compiler.init_values)

        for (name, value) in subs.items():
            if name not in local_subs:
                raise KeyError(f'<{name}> is not a valid fluent in the compiled model.')
            local_subs[name] = self._coerce_like(value, local_subs[name], name)

        wrt_names = list(subs.keys()) if wrt is None else list(wrt)
        wrt_tensors: Dict[str, torch.Tensor] = {}
        for name in wrt_names:
            if name not in local_subs:
                raise KeyError(f'<{name}> is not available in subs.')
            tensor = local_subs[name]
            if not isinstance(tensor, torch.Tensor) or not tensor.dtype.is_floating_point:
                raise TypeError(
                    f'Can only differentiate with respect to floating-point tensors, got '
                    f'<{name}>={type(tensor).__name__}.'
                )
            if not tensor.requires_grad:
                tensor = tensor.requires_grad_(True)
                local_subs[name] = tensor
            wrt_tensors[name] = tensor

        model_params = deepcopy(compiler.model_params)
        key = None
        cpf_outputs: Dict[str, torch.Tensor] = {}
        for (name, cpf) in compiler.cpfs.items():
            value, key, err, model_params = cpf(local_subs, model_params, key)
            if int(err) != 0:
                raise RuntimeError(f'CPF <{name}> evaluation failed with error code {int(err)}.')
            local_subs[name] = value
            cpf_outputs[name] = value

        floating_cpfs = [
            (name, value) for (name, value) in cpf_outputs.items()
            if (
                isinstance(value, torch.Tensor)
                and value.dtype.is_floating_point
                and value.requires_grad
            )
        ]
        if not floating_cpfs:
            return {}

        wrt_values = [wrt_tensors[name] for name in wrt_names]
        gradients: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for index, (cpf_name, value) in enumerate(floating_cpfs):
            grad_values = torch.autograd.grad(
                outputs=value.sum(),
                inputs=wrt_values,
                allow_unused=True,
                retain_graph=create_graph or index < len(floating_cpfs) - 1,
                create_graph=create_graph,
            )
            gradients[cpf_name] = {
                name: grad for (name, grad) in zip(wrt_names, grad_values)
            }
        return gradients

    def cpf_jacobian(self,
                     cpf_name: str,
                     subs: Dict[str, Any],
                     wrt: Optional[list[str]]=None,
                     create_graph: bool=False,
                     context: Optional[NoiseContext]=None) -> Dict[str, torch.Tensor]:
        if not isinstance(subs, dict):
            raise TypeError('subs must be a dict mapping fluent names to values.')

        compiler = self._get_compiler(context)
        if cpf_name not in compiler.cpfs:
            raise KeyError(f'<{cpf_name}> is not a compiled CPF.')

        local_subs = self._clone_structure(compiler.init_values)
        for (name, value) in subs.items():
            if name not in local_subs:
                raise KeyError(f'<{name}> is not a valid fluent in the compiled model.')
            local_subs[name] = self._coerce_like(value, local_subs[name], name)

        wrt_names = list(subs.keys()) if wrt is None else list(wrt)
        wrt_tensors: Dict[str, torch.Tensor] = {}
        for name in wrt_names:
            if name not in local_subs:
                raise KeyError(f'<{name}> is not available in subs.')
            tensor = local_subs[name]
            if not isinstance(tensor, torch.Tensor) or not tensor.dtype.is_floating_point:
                raise TypeError(
                    f'Can only differentiate with respect to floating-point tensors, got '
                    f'<{name}>={type(tensor).__name__}.'
                )
            wrt_tensors[name] = tensor.detach().clone().requires_grad_(True)

        base_subs = self._clone_structure(local_subs)

        def evaluate_selected_cpf(*inputs: torch.Tensor) -> torch.Tensor:
            working_subs = self._clone_structure(base_subs)
            for (name, tensor) in zip(wrt_names, inputs):
                working_subs[name] = tensor

            model_params = deepcopy(compiler.model_params)
            key = None
            for (current_name, cpf) in compiler.cpfs.items():
                value, key, err, model_params = cpf(working_subs, model_params, key)
                if int(err) != 0:
                    raise RuntimeError(
                        f'CPF <{current_name}> evaluation failed with error code {int(err)}.'
                    )
                working_subs[current_name] = value
                if current_name == cpf_name:
                    if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                        raise TypeError(
                            f'CPF <{cpf_name}> must evaluate to a floating-point tensor.'
                        )
                    return value

            raise RuntimeError(f'CPF <{cpf_name}> was not evaluated.')

        jacobian_values = torch.autograd.functional.jacobian(
            evaluate_selected_cpf,
            tuple(wrt_tensors[name] for name in wrt_names),
            create_graph=create_graph,
            vectorize=True,
        )
        return {
            name: jacobian
            for (name, jacobian) in zip(wrt_names, jacobian_values)
        }

    def cpf_jacobian_norm(self,
                          cpf_name: str,
                          subs: Dict[str, Any],
                          wrt: Optional[list[str]]=None,
                          create_graph: bool=False,
                          context: Optional[NoiseContext]=None) -> torch.Tensor:
        jacobians = self.cpf_jacobian(
            cpf_name=cpf_name,
            subs=subs,
            wrt=wrt,
            create_graph=create_graph,
            context=context,
        )
        flat_jacobians = [jacobian.reshape(-1) for jacobian in jacobians.values()]
        if not flat_jacobians:
            return torch.tensor(0.0)
        return torch.linalg.vector_norm(torch.cat(flat_jacobians))

    def gradient_norm_noise(self,
                            subs: Dict[str, Any],
                            wrt: Optional[list[str]]=None,
                            noise_scale: Optional[float]=None,
                            create_graph: bool=False,
                            context: Optional[NoiseContext]=None) -> torch.Tensor:
        gradients = self.cpf_gradients(
            subs=subs,
            wrt=wrt,
            create_graph=create_graph,
            context=context,
        )
        flat_grads = [
            grad.reshape(-1)
            for cpf_grads in gradients.values()
            for grad in cpf_grads.values()
            if grad is not None
        ]
        if not flat_grads:
            return torch.tensor(0.0)
        scale = self.noise_scale if noise_scale is None else float(noise_scale)
        return torch.linalg.vector_norm(torch.cat(flat_grads)) * scale


class JacobianBasedAdditiveNoise(_BaseModelBasedAdditiveNoise):
    """Additive Gaussian noise whose scale is based on a CPF Jacobian norm."""

    def __init__(self,
                 *,
                 cpf_name: str,
                 model: Optional[Any]=None,
                 wrt: Optional[list[str]]=None,
                 noise_scale: float=1.0,
                 bounded: bool=False,
                 action_dim: Optional[int]=None,
                 max_action: Optional[ActionBounds]=None) -> None:
        if not cpf_name:
            raise ValueError('cpf_name must be provided for Jacobian-based noise.')
        super().__init__(
            model=model,
            wrt=wrt,
            noise_scale=noise_scale,
            bounded=bounded,
            action_dim=action_dim,
            max_action=max_action,
        )
        self.cpf_name = cpf_name

    def _noise_std_from_context(self, context: NoiseContext) -> float:
        value = self.cpf_jacobian_norm(
            cpf_name=self.cpf_name,
            subs=self._require_subs(context),
            wrt=self.wrt,
            create_graph=False,
            context=context,
        )
        return float((value * self.noise_scale).detach())


class GradientNormAdditiveNoise(_BaseModelBasedAdditiveNoise):
    """Additive Gaussian noise whose scale is based on aggregated CPF gradients."""

    def _noise_std_from_context(self, context: NoiseContext) -> float:
        value = self.gradient_norm_noise(
            subs=self._require_subs(context),
            wrt=self.wrt,
            noise_scale=self.noise_scale,
            create_graph=False,
            context=context,
        )
        return float(value.detach())


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
               bounded: bool=False) -> AdditiveNoise:
        normalized_type = noise_type.strip().lower()
        action_dim, max_action = cls._extract_action_metadata(source)
        resolved_model = cls._extract_rddl_model(source) if model is None else model

        if normalized_type == 'constant':
            if std == 0.0:
                return NoAdditiveNoise()
            return ConstantAdditiveNoise(
                std=std,
                bounded=bounded,
                action_dim=action_dim,
                max_action=max_action,
            )

        if normalized_type in {'jacobian', 'jacobian_norm', 'model_jacobian'}:
            return JacobianBasedAdditiveNoise(
                cpf_name=cls._require_cpf_name(cpf_name),
                model=resolved_model,
                wrt=wrt,
                noise_scale=noise_scale,
                bounded=bounded,
                action_dim=action_dim,
                max_action=max_action,
            )

        if normalized_type in {'gradient_norm', 'model_gradient'}:
            return GradientNormAdditiveNoise(
                model=resolved_model,
                wrt=wrt,
                noise_scale=noise_scale,
                bounded=bounded,
                action_dim=action_dim,
                max_action=max_action,
            )

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
                bounded=bounded,
                action_dim=action_dim,
                max_action=max_action,
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

    @classmethod
    def _extract_action_metadata(cls, source: Any) -> tuple[int, ActionBounds]:
        action_template = cls._extract_action_template(source)
        rddl_model = cls._extract_rddl_model(source)

        action_dim = sum(
            int(value.numel()) if isinstance(value, torch.Tensor) else 1
            for value in action_template.values()
        )
        max_action = cls._extract_action_bounds(action_template, rddl_model)
        return action_dim, max_action

    @staticmethod
    def _extract_action_template(source: Any) -> Dict[str, Any]:
        if hasattr(source, 'noop_actions'):
            action_template = getattr(source, 'noop_actions')
            if isinstance(action_template, dict):
                return action_template
        if hasattr(source, 'variable_types') and hasattr(source, 'action_ranges'):
            compiler = TorchRDDLCompiler(source)
            compiler.compile(log_expr=False)
            return {
                name: value
                for (name, value) in compiler.init_values.items()
                if source.variable_types[name] == 'action-fluent'
            }
        raise ValueError(
            'Could not extract action template from source. '
            'Pass a rollout-like object with a noop_actions dict or an RDDL model.'
        )

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

    @staticmethod
    def _extract_action_bounds(action_template: Dict[str, Any], rddl_model: Any) -> ActionBounds:
        if hasattr(rddl_model, 'action_ranges'):
            action_ranges = getattr(rddl_model, 'action_ranges')
            if isinstance(action_ranges, dict):
                bounds: Dict[str, ActionBound] = {}
                for name, template in action_template.items():
                    if name not in action_ranges:
                        raise KeyError(f'Missing action range for action fluent <{name}>.')
                    # Assumes action_ranges already stores numeric upper bounds per lifted
                    # action fluent. Revisit this if a domain uses richer/non-scalar ranges.
                    bound_value = action_ranges[name]
                    if isinstance(template, torch.Tensor):
                        bounds[name] = torch.full_like(
                            template,
                            float(bound_value),
                            dtype=template.dtype,
                            device=template.device,
                        )
                    else:
                        bounds[name] = float(bound_value)
                return bounds
        raise ValueError('Could not extract action bounds from the provided source.')
