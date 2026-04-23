from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch

from StochasticPBBP.core.Compiler import FuzzyLogic, TorchRDDLCompiler


class noise:
    def __init__(self,
                 action_dim: int,
                 max_action: float,
                 horizon: int,
                 model: Optional[Any]=None) -> None:
        if int(horizon) <= 0:
            raise ValueError(f'horizon must be positive, got {horizon!r}.')
        self.action_dim = action_dim
        self.max_action = max_action
        self.horizon = int(horizon)
        self.model = model
        self._compiler: Optional[TorchRDDLCompiler] = None

    def get_smaller_1(self, start_noise: float, end_noise: float, step: int) -> float:
        return step * (end_noise - start_noise) / self.horizon + start_noise

    def get_smaller_2(self, start_noise: float, end_noise: float, step: int) -> float:
        return max((step / self.horizon) * start_noise, end_noise)

    @staticmethod
    def constant_noise(noise: float=0) -> float:
        return noise

    def _get_compiler(self) -> TorchRDDLCompiler:
        if self.model is None:
            raise ValueError('dynamic_norm_noise requires a model.')
        if self._compiler is None:
            self._compiler = TorchRDDLCompiler(self.model, logic=FuzzyLogic())
            self._compiler.compile()
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
                      wrt: Optional[List[str]]=None,
                      create_graph: bool=False) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        if not isinstance(subs, dict):
            raise TypeError('subs must be a dict mapping fluent names to values.')

        compiler = self._get_compiler()
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
                     wrt: Optional[List[str]]=None,
                     create_graph: bool=False) -> Dict[str, torch.Tensor]:
        if not isinstance(subs, dict):
            raise TypeError('subs must be a dict mapping fluent names to values.')

        compiler = self._get_compiler()
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
                          wrt: Optional[List[str]]=None,
                          noise_scale: float=1.0,
                          create_graph: bool=False) -> torch.Tensor:
        jacobians = self.cpf_jacobian(
            cpf_name=cpf_name,
            subs=subs,
            wrt=wrt,
            create_graph=create_graph,
        )
        flat_jacobians = [jacobian.reshape(-1) for jacobian in jacobians.values()]
        if not flat_jacobians:
            return torch.tensor(0.0)
        return torch.linalg.vector_norm(torch.cat(flat_jacobians)) * noise_scale

    def dynamic_norm_noise(self,
                           subs: Dict[str, Any],
                           wrt: Optional[List[str]]=None,
                           noise_scale: float=1.0,
                           create_graph: bool=False) -> torch.Tensor:
        gradients = self.cpf_gradients(
            subs=subs,
            wrt=wrt,
            create_graph=create_graph,
        )
        flat_grads = [
            grad.reshape(-1)
            for cpf_grads in gradients.values()
            for grad in cpf_grads.values()
            if grad is not None
        ]
        if not flat_grads:
            return torch.tensor(0.0)
        return torch.linalg.vector_norm(torch.cat(flat_grads)) * noise_scale

    def get_noise(self):
        return 0
