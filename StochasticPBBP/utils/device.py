from __future__ import annotations

import time
from typing import Any, Optional, Union

import torch


DeviceLike = Optional[Union[str, torch.device]]


def resolve_torch_device(device: DeviceLike=None) -> torch.device:
    if device is None:
        device = 'auto'
    if isinstance(device, torch.device):
        resolved = device
    else:
        normalized = str(device).strip().lower()
        if normalized == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            mps_backend = getattr(torch.backends, 'mps', None)
            if mps_backend is not None and mps_backend.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        resolved = torch.device(normalized)

    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA was requested via device={resolved}, but torch.cuda.is_available() is False.'
        )
    if resolved.type == 'mps':
        mps_backend = getattr(torch.backends, 'mps', None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                f'MPS was requested via device={resolved}, but torch.backends.mps.is_available() is False.'
            )
    return resolved


def infer_torch_device(*values: Any, default: DeviceLike=None) -> Optional[torch.device]:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    if default is None:
        return None
    return resolve_torch_device(default)


def as_tensor_on_device(value: Any,
                        *,
                        dtype: Optional[torch.dtype]=None,
                        device: DeviceLike=None,
                        like: Optional[torch.Tensor]=None) -> torch.Tensor:
    if like is not None:
        if device is None:
            device = like.device
        if dtype is None:
            dtype = like.dtype

    resolved_device = None if device is None else resolve_torch_device(device)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(
        value,
        dtype=dtype,
        device=resolved_device,
    )
    if dtype is not None or resolved_device is not None:
        tensor = tensor.to(
            dtype=dtype or tensor.dtype,
            device=resolved_device or tensor.device,
        )
    return tensor


def move_structure_to_device(value: Any, device: DeviceLike) -> Any:
    resolved_device = resolve_torch_device(device)
    if isinstance(value, torch.Tensor):
        return value.to(device=resolved_device)
    if isinstance(value, dict):
        return {k: move_structure_to_device(v, resolved_device) for (k, v) in value.items()}
    if isinstance(value, list):
        return [move_structure_to_device(v, resolved_device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_structure_to_device(v, resolved_device) for v in value)
    return value


def make_generator(seed: Optional[int]=None, device: DeviceLike=None) -> torch.Generator:
    if seed is None:
        seed = time.time_ns()
    resolved_device = resolve_torch_device(device)
    generator = torch.Generator(device=str(resolved_device))
    generator.manual_seed(int(seed))
    return generator
