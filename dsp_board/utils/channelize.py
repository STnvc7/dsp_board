from typing import Callable, Tuple
from functools import wraps
import torch

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def channelize(keep_dims: int) -> Callable:
    if keep_dims < 1:
        raise ValueError("keep_dims must be at least 1")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError("The first argument must be a torch.Tensor.")

            x = args[0]
            if x.ndim < keep_dims:
                raise ValueError(f"The input tensor must have at least {keep_dims} dimensions.")

            x_flat, original_shape = to_channelized(x, keep_dims=keep_dims)

            new_args = (x_flat,) + args[1:]
            out = func(*new_args, **kwargs)

            if isinstance(out, torch.Tensor) is False:
                raise TypeError("The function must return a torch.Tensor.")

            out = from_channelized(out, original_shape, keep_dims=keep_dims)
            return out
        return wrapper
    return decorator

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def to_channelized(x: torch.Tensor, keep_dims: int) -> Tuple[torch.Tensor, torch.Size]:
    original_shape = x.shape
    if x.ndim == keep_dims:
        x = x.unsqueeze(0)
    elif x.ndim == keep_dims + 1:
        pass
    else:
        x = x.view(-1, *original_shape[-keep_dims:])

    return x, original_shape

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def from_channelized(x: torch.Tensor, original_shape: torch.Size, keep_dims: int) -> torch.Tensor:
    if len(original_shape) == keep_dims:
        x = x.squeeze(0)
    elif len(original_shape) == keep_dims + 1:
        pass
    else:
        x = x.view(original_shape[:-keep_dims] + x.shape[1:])
    return x
