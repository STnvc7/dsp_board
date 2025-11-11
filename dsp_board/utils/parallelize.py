from typing import Callable
import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import numpy as np

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
n_cpus = os.cpu_count()
_MAX_WORKERS = n_cpus // 2 if n_cpus is not None else 1
_ENABLE_PARALLEL = False

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def set_max_workers(max_workers: int):
    global _MAX_WORKERS
    _MAX_WORKERS = max_workers

def get_max_workers() -> int:
    return _MAX_WORKERS

def set_enable_parallel(enable_parallel: bool):
    global _ENABLE_PARALLEL
    _ENABLE_PARALLEL = enable_parallel

def get_enable_parallel() -> bool:
    return _ENABLE_PARALLEL

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def parallelize(func: Callable) -> Callable:
    """
    A decorator that parallelizes a function that processes NumPy arrays in batches using threads.
    It treats the first argument (args[0]) as a stack of batches and calls `func` in parallel for each element (x[i]) using a `ThreadPoolExecutor`.
    This is optimized for parallel processing of C/C++ wrapper functions such as `pyworld` or `pysptk`.
    Parallelization is controlled by the global variables `_ENABLE_PARALLEL` 
    and `_MAX_WORKERS`.    
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args or not isinstance(args[0], np.ndarray):
            raise TypeError("The first argument must be a numpy.ndarray.")
        
        x = args[0]
        if _ENABLE_PARALLEL and _MAX_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                result = list(executor.map(lambda _x: func(_x, *args[1:], **kwargs), x))
        else:
            result = [func(x[i], *args[1:], **kwargs) for i in range(x.shape[0])]
        
        if not isinstance(result[0], np.ndarray):
            raise ValueError("The function must return a numpy.ndarray.")
            
        return np.stack(result, axis=0)
    return wrapper