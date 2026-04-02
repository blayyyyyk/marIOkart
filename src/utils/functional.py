import multiprocessing
from typing import Callable
from gym_mkds.wrappers import GtkVecWindow
from gymnasium.vector import AsyncVectorEnv
from gymnasium import Env
from itertools import batched
import sys, os



def _async_env_worker(func: Callable, create_env_fn: Callable, vec_class: Callable, *list_args):
    env_fns = [
        (lambda k=k: create_env_fn(*k)) 
        for k in zip(*list_args, strict=True)
    ]
    
    if vec_class.__name__ == "SubprocVecEnv":
        env = vec_class(env_fns, start_method='spawn') # sb3
    elif vec_class.__name__ == "AsyncVectorEnv":
        env = vec_class(env_fns, context='spawn') # gymnasium
        env = GtkVecWindow(env)
    else:
        env = vec_class(env_fns)
    
    try:
        func(env)
    finally:
        # Guarantee cleanup if the underlying 'func' loops indefinitely or crashes
        env.close()


def sub_process_func(func: Callable, create_env_fn: Callable, vec_class: Callable = AsyncVectorEnv):
    def _wrapper(*args):
        # *args expects parallel lists (e.g., [m1, m2], [o1, o2])
        p = multiprocessing.Process(
            target=_async_env_worker, 
            args=(func, create_env_fn, vec_class, *args)
        )
        p.start()
        p.join()  # Wait for the batch to finish entirely
        
        # Catch if the C++ core threw a SIGBUS/Segfault and crashed the batch
        if p.exitcode != 0:
            print(f"Warning: Subprocess crashed with exit code {p.exitcode}.")
            
    return _wrapper
    
    
def delay_frames(func: Callable[[Env], bool], n_frames: int):
    count = 0
    def _wrapper(env: Env) -> bool:
        nonlocal count, n_frames, func
        if not func(env):
            count += 1
            
        return count < n_frames
    return _wrapper