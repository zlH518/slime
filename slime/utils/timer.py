from contextlib import contextmanager
from functools import wraps
from time import time

from .misc import SingletonMeta

__all__ = ["Timer", "timer"]

SAVE = {
    "task-runtime",
    "task-preprocess-update",
    "task-update",
    "task-postprocess-update",
    "task-preprocess-rollout",
    "task-rollout",
    "task-postprocess-rollout",
    "task-preprocess-train",
    "task-train",
    "task-postprocess-train"
}


class Timer(metaclass=SingletonMeta):
    seq_lens = []
    def __init__(self):
        self.timers = [{},{}]
        self.start_time = [{},{}]
    
    def set(self, task_id, key, value):
        self.timers[task_id][key]=value

    def get(self, task_id, key):
        return self.timers[task_id][key]

    def start(self, task_id, name):
        assert name not in self.start_time[task_id], f"Timer {name} in task {task_id} already started."
        self.start_time[task_id][name] = time()

    def end(self, task_id, name):
        assert name in self.start_time[task_id], f"Timer {name} in task {task_id} not started."
        elapsed_time = time() - self.start_time[task_id][name]
        self.add(task_id, name, elapsed_time)
        del self.start_time[task_id][name]

    def reset(self, task_id, name, save_key):
        save_key = SAVE
        if save_key is None:
            if name is None:
                self.timers[task_id] = {}
            elif name in self.timers[task_id]:
                del self.timers[task_id][name]
        else:
            save_value = {}
            for key, value in self.timers[task_id].items():
                if key in save_key:
                    save_value[key] = value
            
            if name is None:
                self.timers[task_id] = save_value
            elif name in self.timers[task_id]:
                save_value[name] = self.timers[task_id][name]
                self.timers[task_id] = save_value

    def add(self, task_id, name, elapsed_time):
        if name not in self.timers[task_id]:
            self.timers[task_id][name] = elapsed_time
        else:
            self.timers[task_id][name] += elapsed_time

    def log_dict(self, task_id):
        return self.timers[task_id]

    @contextmanager
    def context(self, task_id, name):
        self.start(task_id, name)
        try:
            yield
        finally:
            self.end(task_id, name)


def timer(task_id, name_or_func):
    """
    Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...
    """
    # When used as a context manager
    if isinstance(name_or_func, str):
        name = name_or_func
        return Timer().context(task_id, name)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(task_id, func.__name__):
            return func(*args, **kwargs)

    return wrapper
