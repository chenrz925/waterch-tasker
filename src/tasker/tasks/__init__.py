__all__ = [
    'Task',
    'ForkTask',
    'MapTask',
    'ReduceTask',
    'utils',
]

from sys import stderr

try:
    from tasker.tasks import torch
    __all__.extend('torch')
except RuntimeError:
    print(
        'Extra "pytorch" has not been initialized, please install torch and pytorch-ignite modules to enable this '
        'extra.',
        file=stderr
    )
from tasker.tasks import utils
from tasker.tasks.containers import Task, ForkTask, MapTask, ReduceTask
