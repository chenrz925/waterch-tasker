from collections import defaultdict
from inspect import signature, Parameter
from logging import Logger
from os import environ
from typing import Text, List

from .mixin import ProfileMixin, value
from .storages.basic import Storage
from .tasks import Task
from .typedef import Profile, Context


def def_task(require: List[Text] = (), provide: List[Text] = (), remove: List[Text] = (), **kwargs):
    """
    Decorator to convert a function to a `Task` instance.

    Args:
        require: required
        provide:
        remove:
        **kwargs:

    Returns:

    """
    def name_mapping(name):
        return ''.join(map(lambda it: it.capitalize(), name.split('_'))) + 'Task'

    def wrapper(func):
        parameter_names = tuple(signature(func).parameters)
        assert 'ctx' in parameter_names or 'context' in parameter_names

        def new_invoke(self, profile: Profile, shared: Storage, logger: Logger):
            return func(Context(
                logger=logger,
                shared=shared,
                environ=environ,
            ), **profile)

        def new_require(self):
            return require

        def new_provide(self):
            return provide

        def new_remove(self):
            return remove

        def new_define():
            annotation_map = defaultdict()
            annotation_map.default_factory = lambda: str
            for key, _value in {
                str: str,
                int: int,
                float: float,
                bool: bool,
                List[Text]: list,
                List[int]: list,
                List[float]: list,
                List[str]: list,
                List[bool]: list,
            }.items():
                annotation_map[key] = _value

            return tuple(map(
                lambda it: value(it.name, annotation_map[it.annotation]),
                filter(
                    lambda it: (
                                       it.kind == Parameter.POSITIONAL_OR_KEYWORD or it.kind == Parameter.KEYWORD_ONLY) and it.name not in (
                                   'ctx', 'context'
                               ),
                    signature(func).parameters.values()
                )
            ))

        task_cls = type('.'.join((func.__module__, name_mapping(func.__name__))), (Task, ProfileMixin), {
            'invoke': new_invoke,
            'require': new_require,
            'provide': new_provide,
            'remove': new_remove,
            'define': new_define,
        })
        return task_cls

    return wrapper
