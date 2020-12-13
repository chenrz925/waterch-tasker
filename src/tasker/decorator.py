from collections import defaultdict
from logging import Logger
from typing import Text, List, Type
from inspect import signature, Parameter

from tasker.mixin import ProfileMixin, value
from tasker.storage import Storage
from tasker.typedef import Profile
from tasker.tasks import Task


def def_task(require: List[Text]=(), provide: List[Text]=(), remove: List[Text]=(), **kwargs):
    def name_mapping(name):
        return ''.join(map(lambda it: it.capitalize(), name.split('_'))) + 'Task'

    def wrapper(func):
        parameter_names = tuple(signature(func).parameters)
        assert 'ctx' in parameter_names or 'context' in parameter_names

        def new_invoke(self, profile: Profile, shared: Storage, logger: Logger):
            return func({
                'shared': shared,
                'logger': logger
            }, **profile)

        def new_require(self):
            return require

        def new_provide(self):
            return provide

        def new_remove(self):
            return remove

        def new_define(self):
            annotation_map = defaultdict(default_factory=lambda it: str, map={
                str: str,
                int: int,
                float: float,
                bool: bool,
                List[Text]: list,
                List[int]: list,
                List[float]: list,
                List[str]: list,
                List[bool]: list,
            })
            return tuple(map(
                lambda it: value(it.name, annotation_map[it.annotation]),
                filter(
                    lambda it: (it.kind == Parameter.POSITIONAL_OR_KEYWORD or it.kind == Parameter.KEYWORD_ONLY) and it.name not in ('ctx', 'context'),
                    signature(func).parameters.values()
                )
            ))

        task_cls = type(name_mapping(func.__name__), (Task, ProfileMixin), {
            'invoke': new_invoke,
            'require': new_require,
            'provide': new_provide,
            'remove': new_remove,
            'define': new_define,
        })
        return task_cls

    return wrapper


