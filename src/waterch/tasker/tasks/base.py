__all__ = [
    'Task',
]

from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Text

from waterch.tasker import Profile, Definition, Return, value
from waterch.tasker.mixin import ProfileMixin
from waterch.tasker.storage import Storage


class Task(ProfileMixin, metaclass=ABCMeta):
    """
    The container class of a task.

    To interactive with other task, you need to implement `require`, `provide`
    methods to declare the fields in shared storage.

    To implement action of a task, you need to implement `invoke` method.

    To define the schema of task profile, you need to implement `define` method
    which override from [`ProfileMixin` class][waterch.tasker.mixin.ProfileMixin].
    """
    @abstractmethod
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        raise NotImplementedError('Please implement the task process.')

    @classmethod
    @abstractmethod
    def require(cls) -> List[Text]:
        raise NotImplementedError('Please define required keys.')

    @classmethod
    @abstractmethod
    def provide(cls) -> List[Text]:
        raise NotImplementedError('Please define provided keys.')
