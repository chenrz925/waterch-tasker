__all__ = [
    'SetEnvironmentTask',
]

from logging import Logger
from os import environ
from typing import List, Text

from ..mixin import value
from ..storages.basic import Storage
from ..tasks.containers import Task
from ..typedef import Definition, Profile, Return


class SetEnvironmentTask(Task):
    """
    <i>tasker.tasks.utils.SetEnvironmentTask</i>

    Task to set environment values.
    """

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        """
        Set environment values.
        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            Always return [SUCCESS][tasker.typedef.Return.SUCCESS].
        """
        for name, content in profile.items():
            environ[name] = content
        logger.debug(f'Environment:')
        for item in environ.items():
            logger.debug(f'  {"=".join(item)}')
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        """
        This task requires nothing.

        Returns:
            Nothing
        """
        return []

    def provide(self) -> List[Text]:
        """
        This task provides nothing.

        Returns:
            Nothing
        """
        return []

    def remove(self) -> List[Text]:
        """
        This task removes nothing.

        Returns:
            Nothing
        """
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        """
        SetEnvironmentTask will not generate any template profiles.
        Each environment values can be filled as key-value pairs.

        Returns:
            Schema of profile.
        Examples:
            Set environment value of `CUDA_VISIBLE_DEVICES` to configure valid GPUs.
            ```toml
            "CUDA_VISIBLE_DEVICES" = "0,2,4"
            ```
        """
        return []


class StorageKeyMoveTask(Task):
    """
    <i>tasker.tasks.utils.StorageKeyMoveTask</i>

    Move a key to another key.
    """
    def __init__(self, key_from: Text, key_to: Text, *args):
        self.KEY_FROM = key_from
        self.KEY_TO = key_to

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        shared[self.KEY_TO] = shared[self.KEY_FROM]
        del shared[self.KEY_FROM]
        return Return.SUCCESS

    def require(self) -> List[Text]:
        return [self.KEY_FROM, self.KEY_TO]

    def provide(self) -> List[Text]:
        return [self.KEY_FROM, self.KEY_TO]

    def remove(self) -> List[Text]:
        return [self.KEY_FROM]

    @classmethod
    def define(cls) -> List[Definition]:
        return []


class StorageKeyCopyTask(Task):
    """
    <i>tasker.tasks.utils.StorageKeyCopyTask</i>

    Move a key to another key.
    """
    def __init__(self, key_from: Text, key_to: Text, *args):
        self.KEY_FROM = key_from
        self.KEY_TO = key_to

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        shared[self.KEY_TO] = shared[self.KEY_FROM]
        return Return.SUCCESS

    def require(self) -> List[Text]:
        return [self.KEY_FROM, self.KEY_TO]

    def provide(self) -> List[Text]:
        return [self.KEY_FROM, self.KEY_TO]

    def remove(self) -> List[Text]:
        return [self.KEY_FROM]

    @classmethod
    def define(cls) -> List[Definition]:
        return []

