from datetime import datetime
from logging import Logger
from os import environ
from typing import List, Text
from uuid import uuid4

from waterch.tasker.typedef import Definition, Profile, Return
from waterch.tasker.mixin import value
from waterch.tasker.storage import Storage
from waterch.tasker.tasks.containers import Task


class GenerateIdTask(Task):
    """
    <b>waterch.tasker.tasks.utils.GenerateIdTask</b>

    Task to generate ID.
    The task will declare a field named `id` in shared storage.

    Currently, `GenerateIdTask` supports 3 type of items to generate ID,
    including "datetime", "uuid", "label". If you need to label something,
    considerate using "label" into "join" fields. Items will be joined by
    string defined in field "by".
    """

    def generate_datetime(self, profile) -> Text:
        return datetime.now().strftime('%Y%m%d%H%M%S')

    def generate_uuid(self, profile) -> Text:
        return uuid4().hex.replace('-', '')

    def generate_label(self, profile) -> Text:
        try:
            return profile.label
        except Exception:
            raise RuntimeError('Please define "label" attribute in profile when including label item.')

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> Return:
        """
        Generate ID.
        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            Always return [SUCCESS][Return.SUCCESS].
        """

        def generate(name: Text) -> Text:
            generator = getattr(self, f'generate_{name}', None)
            if generator is None:
                raise RuntimeError(f'ID item {name} is invalid, please check and fix.')
            return generator(profile)

        shared['id'] = profile.by.join(filter(lambda t: t != '', map(generate, profile.join)))
        logger.debug(f'ID: {shared["id"]}')
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        """
        This task requires nothing.
        Returns:
            Nothing
        """
        return ['id']

    def provide(self) -> List[Text]:
        """
        This task provides:

        | Key  | Value                                 |
        | ---- | ------------------------------------- |
        | id   | Generated ID provided to other tasks. |

        Returns:
            Only contains "id"
        """
        return ['id']

    def remove(self) -> List[Text]:
        """
        This task removes nothing.
        Returns:
            Nothing
        """

    @classmethod
    def define(cls) -> List[Definition]:
        """
        Examples:
            ```toml
            __schema__ = "waterch.tasker.tasks.utils.GenerateIdTask"
            by = ""
            label = ""
            ```
        Returns:
            Schema of profile
        """
        return [
            value('by', str),
            value('join', list, [str]),
            value('label', str)
        ]


class SetEnvironmentTask(Task):
    """
    <b>waterch.tasker.tasks.utils.SetEnvironmentTask</b>

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
            Always return [SUCCESS][Return.SUCCESS].
        """
        for item in profile.env:
            assert len(item) == 2
            environ[item[0]] = item[1]
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
        ```toml
        __schema__ = "waterch.tasker.tasks.utils.SetEnvironmentTask"
        env = [
            ["CUDA_VISIBLE_DEVICES", "0"],
        ]
        ```

        Returns:
            Schema of profile.
        """
        return [
            value('env', list, [[str]]),
        ]


