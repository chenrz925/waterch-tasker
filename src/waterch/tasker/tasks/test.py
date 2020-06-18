from logging import Logger
from typing import List, Text

from waterch.tasker.mixin import value
from waterch.tasker.storage import Storage
from waterch.tasker.tasks.containers import Task
from waterch.tasker.typedef import Definition, Profile, Return


class TestMultiTask(Task):
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        logger.debug(f'RUN {profile.index}')
        shared['index'] = profile.index
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return []

    def provide(self) -> List[Text]:
        return ['index']

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('index', int),
        ]


class TwiceTask(Task):
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        logger.debug(f'Twice {shared["index"] * 2}')
        shared['twice'] = 2 * shared['index']
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return ['index']

    def provide(self) -> List[Text]:
        return ['twice']

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return []
