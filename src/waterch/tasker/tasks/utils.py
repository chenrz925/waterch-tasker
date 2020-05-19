from logging import Logger
from typing import List, Text
from datetime import datetime
from uuid import uuid4

from waterch.tasker import Definition, Profile, Return, value
from waterch.tasker.storage import Storage
from waterch.tasker.tasks._base import Task


class GenerateIdTask(Task):
    def generate_datetime(self) -> Text:
        return datetime.now().strftime('%Y%m%d%H%M%S')

    def generate_uuid(self) -> Text:
        return uuid4().hex.replace('-', '')

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> Return:
        def generate(name: Text) -> Text:
            if name == 'datetime':
                return self.generate_datetime()
            elif name == 'uuid':
                return self.generate_uuid()
            elif name == 'label':
                return profile.label
            else:
                return ''

        shared['id'] = profile.by.join(filter(lambda t: t != '', map(generate, profile.join)))
        logger.debug(f'ID: {shared["id"]}')
        return Return.SUCCESS.value

    @classmethod
    def require(cls) -> List[Text]:
        return []

    @classmethod
    def provide(cls) -> List[Text]:
        return ['id']

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('by', str),
            value('join', list, [str]),
            value('label', str)
        ]
