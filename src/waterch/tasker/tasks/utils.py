from logging import Logger
from typing import List, Text
from datetime import datetime
from uuid import uuid4

from waterch.tasker import Definition, Profile, Return, value
from waterch.tasker.storage import Storage
from waterch.tasker.tasks.base import Task


class GenerateIdTask(Task):
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
        def generate(name: Text) -> Text:
            generater = getattr(self, f'generate_{name}', None)
            if generater is None:
                raise RuntimeError(f'ID item {name} is invalid, please check and fix.')
            return generater(profile)

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
