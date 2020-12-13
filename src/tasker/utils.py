from enum import Enum
from typing import Text, Type, List, Tuple
from os import path, makedirs
from logging import getLogger as get_logger


def import_reference(reference: Text) -> Type:
    path = reference.split('.')
    if reference and len(path) > 0:
        root = path[0]
        module = __import__(root)
        if len(path) == 1:
            return module
        else:
            for layer in path[1:]:
                module = getattr(module, layer, None)
                if module is None:
                    raise RuntimeError(f'Reference {reference} NOT found.')
            return module
    else:
        raise RuntimeError(f'Wrong reference {reference}.')


def extract_reference(reference: Text) -> Tuple[Text, List[Text]]:
    reference_splits = reference.split(':')
    return reference_splits[0], reference_splits[1:] if len(reference_splits) > 1 else []


class LocalPath(object):
    _logger = get_logger('tasker.utils.LocalPath')

    class CATEGORY(Enum):
        CACHE = 'cache'
        LOG = 'log'
        OUTPUT = 'output'
        STORAGE = 'storage'
        TEMP = 'temp'

    @classmethod
    def create(cls, category, *name: Text):
        assert isinstance(category, cls.CATEGORY)
        if not name:
            cls._logger.warning('You are operating in a root category folder! Please re-check it.')

        absolute_root = path.abspath(path.curdir)
        absolute_path = path.join(absolute_root, category.value, *name)

        if not path.exists(absolute_path):
            makedirs(absolute_path)

        return absolute_path
