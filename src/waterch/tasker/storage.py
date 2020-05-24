__all__ = [
    'Storage',
    'DictStorage',
    'MultiTaskStorageView'
]

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from sys import stderr
from typing import Text, Any

from more_itertools import flatten


class Storage(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError('Please implement __getitem__ function instead of using abstract storage.')

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError('Please implement __setitem__ function instead of using abstract storage.')

    @abstractmethod
    def __delitem__(self, key):
        raise NotImplementedError('Please implement __delitem__ function instead of using abstract storage.')

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError('Please implement __contains__ function instead of using abstract storage.')

    @abstractmethod
    def load(self):
        raise NotImplementedError('Please implement load function instead of using abstract storage.')

    @abstractmethod
    def dump(self):
        raise NotImplementedError('Please implement dump function instead of using abstract storage.')


class DictStorage(Storage):
    __slots__ = ['_data']

    def __init__(self, **kwargs):
        super(DictStorage, self).__init__(**kwargs)
        self._data = OrderedDict()

    def __getitem__(self, item: Text) -> Any:
        return self._data.__getitem__(item)

    def __setitem__(self, key: Text, value: Any):
        return self._data.__setitem__(key, value)

    def __delitem__(self, key: Text):
        return self._data.__delitem__(key)

    def __contains__(self, item):
        return self._data.__contains__(item)

    def load(self):
        print('Function load is invalid in DictStorage.', file=stderr)

    def dump(self):
        print('Function dump is invalid in DictStorage.', file=stderr)


class TaskStorageView(Storage):
    pass


class MultiTaskStorageView(Storage):
    _REGISTER_KEY = f'_multitask'

    def __init__(self, **kwargs):
        super(MultiTaskStorageView, self).__init__(**kwargs)
        self._storage: Storage = kwargs['storage'] if 'storage' in kwargs else None
        if self._storage is None:
            raise RuntimeError(f'Argument "storage" is missing.')
        self._task = kwargs['task'] if 'task' in kwargs else None
        if self._task is None:
            raise RuntimeError(f'Argument "task" is missing.')
        self._links = kwargs['links'] if 'links' in kwargs else []
        if self._REGISTER_KEY not in self._storage:
            self._storage[self._REGISTER_KEY] = []
        self._storage[self._REGISTER_KEY].append((hex(hash(self._task)), type(self._task)))
        self._suffix = hex(hash(self._task))

    def __getitem__(self, item):
        mirror_items = list(filter(
            lambda suffix: self._storage.__contains__(f'{item}@{suffix}'),
            flatten([[self._suffix], self._links])
        ))
        has_root = self._storage.__contains__(item)
        if len(mirror_items) == 1 and not has_root:
            return self._storage.__getitem__(self._storage.__getitem__(f'{item}@{mirror_items}'))
        elif len(mirror_items) == 0 and has_root:
            return self._storage.__getitem__(item)
        else:
            if len(mirror_items) + 1 if has_root else 0 > 1:
                including = [] if not has_root else [item]
                including.extend(mirror_items)
                raise KeyError(f'Duplicated key {item}, including {", ".join(including)}')
            else:
                raise KeyError(f'Not found {item}')

    def __setitem__(self, key, value):
        return self._storage.__setitem__(f'{key}@{self._suffix}', value)

    def __delitem__(self, key):
        try:
            return self._storage.__delitem__(key)
        except KeyError:
            return self._storage.__delitem__(f'{key}@{self._suffix}')

    def __contains__(self, item):
        mirror_items = list(filter(
            lambda suffix: self._storage.__contains__(f'{item}@{suffix}'),
            flatten([[self._suffix], self._links])
        ))
        return self._storage.__contains__(item) or len(mirror_items) > 0

    def load(self):
        return self._storage.load()

    def dump(self):
        return self._storage.dump()
