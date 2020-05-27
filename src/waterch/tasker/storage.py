__all__ = [
    'Storage',
    'DictStorage',
    'StorageView',
    'CommonStorageView',
    'ForkStorageView'
]

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from sys import stderr
from typing import Text, Any


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


class StorageView(Storage):
    @abstractmethod
    def storage(self) -> Storage:
        raise RuntimeError(f'Please implement storage function instead of using abstract storage view.')

    @staticmethod
    def require_check(task, item):
        if item not in task.require():
            raise RuntimeError(f'Forbidden reading on key "{item}", please declare it in method require.')

    @staticmethod
    def provide_check(task, item):
        if item not in task.provide():
            raise RuntimeError(f'Forbidden writing on key "{item}", please declare it in method provide.')

    @staticmethod
    def remove_check(task, item):
        if item not in task.remove():
            raise RuntimeError(f'Forbidden deleting on key "{item}", please declare it in method remove.')


class CommonStorageView(StorageView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'storage' in kwargs
        assert 'task' in kwargs
        self._task = kwargs['task']
        self._storage = kwargs['storage']

    def storage(self) -> Storage:
        return self._storage

    def __getitem__(self, item):
        self.require_check(self._task, item)
        return self._storage.__getitem__(item)

    def __setitem__(self, key, value):
        self.provide_check(self._task, key)
        return self._storage.__setitem__(key, value)

    def __delitem__(self, key):
        self.remove_check(self._task, key)
        return self._storage.__delitem__(key)

    def __contains__(self, item):
        return self._storage.__contains__(item)

    def load(self):
        return self._storage.load()

    def dump(self):
        return self._storage.dump()


class ForkStorageView(StorageView):
    _META_KEY = '__tasks__'
    _ALLOW = [_META_KEY]

    def __init__(self, **kwargs):
        super(ForkStorageView, self).__init__(**kwargs)
        assert 'storage' in kwargs
        assert 'task' in kwargs

        self._storage: Storage = kwargs['storage']
        self._task = kwargs['task']
        if self._META_KEY in self._storage:
            self._storage[self._META_KEY].append(hex(hash(self._task)))
        else:
            self._storage[self._META_KEY] = [hex(hash(self._task))]

    def _mirror_key(self, key) -> Text:
        return f'{key}@{hex(hash(self._task))}'

    def __getitem__(self, item):
        if item not in self._ALLOW:
            self.require_check(self._task, item)
        if item in self._storage:
            return self._storage.__getitem__(item)
        else:
            return self._storage.__getitem__(self._mirror_key(item))

    def __setitem__(self, key, value):
        if key not in self._ALLOW:
            self.provide_check(self._task, key)
        if key in self._storage:
            return self._storage.__setitem__(key, value)
        else:
            return self._storage.__setitem__(self._mirror_key(key), value)

    def __delitem__(self, key):
        if key not in self._ALLOW:
            self.remove_check(self._task, key)
        if key in self._storage:
            return self._storage.__delitem__(key)
        else:
            return self._storage.__delitem__(self._mirror_key(key))

    def __contains__(self, item):
        return self._storage.__contains__(item) or self._storage.__contains__(self._mirror_key(item))

    def load(self):
        return self._storage.load()

    def dump(self):
        return self._storage.dump()

    def storage(self) -> Storage:
        return self._storage
