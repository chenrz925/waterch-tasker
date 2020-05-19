__all__ = [
    'Storage',
    'DictStorage',
]

from abc import ABCMeta, abstractmethod
from sys import stderr
from typing import Text, Any
from collections import OrderedDict


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

    def load(self):
        print('Function load is invalid in DictStorage.', file=stderr)

    def dump(self):
        print('Function dump is invalid in DictStorage.', file=stderr)
