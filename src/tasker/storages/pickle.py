__all__ = [
    'HardPickleStorage',
    'SoftPickleStorage',
]

import pickle
from datetime import datetime
from os import remove, listdir, makedirs
from pathlib import Path
from uuid import uuid4

from .basic import Storage


class HardPickleStorage(Storage):
    def __init__(self, **kwargs):
        super(HardPickleStorage, self).__init__(**kwargs)
        self.id = f'{datetime.now().strftime("%Y%m%d%H%M%S")}-{uuid4().hex}' if 'id' not in kwargs else kwargs['id']
        self.storage_folder = Path('.tasker') / 'storage' / 'pickle' / self.id
        try:
            makedirs(self.storage_folder)
        except Exception:
            pass

    def __getitem__(self, item):
        with open(self.storage_folder / item, 'rb') as fp:
            return pickle.load(fp)

    def __setitem__(self, key, value):
        with open(self.storage_folder / key, 'wb') as fp:
            return pickle.dump(value, fp)

    def __delitem__(self, key):
        return remove(self.storage_folder / key)

    def __contains__(self, item):
        return listdir(self.storage_folder).__contains__(item)

    def load(self):
        pass

    def dump(self):
        pass


class SoftPickleStorage(Storage):
    def __init__(self, **kwargs):
        super(SoftPickleStorage, self).__init__(**kwargs)
        self.id = f'{datetime.now().strftime("%Y%m%d%H%M%S")}-{uuid4().hex}' if 'id' not in kwargs else kwargs['id']
        self.storage_folder = Path('.tasker') / 'storage' / 'pickle' / self.id
        try:
            makedirs(self.storage_folder)
        except Exception:
            pass
        self.cache_dict = {}

    def __getitem__(self, item):
        return self.cache_dict.__getitem__(item)

    def __setitem__(self, key, value):
        return self.__setitem__(key, value)

    def __delitem__(self, key):
        return self.__delitem__(key)

    def __contains__(self, item):
        return self.__contains__(item)

    def load(self):
        for key in listdir(self.storage_folder):
            with open(self.storage_folder / key, 'rb') as fp:
                self.cache_dict[key] = pickle.load(fp)

    def dump(self):
        for key in self.cache_dict.keys():
            with open(self.storage_folder / key, 'wb') as fp:
                pickle.dump(self.cache_dict[key], fp)
