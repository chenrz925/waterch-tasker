from tasker.storage import Storage
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from sys import stderr
import pickle
from os import remove, listdir, makedirs

try:
    import torch
    from torch import nn
except ImportError as ie:
    print('If certain keys saved by PyTorch, loading will cause runtime error.', file=stderr)
    torch = None


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
            try:
                return pickle.load(fp)
            except pickle.UnpicklingError as ue:
                return torch.load(fp)

    def __setitem__(self, key, value):
        with open(self.storage_folder / key, 'wb') as fp:
            if isinstance(value, nn.Module) or isinstance(value, torch.Tensor):
                return torch.save(value, fp)
            else:
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
