from logging import Logger
from typing import List, Text

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from waterch.tasker.mixin import value

from waterch.tasker.typedef import Definition, Profile
from waterch.tasker.storage import Storage
from waterch.tasker.tasks import Task
from waterch.tasker.tasks.torch import DataLoaderTask
from waterch.tasker.typedef import Return


class CIFARDataLoaderTask(DataLoaderTask):
    def create_dataset(self, profile: Profile, shared: Storage, logger: Logger):
        return datasets.CIFAR100('datasets', transform=transforms.ToTensor())

    @classmethod
    def define_dataset(cls):
        return [
            value('train', bool)
        ]