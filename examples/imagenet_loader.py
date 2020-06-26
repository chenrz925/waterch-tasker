from logging import Logger
from typing import List, Text

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from waterch.tasker.mixin import value

from waterch.tasker.typedef import Definition, Profile
from waterch.tasker.storage import Storage
from waterch.tasker.tasks import Task
from waterch.tasker.typedef import Return


class FashionMNISTDataLoaderTask(Task):
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        shared['train_loader'] = DataLoader(
            datasets.FashionMNIST('datasets', download=True, transform=transforms.ToTensor()),
            batch_size=profile.batch_size,
        )
        shared['validate_loader'] = DataLoader(
            datasets.FashionMNIST('datasets', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=profile.batch_size
        )
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return []

    def provide(self) -> List[Text]:
        return ['train_loader', 'validate_loader']

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('batch_size', int),
        ]
