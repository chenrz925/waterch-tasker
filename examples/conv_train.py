from logging import Logger

from torch import nn

from waterch.tasker.typedef import Profile
from waterch.tasker.storage import Storage
from waterch.tasker.tasks.torch import SimpleTrainTask


class ConvTrainTask(SimpleTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.SELU(),
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            nn.SELU(),
            nn.Conv2d(64, 128, 9),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.Conv2d(128, 256, 14),
            nn.BatchNorm2d(256),
            nn.Flatten()
        )

    @classmethod
    def define_model(cls):
        return []

    def prepare_batch(self, batch, device=None, non_blocking=False):
        return super(ConvTrainTask, self).prepare_batch(batch, device, non_blocking)
