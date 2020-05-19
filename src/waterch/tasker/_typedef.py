from collections import namedtuple

from box import Box
from enum import Enum

Definition = namedtuple('Definition', ('name', 'type', 'children'))


class Profile(Box):
    def __init__(self, *args, **kwargs):
        kwargs['frozen_box'] =True
        super(Profile, self).__init__(*args, **kwargs)


class Return(Enum):
    SUCCESS = 0b0000
    ERROR = 0b1000
    EXIT = 0b0001
    WRITE = 0b0010
    READ = 0b0100

