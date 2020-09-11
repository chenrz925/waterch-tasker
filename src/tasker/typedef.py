from collections import namedtuple
from enum import Enum

from box import Box

Definition = namedtuple('Definition', ('name', 'type', 'children'))


class Profile(Box):
    def __init__(self, *args, **kwargs):
        kwargs['frozen_box'] = True
        super(Profile, self).__init__(*args, **kwargs)


class Return(Enum):
    SUCCESS = 0b00000
    ERROR = 0b01000
    EXIT = 0b00001
    WRITE = 0b00010
    READ = 0b00100
    RETRY = 0b10000
