from collections import namedtuple
from enum import IntEnum

from box import Box

Definition = namedtuple('Definition', ('name', 'type', 'children', 'default'))


class Profile(Box):
    def __init__(self, *args, **kwargs):
        kwargs['frozen_box'] = True
        super(Profile, self).__init__(*args, **kwargs)

    def __repr__(self):
        kwargs = ' '.join(map(
            lambda it: f'(\'{it[0]}\', {repr(it[1])}),',
            self.items()
        ))
        return f'Profile(({kwargs}))'

    def __str__(self):
        return repr(self)


class Context(Box):
    def __init__(self, *args, **kwargs):
        kwargs['frozen_box'] = True
        super(Context, self).__init__(*args, **kwargs)

    def __repr__(self):
        kwargs = ' '.join(map(
            lambda it: f'(\'{it[0]}\', {repr(it[1])}),',
            self.items()
        ))
        return f'Context(({kwargs}))'

    def __str__(self):
        return repr(self)


class Return(IntEnum):
    SUCCESS = 0b000000
    ERROR = 0b001000
    EXIT = 0b000001
    WRITE = 0b000010
    READ = 0b000100
    RETRY = 0b010000
    RELOAD = 0b100000
