__all__ = [
    'Launcher',
    'launch'
]

from argparse import ArgumentParser, Namespace
from datetime import datetime
from logging import getLogger as get_logger
from logging.config import dictConfig as dict_config
from os import makedirs, path, curdir
from sys import stdout, stderr, path as syspath
from typing import List

from toml import dump as toml_dump

from ._version import version
from .mixin import ProfileMixin, value
from .storages.basic import DictStorage, CommonStorageView
from .tasks import Task
from .typedef import Profile, Return, Definition
from .utils import import_reference, parse_profile


class Launcher(ProfileMixin):
    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('__name__', str),
            value('__author__', str),
            value('__version__', str),
            value('__email__', str),
            value('__abstract__', str),
            value('__setting__', list, [
                value('storage', list, [
                    value('reference', str),
                    value('kwargs', list)
                ]),
                value('log', list, [])
            ]),
            value('__meta__', list, [
                [
                    value('reference', str),
                    value('path', str),
                    value('profile', str),
                    value('execute', bool)
                ]
            ]),
        ]

    def _argparser(self) -> ArgumentParser:
        parser = ArgumentParser(
            prog='waterch-tasker',
            description='A scalable and extendable experiment task scheduler framework'
        )
        parser.add_argument(
            '-v', '--version',
            action='version',
            version=f'%(prog)s {version}',
        )
        subcommands = parser.add_subparsers(
            title='command',
            description='execute a command of waterch-tasker',
            dest='command',
            help='choose a command',
        )
        launch_parser = subcommands.add_parser(
            'launch',
            help='launch example_tasks by a profile'
        )
        launch_parser.add_argument(
            '-f', '--file',
            nargs=1,
            required=True,
            help='launch example_tasks defined by this file'
        )
        template_parser = subcommands.add_parser(
            'template',
            help='generate a profile template by a class reference',
        )
        template_parser.add_argument(
            '-r', '--reference',
            nargs='+',
            required=True,
            help='generate profile template of those example_tasks'
        )
        template_parser.add_argument(
            '-o', '--output',
            nargs=1,
            required=False,
            help='output folder path',
            default=['-']
        )
        return parser

    def argparse(self) -> Namespace:
        return self._argparser().parse_args()

    def invoke(self, namespace: Namespace):
        syspath.append(curdir)
        command = getattr(self, f'command_{namespace.command}', self.command_unknown)
        command(namespace)

    def command_unknown(self, namespace: Namespace):
        print(
            f'Please check the input arguments,\n'
            f'command "{namespace.command}" not exists.'
        )

    def command_template(self, namespace: Namespace):
        slash_number = 20
        enable_stdout = namespace.output[0] == '-'
        if not enable_stdout and not path.exists(namespace.output[0]):
            makedirs(namespace.output[0])

        for reference in namespace.reference:
            stream = stdout if enable_stdout else open(path.join(namespace.output[0], f'{reference}.toml'), 'w')
            if enable_stdout:
                print(f'Profile of {reference}')
                print('-' * slash_number)
            cls = import_reference(reference)
            if not issubclass(cls, ProfileMixin):
                raise RuntimeError(f'Class {cls} do NOT supported profile.')
            if not issubclass(cls, Task):
                print(f'Class {cls} is NOT a Task, this may cause unpredictable issue.', file=stderr)
            toml_dump(cls.profile_template(), stream)
            if not enable_stdout:
                stream.flush()
                stream.close()
            else:
                print('-' * slash_number)

    def command_launch(self, namespace):
        slash_number = 20
        raw_profile = Profile.from_toml(filename=namespace.file[0])
        profile = parse_profile(raw_profile)
        # Configure logging
        log_datetime_format = '%Y-%m-%dT%H:%M:%S%z'
        log_format = '%(asctime)s|%(process)d|%(thread)d|%(levelname)s|%(name)s> %(message)s'
        if '__setting__' in profile \
                and 'log' in profile.__setting__:
            handlers = profile.__setting__.log
        else:
            handlers = {
                'default': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'level': 'INFO',
                    'stream': 'ext://sys.stderr'
                }
            }
        log_config = {
            'version': 1,
            'formatters': {
                'default': {
                    'format': log_format,
                    'datefmt': log_datetime_format
                },
            },
            'handlers': handlers,
            'root': {
                'level': 'NOTSET',
                'handlers': list(handlers.keys())
            },
        }
        dict_config(log_config)
        logger = get_logger('tasker.launcher.Launcher')
        # Check the profile
        missing_key = list(
            map(
                lambda definition: definition.name,
                filter(
                    lambda value: value.name not in profile,
                    self.define()
                ))
        )
        if len(missing_key) > 0:
            raise RuntimeError(f'Key {",".join(missing_key)} is/are missing.')
        # Print header info of profile
        print('-' * slash_number)
        logger.debug('-' * slash_number)
        print(f'{profile.__name__} ({profile.__version__})')
        logger.debug(f'{profile.__name__} ({profile.__version__})')
        print(f'Author: {profile.__author__}')
        logger.debug(f'Author: {profile.__author__}')
        print(f'E-Mail: {profile.__email__}')
        logger.debug(f'E-Mail: {profile.__email__}')
        print(profile.__abstract__)
        logger.debug(profile.__abstract__)
        print('-' * slash_number)
        logger.debug('-' * slash_number)
        print()
        # Create shared storage
        if '__setting__' in profile \
                and 'storage' in profile.__setting__ \
                and 'reference' in profile.__setting__.storage:
            storage_cls = profile.__setting__.storage.reference
        else:
            storage_cls = DictStorage
        if '__setting__' in profile \
                and 'storage' in profile.__setting__ \
                and 'kwargs' in profile.__setting__.storage:
            shared = storage_cls(**profile.__setting__.storage.kwargs)
        else:
            shared = storage_cls()
        shared.load()
        # Launch example_tasks
        meta_index = 0
        while True:
            if not meta_index < len(profile.__meta__):
                break
            meta = profile.__meta__[meta_index]
            raw_meta = raw_profile.__meta__[meta_index]

            if not meta.execute:
                meta_index += 1
                continue

            if isinstance(meta.profile, Profile):
                task_profile = meta.profile
                raw_task_profile = profile.from_toml(raw_meta.profile[:-2])
            else:
                task_profile = profile[meta.profile]
                raw_task_profile = raw_profile[raw_meta.profile]
            task: Task = meta.reference
            task_display = f'{task.__class__.__name__}[{hex(hash(task))}]'
            task_logger = get_logger(task_display)
            print(task_display)
            print(f'require: {" ".join(task.require())}')
            logger.debug(f'require: {" ".join(task.require())}')
            print(f'provide: {" ".join(task.provide())}')
            logger.debug(f'provide: {" ".join(task.provide())}')
            print(f'{"-" * (slash_number - 1)}>')
            logger.debug(f'{"-" * (slash_number - 1)}>')
            start_time = datetime.now()
            user_kill = False
            try:
                task_logger.info('=' * slash_number)
                try:
                    task_logger.info('\n' + task_profile.to_toml())
                except Exception:
                    task_logger.info('\n' + raw_task_profile.to_toml())
                task_logger.info('=' * slash_number)
                state = task.invoke(task_profile, CommonStorageView(storage=shared, task=task), task_logger)
            except KeyboardInterrupt:
                state = Return.WRITE.value
                user_kill = True
            end_time = datetime.now()
            print(f'<{"-" * (slash_number - 1)}')
            logger.debug(f'<{"-" * (slash_number - 1)}')
            state_label = 'Failed' if state & Return.ERROR.value else 'Successfully finished'
            print(f'{state_label} in {(end_time - start_time).total_seconds()} seconds.')
            logger.debug(f'{state_label} in {(end_time - start_time).total_seconds()} seconds.')
            print()
            if state & Return.WRITE:
                shared.dump()
            if state & Return.READ:
                shared.load()
            if state & Return.EXIT:
                print('Stopped by task.')
                logger.debug('Stopped by task.')
                break
            if state & Return.RELOAD:
                profile = parse_profile(Profile.from_toml(filename=namespace.file[0]))
            if user_kill:
                print('Stopped by user.')
                logger.debug('Stopped by user.')
                break
            if not (state & Return.RETRY):
                meta_index += 1


def launch():
    launcher = Launcher()
    launcher.invoke(launcher.argparse())
