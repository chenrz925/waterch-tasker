__all__ = [
    'Launcher',
    'launch'
]

from argparse import ArgumentParser, Namespace
from datetime import datetime
from logging import getLogger as get_logger, basicConfig as basic_config
from os import makedirs, path, curdir
from sys import stdout, stderr, path as syspath
from typing import List

from toml import dump as toml_dump

from waterch.tasker.mixin import ProfileMixin, value
from waterch.tasker.storage import DictStorage, CommonStorageView
from waterch.tasker.tasks import Task
from waterch.tasker.typedef import Profile, Return, Definition
from waterch.tasker.utils import import_reference, extract_reference
from waterch.tasker._version import version


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
                    value('reference', str)
                ]),
                value('log', list, [
                    value('stdout', bool),
                    value('level', str),
                ])
            ]),
            value('__meta__', list, [
                [
                    value('reference', str),
                    value('include', bool),
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
            help='launch tasks by a profile'
        )
        launch_parser.add_argument(
            '-f', '--file',
            nargs=1,
            required=True,
            help='launch tasks defined by this file'
        )
        template_parser = subcommands.add_parser(
            'template',
            help='generate a profile template by a class reference',
        )
        template_parser.add_argument(
            '-r', '--reference',
            nargs='+',
            required=True,
            help='generate profile template of those tasks'
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
        profile = Profile.from_toml(filename=namespace.file[0])
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
        print(f'{profile.__name__} ({profile.__version__})')
        print(f'Author: {profile.__author__}')
        print(f'E-Mail: {profile.__email__}')
        print(profile.__abstract__)
        print('-' * slash_number)
        print()
        # Create shared storage
        if '__setting__' in profile \
                and 'storage' in profile.__setting__ \
                and 'reference' in profile.__setting__.storage:
            try:
                storage_cls = import_reference(profile.__setting__.storage.reference)
            except RuntimeError:
                print('Failed to get storage class, fall back to DictStorage.', file=stderr)
                storage_cls = DictStorage
        else:
            storage_cls = DictStorage
        shared = storage_cls(**profile.__setting__.storage)
        # Configure logging
        log_datetime_format = '%Y-%m-%dT%H:%M:%S'
        log_format = '%(asctime)s|%(levelname)s|%(name)s> %(message)s'
        if profile.__setting__.log.stdout:
            basic_config(
                stream=stdout,
                level=profile.__setting__.log.level,
                format=log_format,
                datefmt=log_datetime_format,
            )
        else:
            basic_config(
                filename=f'logs/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.log")}',
                format=log_format,
                datefmt=log_datetime_format,
            )
        # Launch tasks
        meta_index = 0
        while meta_index < len(profile.__meta__):
            meta = profile.__meta__[meta_index]

            if not meta.execute:
                continue

            reference, rparams = extract_reference(meta.reference)
            task_cls = import_reference(reference)
            try:
                if meta.include:
                    task_profile = Profile.from_toml(filename=meta.path)
                else:
                    task_profile = profile[meta.profile]
            except Exception:
                print('Failed to access task profile, stop running.')
                shared.dump()
                break
            task: Task = task_cls(*rparams)
            task_display = f'{reference}[{hex(hash(task))}]'
            task_logger = get_logger(task_display)
            print(task_display)
            print(f'require: {" ".join(task.require())}')
            print(f'provide: {" ".join(task.provide())}')
            print(f'{"-" * (slash_number - 1)}>')
            start_time = datetime.now()
            user_kill = False
            try:
                state = task.invoke(task_profile, CommonStorageView(storage=shared, task=task), task_logger)
            except KeyboardInterrupt:
                state = Return.WRITE.value
                user_kill = True
            end_time = datetime.now()
            print(f'<{"-" * (slash_number - 1)}')
            state_label = 'Failed' if state & Return.ERROR.value else 'Successfully finished'
            print(f'{state_label} in {(end_time - start_time).total_seconds()} seconds.')
            print()
            if state & Return.WRITE.value:
                shared.dump()
            if state & Return.READ.value:
                shared.load()
            if state & Return.EXIT.value:
                print('Stopped by task.')
                break
            if user_kill:
                print('Stopped by user.')
                break
            if not (state & Return.RETRY.value):
                meta_index += 1


def launch():
    launcher = Launcher()
    launcher.invoke(launcher.argparse())
