__all__ = [
    'Task',
]

from abc import ABCMeta, abstractmethod
from functools import reduce
from logging import Logger, getLogger as get_logger
from multiprocessing.dummy import Pool
from typing import List, Text

from waterch.tasker.mixin import ProfileMixin, value
from waterch.tasker.storage import Storage, MultiTaskStorageView
from waterch.tasker.typedef import Definition, Return, Profile
from waterch.tasker.utils import import_reference


class Task(ProfileMixin, metaclass=ABCMeta):
    """
    The container class of a task.

    To interactive with other task, you need to implement `require`, `provide`
    methods to declare the fields in shared storage.
    To implement action of a task, you need to implement `invoke` method.
    To define the schema of task profile, you need to implement `define` method
    which override from [`ProfileMixin` class][waterch.tasker.mixin.ProfileMixin].

    We will introduce a example to describe the usage of `Task` class.
    For example, we constrct a class named `ExampleTask` in "example.py",
    so the reference of `ExampleTask` is `example.ExampleTask`.

    ```python
    from logging import Logger
    from typing import List, Text

    from waterch.tasker import Definition, Profile, Return, value
    from waterch.tasker.storage import Storage
    from waterch.tasker.tasks import Task


    class ExampleTask(Task):
        def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
            print('This is an example.')
            print(f'{profile.example}')
            logger.info('Example INFO')
            logger.debug('Example DEBUG')
            logger.warning('Example WARNING')
            logger.error('Example ERROR')
            shared['example'] = 'This is an example'
            return Return.SUCCESS.value

        @classmethod
        def require(cls) -> List[Text]:
            return []

        @classmethod
        def provide(cls) -> List[Text]:
            return ['example']

        @classmethod
        def define(cls) -> List[Definition]:
            return [
                value('example', str)
            ]

    ```

    The profile is configured below named `example.toml`. When using `waterch-tasker`,
    a task must be configured with a profile file.
    You can create profile files separately using `include` feature, as well as
    define action of all tasks in one profile file.

    ```toml
    __schema__ = "waterch.tasker.launcher.Launcher"
    __name__ = "Example"
    __author__ = "Author"
    __version__ = "1.0.0"
    __email__ = "author@example.com"
    __abstract__ = ""
    [__setting__]
        [__setting__.storage]
        reference = "waterch.tasker.storage.DictStorage"
        [__setting__.log]
        stdout = true
        level = "DEBUG"
    [[__meta__]]
    reference = "example.ExampleTask"
    include = false
    path = ""
    profile = "example"
    execute = true

    [example]
    example = "Hello world"
    ```

    Run the following command.

    ```bash
    waterch-tasker launch -f example.toml
    ```

    If it run like this, your first task using waterch-tasker has been created successfully.

    ```
    --------------------
    Example (1.0.0)
    Author: Author
    E-Mail: author@example.com

    --------------------

    example.ExampleTask[0x2d76307e28]
    require:
    provide: example
    ------------------->
    This is an example.
    Hello world
    2020-05-21T11:46:25|INFO|example.ExampleTask[0x2d76307e28]>Example INFO
    2020-05-21T11:46:25|DEBUG|example.ExampleTask[0x2d76307e28]>Example DEBUG
    2020-05-21T11:46:25|WARNING|example.ExampleTask[0x2d76307e28]>Example WARNING
    2020-05-21T11:46:25|ERROR|example.ExampleTask[0x2d76307e28]>Example ERROR
    <-------------------
    Successfully finished in 0.000998 seconds.
    ```

    """

    @abstractmethod
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        """
        All activities of a task defined in this method.
        You can access configurations from profile object,
        access data from other tasks or provide data to other tasks by
        using shared. A logger is also provide with a task.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            The state defined in [`Return` enumeration][waterch.tasker.typedef.Return].
        """
        raise NotImplementedError('Please implement the task process.')

    @classmethod
    @abstractmethod
    def require(cls) -> List[Text]:
        """
        Declare the keys required by this task.
        Returns:
            A list contains all keys which your task requires.
        Warnings:
            If you want access any key from shared, please declare them in this method.
            Reading a key without declaration in `require` will be forbidden.
        """
        raise NotImplementedError('Please define required keys.')

    @classmethod
    @abstractmethod
    def provide(cls) -> List[Text]:
        """
        Declare the keys provided by this task.
        Returns:
            A list contains all keys which your task provides.
        Warnings:
            If you want access any key from shared, please declare them in this method.
            Writing a key without declaration in `provide` will be forbidden.
        """
        raise NotImplementedError('Please define provided keys.')


class ForkTask(Task):
    class ExitSignal(Exception):
        pass

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        def invoke_check(task, profile, shared, logger) -> int:
            state = task.invoke(profile, shared, logger)
            if state & Return.WRITE.value:
                shared.dump()
            if state & Return.READ.value:
                shared.load()
            if state & Return.EXIT.value:
                print('Stopped by task.')
                raise self.ExitSignal
            return state

        def execute(meta):
            task_cls = import_reference(profile.reference)
            task = task_cls()
            task_logger_name = f'{meta.reference}[{hex(hash(task))}]@{hex(hash(self))}'
            task_logger = get_logger(task_logger_name)
            task_shared = MultiTaskStorageView(storage=shared, task=task, links=[])
            if meta.include:
                task_profile = Profile.from_toml(filename=meta.path)
            else:
                task_profile = profile[meta.profile]
            state = invoke_check(task, task_profile, task_shared, task_logger)
            return \
                state, \
                task, \
                (task_profile, task_shared, task_logger)

        def retry(state: int, task, context):
            if state | Return.RETRY.value:
                return invoke_check(task, *context), task, context
            else:
                return state, task, context

        pool = Pool(profile.worker)
        execute_tasks = list(filter(
            lambda meta: meta.execute,
            profile.tasks
        ))
        try:
            results = pool.map(
                func=execute,
                iterable=execute_tasks,
                chunksize=int(len(execute_tasks) // profile.worker)
            )

            while reduce(lambda t, s: t | s[0], results, Return.SUCCESS.value):
                results = pool.map(
                    func=lambda args: retry(*args),
                    iterable=results,
                    chunksize=int(len(execute_tasks) // profile.worker)
                )

        except self.ExitSignal:
            pool.close()
            return Return.ERROR.value | Return.EXIT.value

        pool.close()
        return Return.SUCCESS.value

    @classmethod
    def require(cls) -> List[Text]:
        return []

    @classmethod
    def provide(cls) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('worker', int),
            value('reference', str),
            value('tasks', list, [
                [
                    value('include', bool),
                    value('path', str),
                    value('profile', str),
                    value('execute', bool)
                ]
            ])
        ]
