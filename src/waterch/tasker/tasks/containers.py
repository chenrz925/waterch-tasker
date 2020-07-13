__all__ = [
    'Task',
]

from abc import ABCMeta, abstractmethod
from functools import reduce
from logging import Logger, getLogger as get_logger
from multiprocessing.dummy import Pool
from typing import List, Text

from waterch.tasker.mixin import ProfileMixin, value
from waterch.tasker.storage import Storage, ForkStorageView, MapStorageView, ReduceStorageView
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

    Examples:
        We will introduce a example to describe the usage of `Task` class.
        For example, we construct a class named `ExampleTask` in "example.py",
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

            def require(self) -> List[Text]:
                return []

            def provide(self) -> List[Text]:
                return ['example']

            def remove(self) -> List[Text]:
                return []

            def define(self) -> List[Definition]:
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
        using shared. A logger is also provided with a task.
        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
        Returns:
            The state defined in [`Return` enumeration][waterch.tasker.typedef.Return].
        """
        raise NotImplementedError('Please implement the task process.')

    @abstractmethod
    def require(self) -> List[Text]:
        """
        Declare the keys required by this task.
        Returns:
            A list contains all keys which your task requires.
        Warnings:
            If you want access any key from shared, please declare them in this method.
            Reading a key without declaration in `require` will be forbidden.
        """
        raise NotImplementedError('Please define required keys.')

    @abstractmethod
    def provide(self) -> List[Text]:
        """
        Declare the keys provided by this task.
        Returns:
            A list contains all keys which your task provides.
        Warnings:
            If you want access any key from shared, please declare them in this method.
            Writing a key without declaration in `provide` will be forbidden.
        """
        raise NotImplementedError('Please define provided keys.')

    @abstractmethod
    def remove(self) -> List[Text]:
        """
        Declare the keys removed by this task.
        Returns:
            A list contains all keys which your task removes.
        Warnings:
            If you want access any key from shared, please declare them in this method.
            Deleting a key without declaration in `remove` will be forbidden.
        """
        raise NotImplementedError('Please define removed keys.')


class _ContainerTask(Task, metaclass=ABCMeta):
    def __init__(self):
        super(_ContainerTask, self).__init__()
        self._require = []
        self._provide = []
        self._remove = []

    class ExitSignal(Exception):
        pass

    @classmethod
    def _invoke_check(cls, task, profile, shared, logger) -> int:
        state = task.invoke(profile, shared, logger)
        if state & Return.WRITE.value:
            shared.dump()
        if state & Return.READ.value:
            shared.load()
        if state & Return.EXIT.value:
            print('Stopped by task.')
            raise cls.ExitSignal
        return state

    def _register_task(self, task: Task):
        for key in task.require():
            if key not in self._require:
                self._require.append(key)
        for key in task.provide():
            if key not in self._provide:
                self._provide.append(key)
        for key in task.remove():
            if key not in self._remove:
                self._remove.append(key)

    def require(self) -> List[Text]:
        """
        This task requires fields mirrored from referenced task.

        Returns:
            Same to referenced task.
        """
        return self._require

    def provide(self) -> List[Text]:
        """
        This task provides fields mirrored from referenced task.

        Returns:
            Same to referenced task.
        """
        return self._provide

    def remove(self) -> List[Text]:
        """
        This task removes fields mirrored from referenced task.

        Returns:
            Same to referenced task.
        """
        return self._remove


class ForkTask(_ContainerTask, ProfileMixin):
    """
    <b>waterch.tasker.tasks.containers.ForkTask</b>

    The container task contains same tasks with different profiles.

    Provided fields in `shared` will be forked to multiple fields.
    To process them, you should execute a [`MapTask` instance][waterch.tasker.tasks.containers.MapTask]
    or a [`ReduceTask` instance][waterch.tasker.tasks.containers.ReduceTask].

    Examples:
        You can execute a `ForkTask` using a profile file just like the following one.
        ```toml
        __schema__ = "waterch.tasker.tasks.containers.ForkTask"
        worker = 10
        reference = "example.ExampleTask"

        [[tasks]]
        include = false
        path = ""
        profile = "task1"
        execute = true

        [task1]
        example = "Hello world 1"

        [[tasks]]
        include = false
        path = ""
        profile = "task2"
        execute = true

        [task2]
        example = "Hello world 2"

        [[tasks]]
        include = false
        path = ""
        profile = "task3"
        execute = true

        [task3]
        example = "Hello world 3"
        ```
    """

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        def execute(meta):
            task_cls = import_reference(profile.reference)
            task = task_cls()
            self._register_task(task)
            task_logger_name = f'{profile.reference}[{hex(hash(task))}]@{hex(hash(self))}'
            task_logger = get_logger(task_logger_name)
            task_shared = ForkStorageView(storage=shared.storage(), task=task)
            if meta.include:
                task_profile = Profile.from_toml(filename=meta.path)
            else:
                task_profile = profile[meta.profile]
            state = self._invoke_check(task, task_profile, task_shared, task_logger)
            return \
                state, \
                task, \
                (task_profile, task_shared, task_logger)

        def retry(state: int, task, context):
            if state | Return.RETRY.value:
                return self._invoke_check(task, *context), task, context
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
    def define(cls) -> List[Definition]:
        """
        Examples:
            ```toml
            __schema__ = "waterch.tasker.tasks.containers.ForkTask"
            worker = 0
            reference = ""
            [[tasks]]
            include = true
            path = ""
            profile = ""
            execute = true
            ```

        Returns:
            Schema of profile
        """
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


class MapTask(_ContainerTask, ProfileMixin):
    """
    <b>waterch.tasker.tasks.containers.MapTask</b>

    The container task mapping forked shared data to new fields in `shared` storage.

    Provided fields in `shared` will be mapped to new fields.
    You can also combine fields into dependent single field by a
    [`ReduceTask` instance][waterch.tasker.tasks.containers.ReduceTask]
    """
    STORAGE_VIEW_CLASS = MapStorageView

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        def execute(idx: Text):
            task = task_cls()
            self._register_task(task)
            task_storage = self.STORAGE_VIEW_CLASS(storage=shared.storage(), task=task, mirror=idx)
            task_logger_name = f'{profile.reference}[{hex(hash(task))}]@{hex(hash(self))}'
            task_logger = get_logger(task_logger_name)
            return self._invoke_check(task, task_profile, task_storage, task_logger), task, (
                task_profile, task_storage, task_logger
            )

        def retry(state, task, context):
            if state & Return.RETRY.value:
                return self._invoke_check(task, *context), task, context
            else:
                return state, task, context

        task_cls = import_reference(profile.reference)
        pool = Pool(profile.worker)
        if profile.include:
            task_profile = Profile.from_toml(profile.path)
        else:
            task_profile = profile[profile.profile]
        state_map = tuple(pool.map(
            func=execute,
            iterable=shared[self.STORAGE_VIEW_CLASS.META_KEY],
            chunksize=int(len(shared[self.STORAGE_VIEW_CLASS.META_KEY]) // profile.worker)
        ))
        while reduce(lambda t, s: t | s[0], state_map, Return.SUCCESS.value) & Return.RETRY.value:
            state_map = tuple(pool.map(
                func=retry,
                iterable=shared[self.STORAGE_VIEW_CLASS.META_KEY],
                chunksize=int(len(shared[self.STORAGE_VIEW_CLASS.META_KEY]) // profile.worker)
            ))
        return reduce(lambda t, s: t | s[0], state_map, Return.SUCCESS.value)

    @classmethod
    def define(cls) -> List[Definition]:
        """
        Examples:
            ```toml
            __schema__ = "waterch.tasker.tasks.containers.MapTask"
            worker = 0
            reference = ""
            include = true
            path = ""
            profile = ""
            ```

        Returns:
            Schema of profile
        """
        return [
            value('worker', int),
            value('reference', str),
            value('include', bool),
            value('path', str),
            value('profile', str),
        ]


class ReduceTask(MapTask):
    """
    <b>waterch.tasker.tasks.containers.ReduceTask</b>

    The container task reduce sequenced fields in shared data into single dependent field.

    You can reuse the key of sequenced fields when writing.
    """
    STORAGE_VIEW_CLASS = ReduceStorageView

    def define(cls) -> List[Definition]:
        """
        Examples:
            ```toml
            __schema__ = "waterch.tasker.tasks.containers.ReduceTask"
            worker = 0
            reference = ""
            include = true
            path = ""
            profile = ""
            ```

        Returns:
            Schema of profile
        """
        return super(ReduceTask, cls).define()
