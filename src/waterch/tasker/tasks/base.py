__all__ = [
    'Task',
]

from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Text

from waterch.tasker import Profile, Definition, Return, value
from waterch.tasker.mixin import ProfileMixin
from waterch.tasker.storage import Storage


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

    The profile is configured below named `example.toml`.

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

        Returns:

        """
        raise NotImplementedError('Please define required keys.')

    @classmethod
    @abstractmethod
    def provide(cls) -> List[Text]:
        """
        :return:
        """
        raise NotImplementedError('Please define provided keys.')
