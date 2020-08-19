from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from copy import deepcopy
from logging import Logger
from typing import List, Text, Tuple, Union, Dict, Any

from waterch.tasker.mixin import value
from waterch.tasker.storage import Storage
from waterch.tasker.tasks.containers import Task
from waterch.tasker.typedef import Definition, Profile
from waterch.tasker.typedef import Return
from waterch.tasker.utils import import_reference

try:
    import torch
    from torch import nn
    from torch import optim
    from torch.utils import data
    from ignite import engine
    from ignite import metrics
except ImportError as ie:
    raise RuntimeError('Tasks in this module needs pytorch and pytorch-ignite modules')


class TrainTask(Task, metaclass=ABCMeta):
    """
    <b>waterch.tasker.tasks.torch.TrainTask</b>

    The fundamental task construction to train a PyTorch model by provided data loaders.

    You need to run a task providing two data loaders named "train_loader" and "validate_loader"
    in shared storage before this task as well as the trained model will be stored into "model"
    label in shared storage.

    However, many actions should be redefined by users when implementing `TrainTask`. You can
    also implement [`SimpleTrainTask`][waterch.tasker.tasks.torch.SimpleTrainTask] to boost
    your development.
    """

    def __init__(self, prefix: Text = None):
        if prefix is not None:
            self.PROVIDE_KEY = f'{prefix}_model'
        else:
            self.PROVIDE_KEY = 'model'

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        train_loader = shared['train_loader']
        validate_loader = shared['validate_loader']

        model = self.create_model(profile.model, shared, logger)
        optimizer_return = self.create_optimizer(model, profile.optimizer, shared, logger)
        if isinstance(optimizer_return, tuple):
            optimizer = optimizer_return[0]
            scheduler = optimizer_return[1] if len(optimizer_return) > 1 else None
        else:
            optimizer = optimizer_return
            scheduler = None
        loss = self.create_loss(profile.loss, shared, logger)
        metrics_dict = OrderedDict()
        self.more_metrics(metrics_dict)

        def prepare_batch(batch, device=None, non_blocking=False):
            return self.prepare_batch(deepcopy(batch), device, non_blocking)

        trainer = self.create_trainer(
            model, optimizer, loss,
            device=torch.device(profile.device),
            non_blocking=profile.non_blocking,
            prepare_batch=prepare_batch,
        )

        evaluator = self.create_evaluator(
            model, metrics_dict,
            device=torch.device(profile.device),
            non_blocking=profile.non_blocking,
            prepare_batch=prepare_batch,
        )

        @trainer.on(engine.Events.STARTED)
        def on_epoch_started(engine_):
            return self.on_epoch_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.COMPLETED)
        def on_completed(engine_):
            return self.on_completed(engine_, profile, shared, logger)

        @trainer.on(engine.Events.ITERATION_STARTED)
        def on_iteration_started(engine_):
            return self.on_iteration_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.ITERATION_COMPLETED)
        def on_iteration_completed(engine_):
            return self.on_iteration_completed(engine_, profile, shared, logger)

        @trainer.on(engine.Events.EPOCH_STARTED)
        def on_epoch_started(engine_):
            return self.on_epoch_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def on_epoch_completed(engine_):
            evaluator.run(validate_loader)
            if scheduler is not None:
                scheduler.step(engine_.state.output)
            return self.on_epoch_completed(engine_, evaluator.state.metrics, profile, shared, logger)

        trainer.run(
            train_loader,
            max_epochs=profile.max_epochs
        )
        shared[self.PROVIDE_KEY] = model
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        """
        The task requires 2 items, including "train_loader" and "validate_loader.

        Returns:
            "train_loader" and "validate_loader
        """
        return ['train_loader', 'validate_loader']

    def provide(self) -> List[Text]:
        """
        The task provides 1 item, including "model".

        Returns:
            "model"
        """
        return [self.PROVIDE_KEY]

    def remove(self) -> List[Text]:
        """
        This task removes nothing.

        Returns:
            Nothing
        """
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        """
        ```toml
        __schema__ = "conv_train.ConvTrainTask"
        device = ""
        non_blocking = true
        max_epochs = 0
        [model]
        # You must implement define_optimizer to define the profile of the model.

        [optimizer]
        # You must implement define_optimizer to define the profile of the optimizer.

        [loss]
        # You must implement define_optimizer to define the profile of the loss function.
        ```
        """
        return [
            value('model', list, cls.define_model()),
            value('optimizer', list, cls.define_optimizer()),
            value('loss', list, cls.define_loss()),
            value('device', str),
            value('non_blocking', bool),
            value('max_epochs', int),
        ]

    @abstractmethod
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        """
        The function to create model instance with defined profile.

        Args:
            profile: Runtime profile defined in TOML file of model.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            A new model instance to train in this task.
        """
        raise NotImplementedError(f'Please create the model in create_model')

    @classmethod
    @abstractmethod
    def define_model(cls):
        """
        A profile template of model need to be implemented by user.

        Returns:
            Definition of model profile.
        """
        raise NotImplementedError(f'Please define the model profile in define_model')

    @abstractmethod
    def create_optimizer(self, model: nn.Module, profile: Profile, shared: Storage, logger: Logger) -> Union[
        optim.Optimizer, Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]
    ]:
        """
        The function to create optimizer instance with defined profile.

        Args:
            model: The model need to be optimized.
            profile: Runtime profile defined in TOML file of optimizer.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            A new optimizer instance to train in this task. A optional scheduler can also be returned with
            the optimizer.
        """
        raise NotImplementedError(f'Please create the optimizer (or with a lr_scheduler) in create_optimizer')

    @classmethod
    @abstractmethod
    def define_optimizer(cls):
        """
        A profile template of optimizer need to be implemented by user.

        Returns:
            Definition of optimizer profile.
        """
        raise NotImplementedError(f'Please define the optimizer profile in define_optimizer')

    @abstractmethod
    def create_loss(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        """
        The function to create loss function instance with defined profile.

        Args:
            profile: Runtime profile defined in TOML file of model.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            A new loss function instance to train in this task.
        """
        raise NotImplementedError(f'Please create the loss function in create_loss')

    @classmethod
    @abstractmethod
    def define_loss(cls):
        """
        A profile template of loss function need to be implemented by user.

        Returns:
            Definition of loss function profile.
        """
        raise NotImplementedError(f'Please define the loss function profile in define_loss')

    def more_metrics(self, metrics_: OrderedDict):
        """
        All metric method you want to monitor in training epochs.

        Args:
            metrics_: The metrics collections.

        Returns:
            Nothing
        """
        pass

    def prepare_batch(self, batch, device=None, non_blocking=False):
        """
        The method to prepare batch provided by data loader. The value returned by this method
        should be a tuple with the format such as (x, y).

        Args:
            batch: Raw batch produced by data loader
            device: The device which model should be moved to
            non_blocking: If True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect.

        Returns:
            The output tuple with format (x, y)

        """
        x, y = batch
        return (
            engine.convert_tensor(x, device=device, non_blocking=non_blocking),
            engine.convert_tensor(y, device=device, non_blocking=non_blocking)
        )

    def on_epoch_started(self, engine_: engine.Engine, profile: Profile, shared: Storage, logger: Logger):
        """
        Define the actions when an epoch started.

        Args:
            engine_: Training context
            profile: The profile of task
            shared: Shared storage of task
            logger: Logger of task.

        Returns:
            Nothing
        """
        pass

    def on_epoch_completed(self, engine_: engine.Engine, metrics_: Dict[Text, Any], profile: Profile, shared: Storage,
                           logger: Logger):
        """
        Define the actions when an epoch completed.

        Args:
            engine_: Training context
            metrics_: Calculated metric values of validate dataset
            profile: The profile of task
            shared: Shared storage of task
            logger: Logger of task.

        Returns:
            Nothing
        """
        pass

    def on_started(self, engine_: engine.Engine, profile: Profile, shared: Storage, logger: Logger):
        """
        Define the actions when the task completed.

        Args:
            engine_: Training context
            profile: The profile of task
            shared: Shared storage of task
            logger: Logger of task.

        Returns:
            Nothing
        """
        pass

    def on_completed(self, engine_: engine.Engine, profile: Profile, shared: Storage, logger: Logger):
        """
        Define the actions when the task completed.

        Args:
            engine_: Training context
            profile: The profile of task
            shared: Shared storage of task
            logger: Logger of task.

        Returns:
            Nothing
        """
        pass

    def on_iteration_started(self, engine_: engine.Engine, profile: Profile, shared: Storage, logger: Logger):
        """
        Define the actions when an iteration started.

        Args:
            engine_: Training context
            profile: The profile of task
            shared: Shared storage of task
            logger: Logger of task.

        Returns:
            Nothing
        """
        pass

    def on_iteration_completed(self, engine_: engine.Engine, profile: Profile, shared: Storage, logger: Logger):
        """
        Define the actions when an iteration completed.

        Args:
            engine_: Training context
            profile: The profile of task
            shared: Shared storage of task
            logger: Logger of task.

        Returns:
            Nothing
        """
        pass

    def create_trainer(
            self, model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform=lambda x, y, y_pred, loss: loss.item()
    ):
        return engine.create_supervised_trainer(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform
        )

    def create_evaluator(
            self, model, metrics, device, non_blocking, prepare_batch, output_transform=lambda x, y, y_pred: (y_pred, y,)
    ):
        return engine.create_supervised_evaluator(
            model, metrics, device, non_blocking, prepare_batch, output_transform
        )


class SimpleTrainTask(TrainTask, metaclass=ABCMeta):
    """
    <b>waterch.tasker.tasks.torch.SimpleTrainTask</b>

    An easy to use base class of task for training models. You don't need to modify the code
    to create optimizer and loss function, instead, you only need to implement the model.
    """

    def create_optimizer(self, model: nn.Module, profile: Profile, shared: Storage, logger: Logger) -> Union[
        optim.Optimizer, Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]
    ]:
        """
        Automatically create optimizer from profile.

        ```toml
        [optimizer]
        reference = ""
            [optimizer.kwargs]
            # the fields in kwargs should be filled according to the definition of optimizer class.
            [optimizer.scheduler] # optional
            reference = ""
                [optimizer.scheduler.kwargs]
                # the fields in kwargs should be filled according to the definition of scheduler class.
        ```

        Args:
            model: The model need to be optimized.
            profile: Runtime profile defined in TOML file of optimizer.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            A new optimizer instance to train in this task. A optional scheduler can also be returned with
            the optimizer.
        """
        optimizer_class = import_reference(profile.reference)
        optimizer = optimizer_class(params=model.parameters(), **profile.kwargs)
        if 'scheduler' in profile:
            scheduler_class = import_reference(profile.scheduler.reference)
            scheduler = scheduler_class(optimizer=optimizer, **profile.scheduler.kwargs)
            return optimizer, scheduler
        else:
            return optimizer

    @classmethod
    def define_optimizer(cls):
        return [
            value('reference', str),
            value('kwargs', list),
            value('scheduler', list, [
                value('reference', str),
                value('kwargs', list),
                value('# OPTIONAL', str)
            ])
        ]

    def create_loss(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        """
        Automatically create loss function from profile.

        Args:
            profile: Runtime profile defined in TOML file of loss function.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            A new loss function instance to train in this task.
        """
        loss_class = import_reference(profile.reference)
        loss = loss_class(**profile.kwargs)
        return loss

    @classmethod
    def define_loss(cls):
        return [
            value('reference', str),
            value('kwargs', list)
        ]

    def on_epoch_completed(self,
                           engine_: engine.Engine, metrics_: Dict[Text, Any], profile: Profile, shared: Storage,
                           logger: Logger):
        """
        Display metrics output of each epochs which declared in `more_metrics`.
        """
        logger.info(f'EPOCH {engine_.state.epoch} | ' + ' | '.join(
            map(
                lambda it: f'{it[0]}: {it[1]}',
                metrics_.items()
            )
        ))

    def on_iteration_completed(self, engine_: engine.Engine, profile: Profile, shared: Storage, logger: Logger):
        """
        Display loss output of each iterations.
        """
        epoch_iteration = engine_.state.iteration % engine_.state.epoch_length
        if epoch_iteration == 0:
            epoch_iteration = engine_.state.epoch_length
        logger.debug(f'ITERATION {epoch_iteration} | output: {engine_.state.output}')


class DataLoaderTask(Task):
    """
    <b>waterch.tasker.tasks.torch.DataLoaderTask</b>

    The fundamental task construction to provide data loaders.

    Please declare the prefix of data loader in shared storage in reference
    field of meta definitions.

    Examples:
        ```toml
        # Data loader of validation.
        [[__meta__]]
        reference = "cifar_loader.CIFARDataLoaderTask:validate"
        include = false
        path = ""
        profile = "validate_loader"
        execute = true

        # Data Loader of training.
        [[__meta__]]
        reference = "conv_train.ConvTrainTask:train"
        include = false
        path = ""
        profile = "train"
        execute = true
        ```
    """

    def __init__(self, _type: Text = None):
        if _type is None:
            self.PROVIDE_KEY = 'loader'
        else:
            self.PROVIDE_KEY = f'{_type}_loader'

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        dataset = self.create_dataset(profile.dataset, shared, logger)

        assert isinstance(dataset, data.Dataset)

        loader_params = dict(profile.loader)
        loader_params['dataset'] = dataset

        assert profile.sampler_type in ('sampler', 'batch_sampler', 'none')

        if profile.sampler_type != 'none':
            loader_params[profile.sampler_type] = self.create_sampler(
                dataset, profile.sampler_type == 'batch_sampler', profile.loader,
                shared, logger
            )
            if profile.sampler_type == 'batch_sampler':
                if 'batch_size' in loader_params:
                    loader_params.pop('batch_size')
                if 'shuffle' in loader_params:
                    loader_params.pop('shuffle')
                if 'sampler' in loader_params:
                    loader_params.pop('sampler')
                if 'drop_last' in loader_params:
                    loader_params.pop('drop_last')

        shared[self.PROVIDE_KEY] = data.DataLoader(**loader_params)
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return []

    def provide(self) -> List[Text]:
        return [self.PROVIDE_KEY]

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        """
        ```toml
        __schema__ = "waterch.tasker.tasks.torch.DataLoaderTask"
        sampler_type = ""

        [loader]
        batch_size = 0
        shuffle = true
        num_workers = 0
        pin_memory = true
        drop_last = true
        ```
        """
        return [
            value('dataset', list, cls.define_dataset()),
            value('loader', list, [
                value('batch_size', int),
                value('shuffle', bool),
                value('num_workers', int),
                value('pin_memory', bool),
                value('drop_last', bool),
            ]),
            value('sampler_type', str),
            value('sampler', list, cls.define_sampler())
        ]

    @abstractmethod
    def create_dataset(self, profile: Profile, shared: Storage, logger: Logger) -> data.Dataset:
        """
        The function to create dataset instance with defined profile.

        Args:
            profile: Runtime profile defined in TOML file of dataset.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            A new dataset instance.
        """
        raise NotImplementedError('Please create the dataset in create_dataset')

    @classmethod
    @abstractmethod
    def define_dataset(cls):
        """
        A profile template of dataset need to be implemented by user.

        Returns:
            Definition of dataset profile.
        """
        raise NotImplementedError('Please define the dataset profile in define_dataset')

    @abstractmethod
    def create_sampler(self, dataset: data.Dataset, batch_sampler: bool, profile: Profile, shared: Storage,
                       logger: Logger):
        """
        The function to create sampler instance with defined profile.

        Args:
            dataset: The dataset instance need to be loaded.
            batch_sampler: Whether to use batch_sampler.
            profile: Runtime profile defined in TOML file of sampler or batch sampler.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            A new sampler or batch sampler instance.
        """
        if batch_sampler:
            raise NotImplementedError('Please create the batch sampler in create_sampler')
        else:
            raise NotImplementedError('Please create the sampler in create_sampler')

    @classmethod
    @abstractmethod
    def define_sampler(cls):
        raise NotImplementedError('Please define the sampler or batch sampler profile in define_sampler')


class SimpleDataLoaderTask(DataLoaderTask):
    """
    <b>waterch.tasker.tasks.torch.SimpleDataLoaderTask</b>

    An easy to use base class of task for providing data loader. You can
    create data loader only with reference of dataset and related profile.
    """

    def create_dataset(self, profile: Profile, shared: Storage, logger: Logger) -> data.Dataset:
        dataset_cls = import_reference(profile.reference)
        return dataset_cls(**profile.kwargs)

    @classmethod
    def define_dataset(cls):
        return [
            value('reference', str),
            value('kwargs', list, [])
        ]

    def create_sampler(self, dataset: data.Dataset, batch_sampler: bool, profile: Profile, shared: Storage,
                       logger: Logger):
        return None

    @classmethod
    def define_sampler(cls):
        return []
