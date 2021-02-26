__all__ = [
    'DataLoaderTask',
    'SimpleDataLoaderTask',
    'TrainTask',
    'SimpleTrainTask'
]

from abc import abstractmethod
from copy import deepcopy
from logging import Logger
from typing import List, Text, Tuple, Dict, Any

from ..mixin import value
from ..storages.basic import Storage
from ..tasks.containers import Task
from ..typedef import Definition, Profile
from ..typedef import Return

try:
    import torch
    from torch import nn
    from torch import optim
    from torch.utils import data
    from ignite import engine
    from ignite import metrics
    from ignite.utils import convert_tensor
except ImportError as ie:
    raise RuntimeError('Tasks in module tasker.contrib.torch needs pytorch and pytorch-ignite modules')


class DataLoaderTask(Task):
    """
    <i>tasker.contrib.torch.DataLoaderTask</i>

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
        __schema__ = "tasker.contrib.torch.DataLoaderTask"
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
    <i>tasker.contrib.torch.SimpleDataLoaderTask</i>

    An easy to use base class of task for providing data loader. You can
    create data loader only with reference of dataset and related profile.
    """

    def create_dataset(self, profile: Profile, shared: Storage, logger: Logger) -> data.Dataset:
        dataset_cls = profile.reference
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


class TrainTask(Task):
    """
    <i>tasker.contrib.torch.TrainTask</i>

    The fundamental task construction to train a model by
    provided data loaders.

    You need to run a task providing two data loaders named
    "train_loader" and "validate_loader" in shared storage
    before this task as well as the trained model will be
    stored into "<something>_model" or "model" label in shared
    storage.

    However, many actions should be redefined by user when
    implementing `TrainTask`. You can also implement [SimpleTrainTask][tasker.contrib.torch.SimpleTrainTask]
    to boost your development.

    If you want to store the model with a prefix, please fill the
    prefix name into the first parameter when referencing it.
    """
    def __init__(self, *args, **kwargs):
        if len(args) >= 1:
            self.prefix = args[0]
        else:
            self.prefix = None

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        torch.manual_seed(profile.seed if 'seed' in profile else 0x3a4e)

        if 'model' in profile:
            device = profile.device if 'device' in profile else 'cpu'
            model = self.create_model(profile.model, shared, logger).to(device)
        else:
            raise RuntimeError('Missing profile field "model" to define the model.')

        if 'train_loader' in shared:
            train_loader: torch.utils.data.DataLoader = shared['train_loader']
        else:
            raise RuntimeError('Missing shared object "train_loader" to provide training sets.')

        if 'validate_loader' in shared:
            validate_loader: torch.utils.data.DataLoader = shared['validate_loader']
        else:
            raise RuntimeError('Missing shared object "validate_loader" to provide validating sets.')

        if 'optimizer' in profile:
            optimizer, lr_scheduler = self.create_optimizer(profile.optimizer, shared, logger, model)
        else:
            raise RuntimeError('Missing profile field "optimizer" to define the optimizer.')

        if 'loss_function' in profile:
            loss_function = self.create_loss_function(profile.loss_function, shared, logger)
        else:
            raise RuntimeError('Missing profile field "loss_function" to define the loss function.')

        trainer = self.create_trainer(
            profile, shared, logger, model, loss_function, optimizer, lr_scheduler,
            profile.train_output_transform if 'train_output_transform' in profile else lambda x, y, y_pred,
                                                                                              loss: loss.item()
        )
        evaluator = self.create_evaluator(
            profile, shared, logger, model, loss_function, optimizer, lr_scheduler,
            profile.evaluate_output_transform if 'evaluate_output_transform' in profile else lambda x, y, y_pred: (
                y_pred, y)
        )

        context = {}

        @evaluator.on(engine.Events.COMPLETED)
        def display_metrics(_engine: engine.Engine):
            logger.info('EVALUATE EPOCH {} | {}'.format(trainer.state.epoch, ' | '.join(map(
                lambda it: '{}: {}'.format(it[0], repr(it[1]).replace('\n', ' ')),
                _engine.state.metrics.items(),
            ))))

        @evaluator.on(engine.Events.COMPLETED)
        def store_model(_engine: engine.Engine):
            if 'compare_metric' not in context:
                context['compare_metric'] = float('-inf')

            if 'compare_by' not in profile or len(profile.compare_by) == 0:
                compare_by = 'loss'
                sign = '-'
            else:
                compare_by = profile.compare_by
                if compare_by[0] in '+-':
                    sign = compare_by[0]
                    compare_by = compare_by[1:]
                else:
                    sign = '+'

            if compare_by not in _engine.state.metrics:
                logger.warning(f'Not found "{compare_by}" in metrics. Fall back to loss.')
                compare_by = 'loss'
                sign = '-'

            metric_value = _engine.state.metrics[compare_by]
            if sign == '-':
                metric_value = -metric_value

            if metric_value > context['compare_metric']:
                context['compare_metric'] = metric_value
                shared[self._model_key] = deepcopy(model.eval())
                logger.info(f'Stored the model with {compare_by} of {metric_value}.')

        @trainer.on(engine.Events.ITERATION_COMPLETED(
            every=int(len(train_loader) * (profile.loss_display if 'loss_display' in profile else 0.1))
        ))
        def display_loss(_engine: engine.Engine):
            epoch_iteration = _engine.state.iteration % _engine.state.epoch_length
            if epoch_iteration == 0:
                epoch_iteration = _engine.state.epoch_length
            logger.info('TRAIN EPOCH {} ITERATION {} | output: {}'.format(
                _engine.state.epoch, epoch_iteration, _engine.state.output
            ))

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def evaluate(_engine: engine.Engine):
            evaluator.run(
                validate_loader,
            )

        self.register_handlers(profile, shared, logger, model, trainer, evaluator, context)

        trainer.run(
            train_loader,
            max_epochs=profile.max_epochs if 'max_epochs' in profile else 100,
        )

        return Return.SUCCESS

    @property
    def _model_key(self):
        return 'model' if self.prefix is None else f'{self.prefix}_model'

    def require(self) -> List[Text]:
        """
        The task requires 2 items, including "train_loader" and "validate_loader.

        Returns:
            "train_loader" and "validate_loader"
        """
        return [
            'train_loader',
            'validate_loader',
        ]

    def provide(self) -> List[Text]:
        """
        The task provides 1 item, including "model" or "<something>_model".

        Returns:
            "model" or "<something>_model"
        """
        return [
            self._model_key
        ]

    def remove(self) -> List[Text]:
        """
        This task removes nothing.

        Returns:
            nothing
        """
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        """
        Define the schema of `TrainTask`.

        Returns:
            Schema of `TrainTask`.

        Examples:
            See Also [Train AlexNet on Place 365 dataset](https://github.com/chenrz925/waterch-tasker/blob/master/examples/place365_alexnet.toml)
        """
        return [
            value('model', list, [
                value('reference', str),
                value('kwargs', list, [])
            ]),
            value('loss_function', list, [
                value('reference', str),
                value('kwargs', list, [])
            ]),
            value('metrics', list, []),
            value('optimizer', list, [
                value('reference', str),
                value('kwargs', list, []),
                value('scheduler', list, [
                    value('reference', str),
                    value('kwargs', list)
                ])
            ])
        ]

    @abstractmethod
    def create_model(self, profile: Profile, shared: Storage, logger: Logger, **kwargs) -> nn.Module:
        """
        Implement `create_model` to build the PyTorch model.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            Always return [SUCCESS][tasker.typedef.Return.SUCCESS].

        Notes:
            The profile should be attribute "model" in the task profile.
        """
        raise NotImplementedError('Please create the model in create_model')

    @abstractmethod
    def create_loss_function(self, profile: Profile, shared: Storage, logger: Logger, **kwargs) -> nn.Module:
        """
        Implement `create_loss_function` to build the PyTorch loss function.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            Always return [SUCCESS][tasker.typedef.Return.SUCCESS].

        Notes:
            The profile should be attribute "loss_function" in the task profile.
        """
        raise NotImplementedError('Please create the loss function in create_loss_function')

    @abstractmethod
    def create_optimizer(self, profile: Profile, shared: Storage, logger: Logger, model: nn.Module,
                         **kwargs) -> optim.Optimizer:
        """
        Implement `create_optimizer` to build the PyTorch optimizer.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            Always return [SUCCESS][tasker.typedef.Return.SUCCESS].

        Notes:
            The profile should be attribute "optimizer" in the task profile.
        """
        raise NotImplementedError('Please create the optimizer in create_optimizer')

    def prepare_train_batch(
            self, profile: Profile, shared: Storage, logger: Logger,
            batch: Tuple[torch.Tensor], device: Text, non_blocking: bool = False
    ):
        """
        Preparing batch of samples when training. Implement this function to
        customize.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            batch: Raw batch provided by the data loader.
            device: Which device of the batch.
            non_blocking: Whether the action of moving the batch is blocking.

        Returns:
            Prepared batch.
        """
        x, y = batch
        return (
            convert_tensor(x, device=torch.device(device), non_blocking=non_blocking),
            convert_tensor(y, device=torch.device(device), non_blocking=non_blocking),
        )

    def prepare_validate_batch(
            self, profile: Profile, shared: Storage, logger: Logger,
            batch: Tuple[torch.Tensor], device: Text, non_blocking: bool = False
    ):
        """
        Preparing batch of samples when validating. Implement this function to
        customize.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            batch: Raw batch provided by the data loader.
            device: Which device of the batch.
            non_blocking: Whether the action of moving the batch is blocking.

        Returns:
            Prepared batch.
        """
        x, y = batch
        return (
            convert_tensor(x, device=torch.device(device), non_blocking=non_blocking),
            convert_tensor(y, device=torch.device(device), non_blocking=non_blocking),
        )

    def create_trainer(
            self, profile: Profile, shared: Storage, logger: Logger,
            model: nn.Module, loss_function: nn.Module, optimizer: optim.Optimizer, lr_scheduler: Any,
            output_transform=lambda x, y, y_pred, loss: loss.item(),
            **kwargs
    ) -> engine.Engine:
        """
        Build the trainer engine. Re-implement this function when you
        want to customize the updating actions of training.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            model: The model to train.
            loss_function: The loss function to train.
            optimizer: The optimizer to train.
            lr_scheduler: The scheduler to control the learning rate.
            output_transform: The action to transform the output of the model.

        Returns:
            The trainer engine.
        """
        if 'device' in profile:
            device_type = profile.device
        else:
            device_type = 'cpu'

        if 'non_blocking' in profile:
            non_blocking = profile.non_blocking
        else:
            non_blocking = False

        if 'deterministic' in profile:
            deterministic = profile.deterministic
        else:
            deterministic = False

        def _update(_engine: engine.Engine, _batch: Tuple[torch.Tensor]):
            model.train()
            optimizer.zero_grad()
            x, y = self.prepare_train_batch(profile, shared, logger, _batch, device=device_type,
                                            non_blocking=non_blocking)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step(loss)

            return output_transform(x, y, y_pred, loss)

        trainer = engine.Engine(_update) if not deterministic else engine.DeterministicEngine(_update)

        return trainer

    def register_metrics(
            self, profile: Profile, shared: Storage, logger: Logger,
            _metrics: Dict
    ):
        """
        Register the metric methods.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            _metrics: The metrics dictionary to register.

        Returns:
            The metrics dictionary.
        """

        return _metrics

    def register_handlers(self, profile, shared, logger, model, trainer, evaluator, context):
        """

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            model: The model to train.
            trainer: The trainer of the model.
            evaluator: The evaluator of the model.
            context: The context dictionary to store states in handlers.

        Returns:
            Nothing.
        """
        pass

    def create_evaluator(
            self, profile: Profile, shared: Storage, logger: Logger,
            model: nn.Module, loss_function: nn.Module, optimizer: optim.Optimizer, lr_scheduler: Any,
            output_transform=lambda x, y, y_pred: (y_pred, y),
            **kwargs
    ) -> engine.Engine:
        """

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            model: The model to train.
            loss_function: The loss function to train.
            optimizer: The optimizer to train.
            lr_scheduler: The scheduler to control the learning rate.
            output_transform: The action to transform the output of the model.

        Returns:
            The evaluator engine.
        """

        if 'device' in profile:
            device_type = profile.device
        else:
            device_type = 'cpu'

        if 'non_blocking' in profile:
            non_blocking = profile.non_blocking
        else:
            non_blocking = False

        if 'deterministic' in profile:
            deterministic = profile.deterministic
        else:
            deterministic = False

        _metrics = {}
        self.register_metrics(profile, shared, logger, _metrics)

        def _inference(_engine: engine.Engine, _batch: Tuple[torch.Tensor]):
            model.eval()
            with torch.no_grad():
                x, y = self.prepare_validate_batch(profile, shared, logger, _batch, device=device_type,
                                                   non_blocking=non_blocking)
                y_pred = model(x)
                return output_transform(x, y, y_pred)

        evaluator = engine.DeterministicEngine(_inference) if deterministic else engine.Engine(_inference)

        for name, metric in _metrics.items():
            metric.attach(evaluator, name)

        return evaluator


class SimpleTrainTask(TrainTask):
    """
    <i>tasker.contrib.torch.SimpleTrainTask</i>

    An easy to use base class of task for training model. You can
    create model only with reference of dataset and related profile.

    Examples:
        See Also [Train AlexNet on Place 365 dataset](https://github.com/chenrz925/waterch-tasker/blob/master/examples/place365_alexnet.toml)

    """
    def create_model(self, profile: Profile, shared: Storage, logger: Logger, **kwargs) -> nn.Module:
        """
        You can build the model class implementing [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
        And the parameters of the model class can be passed by `kwargs`.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            New model instance.
        """
        if 'reference' in profile:
            clz = profile.reference
            if not issubclass(clz, nn.Module):
                raise RuntimeError('Referenced class is not a subclass of torch.nn.Module.')
        else:
            raise RuntimeError('Missing field "reference" in the model profile.')

        if 'kwargs' in profile:
            kwargs = profile.kwargs
        else:
            kwargs = {}

        model = clz(**kwargs)
        return model

    def create_loss_function(self, profile: Profile, shared: Storage, logger: Logger, **kwargs) -> nn.Module:
        """
        You can build the loss function class implementing [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
        And the parameters of the model class can be passed by `kwargs`.
        All loss functions provided PyTorch officially can be referenced.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.

        Returns:
            New model instance.
        """
        if 'reference' in profile:
            clz = profile.reference
            if not issubclass(clz, nn.Module):
                raise RuntimeError('Referenced class is not a subclass of torch.nn.Module.')
        else:
            raise RuntimeError('Missing field "reference" in the loss_function profile.')

        if 'kwargs' in profile:
            kwargs = profile.kwargs
        else:
            kwargs = {}

        loss_function = clz(**kwargs)
        return loss_function

    def create_optimizer(self, profile: Profile, shared: Storage, logger: Logger, model: nn.Module, **kwargs) -> Tuple[
        optim.Optimizer, Any]:
        """
        You can build the optimizer class implementing [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer).
        And the parameters of the optimizer class can be passed by `kwargs`.
        All optimizers provided PyTorch officially can be referenced.
        You can also build a learning rate scheduler through `lr_scheduler` field.

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            model: The model to train.

        Returns:
            New optimizer instance.
        """
        if 'reference' in profile:
            clz = profile.reference
            if not issubclass(clz, optim.Optimizer):
                raise RuntimeError('Referenced class is not a subclass of torch.optim.Optimizer.')
        else:
            raise RuntimeError('Missing field "reference" in the optimizer profile.')

        if 'kwargs' in profile:
            kwargs = profile.kwargs
        else:
            kwargs = {}

        optimizer = clz(model.parameters(), **kwargs)

        if 'lr_scheduler' in profile:
            if 'reference' in profile.lr_scheduler:
                lr_scheduler_clz = profile.lr_scheduler.reference
                if 'kwargs' in profile.lr_scheduler:
                    lr_scheduler_kwargs = profile.lr_scheduler.kwargs
                else:
                    lr_scheduler_kwargs = {}
                lr_scheduler = lr_scheduler_clz(optimizer, **lr_scheduler_kwargs)
            else:
                lr_scheduler = None
        else:
            lr_scheduler = None

        return optimizer, lr_scheduler

    def register_metrics(
            self, profile: Profile, shared: Storage, logger: Logger,
            _metrics: Dict
    ):
        """
        Register the metric methods. In `SimpleTrainTask`,
        all the metrics can be initialized in profile by
        "M" type field.

        Examples:
            Register accuracy as metric method.

            ```toml
            accuracy = '$M$ignite.metrics.Accuracy'
            ```

            Register F1 macro as metric method.

            ```toml
            f1macro = '1$M$tasker.contrib.torch.FBetaMacro$I'
            ```

        Args:
            profile: Runtime profile defined in TOML file.
            shared: Shared storage in the whole lifecycle.
            logger: The logger named with this Task.
            _metrics: The metrics dictionary to register.

        Returns:
            The metrics dictionary.
        """
        _metrics['loss'] = metrics.Loss(self.create_loss_function(profile.loss_function, shared, logger))

        if 'metrics' in profile:
            _metrics.update(profile.metrics)

        return _metrics


class FBetaMacro(metrics.MetricsLambda):
    """
    <i>tasker.contrib.torch.FBetaMacro</i>

    Metric of F-beta score.
    See Also [Metrics](https://pytorch.org/ignite/metrics.html)
    """
    def __init__(self, beta: int):
        def f_beta(p, r, beta):
            return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

        super(FBetaMacro, self).__init__(f_beta, metrics.Precision(), metrics.Recall(), beta)
