from enum import Enum
from logging import getLogger as get_logger
from os import path, makedirs, environ
from random import getrandbits
from typing import Text, Type, List, Callable

from box import BoxError

from .typedef import Profile


def import_reference(reference: Text) -> Type:
    path = reference.split('.')
    if reference and len(path) > 0:
        root = path[0]
        module = __import__(root)
        if len(path) == 1:
            return module
        else:
            for layer in path[1:]:
                module = getattr(module, layer, None)
                if module is None:
                    raise RuntimeError(f'Reference {reference} NOT found.')
            return module
    else:
        raise RuntimeError(f'Wrong reference {reference}.')


class LocalPath(object):
    _logger = get_logger('tasker.utils.LocalPath')

    class CATEGORY(Enum):
        CACHE = 'cache'
        LOG = 'log'
        OUTPUT = 'output'
        STORAGE = 'storage'
        TEMP = 'temp'
        CONfig = 'config'

    @classmethod
    def create(cls, category, *name: Text):
        assert isinstance(category, cls.CATEGORY)
        if not name:
            cls._logger.warning('You are operating in a root category folder! Please re-check it.')

        absolute_root = path.abspath(path.curdir)
        absolute_path = path.join(absolute_root, category.value, *name)

        if not path.exists(absolute_path):
            makedirs(absolute_path)

        return absolute_path


class LambdaWrapper(object):
    def __init__(self, expression):
        self.expression = expression

        try:
            instance = eval(self.expression)
        except Exception:
            raise RuntimeError(f'Please check expression "{self.expression}"')

        if not isinstance(instance, Callable):
            raise RuntimeError(f'"{self.expression}" is not a lambda expression')

    def __call__(self, *args, **kwargs):
        return eval(self.expression)(*args, **kwargs)


class FieldParser(object):
    def _single(self, text: Text):
        dollar_controller = f'<{hex(getrandbits(16))}>'
        comma_controller = f'<{hex(getrandbits(16))}>'
        escaped_text = text.replace(r'\$', dollar_controller).replace(r'\,', comma_controller)
        raw_sequences = escaped_text.split('$')
        length = len(raw_sequences)

        if length == 1:
            return text
        else:
            field_type = raw_sequences[1]

            try:
                parse = getattr(self, f'_parse_{field_type}')
            except AttributeError:
                raise RuntimeError(f'Field type {field_type} is not supported.')

            return parse(raw_sequences, dollar_controller, comma_controller)

    def _parse_B(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <value>$B
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0]
        if content in ('true', 'True', 't', 'T', 'TRUE', 'yes', 'Yes', 'Y', 'y', 'YES', 'on', 'ON'):
            return True
        else:
            return False

    def _parse_E(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <environment-key>$E($<value-type>)?
        ```

        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')
        length = len(raw_sequences)
        if length == 2:
            return environ[content] if content in environ else ''
        elif length >= 3:
            field_type = raw_sequences[2]

            try:
                parse = getattr(self, f'_parse_{field_type}')
            except AttributeError:
                raise RuntimeError(f'Field type {field_type} is not supported.')

            return parse([environ[content], field_type], dollar_controller, comma_controller)

    def _parse_F(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <value>$F
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')
        return float(content)

    def _parse_I(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <value>$I
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')
        return int(content)

    def _parse_M(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <parameter>?(,<parameter>)*$M$<method-reference>$<parameter-type>?(,<parameter-type>)*
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """

        raw_parameters = raw_sequences[0].split(',')
        if len(raw_sequences) >= 4:
            parameter_types = raw_sequences[3].split(',')
            assert len(parameter_types) == len(raw_parameters)
        else:
            parameter_types = ['S'] * len(raw_parameters)

        parameters = []
        for raw_parameter, parameter_type in zip(raw_parameters, parameter_types):
            try:
                parse = getattr(self, f'_parse_{parameter_type}')
            except AttributeError:
                raise RuntimeError(f'Field type {parameter_type} is not supported.')

            parameters.append(parse([raw_parameter, parameter_type], dollar_controller, comma_controller))

        method_reference = raw_sequences[2]
        try:
            method = eval(method_reference)
        except NameError:
            method = import_reference(method_reference)

        if len(parameter_types) == 1 and parameter_types[0] == 'P':
            return method(**parameters[0])
        else:
            return method() if parameters == [''] else method(*parameters)

    def _parse_T(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <reference>$T$<parameter>*(,<parameter)*
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$')
        clz = import_reference(content)

        if len(raw_sequences) >= 3:
            parameters = list(map(
                lambda it: it.replace(comma_controller, ','),
                raw_sequences[2].split(',')
            ))
        else:
            parameters = None
        instance = clz(*parameters) if parameters is not None and parameters != [''] else clz()
        assert hasattr(instance, 'invoke')

        return instance

    def _parse_R(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <reference>$R
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')
        return import_reference(content)

    def _parse_X(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <expression>$X
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')
        return eval(content)

    def _parse_L(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <expression>$L
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')
        return LambdaWrapper(content)

    def __call__(self, *args, **kwargs):
        if len(args) > 1:
            return tuple(map(self._single, args))
        else:
            return self._single(*args)

    def _parse_S(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <text>$S
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        return raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')

    def _parse_P(self, raw_sequences: List[Text], dollar_controller: Text, comma_controller: Text):
        """
        ```
        <path>$P
        ```
        Args:
            raw_sequences:
            dollar_controller:
            comma_controller:

        Returns:

        """
        content = raw_sequences[0].replace(dollar_controller, '$').replace(comma_controller, ',')
        try:
            return self(Profile.from_toml(filename=content))
        except BoxError:
            raise RuntimeError(f'Profile "{content}" does NOT exist')


_parse_field = FieldParser()


def parse_profile(instance):
    if isinstance(instance, Profile):
        return Profile(tuple(map(
            lambda it: (it[0], parse_profile(it[1])),
            instance.items(),
        )))
    elif isinstance(instance, (list, tuple)):
        return tuple(map(parse_profile, instance))
    elif isinstance(instance, str):
        return _parse_field(instance)
    else:
        return instance
