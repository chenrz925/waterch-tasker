__all__ = [
    'ProfileMixin',
    'value',
    'include',
]

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from re import match as re_match
from typing import List, Tuple, Text, Any, Union, Dict, Type

from waterch.tasker.typedef import Definition


def value(name: Text, type: Type, children: Union[List, Tuple] = None) -> Definition:
    return Definition(name, type, children)


def include(name: Text, cls: Type):
    if issubclass(cls, ProfileMixin):
        return Definition(name, list, cls.define())
    else:
        raise RuntimeError(f'Class {cls} is NOT a ProfileMixin subclass')


class ProfileMixin(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def define(cls) -> List[Definition]:
        raise NotImplementedError('Please define the profile of the task.')

    @classmethod
    def profile_template(cls) -> Dict[Text, Any]:
        def sequence_closure(sequence: Union[List, Tuple]) -> Union[Dict[Text, Any], List[Any]]:
            if len(sequence) > 0:
                if isinstance(sequence[0], Definition):
                    return dict(map(closure, sequence))
                elif isinstance(sequence[0], list) or isinstance(sequence[0], tuple):
                    return list(map(sequence_closure, sequence))
                elif sequence[0] == int:
                    return [0]
                elif sequence[0] == float:
                    return [0.0]
                elif sequence[0] == bool:
                    return [True]
                elif sequence[0] == str:
                    return ['']
                else:
                    return []
            else:
                return []

        def closure(definition: Definition) -> Tuple[Text, Any]:
            if definition.type == str:
                return definition.name, ''
            elif definition.type == int:
                return definition.name, 0,
            elif definition.type == float:
                return definition.name, 0.0
            elif definition.type == bool:
                return definition.name, True
            elif definition.type == list or definition.type == tuple:
                if definition.children is not None and len(definition.children) > 0:
                    if isinstance(definition.children[0], Definition):
                        return definition.name, dict(map(closure, definition.children))
                    elif isinstance(definition.children[0], list) or isinstance(definition.children[0], tuple):
                        return definition.name, list(
                            filter(lambda it: len(it) > 0, map(sequence_closure, definition.children)))
                    return '_', None
                else:
                    return definition.name, []
            else:
                return '_', None

        class_str = str(cls)
        schema = class_str[slice(*re_match(r'.*\'([a-zA-Z._]+)\'.*', class_str).regs[-1])]
        template = OrderedDict({'__schema__': schema})
        template.update(filter(lambda it: it[0] != '_' and it[1] != [], map(closure, cls.define())))
        return template
