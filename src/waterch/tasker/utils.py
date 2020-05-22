from typing import Text, Type


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
