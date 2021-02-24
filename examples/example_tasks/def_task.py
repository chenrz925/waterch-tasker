from typing import Text

from tasker import def_task, Return


@def_task()
def print(ctx, text: str):
    ctx.logger.info(text)
    return Return.SUCCESS


@def_task()
def environ(ctx, **kwargs):
    for key, value in kwargs.items():
        ctx.environ[key] = value

    for key, value in ctx.environ.items():
        ctx.logger.info(f'{key}={value}')
    return Return.SUCCESS
