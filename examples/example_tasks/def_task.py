from tasker import def_task, Return


@def_task()
def print(ctx, text: str):
    ctx['logger'].info(text)
    return Return.SUCCESS.value
