import asyncio
from functools import wraps


def run_async(coro):
    @wraps(coro)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))

    return wrapper
