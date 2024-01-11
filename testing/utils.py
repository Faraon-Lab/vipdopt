"""Common utility functions for use throughout the project."""

import functools
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from vipdopt.utils import Number

__all__ = ['assert_equal', 'assert_close', 'catch_exits', 'sleep_and_raise']


def assert_equal(result: Any, expected: Any) -> None:
    """Assert the two inputs are equivalent."""
    assert np.array_equal(result, expected)


def assert_close(
        n1: Any,
        n2: Any,
        err: Number=1e-3
) -> None:
    """Assert two numbers are close to equal, within some error bound."""
    if not isinstance(n1, Number | list | np.ndarray) or \
       not isinstance(n2, Number | list | np.ndarray):
        assert_equal(n1, n2)
        return
    assert np.allclose(n1, n2, rtol=err)


def catch_exits(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit as e:
            assert e.code == 0
    return wrapper

def sleep_and_raise(n: int):
    time.sleep(n)
    raise RuntimeError('expected raise')
