"""Common utility functions for use throughout the project."""

import functools
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

from vipdopt.utils import Number

__all__ = ['assert_equal', 'assert_close', 'catch_exits', 'sleep_and_raise', 'assert_equal_dict']


def assert_equal_dict(result: dict, expected: dict) -> bool:
    """Return whether two dictionaries are equal."""
    try:
        vals = (assert_close(result[k], expected[k]) for k in expected)
    except KeyError:
        assert False
    print(list(vals))
    assert all(vals)

def assert_equal(result: Any, expected: Any) -> None:
    """Assert the two inputs are equivalent."""
    if isinstance(result, dict) and isinstance(expected, dict):
        assert_equal_dict(result, expected)
        return
    assert np.array_equal(result, expected)


def assert_close(
        n1: Any,
        n2: Any,
        err: float=1e-3
) -> None:
    """Assert two numbers are close to equal, within some error bound."""
    if isinstance(n1, Number | np.ndarray) and isinstance(n2, Number | np.ndarray):
        assert np.allclose(n1, n2, rtol=err)
    elif isinstance(n1, list | np.ndarray) and isinstance(n2, list | np.ndarray) and \
        len(n1) > 0 and isinstance(n1[0], Number):
        assert np.allclose(n1, n2, rtol=err)
    else:
        assert_equal(n1, n2)


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
