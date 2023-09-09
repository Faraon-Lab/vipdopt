"""Common utility functions for use throughout the project."""

from typing import Any

import numpy as np
import numpy.typing as npt

from vipdopt.utils import Number

__all__ = ['assert_equal', 'assert_close']


def assert_equal(result: Any, expected: Any) -> None:
    """Assert the two inputs are equivalent."""
    assert np.array_equal(result, expected)


def assert_close(
        n1: npt.ArrayLike | Number,
        n2: npt.ArrayLike | Number,
        err: Number=1e-3
) -> None:
    """Assert two numbers are close to equal, within some error bound."""
    assert np.allclose(n1, n2, rtol=err)
