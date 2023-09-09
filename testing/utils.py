"""Common utility functions for use throughout the project."""

from typing import Any

__all__ = ['assert_equal', 'assert_close']


def assert_equal(result: Any, expected: Any) -> None:
    """Assert the two inputs are equivalent."""
    assert type(result) == type(expected)
    assert result == expected


def assert_close(n1: float, n2: float, err: float=1e-3) -> None:
    """Assert two numbers are close to equal, within some error bound."""
    assert abs(n1 - n2) <= err
