"""Common utility functions for use throughout the project."""

from typing import Any

__all__ = ['assert_equal']


def assert_equal(result: Any, expected: Any) -> None:
    """Assert the two inputs are equivalent."""
    assert type(result) == type(expected)
    assert result == expected
