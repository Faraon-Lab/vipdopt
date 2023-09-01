from typing import Any

__all__ = ["assert_equal"]


def assert_equal(result: Any, expected: Any) -> None:
    assert type(result) == type(expected) and result == expected
