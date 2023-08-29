from typing import Any

def is_equal(result: Any, expected: Any) -> bool:
    if type(expected) is Exception:
        return is_error(result, expected)
    return type(result) == type(expected) and result == expected

def is_error(result: Any, expected: Exception) -> bool:
    return type(result) == type(expected)