import pytest

from vipdopt.filter import Filter
from testing import assert_equal

import numpy as np

@pytest.mark.smoke
@pytest.mark.parametrize(
    'min, max, var, expected',
    [
        (0, 100, 50, True),
        (-100, 0, -50, True),
        (0, 100, 101, False),
        (0, 100, 100, True),
        (0, 100, 0, True),
        (0, 0, 1, False),
        (0, 1, 0.5, True)
    ]
)
def test_verify_bounds(min: float, max: float, var: np.ndarray, expected: bool):
    filter = Filter((min, max))
    result = filter.verify_bounds(var)

    print(result)

    assert_equal(result, expected)


@pytest.mark.smoke
def test_base_filter_not_impl():
    filter = Filter((0, 1))

    with pytest.raises(NotImplementedError):
        filter.forward(None)

    with pytest.raises(NotImplementedError):
        filter.chain_rule(None, None, None)