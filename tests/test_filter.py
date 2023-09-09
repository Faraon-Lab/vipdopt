"""Tests for filter.py"""

import numpy as np
import numpy.typing as npt
import pytest

from testing import assert_close, assert_equal
from vipdopt.filter import Sigmoid

# Generic values for instantiating Sigmoid filters
ETA = 0.5
BETA = 0.05

@pytest.mark.smoke()
@pytest.mark.parametrize(
    'var, expected',
    [
        (0.0, True),
        (1.0, True),
        (0.5, True),
        (-1.0, False),
        (1.1, False),
        ([0.0, 1.0, 0.5], True),
        ([-0.1, 1.1], False),
        ([-0.1, 0.5, 1.1], False),
    ]
)
def test_verify_bounds_sigmoid(var: npt.ArrayLike | np.number, expected: bool):
    # Sigmoid always has bounds [0, 1]
    sig = Sigmoid(ETA, BETA)
    result = sig.verify_bounds(var)
    assert_equal(result, expected)


@pytest.mark.smoke()
@pytest.mark.parametrize('eta', [-0.1, 1.1])
def test_sigmoid_bad_eta(eta: float):
    with pytest.raises(ValueError, match=r'Eta must be in the range \[0, 1\]'):
        Sigmoid(eta, BETA)

@pytest.mark.smoke()
@pytest.mark.parametrize(
    'eta, beta, x, y',
    [
        # Check that \rho=0.5 always get projected to 0.5, for lare and small beta
        (0.5, 1, 0.5, 0.5),
        (0.5, 1e9, 0.5, 0.5),

        # Check that going around the center pushes towards the ends
        (0.5, 1, 0.4, 0.39216),
        (0.5, 1, 0.6, 0.607838),
        (0.5, 1e3, 0.4, 0),
        (0.5, 1e3, 0.6, 1),
    ]
)
def test_sigmoid_forward(eta: float, beta: float, x: float, y: float):
    sig = Sigmoid(eta, beta)
    assert_close(sig.forward(x), y)
