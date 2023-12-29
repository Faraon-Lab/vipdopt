"""Tests for device.py"""
from numbers import Number

import numpy as np
import pytest

from testing import assert_close, assert_equal
from vipdopt.device import Device
from vipdopt.filter import Sigmoid

ETA = 0.5
BETA = 1
FILTER_LIST = [Sigmoid(ETA, BETA), Sigmoid(ETA, BETA)]
INITIAL_DENSITY = 0.6

@pytest.mark.smoke()
@pytest.mark.parametrize(
        'size, match',
        [
            ((1, 1, 1), r'Device size must be 2 dimensional.*'),
            ((0, 1, 1), r'Device size must be 2 dimensional.*'),
            ((0, 1), r'Expected positive, integer dimensions;.*'),
            ((1, 0), r'Expected positive, integer dimensions;.*'),
            ((-1, 1), r'Expected positive, integer dimensions;.*'),
            ((1, -1), r'Expected positive, integer dimensions;.*'),
            ((1.1, 1.1), r'Expected positive, integer dimensions;.*'),
            ((1.0, 1.0), r'Expected positive, integer dimensions;.*'),
        ],
)
def test_bad_size(size: tuple[Number, ...], match):
    with pytest.raises(ValueError, match=match):
        Device(size, (1, 1), (1, 1, 1))


pytest.mark.smoke()
@pytest.mark.parametrize(
        'permittivity, match',
        [
            ((1, 1, 1), r'Expected 2 permittivity constraints;.*'),
            ((1, 1j, 1), r'Expected 2 permittivity constraints;.*'),
            ((0, 1, 2.0), r'Expected 2 permittivity constraints;.*'),
            ((1j, 1), r'Expected real permittivity;.*'),
            ((1, 1j), r'Expected real permittivity;.*'),
            ((1j, 1j), r'Expected real permittivity;.*'),
            ((1, 1), r'Maximum permittivity must be greater than minimum'),
            ((1.0, 1.0), r'Maximum permittivity must be greater than minimum'),
            ((0, 0), r'Maximum permittivity must be greater than minimum'),
            ((1, 0), r'Maximum permittivity must be greater than minimum'),
            ((1+2j, 0), r'Expected real permittivity;.*'),
            ((1.0, 1), r'Maximum permittivity must be greater than minimum'),
            ((np.pi, 3.14), r'Maximum permittivity must be greater than minimum'),
        ],
)
def test_bad_permittivity(permittivity: tuple[Number, Number], match):
    with pytest.raises(ValueError, match=match):
        Device((1, 1), permittivity, (1, 1, 1))


@pytest.mark.smoke()
@pytest.mark.parametrize(
        'coords, match',
        [
            ((1, 1, 1, 1), r'Expected device coordinates to be 3 dimensional;.*'),
            ((1, 1), r'Expected device coordinates to be 3 dimensional;.*'),
            ((1.0, 1j, 1), r'Expected device coordinates to be rational;.*'),
            ((0.0, 1.1, np.pi), r'Expected device coordinates to be rational;.*'),
            (
                (0.0, 1.1, np.exp(1j * np.pi / 3)),
                r'Expected device coordinates to be rational;.*',
            ),
        ],
)
def test_bad_coords(coords: tuple[Number, ...], match):
    with pytest.raises(ValueError, match=match):
        Device((1, 1), (0, 1), coords)


@pytest.mark.smoke()
def test_init_vars():
    device = Device(
        (3, 3),
        (0, 1),
        (0, 0, 0),
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
    )
    x = INITIAL_DENSITY * np.ones((3, 3), dtype=np.complex128)
    assert_equal(device.w[..., 0], x)
    assert_equal(device.w.shape, (3, 3, 3))


@pytest.mark.smoke()
def test_update_density():
    device = Device(
        (3, 3),
        (0, 1),
        (0, 0, 0),
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
    )
    x = np.ones((3, 3, 3), dtype=np.complex128)
    x[..., 0] *= 0.6
    x[..., 1] *= 0.607838
    x[..., 2] *= 0.616228

    device.update_density()

    assert_close(device.w, x)


@pytest.mark.smoke()
def test_pass_through_filters():
    device = Device(
        (3, 3),
        (0, 1),
        (0, 0, 0),
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
    )

    x_og = INITIAL_DENSITY * np.ones((3, 3), dtype=np.complex128)
    x_copy = INITIAL_DENSITY * np.ones((3, 3), dtype=np.complex128)
    expected_output  = 0.616228 * np.ones((3, 3), dtype=np.complex128)

    y = device.pass_through_filters(x_og)

    assert_close(y, expected_output)

    # Original should not be modified
    assert_equal(x_og, x_copy)
