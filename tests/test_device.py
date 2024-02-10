"""Tests for device.py"""
from numbers import Number

import numpy as np
import pytest

from testing import assert_close, assert_equal
from vipdopt.optimization.device import Device
from vipdopt.optimization.filter import Sigmoid

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


@pytest.mark.smoke()
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
def test_init_random():
    d1 = Device(
        (3, 3),
        (0, 1),
        (0, 0, 0),
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
        randomize=True,
        init_seed=0,
    )
    d2 = Device(
        (3, 3),
        (0, 1),
        (0, 0, 0),
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
        randomize=True,
        init_seed=0,
    )

    assert_close(d1.w, d2.w)

@pytest.mark.smoke()
def test_init_symmetric():
    device = Device(
        (3, 3),
        (0, 1),
        (0, 0, 0),
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
        randomize=True,
        init_seed=0,
        symmetric=True,
    )

    # check symmetry
    w = device.w
    a = w[..., 0] = np.flip(w[..., 0], axis=1)
    assert np.allclose(a, a.T, rtol=1e-5, atol=1e-8)


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


@pytest.mark.smoke()
def test_save_load(tmpdir, device_dict):
    dev1 = Device.from_source(device_dict)
    p = tmpdir / 'device.npy'
    dev1.save(p)

    dev2 = Device.from_source(p)

    # Should still work when loading a device with different size (i.e. 4x4 vs 40x40)
    dev3 = Device((4, 4), (0, 1), (0, 0, 0))
    dev3.load_file(p)

    dev4 = Device((4, 4), (0, 1), (0, 0, 0))
    dev4.load_dict(device_dict)

    for attr in vars(dev1):
        assert_equal(dev2.__getattribute__(attr), dev1.__getattribute__(attr))
        assert_equal(dev3.__getattribute__(attr), dev1.__getattribute__(attr))
        assert_equal(dev4.__getattribute__(attr), dev1.__getattribute__(attr))

