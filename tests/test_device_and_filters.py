"""Tests for device.py"""

import sys
from pathlib import Path
import math
from numbers import Number
import numpy as np

from vipdopt.configuration import Config
from vipdopt.optimization.device import Device
from vipdopt.optimization.filter import Sigmoid, Scale

import pytest
from testing import assert_close, assert_equal
from tests.conftest import TEST_YAML_PATH

ETA = 0.5
BETA = 1
PERM_BOUNDS = (2.25, 5.76)
FILTER_LIST = [Sigmoid(ETA, BETA), Sigmoid(ETA, BETA), Scale(PERM_BOUNDS)]
INITIAL_DENSITY = 0.6
BASE_COORDS = {
    'x': np.array([0, 1]),
    'y': np.array([0, 1]),
    'z': np.array([0, 1]),
}


def test_device_from_config():
    cfg_file = Path.cwd() / TEST_YAML_PATH
    cfg = Config.from_file(cfg_file)

    device = Device.load_config(cfg)
    
    assert device is not None
    # Quick check if cfg values imported correctly
    assert_equal(len(device.coords['x']), cfg['device_voxels_lateral_bordered'] )
    assert_equal(device.permittivity_constraints, 
                 (cfg['min_device_permittivity'], cfg['max_device_permittivity'])
                )
    # todo: write test for filter import

# NOTE: fixture "example_device" depends on Device.load_config() which is tested above
def test_example_device(example_device):
    assert example_device is not None


@pytest.mark.smoke()
@pytest.mark.parametrize(
    'size, match',
    [
        ((1, 1), r'Device size must be 3 dimensional.*'),
        ((0, 1), r'Device size must be 3 dimensional.*'),
        ((1, 0, 1), r'Expected positive, integer dimensions;.*'),
        ((1, 1, 0), r'Expected positive, integer dimensions;.*'),
        ((1, -1, 1), r'Expected positive, integer dimensions;.*'),
        ((1, 1, -1), r'Expected positive, integer dimensions;.*'),
        ((1, 1.1, 1.1), r'Expected positive, integer dimensions;.*'),
        ((1, 1.0, 1.0), r'Expected positive, integer dimensions;.*'),
    ],
)
def test_bad_size(size: tuple[Number, ...], match):
    with pytest.raises(ValueError, match=match):
        Device(size, (1, 1), BASE_COORDS)


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
        ((1 + 2j, 0), r'Expected real permittivity;.*'),
        ((1.0, 1), r'Maximum permittivity must be greater than minimum'),
        ((np.pi, math.pi), r'Maximum permittivity must be greater than minimum'),
    ],
)
def test_bad_permittivity(permittivity: tuple[Number, Number], match):
    with pytest.raises(ValueError, match=match):
        Device((1, 1, 1), permittivity, BASE_COORDS)


@pytest.mark.smoke()
@pytest.mark.parametrize(
    'err_type, coords, match',
    [
        (
            TypeError,
            (1, 1, 1),
            r'Expected device coordinates to be a dictionary with.*',
        ),
        (
            TypeError,
            {'x': 1, 'y': 1, 'z': 1, 'a': 1},
            r'Expected device coordinates to be a dictionary with.*',
        ),
        (
            TypeError,
            {'x': 1, 'y': 1},
            r'Expected device coordinates to be a dictionary with.*',
        ),
        (
            TypeError,
            {'a': 1, 'b': 1, 'c': 1},
            r'Expected device coordinates to be a dictionary with.*',
        ),
        (
            TypeError,
            {'x': 1, 'y': 1, 'z': 1},
            r'Expected device coordinates to be ndarrays;.*',
        ),
        (
            TypeError,
            {'x': np.array(1), 'y': np.array(1), 'z': 1},
            r'Expected device coordinates to be ndarrays;.*',
        ),
    ],
)
def test_bad_coords(err_type: type[BaseException], coords: tuple[Number, ...], match):
    with pytest.raises(err_type, match=match):
        Device((1, 1, 1), (0, 1), coords)

@pytest.mark.smoke()
def test_init_vars():
    device = Device(
        (3, 3, 1),
        (0, 1),
        BASE_COORDS,
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
    )
    x = INITIAL_DENSITY * np.ones((3, 3, 1), dtype=np.complex128)
    assert_close(device.w[..., 0], x)
    assert_equal(device.w.shape, (3, 3, 1, 4))


@pytest.mark.smoke()
def test_init_random():
    d1 = Device(
        (3, 3, 1),
        (0, 1),
        BASE_COORDS,
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
        randomize=True,
        init_seed=0,
    )
    d2 = Device(
        (3, 3, 1),
        (0, 1),
        BASE_COORDS,
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
        randomize=True,
        init_seed=0,
    )

    assert_close(d1.w, d2.w)


@pytest.mark.smoke()
def test_init_symmetric():
    device = Device(
        (3, 3, 1),
        (0, 1),
        BASE_COORDS,
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
        randomize=True,
        init_seed=0,
        symmetric=True,
    )

    # check symmetry
    w = device.w
    a = np.flip(w[..., 0, 0], axis=1)
    assert np.allclose(a, a.T, rtol=1e-5, atol=1e-8)


@pytest.mark.smoke()
def test_update_density():
    device = Device(
        (3, 3, 1),
        (0, 1),
        BASE_COORDS,
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
    )
    x = np.ones((3, 3, 1, 4), dtype=np.complex128)
    x[..., 0] *= 0.6
    x[..., 1] *= 0.607838
    x[..., 2] *= 0.616228
    x[..., 3] *= 4.412962

    device.update_density()

    assert_close(device.w, x)



@pytest.mark.smoke()
def test_pass_through_filters():
    device = Device(
        (3, 3, 1),
        (0, 1),
        BASE_COORDS,
        init_density=INITIAL_DENSITY,
        filters=FILTER_LIST,
    )

    x_og = INITIAL_DENSITY * np.ones((3, 3), dtype=np.complex128)
    x_copy = INITIAL_DENSITY * np.ones((3, 3), dtype=np.complex128)
    expected_output = 4.412962 * np.ones((3, 3), dtype=np.complex128)

    y = device.pass_through_filters(x_og)

    assert_close(y, expected_output)
    # Original should not be modified
    assert_equal(x_og, x_copy)



@pytest.mark.smoke()
def test_save_load(tmpdir, example_device):
    dev1 = example_device
    p = tmpdir / 'device.npy'
    dev1.save(p)

    dev2 = Device.from_source(p)

    # Should still work when loading a device with different size (i.e. 4x4 vs 40x40)
    dev3 = Device((4, 4, 5), (0, 1), BASE_COORDS)
    dev3.load_file(p)

    dev4 = Device((4, 4, 5), (0, 1), BASE_COORDS)
    dev4.load_dict(dev1.as_dict())      # Serves also as a test of Device.as_dict()

    for attr in vars(dev1):
        assert_close(dev2.__getattribute__(attr), dev1.__getattribute__(attr))
        assert_close(dev3.__getattribute__(attr), dev1.__getattribute__(attr))
        assert_close(dev4.__getattribute__(attr), dev1.__getattribute__(attr))

# # todo: Test more complicated filters like Layering
# def test_layering(example_device):
#     pass

if __name__ == '__main__':
    print('End of code reached.')