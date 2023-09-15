
"""Tests for filter.py"""

import pytest

from testing import assert_equal
from vipdopt.configuration import Config, SonyBayerConfig

TEST_YAML_PATH = 'config.yml.example'

@pytest.mark.smoke()
@pytest.mark.usefixtures('_mocker_empty_config')
def test_load_empty_yaml():
    cfg = Config()
    cfg.read_file('fakefile')

    with pytest.raises(KeyError):
        _ = cfg.fixed_layers  # Accessing a property that doesn't exist


@pytest.mark.smoke()
def test_load_yaml():
    cfg = Config()
    cfg.read_file(TEST_YAML_PATH)

    expected_pairs = [
        ('fixed_layers', [1, 3]),
        ('do_rejection', False),
        ('start_from_step', 0),
    ]

    for (k, v) in expected_pairs:
        assert_equal(cfg.__getattr__(k), v)

    assert_equal(cfg.fixed_layers, [1, 3])
    assert_equal(cfg.do_rejection, False)
    assert_equal(cfg.start_from_step, 0)


def test_cascading_params():
    cfg = SonyBayerConfig()
    cfg.read_file(TEST_YAML_PATH)

    assert_equal(cfg.min_feature_size_voxels, 0.085 / 0.085)
    assert_equal(cfg.min_device_permittivity, 1.5 ** 2)
    assert_equal(cfg.max_device_permittivity, 2.4 ** 2)
    assert_equal(cfg.focal_length_um, 0.051 * 30)
    assert_equal(cfg.focal_plane_center_vertical_um, -0.051 * 30)
    assert_equal(cfg.vertical_layer_height_voxels, round(0.408 / 0.051))
    assert_equal(cfg.device_size_vertical_um, 0.408 * 5)
    assert_equal(cfg.device_voxels_lateral, round(2.04 / 0.085))
    assert_equal(cfg.device_voxels_vertical, round(0.408 * 5 / 0.051))
    assert_equal(cfg.device_voxels_simulation_mesh_lateral, 1 + int(2.04 / 0.017))

@pytest.mark.smoke
@pytest.mark.usefixtures('_mocker_empty_config')
def test_unsupported_filetype():
    cfg = Config()
    with pytest.raises(NotImplementedError):
        cfg.read_file('fakefilename', 'json')
