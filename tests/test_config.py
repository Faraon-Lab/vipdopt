
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

    assert_equal(cfg.min_device_permittivity, 1.5 ** 2)
