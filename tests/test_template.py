"""Tests for vipdopt.configuration.template"""


import pytest

from vipdopt.configuration import Config

TEST_YAML_PATH = 'vipdopt/configuration/config_example.yml'

@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_empty_config')
def test_render():
    cfg = Config()
    cfg.read_file('fakefile.yaml')
