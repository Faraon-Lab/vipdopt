"""Fixtures for use in multiple tests."""

import pytest

_mock_bad_config_data = {
    'bad_config_1.yml': 'border_optimization: true\nuse_smooth_blur: true',
    'bad_config_2.yml': 'border_optimization: true\nnum_sidewalls: 1',
    'bad_config_3.yml': 'add_pdaf: true',
    'bad_config_4.yml': 'add_pdaf: true\ndevice_size_lateral_um: 3\ngeometry_spacing_lateral_um: 1',
    'bad_config_5.yml': 'add_pdaf: true\ndevice_size_lateral_um: 4\ngeometry_spacing_lateral_um: 1\nadd_infrared: true',
    'bad_config_6.yml': 'reinterpolate_permittivity: False\nreinterpolate_permittivity_factor: 2',
}

@pytest.fixture()
def mock_empty_config(mocker):
    """Read a mocked empty config file"""
    mocked_config = mocker.mock_open(
        read_data=''
    )
    mocker.patch('builtins.open', mocked_config)

def _open_bad_config(m):
    def open_mock(fname):
        file_obj = m.mock_open(read_data=_mock_bad_config_data[fname])
        return file_obj



@pytest.fixture()
def mock_bad_config(mocker):
    """Read a config file with disallowed settings"""
    def open_mock(fname, *args):
        return mocker.mock_open(read_data=_mock_bad_config_data[fname]).return_value
    mocker.patch('builtins.open', open_mock)

    # mocked_config = mocker.mock_open(
    #     # read_data= [
    #     #     'border_optimization: true\nuse_smooth_blur: true',
    #     #     'border_optimization: true\nnum_sidewalls: 1',
    #     # ]
    # )
    # # m = mocker.patch('builtins.open', mocked_config)
    # mocked_config.__r.read.side_effect = lambda : read_data.pop(0)
    