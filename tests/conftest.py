"""Fixtures for use in multiple tests."""

import pytest

mock_bad_config_data = {
    'bad_config_1.yml': 'border_optimization: true\nuse_smooth_blur: true',
    'bad_config_2.yml': 'border_optimization: true\nnum_sidewalls: 1',
    'bad_config_3.yml': 'add_pdaf: true',
    'bad_config_4.yml':
        'add_pdaf: true\ndevice_size_lateral_um: 3\ngeometry_spacing_lateral_um: 1',
    'bad_config_5.yml':
        'add_pdaf: true\ndevice_size_lateral_um: 4\n'
        'geometry_spacing_lateral_um: 1\nadd_infrared: true',
    'bad_config_6.yml':
    'reinterpolate_permittivity: False\nreinterpolate_permittivity_factor: 2',
}

_mock_sim_file_data = """{
    "objects": {
        "source_aperture": {
            "name": "source_aperture",
            "obj_type": "rect",
            "properties": {
                "x": 0,
                "x span": 1.5e-06,
                "y": 0,
                "y span": 1.5e-06,
                "index": 3e-07
            }
        },
        "device_mesh": {
            "name": "device_mesh",
            "obj_type": "mesh",
            "properties": {
                "x": 0,
                "x span": 1.5e-06,
                "y": 0,
                "y span": 1.5e-06
            }
        }
    }
}"""

@pytest.fixture()
def mock_sim_file():
    return _mock_sim_file_data

@pytest.fixture()
def _mock_empty_config(mocker):
    """Read a mocked empty config file"""
    mocked_config = mocker.mock_open(
        read_data=''
    )
    mocker.patch('builtins.open', mocked_config)

# def _open_bad_config(m):
#     def open_mock(fname):


@pytest.fixture()
def _mock_bad_config(mocker):
    """Read a config file with disallowed settings"""
    def open_mock(fname, *args):
        return mocker.mock_open(read_data=mock_bad_config_data[fname]).return_value
    mocker.patch('builtins.open', open_mock)


@pytest.fixture(scope='session')
def simulation_json(tmpdir_factory):
    fn = tmpdir_factory.mktemp('sim').join('sim.json')
    with open(str(fn), 'w') as f:
        f.write(_mock_sim_file_data)
    return fn
