import pytest
import sys
from pathlib import Path
from copy import copy

import numpy as np

sys.path.append(Path.cwd())
from vipdopt.configuration import Config, MetasurfaceRenderer
from vipdopt.optimization.device import Device

TEST_YAML_PATH = Path('testing/processed_config.yml')
TEST_TEMPLATE_PATH = Path('testing/derived_simulation_properties.j2')
PROJECT_INPUT_DIR = Path('runs/test_project/')
PROJECT_OUTPUT_DIR = Path('runs/test_output_project/')
MONITOR_DATA_DIR = Path('testing/monitor_data/')

_mock_bad_config_data = {
    'bad_config_1.yml': 'border_optimization: true\nuse_smooth_blur: true',
    'bad_config_2.yml': 'border_optimization: true\nnum_sidewalls: 1',
    'bad_config_3.yml': 'add_pdaf: true',
    'bad_config_4.yml': 'add_pdaf: true\ndevice_size_lateral_um: 3\n'
    'geometry_spacing_lateral_um: 1',
    'bad_config_5.yml': 'add_pdaf: true\ndevice_voxels_lateral_um: 4\n'
    'geometry_spacing_lateral_um: 1\nadd_infrared: true',
    'bad_config_6.yml': 'reinterpolate_permittivity: False\n'
    'reinterpolate_permittivity_factor: 2',
}

_mock_sim_file_data = """{
    "objects": {
        "source_aperture": {
            "name": "source_aperture",
            "obj_type": "rect",
            "info": {
                "name": ""
            },
            "properties": {
                "name": "source_aperture",
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
            "info": {
                "name": ""
            },
            "properties": {
                "name": "device_mesh",
                "x": 0,
                "x span": 1.5e-06,
                "y": 0,
                "y span": 1.5e-06
            }
        }
    }
}"""

_mock_device_dict = {
    'size': (40, 40, 5),
    'permittivity_constraints': (0.0, 1.0),
    'coords': {'x': [0, 1], 'y': [0, 1], 'z': [0, 1]},
    'name': 'device',
    'init_density': 0.5,
    'randomize': True,
    'init_seed': 0,
    'symmetric': True,
    'filters': [
        {'type': 'Sigmoid', 'parameters': {'eta': 0.5, 'beta': 1.0}},
        {'type': 'Sigmoid', 'parameters': {'eta': 0.5, 'beta': 1.0}},
    ],
}

_mock_fom_dict = {
    'fom_0': {
        'type': 'BayerFilterFoM',
        'fom_monitors': [['forward_srcx', 'focal_monitor_0']],
        'grad_monitors': [
            ['forward_srcx', 'design_efield_monitor'],
            ['adj_src_0x', 'design_efield_monitor'],
        ],
        'polarization': 'TE',
        'freq': [0.0, 0.1, 0.2],
        'weight': 1.0,
    }
}


EXAMPLE_CONFIG_DERIVED_PROPERTIES = {
    # Mesh Properties
    'min_feature_size_voxels': 0.085 / 0.102,
    # Optical Properties
    'min_device_permittivity': 1.5**2,
    'max_device_permittivity': 2.4**2,
    'focal_length_um': 0.102 * 30,
    'focal_plane_center_vertical_um': -0.102 * 30,
    # Device Properties
    'layer_height_voxels': round(0.204 / 0.102),
    'device_size_vertical_um': 0.408 * 5,
    'device_voxels_lateral': round(3.96 / 0.102),
    'device_voxels_z': round(0.408 * 5 / 0.102),
    # todo: the rest
    # todo: or honestly just set something to match the derived_simulation_properties.j2
    # 'device_voxels_simulation_mesh_lateral': 1 + int(3.96 / 0.034),
    # 'device_voxels_simulation_mesh_vertical': 2 + int((0.408 * 5) / 0.017),
    # 'device_vertical_maximum_um': 0.408 * 5,
    # 'device_size_lateral_bordered_um': 2.04,
    # 'device_voxels_simulation_mesh_lateral_bordered': 1 + int(2.04 / 0.017),
    # 'border_size_um': 5 * 0.051,
    # 'border_size_voxels': int(round((5 * 0.051) / 0.085)),
    # 'device_voxels_lateral_bordered': int(round(2.04 / 0.085)),
    # # FDTD Properties
    # 'sim_edge_gap_z_um': 0.085 * 25,
    # 'sim_edge_gap_x_um': 0.051 * 10,
    # 'fdtd_size_z_um': 2 * (0.085 * 25) + (0.408 * 5) + (0.051 * 30),
    # 'fdtd_size_x_um': 2 * (0.051 * 10) + 2.04,
    # 'fdtd_region_maximum_z_um': 2.04 + (0.085 * 25),
    # 'fdtd_region_minimum_z_um': -1 * 0.051 * 30 - (0.085 * 25),
    # # Surrounding Properties
    # 'pec_aperture_thickness_um': 3 * 0.017,
    # 'sidewall_x_positions_um': [(2.04 + 0.085) / 2, 0, -(2.04 + 0.085) / 2, 0],
    # 'sidewall_y_positions_um': [0, (2.04 + 0.085) / 2, 0, -(2.04 + 0.085) / 2],
    # 'sidewall_xspan_positions_um': [0.085, 2.04 + 2 * 0.085, 0.085, 2.04 + 2 * 0.085],
    # 'sidewall_yspan_positions_um': [2.04 + 0.085 * 2, 0.085, 2.04 + 2 * 0.085, 0.085],
    # # Spectral Properties
    # 'bandwidth_um': 0.725 - 0.375,
    # 'num_design_frequency_points': 3 * 20,
    # 'num_eval_frequency_points': 6 * 3 * 20,
    # 'lambda_values_um': np.linspace(0.375, 0.725, 3 * 20),
    # 'max_intensity_by_wavelength': (2.04**4)
    # / ((0.051 * 30) ** 2 * np.power(np.linspace(0.375, 0.725, 3 * 20), 2)),
    # 'dispersive_range_size_um': (0.725 - 0.375) / 1,
    # 'dispersive_ranges_um': [[0.375, 0.725]],
    # # PDAF Functionality Properties
    # 'pdaf_focal_spot_x_um': 0.25 * 2.04,
    # 'pdaf_focal_spot_y_um': 0.25 * 2.04,
    # 'pdaf_lambda_min_um': 0.375,
    # 'pdaf_lambda_max_um': 0.375 + (0.725 - 0.375) / 3,
    # 'pdaf_lambda_values_um': np.linspace(0.375, 0.375 + (0.725 - 0.375) / 3, 3 * 20),
    # 'pdaf_max_intensity_by_wavelength': (2.04**4)
    # / (
    #     (0.051 * 30) ** 2
    #     * np.power(np.linspace(0.375, 0.375 + (0.725 - 0.375) / 3, 3 * 20), 2)
    # ),
    # # Forward Properties
    # 'lateral_aperture_um': 1.1 * 2.04,
    # 'src_maximum_vertical_um': 0.408 * 5 + (0.085 * 25 * 5 / 6),
    # 'src_minimum_vertical_um': -(0.051 * 30) - 0.5 * (0.085 * 25),
    # 'mid_lambda_um': (0.375 + 0.725) / 2,
    # 'source_angle_theta_rad': np.arcsin(np.sin(0) * 1.0 / 1.5),
    # 'source_angle_theta_deg': np.arcsin(np.sin(0) * 1.0 / 1.5) * 180 / np.pi,
    # # Adjoint Properties
    # 'adjoint_vertical_um': -0.051 * 30,
    # 'num_adjoint_sources': 4,
    # 'adjoint_x_positions_um': 2.04 * np.array([0.25, -0.25, -0.25, 0.25]),
    # 'adjoint_y_positions_um': 2.04 * np.array([0.25, 0.25, -0.25, -0.25]),
    # # Optimization Properties
    # 'epoch_range_design_change_max': 0.15 - 0.05,
    # 'epoch_range_design_change_min': 0.05,
    # 'spectral_focal_plane_map': [
    #     [0, 20],
    #     [20, 2 * 20],
    #     [2 * 20, 3 * 20],
    #     [20, 2 * 20],
    # ],
    # 'desired_peak_location_per_band': np.argmin(
    #     np.power(
    #         np.linspace(0.375, 0.725, 3 * 20)[:, np.newaxis] - [0.45, 0.54, 0.65],
    #         2,
    #     ),
    #     axis=0,
    # ),
}




@pytest.fixture(scope='session')
def fom_dict() -> dict:
    return _mock_fom_dict


@pytest.fixture(scope='session')
def template_renderer() -> MetasurfaceRenderer:
    rndr = MetasurfaceRenderer('jinja_templates/')

    rndr.set_template('derived_simulation_properties.j2')

    return rndr


@pytest.fixture()
def _mock_empty_config(mocker):
    """Read a mocked empty config file"""
    mocked_config = mocker.mock_open(read_data='')
    mocker.patch('builtins.open', mocked_config)


@pytest.fixture()
def _mock_example_config(mocker):
    with open(TEST_YAML_PATH) as f:
        text = f.read()
    mocked_config = mocker.mock_open(
        read_data=text,
    )
    mocker.patch('builtins.open', mocked_config)


@pytest.fixture()
def _mock_bad_config(mocker):
    """Read a config file with disallowed settings"""

    def open_mock(fname, *args):
        return mocker.mock_open(
            read_data=_mock_bad_config_data[str(fname)]
        ).return_value

    mocker.patch('builtins.open', open_mock)



@pytest.fixture()
def _mock_example_template(mocker):
    with open(TEST_TEMPLATE_PATH) as f:
        text = f.read()
    mocked_template = mocker.mock_open(
        read_data=text,
    )
    mocker.patch('builtins.open', mocked_template)


@pytest.fixture()
def _mock_focal_monitor(mocker):
    with open(MONITOR_DATA_DIR / 'sim_focal_monitor_0.npz', 'rb') as f:
        text = f.read()
    mocked_template = mocker.mock_open(
        read_data=text,
    )
    mocker.patch('builtins.open', mocked_template)


@pytest.fixture(scope='session')
def focal_monitor_efield():
    data = np.load(MONITOR_DATA_DIR / 'sim_focal_monitor_0.npz', allow_pickle=True)
    return data['e']


@pytest.fixture(scope='session')
def transmission_monitor_t():
    data = np.load(
        MONITOR_DATA_DIR / 'sim_transmission_monitor_0.npz', allow_pickle=True
    )
    return data['t']


@pytest.fixture(scope='session')
def example_derived_properties():
    return EXAMPLE_CONFIG_DERIVED_PROPERTIES


@pytest.fixture()
def _mock_project_json(mocker):
    with (PROJECT_INPUT_DIR / 'config.json').open('r') as f:
        data = f.read()
    mocked_project = mocker.mock_open(
        read_data=data,
    )
    mocker.patch('builtins.open', mocked_project)

#* Device Fixtures

@pytest.fixture(scope='session')
def mock_device_dict() -> dict:
    return {'device': _mock_device_dict}

@pytest.fixture(scope='session')
def device_dict( cfg_file:Path=Path.cwd()/TEST_YAML_PATH ) -> dict:
    # NOTE: Can change the below line if we wish to use some other dictionary for the device dict params
    # NOTE: e.g. loading device dict from a config
    if cfg_file == Path.cwd():
        # Allows the use of Device.load_config(dict) instead of Device(**dict)
        return {'device': _mock_device_dict}
    else:
        return Config.from_file(cfg_file)


@pytest.fixture()
def example_indirect_mock_device(request) -> Device:
    dd = copy(_mock_device_dict)
    dd.update(request.param)
    return Device.load_config({'device': dd})

@pytest.fixture()
def example_indirect_device(request, device_dict) -> Device:
    device_dict.update(request.param)

    # return Device(**device_dict)
    # # equivalent to the above but allows the loading of a config
    return Device.load_config(device_dict)

@pytest.fixture()
def example_device(device_dict) -> Device:
    # return Device(**device_dict)
    return Device.load_config(device_dict)

#* Simulation Fixtures

@pytest.fixture(scope='session')
def device_dict( cfg_file:Path=Path.cwd()/TEST_YAML_PATH ) -> dict:
    # NOTE: Can change the below line if we wish to use some other dictionary for the device dict params
    # NOTE: e.g. loading device dict from a config
    if cfg_file == Path.cwd():
        # Allows the use of Device.load_config(dict) instead of Device(**dict)
        return {'device': _mock_device_dict}
    else:
        return Config.from_file(cfg_file)

@pytest.fixture(scope='session')
def sim_file() -> str:
    return _mock_sim_file_data

@pytest.fixture(scope='session')
def simulation_json(tmpdir_factory, sim_file):
    fn = tmpdir_factory.mktemp('sim').join('sim.json')
    with open(str(fn), 'w') as f:
        f.write(sim_file)
    return fn

@pytest.fixture()
def _mock_sim_json(mocker, sim_file):
    # todo: figure out what this does.
    mocked_sim = mocker.mock_open(
        read_data=sim_file,
    )
    mocker.patch('builtins.open', mocked_sim)