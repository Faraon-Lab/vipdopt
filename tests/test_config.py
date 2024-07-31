"""Tests for filter.py"""

from contextlib import nullcontext as does_not_raise
from re import Pattern
from typing import Any

import numpy as np
import pytest

from testing import assert_close, assert_equal, assert_equal_dict
from tests.conftest import TEST_YAML_PATH
from vipdopt.configuration import Config, SonyBayerConfig


@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_empty_config')
def test_load_empty_yaml():
    cfg = Config()
    cfg.read_file('fakefile.yaml')

    with pytest.raises(KeyError):
        _ = cfg['fixed_layers']  # Accessing a property that doesn't exist


@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_example_config')
def test_load_yaml():
    cfg = Config()
    cfg.read_file('fakefile.yaml')

    expected_pairs = [
        ('fixed_layers', [1, 3]),
        ('do_rejection', False),
        ('start_from_step', 0),
        ('adam_betas', (0.9, 0.999)),
    ]

    for k, v in expected_pairs:
        assert_equal(cfg[k], v)


@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_example_config')
def test_from_file():
    cfg = Config.from_file('fakefile.yaml')

    expected_pairs = [
        ('fixed_layers', [1, 3]),
        ('do_rejection', False),
        ('start_from_step', 0),
        ('adam_betas', (0.9, 0.999)),
    ]

    for k, v in expected_pairs:
        assert_equal(cfg[k], v)


@pytest.mark.smoke()
@pytest.mark.parametrize(
    'filetype, expectation',
    [
        ('.yaml', does_not_raise()),
        ('.yml', does_not_raise()),
        ('.YAML', does_not_raise()),
        ('.YML', does_not_raise()),
        ('.json', does_not_raise()),
        ('.JSON', does_not_raise()),
        ('auto', does_not_raise()),
        ('AUTO', does_not_raise()),
        ('.html', pytest.raises(NotImplementedError)),
    ],
)
def test_save_load_config(tmp_path, filetype: str, expectation):
    og_cfg = Config.from_file(TEST_YAML_PATH)

    save_file = tmp_path / 'save'
    if filetype.lower() == 'auto':
        save_file = save_file.with_suffix('.yaml')
    else:
        save_file = save_file.with_suffix(filetype)
    with expectation:
        og_cfg.save(save_file, filetype)

        copy_cfg = Config.from_file(save_file)

        assert_equal_dict(copy_cfg, og_cfg)


@pytest.mark.usefixtures('_mock_example_config')
@pytest.mark.parametrize(
    'param, value',
    [
        # Mesh Properties
        ('min_feature_size_voxels', 0.085 / 0.085),
        # Optical Properties
        ('min_device_permittivity', 1.5**2),
        ('max_device_permittivity', 2.4**2),
        ('focal_length_um', 0.051 * 30),
        ('focal_plane_center_vertical_um', -0.051 * 30),
        # Device Properties
        ('vertical_layer_height_voxels', round(0.408 / 0.051)),
        ('device_size_vertical_um', 0.408 * 5),
        ('device_voxels_lateral', round(2.04 / 0.085)),
        ('device_voxels_vertical', round(0.408 * 5 / 0.051)),
        (
            'device_voxels_simulation_mesh_lateral',
            1 + int(2.04 / 0.017),
        ),
        (
            'device_voxels_simulation_mesh_vertical',
            2 + int((0.408 * 5) / 0.017),
        ),
        ('device_vertical_maximum_um', 0.408 * 5),
        ('device_size_lateral_bordered_um', 2.04),
        (
            'device_voxels_simulation_mesh_lateral_bordered',
            1 + int(2.04 / 0.017),
        ),
        ('border_size_um', 5 * 0.051),
        ('border_size_voxels', int(round((5 * 0.051) / 0.085))),
        ('device_voxels_lateral_bordered', int(round(2.04 / 0.085))),
        # FDTD Properties
        ('vertical_gap_size_um', 0.085 * 25),
        ('lateral_gap_size_um', 0.051 * 10),
        (
            'fdtd_region_size_vertical_um',
            2 * (0.085 * 25) + (0.408 * 5) + (0.051 * 30),
        ),
        ('fdtd_region_size_lateral_um', 2 * (0.051 * 10) + 2.04),
        ('fdtd_region_maximum_vertical_um', 2.04 + (0.085 * 25)),
        (
            'fdtd_region_minimum_vertical_um',
            -1 * 0.051 * 30 - (0.085 * 25),
        ),
        # Surrounding Properties
        ('pec_aperture_thickness_um', 3 * 0.017),
        (
            'sidewall_x_positions_um',
            [(2.04 + 0.085) / 2, 0, -(2.04 + 0.085) / 2, 0],
        ),
        (
            'sidewall_y_positions_um',
            [0, (2.04 + 0.085) / 2, 0, -(2.04 + 0.085) / 2],
        ),
        (
            'sidewall_xspan_positions_um',
            [0.085, 2.04 + 2 * 0.085, 0.085, 2.04 + 2 * 0.085],
        ),
        (
            'sidewall_yspan_positions_um',
            [2.04 + 0.085 * 2, 0.085, 2.04 + 2 * 0.085, 0.085],
        ),
        # Spectral Properties
        ('bandwidth_um', 0.725 - 0.375),
        ('num_design_frequency_points', 3 * 20),
        ('num_eval_frequency_points', 6 * 3 * 20),
        ('lambda_values_um', np.linspace(0.375, 0.725, 3 * 20)),
        (
            'max_intensity_by_wavelength',
            (2.04**4)
            / ((0.051 * 30) ** 2 * np.power(np.linspace(0.375, 0.725, 3 * 20), 2)),
        ),
        ('dispersive_range_size_um', (0.725 - 0.375) / 1),
        ('dispersive_ranges_um', [[0.375, 0.725]]),
        # PDAF Functionality Properties
        ('pdaf_focal_spot_x_um', 0.25 * 2.04),
        ('pdaf_focal_spot_y_um', 0.25 * 2.04),
        ('pdaf_lambda_min_um', 0.375),
        ('pdaf_lambda_max_um', 0.375 + (0.725 - 0.375) / 3),
        (
            'pdaf_lambda_values_um',
            np.linspace(0.375, 0.375 + (0.725 - 0.375) / 3, 3 * 20),
        ),
        (
            'pdaf_max_intensity_by_wavelength',
            (2.04**4)
            / (
                (0.051 * 30) ** 2
                * np.power(np.linspace(0.375, 0.375 + (0.725 - 0.375) / 3, 3 * 20), 2)
            ),
        ),
        # Forward Properties
        ('lateral_aperture_um', 1.1 * 2.04),
        ('src_maximum_vertical_um', 0.408 * 5 + (0.085 * 25 * 5 / 6)),
        (
            'src_minimum_vertical_um',
            -(0.051 * 30) - 0.5 * (0.085 * 25),
        ),
        ('mid_lambda_um', (0.375 + 0.725) / 2),
        ('source_angle_theta_rad', np.arcsin(np.sin(0) * 1.0 / 1.5)),
        (
            'source_angle_theta_deg',
            np.arcsin(np.sin(0) * 1.0 / 1.5) * 180 / np.pi,
        ),
        # Adjoint Properties
        ('adjoint_vertical_um', -0.051 * 30),
        ('num_adjoint_sources', 4),
        (
            'adjoint_x_positions_um',
            2.04 * np.array([0.25, -0.25, -0.25, 0.25]),
        ),
        (
            'adjoint_y_positions_um',
            2.04 * np.array([0.25, 0.25, -0.25, -0.25]),
        ),
        # Optimization Properties
        ('epoch_range_design_change_max', 0.15 - 0.05),
        ('epoch_range_design_change_min', 0.05),
        (
            'spectral_focal_plane_map',
            [
                [0, 20],
                [20, 2 * 20],
                [2 * 20, 3 * 20],
                [20, 2 * 20],
            ],
        ),
        (
            'desired_peak_location_per_band',
            [
                np.argmin(np.power(np.linspace(0.375, 0.725, 3 * 20) - band, 2))
                for band in (0.45, 0.54, 0.65)
            ],
        ),
    ],
)
def test_cascading_params(param, value: Any, template_renderer):
    cfg = SonyBayerConfig()
    cfg.read_file('fakefile.yaml')
    cfg.derive_params(template_renderer)

    assert_close(cfg[param], value, err=1e-5)


@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_empty_config')
def test_unsupported_filetype():
    cfg = Config()
    with pytest.raises(NotImplementedError):
        cfg.read_file('fakefilename', '.html')


@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_bad_config')
@pytest.mark.parametrize(
    'fname, msg',
    [
        ('bad_config_1.yml', r'.*border_optimization.+use_smooth_blur.*'),
        ('bad_config_2.yml', r'.*border_optimization.+num_sidewalls.*> 0.*'),
        (
            'bad_config_3.yml',
            r'.*device_voxels_lateral.+(None|\d+(?!\d)(?<=[13579])).*',
        ),
        (
            'bad_config_4.yml',
            r'.*device_voxels_lateral.+(None|\d+(?!\d)(?<=[13579])).*',
        ),
        ('bad_config_5.yml', r'.*add_pdaf.*add_infrared.*not compatible'),
        ('bad_config_6.yml', r'.*reinterpolate_permittivity_factor.*(?!1)(\d|None).*'),
    ],
)
def test_bad_config(fname: str, msg: Pattern):
    cfg = SonyBayerConfig()
    with pytest.raises(ValueError, match=msg):
        cfg.read_file(fname)
