"""Tests for filter.py"""

import re
from re import Pattern
from typing import Any

import numpy as np
import pytest

from testing import assert_close, assert_equal
from vipdopt.configuration import Config, SonyBayerConfig

TEST_YAML_PATH = 'vipdopt/configuration/config_example.yml'

@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_empty_config')
def test_load_empty_yaml():
    cfg = Config()
    cfg.read_file('fakefile.yaml')

    with pytest.raises(AttributeError):
        _ = cfg.fixed_layers  # Accessing a property that doesn't exist


@pytest.mark.smoke()
def test_load_yaml():
    cfg = Config()
    cfg.read_file(TEST_YAML_PATH)

    expected_pairs = [
        ('fixed_layers', [1, 3]),
        ('do_rejection', False),
        ('start_from_step', 0),
        ('adam_betas', (0.9, 0.999)),
    ]

    for (k, v) in expected_pairs:
        assert_equal(getattr(cfg, k), v)

    assert_equal(cfg.fixed_layers, [1, 3])
    assert_equal(cfg.do_rejection, False)
    assert_equal(cfg.start_from_step, 0)
    assert_equal(cfg.adam_betas, (0.9, 0.999))


@pytest.mark.parametrize(
        'func, value',
        [
            # Mesh Properties
            (SonyBayerConfig.min_feature_size_voxels, 0.085 / 0.085),

            # Optical Properties
            (SonyBayerConfig.min_device_permittivity, 1.5 ** 2),
            (SonyBayerConfig.max_device_permittivity, 2.4 ** 2),
            (SonyBayerConfig.focal_length_um, 0.051 * 30),
            (SonyBayerConfig.focal_plane_center_vertical_um, -0.051 * 30),

            # Device Properties
            (SonyBayerConfig.vertical_layer_height_voxels, round(0.408 / 0.051)),
            (SonyBayerConfig.device_size_vertical_um, 0.408 * 5),
            (SonyBayerConfig.device_voxels_lateral, round(2.04 / 0.085)),
            (SonyBayerConfig.device_voxels_vertical, round(0.408 * 5 / 0.051)),
            (
                SonyBayerConfig.device_voxels_simulation_mesh_lateral,
                1 + int(2.04 / 0.017),
            ),
            (
                SonyBayerConfig.device_voxels_simulation_mesh_vertical,
             2 + int((0.408 * 5) /0.017),
            ),
            (SonyBayerConfig.device_vertical_maximum_um, 0.408 * 5),
            (SonyBayerConfig.device_size_lateral_bordered_um, 2.04),
            (
                SonyBayerConfig.device_voxels_simulation_mesh_lateral_bordered,
                1 + int(2.04 / 0.017),
            ),
            (SonyBayerConfig.border_size_um, 5 * 0.051),
            (SonyBayerConfig.border_size_voxels, int(round((5 * 0.051) /0.085))),
            (SonyBayerConfig.device_voxels_lateral_bordered, int(round(2.04 /0.085))),

            # FDTD Properties
            (SonyBayerConfig.vertical_gap_size_um, 0.085 * 15),
            (SonyBayerConfig.lateral_gap_size_um, 0.051 * 10),
            (
                SonyBayerConfig.fdtd_region_size_vertical_um,
                2 * (0.085 * 15) + (0.408 * 5) + (0.051 * 30),
            ),
            (SonyBayerConfig.fdtd_region_size_lateral_um, 2 * (0.051 * 10) + 2.04),
            (SonyBayerConfig.fdtd_region_maximum_vertical_um, 2.04 + (0.085 * 15)),
            (
                SonyBayerConfig.fdtd_region_minimum_vertical_um,
                -1 * 0.051 * 30 - (0.085 * 15),
            ),

            # Surrounding Properties
            (SonyBayerConfig.pec_aperture_thickness_um, 3 * 0.017),
            (
                SonyBayerConfig.sidewall_x_positions_um,
                [(2.04 + 0.085) / 2, 0, -(2.04 + 0.085)/2, 0],
            ),
            (
                SonyBayerConfig.sidewall_y_positions_um,
                [0, (2.04 + 0.085) / 2, 0, -(2.04 + 0.085)/2],
            ),
            (
                SonyBayerConfig.sidewall_xspan_positions_um,
                [0.085, 2.04 + 2 * 0.085, 0.085, 2.04 + 2 * 0.085],
            ),
            (
                SonyBayerConfig.sidewall_yspan_positions_um,
                [2.04 + 0.085 * 2, 0.085, 2.04 + 2 * 0.085, 0.085],
            ),

            # Spectral Properties
            (SonyBayerConfig.bandwidth_um, 0.725 - 0.375),
            (SonyBayerConfig.num_design_frequency_points, 3 * 20),
            (SonyBayerConfig.num_eval_frequency_points, 6 * 3 * 20),
            (SonyBayerConfig.lambda_values_um, np.linspace(0.375, 0.725, 3 * 20)),
            (
                SonyBayerConfig.max_intensity_by_wavelength,
                (2.04 ** 4) / \
                    ((0.051 * 30)**2 * np.power(np.linspace(0.375, 0.725, 3 * 20), 2)),
            ),
            (SonyBayerConfig.dispersive_range_size_um, (0.725 - 0.375) / 1),
            (SonyBayerConfig.dispersive_ranges_um, [[0.375, 0.725]]),

            # PDAF Functionality Properties
            (SonyBayerConfig.pdaf_focal_spot_x_um, 0.25 * 2.04),
            (SonyBayerConfig.pdaf_focal_spot_y_um, 0.25 * 2.04),
            (SonyBayerConfig.pdaf_lambda_min_um, 0.375),
            (SonyBayerConfig.pdaf_lambda_max_um, 0.375 + (0.725 - 0.375) / 3),
            (
                SonyBayerConfig.pdaf_lambda_values_um,
                np.linspace(0.375, 0.375 + (0.725 - 0.375) / 3, 3 * 20),
            ),
            (
                SonyBayerConfig.pdaf_max_intensity_by_wavelength,
                (2.04 **4) / ((0.051 * 30)**2 * \
                              np.power(
                                  np.linspace(
                                      0.375,
                                      0.375 + (0.725 - 0.375) / 3,
                                      3 * 20),
                                    2)
                            ),
            ),

            # Forward Properties
            (SonyBayerConfig.lateral_aperture_um, 1.1 * 2.04),
            (SonyBayerConfig.src_maximum_vertical_um, 0.408 * 5 + (0.085 * 15 * 2 / 3)),
            (
                SonyBayerConfig.src_minimum_vertical_um,
                -(0.051 * 30) - 0.5 * (0.085 * 15),
            ),
            (SonyBayerConfig.mid_lambda_um, (0.375 + 0.725) / 2),
            (SonyBayerConfig.source_angle_theta_rad, np.arcsin(np.sin(0)  * 1.0 / 1.5)),
            (
                SonyBayerConfig.source_angle_theta_deg,
                np.arcsin(np.sin(0)  * 1.0 / 1.5) * 180 / np.pi,
            ),

            # Adjoint Properties
            (SonyBayerConfig.adjoint_vertical_um, - 0.051 * 30),
            (SonyBayerConfig.num_adjoint_sources, 4),
            (
                SonyBayerConfig.adjoint_x_positions_um,
                2.04 * np.array([.25, -.25, -.25, .25]),
            ),
            (
                SonyBayerConfig.adjoint_y_positions_um,
                2.04 *np.array([.25, .25, -.25, -.25]),
            ),

            # Optimization Properties
            (SonyBayerConfig.epoch_range_design_change_max, 0.15 - 0.05),
            (SonyBayerConfig.epoch_range_design_change_min, 0.05),
            (SonyBayerConfig.spectral_focal_plane_map, [
                [0, 20],
                [20, 2 * 20],
                [2 * 20, 3 * 20],
                [20, 2 * 20],
            ]),
            (SonyBayerConfig.desired_peak_location_per_band,
             [
                np.argmin(np.power(np.linspace(0.375, 0.725, 3 * 20) - band, 2))
                for band in (0.45, 0.54, 0.65)
             ]),

        ]
)
def test_cascading_params(func, value: Any):
    cfg = SonyBayerConfig()
    cfg.read_file(TEST_YAML_PATH)

    assert_close(func.__get__(cfg), value, err=1e-5)


@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_empty_config')
def test_unsupported_filetype():
    cfg = Config()
    with pytest.raises(NotImplementedError):
        cfg.read_file('fakefilename', 'json')

@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_bad_config')
@pytest.mark.parametrize(
    'fname, msg',
    [
        ('bad_config_1.yml', r'.*border_optimization.+use_smooth_blur.*'),
        ('bad_config_2.yml', r'.*border_optimization.+num_sidewalls.*> 0.*'),
        ('bad_config_3.yml',
         r'.*device_voxels_lateral.+(None|\d+(?!\d)(?<=[13579])).*'),
        ('bad_config_4.yml',
         r'.*device_voxels_lateral.+(None|\d+(?!\d)(?<=[13579])).*'),
        ('bad_config_5.yml', r'.*add_pdaf.*add_infrared.*not compatible'),
        ('bad_config_6.yml',
         r'.*reinterpolate_permittivity_factor.*(?!1)(\d|None).*'),
    ]
)
def test_bad_config(fname: str, msg: Pattern):
    cfg = SonyBayerConfig()
    with pytest.raises(ValueError) as e:  # noqa: PT011
        cfg.read_file(fname)
    assert re.match(msg, e.value.args[0])
