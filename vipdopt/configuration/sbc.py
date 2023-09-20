"""Configuration manager for SonyBayerFilter."""

import logging
from typing import Any

import numpy as np
from overrides import override

from vipdopt.configuration.config import Config

# A lookup table that adds additional vertical mesh cells depending on layers
LOOKUP_MESH_CELLS = LOOKUP_MESH_CELLS = {1:3, 2:3, 3:2, 4:3, 5:2, 6:2, 8:3, 10:2,
                                    15:2, 20:2, 30:2, 40:2}

VERTICAL_LAYERS = 10

class SonyBayerConfig(Config):
    """Config object specifically for use with the Sony bayer filter optimization."""

    def __init__(self):
        """Initialize SonyBayerConfig object."""
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> Any:
        """Set the value of an attribute, creating it if it doesn't already exist."""
        self.__dict__[name] = value

    @override
    def read_file(self, filename: str, cfg_format: str = 'yaml'):
        super().read_file(filename, cfg_format)
        self._validate()

    def _explicit_band_centering(self):
        # Determine the wavelengths that will be directed to each focal area
        self.spectral_focal_plane_map = [
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points]
        ]

        dplpb = self.desired_peak_location_per_band
        bandwidth_left_by_quad = [
            ( dplpb[ 1 ] - dplpb[ 0 ] ) // 2,
            ( dplpb[ 1 ] - dplpb[ 0 ] ) // 2,
            ( dplpb[ 2 ] - dplpb[ 1 ] ) // 2,
            ( dplpb[ 1 ] - dplpb[ 0 ] ) // 2
        ]

        bandwidth_right_by_quad = [
            ( dplpb[ 1 ] - dplpb[ 0 ] ) // 2,
            ( dplpb[ 2 ] - dplpb[ 1 ] ) // 2,
            ( dplpb[ 2 ] - dplpb[ 1 ] ) // 2,
            ( dplpb[ 2 ] - dplpb[ 1 ] ) // 2
        ]

        bandwidth_left_by_quad = list(np.array(bandwidth_left_by_quad) / 1.75)
        bandwidth_right_by_quad = list(np.array(bandwidth_right_by_quad) / 1.75)

        quad_to_band = [ 0, 1, 1, 2 ] if self.shuffle_green else [ 0, 1, 2, 1 ]

        weight_individual_wavelengths_by_quad = \
            np.zeros((4,self.num_design_frequency_points))
        for quad_idx in range( 4 ):
            bandwidth_left = bandwidth_left_by_quad[ quad_to_band[ quad_idx ] ]
            bandwidth_right = bandwidth_right_by_quad[ quad_to_band[ quad_idx ] ]

            band_center = dplpb[ quad_to_band[ quad_idx ] ]

            for wl_idx in range( self.num_design_frequency_points ):

                choose_width = bandwidth_right
                if wl_idx < band_center:
                    choose_width = bandwidth_left

                scaling_exp = -1. / np.log( 0.5 )
                weight_individual_wavelengths_by_quad[ quad_idx, wl_idx ] = \
                    np.exp(
                       -( wl_idx - band_center )**2 /\
                          ( scaling_exp * choose_width**2 )
                        )
        self.weight_individual_wavelengths_by_quad = \
            weight_individual_wavelengths_by_quad

    def _add_infrared(self):
        self.lambda_max_um = 1.0
        self.lambda_max_eval_um = 1.0

        wl_idx_ir_start = 0
        for wl_idx in range( len( self.lambda_values_um ) ):
            if self.lambda_values_um[ wl_idx ] > self.infrared_center_um:
                wl_idx_ir_start = wl_idx - 1
                break

        # Determine the wavelengths that will be directed to each focal area
        spectral_focal_plane_map = [
            [ 0, 2 + ( self.num_points_per_band // 2 ) ],
            [ 2 + ( self.num_points_per_band // 2 ), 4 + self.num_points_per_band ],
            [ 4 + self.num_points_per_band, 6 + self.num_points_per_band +\
              ( self.num_points_per_band // 2 ) ],
            [ wl_idx_ir_start, ( wl_idx_ir_start + 3 ) ]
        ]

        ir_band_equalization_weights = [
            1. / ( spectral_focal_plane_map[ idx ][ 1 ] - \
                  spectral_focal_plane_map[ idx ][ 0 ] ) \
                    for idx in range( len( spectral_focal_plane_map ) ) ]

        weight_individual_wavelengths = np.ones( len( self.lambda_values_um ) )
        for band_idx in range( len( spectral_focal_plane_map ) ):

            for wl_idx in range( spectral_focal_plane_map[ band_idx ][ 0 ],
                                spectral_focal_plane_map[ band_idx ][ 1 ] ):
                weight_individual_wavelengths[ wl_idx ] = \
                    ir_band_equalization_weights[ band_idx ]

        self.weight_individual_wavelengths = weight_individual_wavelengths

    def _layer_gradient(self):
        voxels_per_layer = np.array( [ 1, 2, 4, 4, 4, 4, 5, 5, 5, 6 ] )
        assert np.sum( voxels_per_layer ) == self.device_voxels_vertical

        if self.flip_gradient:
            voxels_per_layer = np.flip( voxels_per_layer )

        self.voxels_per_layer = np.array

    def _do_rejection(self):
        # Determine the wavelengths that will be directed to each focal area
        self.spectral_focal_plane_map = [
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points],
            [0, self.num_design_frequency_points]
        ]

        desired_band_width_um = 0.04#0.08#0.12#0.18


        self.wl_per_step_um = self.lambda_values_um[ 1 ] - self.lambda_values_um[ 0 ]

        #
        # weights will be 1 at the center and drop to 0 at the edge of the band
        #

        weighting_by_band = np.zeros( ( self.num_bands,
                                       self.num_design_frequency_points ) )

        for band_idx in range( self.num_bands ):
            wl_center_um = self.desired_peaks_per_band_um[ band_idx ]

            for wl_idx in range( self.num_design_frequency_points ):
                wl_um = self.lambda_values_um[ wl_idx ]
                weight = -0.5 + 1.5 * np.exp(
                    -( ( wl_um - wl_center_um )**2 /\
                       ( desired_band_width_um**2 ) ) )
                weighting_by_band[ band_idx, wl_idx ] = weight

        spectral_focal_plane_map_directional_weights = \
            np.zeros( ( 4, self.num_design_frequency_points ) )
        spectral_focal_plane_map_directional_weights[ 0, : ] = weighting_by_band[ 0 ]
        spectral_focal_plane_map_directional_weights[ 1, : ] = weighting_by_band[ 1 ]
        spectral_focal_plane_map_directional_weights[ 2, : ] = weighting_by_band[ 2 ]
        spectral_focal_plane_map_directional_weights[ 3, : ] = weighting_by_band[ 1 ]

        self.spectral_focal_plane_map_directional_weights = \
            spectral_focal_plane_map_directional_weights

    def _derive_params(self):
        """Derive the parameters that depend on the config files."""

        if self.border_optimization:
            self.device_size_lateral_bordered_um = 2 * self.border_size_um

            if self.evaluate_bordered_extended:
                self.border_size_um = self.device_size_lateral_um

            self.device_size_lateral_bordered_um += 2 * self.border_size_um
            self.device_voxels_lateral_bordered = \
                int(np.round(
                   self.device_size_lateral_bordered_um / \
                    self.geometry_spacing_lateral_um))

            # 1 if mesh_spacing_um == 0.017
            self.device_voxels_simulation_mesh_lateral_bordered = \
                int(self.device_size_lateral_bordered_um / self.mesh_spacing_um) + 1
        else:
           self.device_voxels_simulation_mesh_lateral_bordered = \
            self.device_voxels_simulation_mesh_lateral

        if self.use_airy_approximation:
            self.gaussian_waist_radius_um = self.airy_correction_factor * \
                self.mid_lambda_um * self.f_number
        else:
            self.gaussian_waist_radius_um = self.mid_lambda_um / \
                ( np.pi * ( 1. / ( 2 * self.f_number ) ) )
        self.gaussian_waist_radius_um *= self.beam_size_multiplier

        if self.sidewall_extend_pml:
            self.sidewall_thickness_um = (self.fdtd_region_size_lateral_um - \
                                          self.device_size_lateral_um) / 2

        if self.add_infrared:
           self._add_infrared()

        if self.layer_gradient:
            if self.num_vertical_layers != VERTICAL_LAYERS:
               raise ValueError(f"Expected 'num_vertical_layers'=={VERTICAL_LAYERS},"
                                f' got {self.num_vertical_layers}.')
            self._layer_gradient()

        if self.explicit_band_centering:
            self._explicit_band_centering()

        if self.do_rejection:
            self._do_rejection()

    def _validate(self):
        """Validate the config file and compute conditional attributes."""
        if self.safe_get('border_optimization') and self.safe_get('use_smooth_blur'):
            msg = ("Combining 'border_optimization' and "
            "'use_smooth_blur' is not supported")
            raise ValueError(msg)

        if self.safe_get('border_optimization') and self.safe_get('num_sidewalls') != 0:
            msg = ("Combining 'border_optimization' and "
            "'num_sidewalls' > 0 is not supported")
            raise ValueError(msg)


        if self.safe_get('add_pdaf'):
            dvl = self.safe_get(SonyBayerConfig.device_voxels_lateral)
            if dvl is None or dvl % 2 != 0:
               raise ValueError(("Expected 'device_voxels_lateral' to be even for"
                                ' PDAF implementation ease, got '
                                f'\'{dvl}\'.'))

            if self.safe_get('add_infrared'):
               raise ValueError("'add_pdaf and 'add_infrared' are not compatible.")


        if self.safe_get('reinterpolate_permittivity') == False and \
            self.safe_get('reinterpolate_permittivity_factor') != 1:
               raise ValueError(("Expected 'reinterpolate_permittivity_factor' to be 1 if not"
                                ' reinterpolating permittivity,'
                                f' got \'{self.reinterpolate_permittivity_factor}\'.'))

        self._derive_params()


    #
    # Mesh Properties
    #
    @property
    def min_feature_size_voxels(self):
        """Meant for square_blur and square_blur_smooth.

        Used when mesh cells and minimum feature cells don't match
        """
        return self.min_feature_size_um / self.geometry_spacing_lateral_um

    #! Not Tested
    @property
    def blur_half_width_voxels(self):
        """Meant for square_blur and square_blur_smooth.

        Used when mesh cells and minimum feature cells don't match
        """
        return np.ceil( ( self.min_feature_size_voxels - 1 ) / 2. )

    #
    # Optical Properties
    #
    @property
    def min_device_permittivity(self):
       return self.min_device_index**2

    @property
    def max_device_permittivity(self):
       return self.max_device_index**2

    @property
    def focal_length_um(self):
       return self.device_scale_um * 30

    @property
    def focal_plane_center_vertical_um(self):
       return -self.focal_length_um

    #
    # Device Properties
    #
    @property
    def vertical_layer_height_voxels(self):
       return int( self.vertical_layer_height_um / self.device_scale_um )

    @property
    def device_size_vertical_um(self):
       return self.vertical_layer_height_um * self.num_vertical_layers

    @property
    def device_voxels_lateral(self):
        return np.round( self.device_size_lateral_um / self.geometry_spacing_lateral_um)
    @property
    def device_voxels_vertical(self):
        return int(np.round( self.device_size_vertical_um / self.device_scale_um))

    @property
    def device_voxels_simulation_mesh_lateral(self):
        return  1 + int(self.device_size_lateral_um / self.mesh_spacing_um)

    @property
    def device_voxels_simulation_mesh_vertical(self):
        return LOOKUP_MESH_CELLS[ self.num_vertical_layers ] + \
                 int(self.device_size_vertical_um / self.mesh_spacing_um)

    @property
    def device_vertical_maximum_um(self):
        return self.device_size_vertical_um

    @property
    def device_size_lateral_bordered_um(self):
        return  self.device_size_lateral_um

    @property
    def device_voxels_simulation_mesh_lateral_bordered(self):
        return  self.device_voxels_simulation_mesh_lateral

    @property
    def border_size_um(self):
        return  5 * self.device_scale_um

    @property
    def border_size_voxels(self):
        return  int(np.round(self.border_size_um / self.geometry_spacing_lateral_um))

    @property
    def device_voxels_lateral_bordered(self):
        return  int(np.round( self.device_size_lateral_bordered_um /\
                              self.geometry_spacing_lateral_um))

    #
    # FDTD Properties
    #
    @property
    def vertical_gap_size_um(self):
        return  self.geometry_spacing_lateral_um * 15

    @property
    def lateral_gap_size_um(self):
        return  self.device_scale_um * 10

    @property
    def fdtd_region_size_vertical_um(self):
        return  2 * self.vertical_gap_size_um + self.device_size_vertical_um +\
            self.focal_length_um

    @property
    def fdtd_region_size_lateral_um(self):
        return  2 * self.lateral_gap_size_um + self.device_size_lateral_um

    @property
    def fdtd_region_maximum_vertical_um(self):
        return  self.device_size_vertical_um + self.vertical_gap_size_um

    @property
    def fdtd_region_minimum_vertical_um(self):
        return  -1 * self.focal_length_um - self.vertical_gap_size_um

    #
    # Surrounding Properties
    #
    @property
    def pec_aperture_thickness_um(self):
        return  3 * self.mesh_spacing_um #! new

    @property
    def sidewall_x_positions_um(self):
        return  [
            self.device_size_lateral_um / 2 + self.sidewall_thickness_um / 2,
            0,
            -self.device_size_lateral_um / 2 - self.sidewall_thickness_um / 2,
            0,
        ]

    @property
    def sidewall_y_positions_um(self):
        return  [
            0,
            self.device_size_lateral_um / 2 + self.sidewall_thickness_um / 2,
            0,
            -self.device_size_lateral_um / 2 - self.sidewall_thickness_um / 2,
        ]

    @property
    def sidewall_xspan_positions_um(self):
        return  [
            self.sidewall_thickness_um,
            self.device_size_lateral_um + self.sidewall_thickness_um * 2,
            self.sidewall_thickness_um,
            self.device_size_lateral_um + self.sidewall_thickness_um * 2,
        ]

    @property
    def sidewall_yspan_positions_um(self):
        return  [
            self.device_size_lateral_um + self.sidewall_thickness_um * 2,
            self.sidewall_thickness_um,
            self.device_size_lateral_um + self.sidewall_thickness_um * 2,
            self.sidewall_thickness_um,
        ]

    #
    # Spectral Properties
    #
    @property
    def bandwidth_um(self):
        return  self.lambda_max_um - self.lambda_min_um

    @property
    def num_design_frequency_points(self):
        return self.num_bands * self.num_points_per_band

    @property
    def num_eval_frequency_points(self):
        return  6 * self.num_design_frequency_points

    @property
    def lambda_values_um(self):
        return  np.linspace(
            self.lambda_min_um,
            self.lambda_max_um,
            self.num_design_frequency_points,
        )

    @property
    def max_intensity_by_wavelength(self):
        """Normalization factor for intensity depending on frequency

        Ensures equal weighting. For more information, see
        https://en.wikipedia.org/wiki/Airy_disk#:~:text=at%20the%20center%20of%20the%20diffraction%20pattern
        """
        return  (self.device_size_lateral_um**2)**2 /\
              (self.focal_length_um**2 * self.lambda_values_um**2)

    @property
    def dispersive_range_size_um(self):
        return  self.bandwidth_um / self.num_dispersive_ranges

    @property
    def dispersive_ranges_um(self):
        return  [[ self.lambda_min_um, self.lambda_max_um ]]

    #
    # PDAF Functionality Properties
    #
    @property
    def pdaf_focal_spot_x_um(self):
        return 0.25 * self.device_size_lateral_um

    @property
    def pdaf_focal_spot_y_um(self):
        return  0.25 * self.device_size_lateral_um

    @property
    def pdaf_lambda_min_um(self):
        return  self.lambda_min_um

    @property
    def pdaf_lambda_max_um(self):
        return  self.lambda_min_um + self.bandwidth_um / 3.

    @property
    def pdaf_lambda_values_um(self):
        return np.linspace(
            self.pdaf_lambda_min_um,
            self.pdaf_lambda_max_um,
            self.num_design_frequency_points
        )

    @property
    def pdaf_max_intensity_by_wavelength(self):
        return  (self.device_size_lateral_um**2)**2 /\
              (self.focal_length_um**2 * self.pdaf_lambda_values_um**2)

    #
    # Forward (Input) Source Properties
    #
    @property
    def lateral_aperture_um(self):
        return  1.1 * self.device_size_lateral_um

    @property
    def src_maximum_vertical_um(self):
        return  self.device_size_vertical_um + self.vertical_gap_size_um * 2. / 3.

    @property
    def src_minimum_vertical_um(self):
        return  -self.focal_length_um - 0.5 * self.vertical_gap_size_um

    @property
    def mid_lambda_um(self):
        return  (self.lambda_min_um + self.lambda_max_um)/2

    @property
    def gaussian_waist_radius_um(self):
        return self.beam_size_multiplier

    @property
    def source_angle_theta_rad(self):
        return  np.arcsin( np.sin( self.source_angle_theta_vacuum_deg * np.pi / 180. )\
                           * 1.0 / self.background_index )

    @property
    def source_angle_theta_deg(self):
        return self.source_angle_theta_rad * 180. / np.pi

    #
    # Adjoint Source Properties
    #
    @property
    def adjoint_vertical_um(self):
        return  -self.focal_length_um

    @property
    def num_adjoint_sources(self):
        return  self.num_focal_spots

    @property
    def adjoint_x_positions_um(self):
        return  [
            self.device_size_lateral_um / 4.,
            -self.device_size_lateral_um / 4.,
            -self.device_size_lateral_um / 4.,
            self.device_size_lateral_um / 4.,
        ]

    @property
    def adjoint_y_positions_um(self):
        return  [
            self.device_size_lateral_um / 4.,
            self.device_size_lateral_um / 4.,
            -self.device_size_lateral_um / 4.,
            -self.device_size_lateral_um / 4.,
        ]

    #
    # Optimization Properties
    #
    @property
    def epoch_range_design_change_max(self):
        return  self.epoch_start_design_change_max - self.epoch_end_design_change_max

    @property
    def epoch_range_design_change_min(self):
      return  self.epoch_start_design_change_min - self.epoch_end_design_change_min

    # Determine the wavelengths that will be directed to each focal area
    @property
    def spectral_focal_plane_map(self):
      return  [
        [0, self.num_points_per_band],
        [self.num_points_per_band, 2 * self.num_points_per_band],
        [2 * self.num_points_per_band, 3 * self.num_points_per_band],
        [self.num_points_per_band, 2 * self.num_points_per_band],
    ]

    @property
    def desired_peak_location_per_band(self):
        # Find indices of desired peak locations in lambda_values_um
        res = [
                np.argmin(
                    (self.lambda_values_um - b)**2,
                )
                for b in self.desired_peaks_per_band_um
        ]
        logging.debug(f'Desired peak locations per band are: {res}')
        return res
