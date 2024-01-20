"""Configuration manager for SonyBayerFilter."""


import numpy as np
import yaml
from overrides import override

from vipdopt.configuration.config import Config
from vipdopt.configuration.template import TemplateRenderer
from vipdopt.utils import PathLike

# A lookup table that adds additional vertical mesh cells depending on layers
LOOKUP_MESH_CELLS = {1:3, 2:3, 3:2, 4:3, 5:2, 6:2, 8:3, 10:2,
                                    15:2, 20:2, 30:2, 40:2}

VERTICAL_LAYERS = 10

class SonyBayerConfig(Config):
    """Config object specifically for use with the Sony bayer filter optimization."""

    def __init__(self):
        """Initialize SonyBayerConfig object."""
        super().__init__()

    @override
    def read_file(self, fname: PathLike, cfg_format: str='auto') -> None:
        super().read_file(fname, cfg_format=cfg_format)
        self._validate()


    def derive_params(self, renderer: TemplateRenderer):
        """Derive the parameters that depend on the config files."""
        new_yaml = renderer.render(data=self._parameters, pi=np.pi)
        new_params = yaml.safe_load(new_yaml)
        self._parameters.update(new_params)
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

        if self.safe_get('border_optimization'):
            self.device_size_lateral_bordered_um = 2 * self.border_size_um

            if self.safe_get('evaluate_bordered_extended'):
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

        if self.safe_get('use_airy_approximation'):
            self.gaussian_waist_radius_um = self.airy_correction_factor * \
                self.mid_lambda_um * self.f_number
        else:
            self.gaussian_waist_radius_um = self.mid_lambda_um / \
                ( np.pi * ( 1. / ( 2 * self.f_number ) ) )
        self.gaussian_waist_radius_um *= self.beam_size_multiplier

        if self.safe_get('sidewall_extend_pml'):
            self.sidewall_thickness_um = (self.fdtd_region_size_lateral_um - \
                                          self.device_size_lateral_um) / 2

        if self.safe_get('add_infrared'):
           self._add_infrared()

        if self.safe_get('layer_gradient'):
            if self.num_vertical_layers != VERTICAL_LAYERS:
               raise ValueError(f"Expected 'num_vertical_layers'=={VERTICAL_LAYERS},"
                                f' got {self.num_vertical_layers}.')
            self._layer_gradient()

        if self.safe_get('explicit_band_centering'):
            self._explicit_band_centering()

        if self.safe_get('do_rejection'):
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
            dvl = self.safe_get('device_voxels_lateral_um')
            if dvl is None or dvl % 2 != 0:
               raise ValueError("Expected 'device_voxels_lateral_um' to be even for"
                                ' PDAF implementation ease, got '
                                f"'{dvl}'.")

            if self.safe_get('add_infrared'):
               raise ValueError("'add_pdaf and 'add_infrared' are not compatible.")


        if self.safe_get('reinterpolate_permittivity') is False and \
            self.safe_get('reinterpolate_permittivity_factor') != 1:
               raise ValueError("Expected 'reinterpolate_permittivity_factor' to be 1"
                                ' if not reinterpolating permittivity,'
                                f" got '{self['reinterpolate_permittivity_factor']}'.")
