"""Library of code for working with Jinja and rendering templates."""

import logging
from argparse import ArgumentParser
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from jinja2 import Environment, FileSystemLoader, Undefined

from vipdopt.configuration.config import read_config_file
from vipdopt.utils import ensure_path, setup_logger


class TemplateRenderer:
    """Class for rendering Jinja Templates."""

    def __init__(self, src_directory: Path) -> None:
        """Initialize and TemplateRenderer."""
        p = ensure_path(src_directory)
        self.env = Environment(loader=FileSystemLoader(str(p)))

    def render(self, **kwargs) -> str:
        """Render template with provided data values."""
        return self.template.render(trim_blocks=True, lstrip_blocks=True, **kwargs)

    def render_to_file(self, fname: Path | str, **kwargs):
        """Render template with provided data values and save to file."""
        output = self.render(**kwargs)

        with open(fname, 'w') as f:
            f.write(output)

        logger.debug(f'Succesfully rendered and saved output to {fname}')

    def set_template(self, template: Path) -> None:
        """Set the active template for the renderer."""
        p = ensure_path(template)
        """Set the current template to render."""
        self.template = self.env.get_template(p.name)

    def register_filter(self, name: str, func: Callable) -> None:
        """Add or reassign a filter to use in the environment."""
        self.env.filters[name] = func


class SonyBayerRenderer(TemplateRenderer):
    """TemplateRenderer including various filters for ease of writing templates."""

    def __init__(self, src_directory: Path) -> None:
        """Initialize a SonyBayerRenderer."""
        super().__init__(src_directory)
        self.register_filter('linspace', np.linspace)
        self.register_filter('sin', np.sin)
        self.register_filter('tan', np.tan)
        self.register_filter('arcsin', np.arcsin)
        self.register_filter('argmin', np.argmin)
        self.register_filter('newaxis', SonyBayerRenderer._newaxis)

    @staticmethod
    def _newaxis(iterable: Iterable | Undefined) -> npt.ArrayLike:
        """Return the itertable as a NumPy array with an additional axis."""
        if iterable is None or isinstance(iterable, Undefined):
            return iterable
        return np.array(iterable)[:, np.newaxis]

    @staticmethod
    def _explicit_band_centering(num_points, dplpb, shuffle_green):
        """Determine the wavelengths that will be directed to each focal area.

        Arguments:
            num_points (int): The number of deisgn frequency points
            dplpb (float): Desired Peak Location Per Band
            shuffle_green (bool): Whether to shuffle green channel


        Returns:
            (np.array): 2D Array with shape (4 x num_points) containing the weights
                for each individual wavelength by quad.
        """
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

        quad_to_band = [ 0, 1, 1, 2 ] if shuffle_green else [ 0, 1, 2, 1 ]

        weight_by_quad = np.zeros((4, num_points))
        for quad_idx in range(4):
            bandwidth_left = bandwidth_left_by_quad[ quad_to_band[ quad_idx ] ]
            bandwidth_right = bandwidth_right_by_quad[ quad_to_band[ quad_idx ] ]

            band_center = dplpb[ quad_to_band[ quad_idx ] ]

            for wl_idx in range(num_points):

                choose_width = bandwidth_right
                if wl_idx < band_center:
                    choose_width = bandwidth_left

                scaling_exp = -1. / np.log( 0.5 )
                weight_by_quad[ quad_idx, wl_idx ] = \
                    np.exp(
                       -( wl_idx - band_center )**2 /\
                          ( scaling_exp * choose_width**2 )
                        )
        return weight_by_quad

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

if __name__ == '__main__':
    parser = ArgumentParser('Create a simulation YAML file from a template.')
    parser.add_argument('template', type=Path,
                        help='Jinja2 template file to use')
    parser.add_argument('data_file', type=Path,
                        help='File containing values to substitute into template'
    )
    parser.add_argument('output', type=Path,
                        help='File to output rendered template to.',
    )
    parser.add_argument(
        '-s',
        '--src-directory',
        type=Path,
        default='jinja_templates/',
        help='Directory to search for the jinja template.'
        ' Defaults to "jinja_templates/"',
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output. Takes priority over quiet.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Enable quiet output. Will only show critical logs.')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose \
        else logging.WARNING if args.quiet \
            else logging.INFO
    logger = setup_logger('template_logger', log_level)

    rndr = SonyBayerRenderer(args.src_directory)

    rndr.set_template(args.template)

    data = read_config_file(args.data_file)
    output = rndr.render(data=data, pi=np.pi)
    logger.info(f'Rendered Output:\n{output}')

    with open(args.output, 'w') as f:
        f.write(output)

    logger.info(f'Succesfully saved output to {args.output}')
