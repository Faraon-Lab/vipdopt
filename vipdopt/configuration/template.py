"""Library of code for working with Jinja and rendering templates."""

import logging
import os
import sys
from argparse import ArgumentParser
from collections.abc import Callable, Iterable
from pathlib import Path
import yaml

import numpy as np
import numpy.typing as npt
from jinja2 import Environment, FileSystemLoader, Undefined
from overrides import override

sys.path.append(os.getcwd())

import vipdopt
from vipdopt.configuration.config import read_config_file
from vipdopt.utils import ensure_path, convert_path, setup_logger

class TemplateRenderer:
    """Class for rendering Jinja Templates."""

    @ensure_path
    def __init__(self, src_directory: Path) -> None:
        """Initialize and TemplateRenderer."""

        self.env = Environment(loader=FileSystemLoader(str(src_directory)))

    def render(self, **kwargs) -> str:
        """Render template with provided data values."""
        return self.template.render(trim_blocks=True, lstrip_blocks=True, **kwargs)

    @ensure_path
    def render_to_file(self, fname: Path, **kwargs):
        """Render template with provided data values and save to file."""
        output = self.render(**kwargs)

        with open(fname, 'w') as f:
            f.write(output)

        vipdopt.logger.debug(f'Successfully rendered and saved output to {fname}')

    @ensure_path
    def set_template(self, template: Path) -> None:
        """Set the active template for the renderer."""
        self.template = self.env.get_template(template.name)

    def register_filter(self, name: str, func: Callable) -> None:
        """Add or reassign a filter to use in the environment."""
        self.env.filters[name] = func

class MetasurfaceRenderer(TemplateRenderer):
    """TemplateRenderer including various filters for ease of writing templates."""

    @ensure_path
    @override
    def __init__(self, src_directory: Path) -> None:
        """Initialize a SonyBayerRenderer."""
        super().__init__(src_directory)
        self.register_filter('nparray', np.array)
        self.register_filter('linspace', np.linspace)
        self.register_filter('sin', np.sin)
        self.register_filter('cos', np.cos)
        self.register_filter('tan', np.tan)
        self.register_filter('arcsin', np.arcsin)
        self.register_filter('argmin', np.argmin)
        self.register_filter('newaxis', MetasurfaceRenderer._newaxis)

    @staticmethod
    def _newaxis(iterable: Iterable | Undefined) -> npt.ArrayLike:
        """Return the itertable as a NumPy array with an additional axis."""
        if iterable is None or isinstance(iterable, Undefined):
            return iterable
        return np.array(iterable)[:, np.newaxis]

def default_template_parser():
    parser = ArgumentParser('Create a simulation YAML file from a template.')
    parser.add_argument('template', type=Path, help='Jinja2 template file to use')
    parser.add_argument(
        'data_file', type=Path,
        help='File containing values to substitute into template',
    )
    parser.add_argument(
        'output', type=Path,
        help='File to output rendered template to.',
    )
    parser.add_argument(
        '-s', '--src-directory', type=Path,
        default='runs/test_run/',
        help='Directory to search for the jinja template.'
        ' Defaults to "runs/test_run/"',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose output. Takes priority over quiet.',
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Enable quiet output. Will only show critical logs.',
    )
    
    return parser

def reload_template(template_dir, template, data_file, output_file):
    template_dir = convert_path(template_dir)
    output_file = convert_path(output_file)

    rndr = MetasurfaceRenderer(template_dir)
    rndr.set_template(template)

    data = read_config_file(data_file)
    output = rndr.render(data=data, pi=np.pi)
    vipdopt.logger.info(f'Rendered Output:\n{output}')

    if output_file.suffix in ['.yml','.yaml']:
        output = output.replace('None', 'null')
    with open(output_file, 'w') as f:
        f.write(output)

    vipdopt.logger.info(f'Successfully saved output to {output_file}')

if __name__ == '__main__':
    parser = default_template_parser()
    args = parser.parse_args()

    log_level = (
        logging.DEBUG
        if args.verbose
        else logging.WARNING
        if args.quiet
        else logging.INFO
    )
    logger = setup_logger('template_logger', log_level)

    reload_template(args.src_directory, args.template, args.data_file, args.output)
