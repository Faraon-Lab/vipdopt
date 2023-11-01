"""Library of code for working with Jinja and rendering templates."""

import logging
from argparse import ArgumentParser
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt
from jinja2 import Environment, FileSystemLoader, Undefined

from vipdopt.configuration.config import read_config_file


class TemplateRenderer:
    """Class for rendering Jinja Templates."""

    def __init__(self, src_directory: str) -> None:
        """Initialize and TemplateRenderer."""
        self.env = Environment(loader=FileSystemLoader(src_directory))

    def render(self, **kwargs) -> str:
        """Render template with provided data values."""
        return self.template.render(**kwargs)

    def set_template(self, template: str) -> None:
        """Set the current template to render."""
        self.template = self.env.get_template(template)

    def register_filter(self, name: str, func: Callable) -> None:
        """Add or reassign a filter to use in the environment."""
        self.env.filters[name] = func


def newaxis(iterable: Iterable | Undefined) -> npt.ArrayLike:
    """Return the itertable as a NumPy array with an additional axis."""
    if iterable is None or isinstance(iterable, Undefined):
        return iterable
    return np.array(iterable)[:, np.newaxis]


if __name__ == '__main__':
    parser = ArgumentParser('Create a simulation YAML file from a template.')
    parser.add_argument('template', type=str,
                        help='Jinja2 template file to use')
    parser.add_argument('data_file', type=str,
                        help='File containing values to substitute into template'
    )
    parser.add_argument('output', type=str,
                        help='File to output rendered template to.',
    )
    parser.add_argument(
        '-s',
        '--src-directory',
        type=str,
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
    logging.getLogger().setLevel(log_level)

    rndr = TemplateRenderer(args.src_directory)
    rndr.register_filter('linspace', np.linspace)
    rndr.register_filter('sin', np.sin)
    rndr.register_filter('arcsin', np.arcsin)
    rndr.register_filter('argmin', np.argmin)
    rndr.register_filter('newaxis', newaxis)

    rndr.set_template(args.template)

    data = read_config_file(args.data_file)
    output = rndr.render(data=data, pi=np.pi, trim_blocks=True, lstrip_blocks=True)
    logging.info(f'Rendered Output:\n{output}')

    with open(args.output, 'w') as f:
        f.write(output)

    logging.info(f'Succesfully saved output to {args.output}')
