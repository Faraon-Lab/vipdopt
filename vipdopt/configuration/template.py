"""Library of code for working with Jinja and rendering templates."""

import logging
from argparse import ArgumentParser
import pathlib

from jinja2 import Environment, FileSystemLoader, Undefined
import numpy as np

from vipdopt.simulation import LumericalSimObject
from vipdopt.configuration.config import get_config_loader

# def linspace(start, stop, num):
#     return np.linspace(start, stop, num)

# def sin(x):
#     return np.sin(x)

# def arcsin(x):
#     return np.arcsin(x)

def newaxis(iterable):
    if iterable is None or isinstance(iterable, Undefined):
        return iterable
    
    return np.array(iterable)[:, np.newaxis]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

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
        help='Directory to search for the jinja template. Defaults to "jinja_templates/"',
    )

    args = parser.parse_args()

    env = Environment(loader=FileSystemLoader(args.src_directory))
    env.filters['linspace'] = np.linspace
    env.filters['sin'] = np.sin
    env.filters['arcsin'] = np.arcsin
    env.filters['argmin'] = np.argmin
    env.filters['newaxis'] = newaxis

    template = env.get_template(args.template)

    data = get_config_loader(pathlib.Path(args.data_file).suffix)(args.data_file)
    output = template.render(data=data, pi=np.pi, trim_blocks=True, lstrip_blocks=True)
    print(output)

    with open(args.output, 'w') as f:
        f.write(output)