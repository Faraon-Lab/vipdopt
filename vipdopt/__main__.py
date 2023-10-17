"""Volumetric Inverse Photonic Design Optimizer."""
import logging
import os
import sys
from argparse import ArgumentParser

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

path = os.path.dirname(f)
path = os.path.join(path, '..')
sys.path.insert(0, path)


from vipdopt.configuration.config import Config

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='vipdopt',
        description='Volumetric Inverse Photonic Design Optimizer',
    )
    subparsers = parser.add_subparsers()
    opt_parser = subparsers.add_parser('optimize')
    eval_parser = subparsers.add_parser('evaluate')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Enable verbose output.')

    args = parser.parse_args()

    # Set verbosity
    levels = [logging.INFO, logging.WARNING, logging.DEBUG]
    level = levels[min(args.verbose, len(levels) - 1)]  # cap to last level index
    logging.basicConfig(level=level, format='%(message)s')

    cfg = Config()
    cfg.read_file('config.yml.example')
