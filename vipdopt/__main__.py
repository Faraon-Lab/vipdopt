"""Run the Vipdopt software package."""
import logging
import os
import sys
from argparse import SUPPRESS, ArgumentParser

import vipdopt
from vipdopt.utils import setup_logger

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

path = os.path.dirname(f)
path = os.path.join(path, '..')
sys.path.insert(0, path)


from pathlib import Path

from vipdopt.project import Project

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='vipdopt',
        description='Volumetric Inverse Photonic Design Optimizer',
    )
    parser.add_argument('-v', '--verbose', action='store_const', const=True,
                            default=False, help='Enable verbose output.')
    parser.add_argument(
         '--log',
        type=Path,
        default=None,
        help='Path to the log file.'
    )

    subparsers = parser.add_subparsers()
    opt_parser = subparsers.add_parser('optimize')

    opt_parser.add_argument('-v', '--verbose', action='store_const', const=True,
                            default=SUPPRESS, help='Enable verbose output.')
    opt_parser.add_argument(
        'directory',
        type=Path,
        help='Project directory to use',
    )
    opt_parser.add_argument(
         '--log',
        type=Path,
        default=SUPPRESS,
        help='Path to the log file.'
    )

    args = parser.parse_args()

    if args.log is None:
        args.log = args.directory / 'dev.log'

    # Set verbosity
    level = logging.DEBUG if args.verbose else logging.INFO
    vipdopt.logger = setup_logger('global_logger', level, log_file=args.log)

    #
    # Step 0
    #
    vipdopt.logger.info('Beginnning Step 0: Project Setup...')

    project = Project()
    project.load_project(args.directory)

    # # Load configuration file

    # # Load Simulations...

    # # Create all different versions of simulation that are run




    vipdopt.logger.info('Completed Step 0: Project Setup')

    # #
    # # Step 1
    # #


    # #
    # # Step 2:
    # #

    # # Create sources and figures of merit (FoMs)

    #     (grad_fwd_monitors[i], grad_adj_monitors[j]) for i, j in monitor_pairs

    #     BayerFilterFoM(
    #         'TE',
    #     ) for i, j in monitor_pairs

    # # Add weighted FoMs

    # # Create optimizer


    # Create optimization

    #     all_sims,
    #     bayer_filter,
    #     full_fom,
    #     optimizer,
    #     max_iter,
    #     max_epochs,

    # for sim in all_sims:

    # TODO: Add reinterpolation of the device


    #
    # Step 3
    #
    vipdopt.logger.info('Beginning Step 3: Run Optimization')

    project.optimization.run()

    vipdopt.logger.info('Completed Step 3: Run Optimization')

    #
    # Cleanup
    #
    # for i in range(len(all_sims)):
