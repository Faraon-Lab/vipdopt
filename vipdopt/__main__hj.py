"""Run the Vipdopt software package."""

import logging
import os
import sys
from argparse import SUPPRESS, ArgumentParser

import numpy as np

# Import main package and add program folder to PATH.
sys.path.append( os.getcwd() )  # 20240219 Ian: Only added this so I could debug some things from my local VSCode
import vipdopt
from vipdopt.utils import import_lumapi, setup_logger

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

path = os.path.dirname(f)
path = os.path.join(path, '..')
sys.path.insert(0, path)


from pathlib import Path

# from vipdopt.gui import start_gui
from vipdopt.project import Project

if __name__ == '__main__':
    
    # Set up argument parser
    parser = ArgumentParser(
        prog='vipdopt',
        description='Volumetric Inverse Photonic Design Optimizer',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=True,
        default=False,
        help='Enable verbose output.',
    )
    parser.add_argument(
        '--log', type=Path, default='dev.log', help='Path to the log file.'
    )
    
    # Set up argument subparsers
    subparsers = parser.add_subparsers(help='commands', dest='command')
    opt_parser = subparsers.add_parser('optimize')
    gui_parser = subparsers.add_parser('gui')
    
    # Configure optimizer subparser
    opt_parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=True,
        default=SUPPRESS,
        help='Enable verbose output.',
    )
    opt_parser.add_argument(
        'directory',
        type=Path,
        help='Project directory to use',
    )
    opt_parser.add_argument(
        '--log', type=Path, default=SUPPRESS, help='Path to the log file.'
    )
    opt_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file to use in the optimization; defaults to config.yaml',
    )
    
    args = parser.parse_args()

    # Set up logging
    log_file = args.directory / args.log if args.command == 'optimize' else args.log
    # Set verbosity
    level = logging.DEBUG if args.verbose else logging.INFO
    vipdopt.logger = setup_logger('global_logger', level, log_file=log_file)
    
    # GUI? 
    if args.command == 'gui':
        sys.exit(start_gui([]))
    elif args.command is None:
        print(parser.format_usage())
        sys.exit(1)
    
    
    #
    # * Step 0: Set up simulation conditions and environment. ======================================================================================
    # i.e. current sources, boundary conditions, supporting structures, surrounding regions.
    # Also any other necessary editing of the Lumerical environment and objects.
    vipdopt.logger.info('Beginning Step 0: Project Setup...')

    project = Project()
    project.load_project(args.directory, config_name=args.config)
    # What does the Project class contain?
    # 'dir': directory where it's stored; 'config': SonyBayerConfig object; 'optimization': Optimization object;
    # 'optimizer': [Adam/GradientAscent]Optimizer object; 'device': Device object;
    # 'base_sim': Simulation object; 'src_to_sim_map': dict with source names as keys, Simulation objects as values
    # 'foms': list of FoM objects, 'weights': array of shape (#FoMs, nλ)
    vipdopt.logger.info('Completed Step 0: Project Setup')

    # Now that config is loaded, set up lumapi
    if os.getenv('SLURM_JOB_NODELIST') is None:
        vipdopt.lumapi = import_lumapi(
            project.config.data['lumapi_filepath_local']
        )  # Windows (local machine)
        vipdopt.logger.info(f'Local Lumapi path is: {project.config.data["lumapi_filepath_local"]}')
        hide_fdtd=False
    else:
        vipdopt.lumapi = import_lumapi(
            project.config.data['lumapi_filepath_hpc']
        )  # HPC (Linux)
        vipdopt.logger.info(f'HPC Lumapi path is: {project.config.data["lumapi_filepath_hpc"]}')
        hide_fdtd=True

    fdtd = project.optimization.fdtd    # NOTE: - the instantiation is called in optimization.py
    vipdopt.fdtd = fdtd      # Makes it available to global

    fdtd.connect(hide=hide_fdtd)
    project.start_optimization()
    
    # STL Export final design
    project.device.export_density_as_stl( project.subdirectories['data'] / 'final_device.stl' )
    # GDS Export final design
    project.device.export_density_as_gds( project.subdirectories['data'] / 'gds' )
    
    vipdopt.logger.info('Reached end of code.')