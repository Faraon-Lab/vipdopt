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

from vipdopt.gui import start_gui
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
    # 'optimizer': Adam/GDOptimizer object; 'device': Device object;
    # 'base_sim': Simulation object; 'src_to_sim_map': dict with source names as keys, Simulation objects as values
    # 'foms': list of FoM objects, 'weights': array of shape (#FoMs, nλ)

    # Now that config is loaded, set up lumapi
    if os.getenv('SLURM_JOB_NODELIST') is None:
        vipdopt.lumapi = import_lumapi(
            project.config.data['lumapi_filepath_local']
        )  # Windows (local machine)
    else:
        vipdopt.lumapi = import_lumapi(
            project.config.data['lumapi_filepath_hpc']
        )  # HPC (Linux)
    
    
    # TODO!!!!
    # Nia: also all of the simulations can be made from the FoMs using the create_forward_sims
    # so you just need the base sim, and it’ll make all of the different versions that need to be
    # run automatically from the FoM
    
    # fwd_sim = fom.create_forward_sim(base_sim)[0]
    # fwd_sim.set_path('test_data\\fwd_sim.fsp')
    # adj_sim = fom.create_adjoint_sim(base_sim)[0]
    # adj_sim.set_path('test_data\\adj_sim.fsp')

    # # fdtd = LumericalFDTD()
    # # fdtd.connect(hide=True)

    # # fdtd.save('test_data\\fwd_sim.fsp', fwd_sim)
    # # fdtd.save('test_data\\adj_sim.fsp', adj_sim)
    # # fdtd.addjob('test_data\\fwd_sim.fsp')
    # # fdtd.addjob('test_data\\adj_sim.fsp')
    # # fdtd.runjobs(0)

    # # fdtd.reformat_monitor_data([fwd_sim, adj_sim])
    # fwd_sim.link_monitors()
    # adj_sim.link_monitors()

    # fom_val = fom.compute_fom()
    # vipdopt.logger.debug(f'FoM: {fom_val.shape}')
    # grad_val = fom.compute_grad()
    # vipdopt.logger.debug(f'Gradient: {grad_val.shape}')
    
    print('Reached end of code.')