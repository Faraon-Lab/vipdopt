import sys
from pathlib import Path
from argparse import SUPPRESS, ArgumentParser
import logging

# Import main package and add program folder to PATH.
sys.path.append(Path.cwd()) # 20240219 Ian: Only added this so I could debug some things from my local VSCode
import vipdopt
from vipdopt.project import Project, create_internal_folder_structure
from vipdopt.optimization import (
    Device,
    FoM,
    Optimization,
    UniformMSEFoM,
    MSEFoM,
    # LumericalOptimization,
    # SuperFoM,
)
from vipdopt.utils import setup_logger

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

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

    #
    # * Step 0: Set up simulation conditions and environment. ======================================================================================
    # i.e. current sources, boundary conditions, supporting structures, surrounding regions.
    # Also any other necessary editing of the Lumerical environment and objects.
    vipdopt.logger.info('Beginning Step 0: Project Setup...')

    project = Project()
    project.load_project(args.directory, config_name=args.config)
    # What does the Project class contain?
    # 'dir': directory where it's stored; 'config': SonyBayerConfig object; 'optimization': Optimization object;
    # 'device': Device object; 'base_sim': Simulation object;
    # 'src_to_sim_map': dict with source names as keys, Simulation objects as values
    # 'foms': list of FoM objects, 'weights': array of shape (#FoMs, nλ)
    #! Each Project should correspond only to one Device optimized for a certain functionality and situation.
    #! Optimization parameter sweeps necessitate multiple Projects.

    import numpy as np
    
    FoM_1 = MSEFoM()
    from PIL import Image
    im = Image.open('examples/image_mse_fom_example.jpg').resize(project.device.size[:2], Image.Resampling.LANCZOS)
    im = np.repeat(np.array(im)[:, :, np.newaxis], project.device.size[2], axis=2)
    FoM_1 = MSEFoM(target=im)
    FoM_2 = MSEFoM(target=im)
    project.fom = FoM(None, None, [(FoM_1,), (FoM_2,)], (1.0,2.0))

    # Multiple simulations may be created here due to the need for large-area simulation segmentation, or genetic optimizations
    assert project.base_sim is not None
    assert project.device is not None
    # TODO: Partitioning the base_sim into simulations: i.e. a list of base Simulation objects
    # TODO: And the same with devices.
    # TODO: 1 base sim for 1 device and vice versa. A single device may contain multiple design regions, however.
    # !! Ultimately all the devices and simulation regions will be stitched back together.
    base_sims = project.base_sim.partition()
    devices = project.device.partition(base_sims)
    foms = project.fom.partition()

    # Setup Optimization(s) - 1 for each base_sim + device pair.
    for opt_idx, base_sim in enumerate(base_sims):

        cfg = project.config
        # NOTE: The optimizer is explicitly only a property of the Optimization, not the containing Project.
        # optimizer = project.load_optimizer(cfg)
        from vipdopt.optimization.optimizer import NLOptOptimizer
        optimizer = NLOptOptimizer()
        base_sim.set_solver(None)

        optimization = Optimization(
            base_sim,
            #  sims,
            project.device,
            optimizer,
            foms[opt_idx],
        #     # Explicitly declare the fom kwargs passed to compute_fom() as cfg.
        #     fom_kwargs=cfg,
        #     # It could be different but for now it's not
            cfg=cfg,
            epoch_list=cfg.get('epoch_list'),
            true_iteration=cfg.get('iteration', 0),
        #     env_vars=env_vars,
            dirs=Optimization.create_opt_folder_structure(
                                Path(project.dir)/f'optimizations/opt{opt_idx}',
                                pull_files_debug_mode=cfg.get('pull_sim_files_from_debug_folder')
                            ),
            project=project,
        )
        vipdopt.logger.info(f'Optimization {opt_idx} initialized.')
        project.optimizations.append(optimization)

    vipdopt.logger.info('Completed Step 0: Project Setup')

    # What happens if the optimization breaks? Go into project.subdirectories['checkpoints']
    # Copy project.json and config.json to the current folder (e.g. runs\test_run)
    # Launch vipdopt again but with "--config config.json" i.e. point the config file to the savestate
    # TODO: Write Tests

    project.start_all_optimizations()

    # # Numpy Export final design
    # project.device.save( project.subdirectories['data'] / 'final_device.npy' )
    # project.device.save( project.subdirectories['summary'] / 'last_device.npy' )
    # # STL Export final design
    # project.device.export_density_as_stl( project.subdirectories['data'] / 'final_device.stl' )
    # # GDS Export final design
    # project.device.export_density_as_gds( project.subdirectories['data'] / 'gds' )

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    project.device.visualize_layer()
    plt.show()

    print('End of code reached.')