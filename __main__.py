import sys
from pathlib import Path
from argparse import SUPPRESS, ArgumentParser
import logging
from functools import partial

import vipdopt.optimization

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
from vipdopt.optimization.optimizer import NLOptOptimizer, GradientAscentOptimizer, AdamOptimizer
from vipdopt.utils import setup_logger

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

# Example of how to pass custom functions into the pre-existing / pre-written class instances
def update_histories_em_filter(self):
    # Manually adjust metrics stored and calculated 
    #! These will be fed directly into plotter.py so this is the place to be changing labels / variable names and somesuch.
    for metric in ['transmission', 'intensity']:
        self.fom_hist.update({f'{metric}_overall': []})
        for i, f in enumerate(self.fom.foms):
            self.fom_hist.update( {f'{metric}_{i}': []} )
    self.fom_hist.update( {'intensity_overall_xyzwl': []} )

def generate_plots_simple_mse(self):
    """Generate the plots and save to file."""
    import pickle
    from vipdopt.eval import plotter
    
    folder = self.dirs['eval_info']
    iteration = self.iteration #  if self.iteration==self.epoch_list[-1] else self.iteration+1
    # vipdopt.logger.debug(f'Plotter. Iteration {iteration}: Plot histories length {len(self.fom_hist["intensity_overall"])}')

    # TODO: Copy all to summary folder as well.
    # ! 20240229 Ian - Best to be specifying functions for 2D and for 3D.

    fom_fig = plotter.plot_fom_trace(
        np.array(self.fom_hist['fom_overall']),
        folder)

    quads_to_plot = [0,1] if self.cfg['simulator_dimension']=='2D' else [0,1,2,3]
    quad_trans_fig = plotter.plot_bayer_quadrant_transmission_trace(
        np.array([self.fom_hist[f'fom_{x}'] for x in quads_to_plot]).swapaxes(0,1),
        folder,
    )
    cur_index = self.device.index_from_permittivity(self.device.get_permittivity())
    final_device_layer_fig, _ = plotter.visualize_device(
                                        self.device.coords['x'], self.device.coords['y'], cur_index,
                                        # self.device.coords['x'], self.device.coords['z'],
                                        # np.rot90(cur_index),         # 20241003: Want to see the side view for layering.
                                        folder,
                                        filename=f'_{iteration}'
                                    )

    # Evaluation Plots


    # Create plot pickle files for GUI visualization
    with (folder / 'fom.pkl').open('wb') as f:
        pickle.dump(fom_fig, f)
    with (folder / 'quad_trans.pkl').open('wb') as f:
        pickle.dump(quad_trans_fig, f)
    # with (folder / 'enorm.pkl').open('wb') as f:
    #     pickle.dump(intensity_fig, f)
    with (folder / 'final_device_layer.pkl').open('wb') as f:
        pickle.dump(final_device_layer_fig, f)
#     # TODO: rest of the plots

    plotter.close_all()

#* ==============================================================================

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

    from vipdopt.configuration import SonyBayerConfig
    project = Project(config_type=SonyBayerConfig)
    # project.load_project(args.directory, config_name=args.config)
    project.load_project(args.directory)
    # What does the Project class contain?
    # 'dir': directory where it's stored; 'config': SonyBayerConfig object; 'optimization': Optimization object;
    # 'device': Device object; 'base_sim': Simulation object;
    # 'src_to_sim_map': dict with source names as keys, Simulation objects as values
    # 'foms': list of FoM objects, 'weights': array of shape (#FoMs, nλ)
    #! Each Project should correspond only to one Device optimized for a certain functionality and situation.
    #! Optimization parameter sweeps necessitate multiple Projects.

    import numpy.typing as npt
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # FoM_1 = MSEFoM()
    # from PIL import Image
    # im = Image.open('example.jpg').resize(project.device.size[:2], Image.Resampling.LANCZOS)
    # im = np.repeat(np.array(im).transpose()[:, :, np.newaxis], project.device.size[2], axis=2)
    # FoM_1 = MSEFoM(target=im)
    # FoM_2 = MSEFoM(target=im)

    FoM_1 = MSEFoM(target=3*np.ones(project.device.size))
    FoM_1.set_target_2d_gaussian(arr_shape=project.device.size, peak=5.5-2.25, N=9, std=2, center_point=(6,6))
    FoM_1.target += 2.25
    FoM_2 = MSEFoM(target=4*np.ones(project.device.size))
    FoM_2.set_target_2d_gaussian(arr_shape=project.device.size, peak=5.5-2.25, N=9, std=2, center_point=(24,12))
    FoM_2.target += 2.25
    project.fom = FoM(None, None, [(FoM_1,), (FoM_2,)], (1.0,1.0))
    
    # foms, weights = FoM._load_from_config(project.config, project.base_sim)
    # FoM._setup_spectral_weights(foms, project.config)  #! TODO:
    # project.fom = FoM(None, None, [(f,) for f in foms], tuple(weights))
    

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
        optimizer = NLOptOptimizer()
        base_sim = base_sim.set_solver(None)
        # optimizer = GradientAscentOptimizer()
        # base_sim = base_sim.set_solver('LumericalFDTD')
        # optimizer = AdamOptimizer()
        # optimizer = vipdopt.optimization.optimizer._load_optimizer(project.config)

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
        
        # Example of how to pass custom functions into the Optimization instance
        optimization.update_histories = update_histories_em_filter
        optimization.update_histories(optimization)
        optimization.generate_plots = partial(generate_plots_simple_mse, self=optimization)
        
        project.optimizations.append(optimization)
        
        # evaluation = Evaluation(
            
        # )
        # vipdopt.logger.info(f'Evaluation {opt_idx} initialized.')
        # project.evaluations.append(evaluation)
        
        # todo: What does the Evaluation even need to do?
        # conduct a sweep
        # perturb the base sim
        # run sims (it's a method of self.base_sim)
        # spit out plots
        # so it needs to read in the sweep variables and the corresponding ways of perturbing
        # those ways of perturbing need to be written in!
        # perturb base sim and then run sims with create_forward_sims() and run_sims()
        # grab info and feed into the sweep variable lists

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

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # project.device.visualize_layer()
    # plt.show()
    
    
    project.save_as(project.subdirectories['checkpoints'])
    p2 = Project(config_type=SonyBayerConfig)
    p2.load_project(args.directory)
    print('End of code reached.')
# TO SAVE:
# project
# - base sim / partitioned sims
# - device / partitioned devices
# optimization
# - optimizer
# NO NEED TO SAVE:
# FoM (shouldn't change)
# config
# args