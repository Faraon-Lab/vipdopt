import sys
from pathlib import Path
from argparse import SUPPRESS, ArgumentParser
import logging
from functools import partial

import numpy as np

# Import main package and add program folder to PATH.
sys.path.append(Path.cwd()) # 20240219 Ian: Only added this so I could debug some things from my local VSCode
import vipdopt
from vipdopt.configuration.template import MetasurfaceRenderer, reload_template
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
from vipdopt.simulation import ISimulation, Simulation, LumericalFDTD
from vipdopt.utils import setup_logger, read_config_file

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

def default_parser():
    '''Set up argument parser'''
    
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
    
    return parser

#* ==============================================================================

if __name__ == '__main__':

    parser = default_parser()
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

    solver = LumericalFDTD()
    # solver.promise_env_setup(**LumericalFDTD.get_env_vars(cfg,
    #                                         nsims=len(list(self.base_sim.source_names()))
    #                                 ))
    vipdopt.solver = solver


    def reload_config(dim=2):
        config_file = '2d' if dim == 2 else '3d'
        reload_template(*[
                        Path('runs/test_run_neuton_bs'),
                        Path("derived_simulation_properties.j2"),
                        Path(f"runs/test_run_neuton_bs/config_example_{config_file}.yml"),
                        Path("runs/test_run_neuton_bs/processed_config.yml")
                    ])

    def update_simulation():
        reload_template(*[
                        Path('runs/test_run_neuton_bs'),
                        Path("simulation_template.j2"),
                        Path("runs/test_run_neuton_bs/processed_config.yml"),
                        Path("runs/test_run_neuton_bs/sim.json")
                    ])

    def reload_base_sim(dim=3):
        # Regenerate templates
        reload_config(dim=dim)
        update_simulation()
        
        # Use reloaded templates to reload project and update values
        project.load_project(args.directory, config_name=args.config)

        assert project.base_sim is not None
        base_sim = project.base_sim.set_solver('LumericalFDTD')
        solver.connect(hide=False)
        # Sync up new JSON values of Simulation with solver
        solver.save(base_sim.get_path(), base_sim)
        solver.set_view()

    def derive_sim_properties(dim=3):
        config_file=f'{dim}d'
        data = read_config_file(f"runs/test_run/config_example_{config_file}.yml")

        # min_feature_size_voxels = data['min_feature_size_um'] / data['voxel_size_x_um']
        # blur_half_width_voxels: {{ ((min_feature_size_voxels - 1) / 2) | round(method='ceil') | int }}

        print(3)

    reload_base_sim(dim=2)

    print(3)