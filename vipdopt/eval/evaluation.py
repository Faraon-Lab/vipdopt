"""Class for handling evaluation setup, running, and saving."""

from __future__ import annotations

import os
import pickle
import copy
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt import GDS, STL
from vipdopt.configuration import Config
from vipdopt.eval import plotter_v2
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import BayerFilterFoM, FoM, SuperFoM
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import LumericalFDTD, LumericalSimulation
from vipdopt.utils import glob_first, rmtree, real_part_complex_product, replace_border

DEFAULT_EVAL_FOLDERS = {
    'temp': Path('./evaluation/temp'),
    'eval_info': Path('./evaluation'),
    'eval_plots': Path('./evaluation/plots'),
}

TI02_THRESHOLD = 0.5


class LumericalEvaluation:
    """Class for orchestrating all the pieces of an evaluation."""

    def __init__(
        self,
        base_sim: LumericalSimulation,
        # sims: list[LumericalSimulation, ...],       #! NOT USED
        device: Device,
        # optimizer: GradientOptimizer,               #! NOT USED
        fom: SuperFoM,
        # fom_args: tuple[Any, ...] = tuple(),        #! NOT USED
        fom_kwargs: dict = {},
        # grad_args: tuple[Any, ...] = tuple(),       #! NOT USED
        # grad_kwargs: dict = {},                     #! NOT USED
        cfg: Config = Config(),
        dirs: dict[str, Path] = DEFAULT_EVAL_FOLDERS,
        env_vars: dict = {},
        project: Any = None
    ):
        """Initialize Evaluation object."""
        self.base_sim = base_sim
        # self.sims = sims
        self.device = copy.deepcopy(device)         # Whatever changes we make here cannot affect the original.
        # self.optimizer = optimizer
        if isinstance(fom, FoM):
            self.fom = SuperFoM([(fom,)], [1])
        else:
            self.fom = fom
        # self.fom_args = fom_args
        # self.fom_kwargs = fom_kwargs
        # self.grad_args = grad_args
        # self.grad_kwargs = grad_kwargs
        self.dirs = dirs
        # Ensure directories exist
        for directory in dirs.values():
            directory.mkdir(exist_ok=True, mode=0o777, parents=True)

        # self.sim_files = [dirs['temp'] / f'sim_{i}.fsp' for i in range(self.nsims)]
        self.cfg = (cfg)
        self.fom_kwargs = self.cfg if bool(fom_kwargs) else fom_kwargs
        
        # Setup histories - #! SHOULD MATCH THE ONE IN OPTIMIZATION
        self.fom_hist: dict[
            str, list[npt.NDArray]
        ] = {}  # History of FoM-related parameters with each iteration
        self.param_hist: dict[
            str, list[npt.NDArray]
        ] = {}  # History of all other parameters with each iteration
        #! These will be fed directly into plotter.py so this is the place to be changing labels / variable names and somesuch.
        for metric in ['transmission', 'intensity']:
            self.fom_hist.update({f'{metric}_overall': []})
            for i, f in enumerate(self.fom.foms):
                self.fom_hist.update( {f'{metric}_{i}': []} )
        self.fom_hist.update( {'intensity_overall_xyzwl': []} )

        self.loop = True
        self.iteration = 0

        # self.spectral_weights = np.array(1)
        # self.performance_weights = np.array(1)

        # Setup Lumerical Hook
        self.fdtd = vipdopt.fdtd
        try:
            # # TODO: Are we running it locally or on SLURM or on AWS or?
            self.fdtd.promise_env_setup(**env_vars)
        except Exception as ex:
            vipdopt.logging.debug("Environment variables not promised to Evaluation class' FDTD instance.")

        # Set up parent project
        self.project = project

        # Setup callback functions
        self._callbacks: list[Callable[[LumericalEvaluation], None]] = []

        # Turn off structures that should not exist in periodic BCs.
        # right now that is every dict in simulation_template.j2 that relies on data.boundary_conditions
        # and isn't overwritten by an equivalent in eval_objects
        # PEC_screen; source_aperture; substrate(s)
        self.base_sim.disable(['PEC_screen', 'source_aperture', 'substrate'])


    def add_callback(self, func: Callable[[LumericalEvaluation], None]):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)

    def set_env_vars(self, env_vars: dict):
        """Set env vars that are passed to the runner sim of this Optimization."""
        self.env_vars = env_vars

    def save_histories(self, folder=None):
        """Save the fom and parameter histories to file."""
        if folder is None:
            folder = self.dirs['opt_info']
        foms = np.array(self.fom_hist)
        # Todo: need to explore different compression algorithms
        with (folder / 'fom_history.npy').open('wb') as f:
            np.save(f, foms)
        params = np.array(self.param_hist)
        with (folder / 'parameter_history.npy').open('wb') as f:
            np.save(f, params)

        with (self.dirs['summary'] / 'fom_history.npy').open('wb') as f:
            np.save(f, foms)
        with (self.dirs['summary'] / 'parameter_history.npy').open('wb') as f:
            np.save(f, params)


    def load_histories(self, folder=None):
        """Load the fom and parameter histories from file."""
        if folder is None:
            folder = self.dirs['opt_info']

        fom_hist_file = folder / 'fom_history.npy'
        param_hist_file = folder / 'parameter_history.npy'

        if not fom_hist_file.exists():
            # Search the directory for a configuration file
            fom_hist_file = glob_first(self.dirs['root'], '**/*fom_history*.{npy,npz}')
        if not param_hist_file.exists():
            # Search the directory for a configuration file
            param_hist_file = glob_first(self.dirs['root'], '**/*parameter_history*.{npy,npz}')

        self.fom_hist = np.load(folder / 'fom_history.npy', allow_pickle=True).item()
        self.param_hist = np.load(folder / 'parameter_history.npy', allow_pickle=True).item()

        # Remove the latest history values so as to match up to the iteration.
        for _, v in {**self.fom_hist, **self.param_hist}.items():
            for _ in range( len(v) - self.iteration ):
                try:
                    v.pop(-1)
                except Exception as err:
                    pass

    def generate_plots(self):
        """Generate the plots and save to file."""
        folder = self.dirs['opt_plots']
        iteration = self.iteration #  if self.iteration==self.epoch_list[-1] else self.iteration+1
        vipdopt.logger.debug(f'Plotter. Iteration {iteration}: Plot histories length {len(self.fom_hist["intensity_overall"])}')

        # TODO: Copy all to summary folder as well.

        # Placeholder indiv_quad_trans
        import matplotlib.pyplot as plt
        # getattr(self, f'generate_plots_{self.cfg["simulator_dimension"].lower()}_v2')()
        # self.generate_plots_efield_focalplane_1d()

        # ! 20240229 Ian - Best to be specifying functions for 2D and for 3D.

        # TODO: Assert iteration == len(self.fom_hist['intensity_overall']); if unequal, make it equal.
        # Plot key information such as Figure of Merit evolution for easy visualization and checking in the middle of optimizations
        fom_fig = plotter_v2.plot_fom_trace(
            np.array(self.fom_hist['intensity_overall']),
            folder)
        quads_to_plot = [0,1] if self.cfg['simulator_dimension']=='2D' else [0,1,2,3]
        quad_trans_fig = plotter_v2.plot_quadrant_transmission_trace(
            np.array([self.fom_hist[f'transmission_{x}'] for x in quads_to_plot]).swapaxes(0,1),
            folder,
        )
        overall_trans_fig = plotter_v2.plot_quadrant_transmission_trace(
            np.expand_dims(np.array(self.fom_hist['transmission_overall']), axis=1),
            folder,
            filename='overall_trans_trace',
        )

        if self.cfg['simulator_dimension'] == '2D':
            intensity_f = np.squeeze(self.fom_hist.get('intensity_overall_xyzwl')) #[-1]) only if we're recording more than the most recent one
            spatial_x = np.linspace(self.device.coords['x'][0], self.device.coords['x'][-1], intensity_f.shape[0])
            intensity_fig = plotter_v2.plot_Enorm_focal_2d(
                intensity_f,
                spatial_x,
                self.cfg['lambda_values_um'],
                folder,
                iteration,
                wl_idxs=[7, 22]
            )
        # elif self.cfg['simulator_dimension'] == '3D':
            # intensity_fig = plotter.plot_Enorm_focal_3d(
            #     np.sum(np.abs(np.squeeze(e_focal['E'])) ** 2, axis=-1),
            #     e_focal['x'],
            #     e_focal['y'],
            #     e_focal['lambda'],
            #     folder,
            #     self.iteration,
            #     wl_idxs=[9, 29, 49],
            # )

        trans_quadrants = [0,1] if self.cfg['simulator_dimension']=='2D' else [0,1,2,3]
        indiv_trans_fig = plotter_v2.plot_individual_quadrant_transmission(
            np.array([self.fom_hist[f'transmission_{x}'][-1] for x in trans_quadrants]),
            self.cfg['lambda_values_um'],
            folder,
            self.iteration,
        )  # continuously produces only one plot per epoch to save space

        cur_index = self.device.index_from_permittivity(self.device.get_permittivity())
        final_device_layer_fig, _ = plotter_v2.visualize_device(
            self.device.coords['x'], self.device.coords['y'],
            cur_index,
            # self.device.coords['x'], self.device.coords['z'],
            # np.rot90(cur_index),         # 20241003: Want to see the side view for layering.
            folder, iteration=iteration
        )

    #     # # plotter.plot_moments(adam_moments, OPTIMIZATION_PLOTS_FOLDER)
    #     # # plotter.plot_step_size(adam_moments, OPTIMIZATION_PLOTS_FOLDER)

        # Create plot pickle files for GUI visualization
        with (folder / 'fom.pkl').open('wb') as f:
            pickle.dump(fom_fig, f)
        with (folder / 'quad_trans.pkl').open('wb') as f:
            pickle.dump(quad_trans_fig, f)
        with (folder / 'overall_trans.pkl').open('wb') as f:
            pickle.dump(overall_trans_fig, f)
        # with (folder / 'enorm.pkl').open('wb') as f:
        #     pickle.dump(intensity_fig, f)
        with (folder / 'indiv_trans.pkl').open('wb') as f:
            pickle.dump(indiv_trans_fig, f)
        with (folder / 'final_device_layer.pkl').open('wb') as f:
            pickle.dump(final_device_layer_fig, f)
    #     # TODO: rest of the plots

        plotter_v2.close_all()

    def generate_plots_3d_v2(self):
        import matplotlib.pyplot as plt
        plt.close('all')

        quad_trans = []
        quad_colors = ['blue','xkcd:grass green','red','xkcd:olive green','black','green']
        quad_labels = ['B','G1','R','G2','Total','G1+G2']
        # for idx, num in enumerate([5,4,7,6,8]):
        #     quad_trans.append(fwd_sim.monitors()[num].trans_mag)
        for suffix in ['0','1','2','3','overall']:
            quad_trans.append(self.fom_hist[f'transmission_{suffix}'][-1])
        quad_trans.append(quad_trans[1]+quad_trans[3])

        fig, ax = plt.subplots()
        for idx, num in enumerate([0,2,-1]):
            plt.plot(self.cfg['lambda_values_um'], quad_trans[num], color=quad_colors[num], label=quad_labels[num])
        for idx, num in enumerate([1,3]):
            plt.plot(self.cfg['lambda_values_um'], quad_trans[num], linestyle='--', color=quad_colors[num], label=quad_labels[num])

        plt.plot(self.cfg['lambda_values_um'], quad_trans[-2], color='black', label=f'Overall')
        plt.hlines(0.225, self.cfg['lambda_values_um'][0], self.cfg['lambda_values_um'][-1],
                    color='black', linestyle='--',label="22.5%")
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim([0,1])
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Transmission')
        plt.title('PML BCs')
        plt.tight_layout()
        plt.savefig(self.dirs['opt_plots'] / f'quad_trans_i{self.iteration}.png', bbox_inches='tight')

    def generate_plots_2d_v2(self):
        import matplotlib.pyplot as plt
        plt.close('all')

        quad_trans = []
        quad_colors = ['blue','red','black']
        quad_labels = ['B','R','Overall']
        # for idx, num in enumerate([5,4,7,6,8]):
        #     quad_trans.append(fwd_sim.monitors()[num].trans_mag)
        for suffix in ['0','1','overall']:
            quad_trans.append(self.fom_hist[f'transmission_{suffix}'][-1])
        # quad_trans.append(quad_trans[1]+quad_trans[3])

        fig, ax = plt.subplots()
        for idx, num in enumerate([0,1,-1]):
            plt.plot(self.cfg['lambda_values_um'], quad_trans[num], color=quad_colors[num], label=quad_labels[num])

        # plt.hlines(0.225, self.cfg['lambda_values_um'][0], self.cfg['lambda_values_um'][-1],
        #             color='black', linestyle='--',label="22.5%")
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim([0,1])
        plt.xlabel('Wavelength (um)')
        plt.ylabel('Transmission')
        plt.title('PML BCs')
        plt.tight_layout()
        plt.savefig(self.dirs['opt_plots'] / f'quad_trans_i{self.iteration}.png', bbox_inches='tight')

    def generate_plots_efield_focalplane_1d(self):
        import matplotlib.pyplot as plt
        plt.close('all')

        fig, ax = plt.subplots()
        intensity = np.squeeze(self.fom_hist.get('intensity_overall_xyzwl')[-1])
        spatial_x = np.linspace(self.device.coords['x'][0], self.device.coords['x'][-1], intensity.shape[0])
        for peak_ind in self.cfg['desired_peak_location_per_band']:
            wl = self.cfg['lambda_values_um'][peak_ind]
            plt.plot(spatial_x, intensity[:, peak_ind], label='spatial')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.ylim([0,1])
            plt.xlabel('x (um)')
            plt.ylabel('Intensity')
            plt.title(f'E-field at Focal Plane, {wl:.3f}um')
            plt.tight_layout()
            plt.savefig(self.dirs['opt_plots'] / f'efield_focal_wl{wl:.3f}um_i{self.iteration}.png', bbox_inches='tight')
            plt.close()

    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        # Connect to Lumerical. #! Warning - starts a new project if already connected
        self.loop = True
        self.fdtd.connect(hide=True)

    def _post_run(self):
        """Final post-processing after running the optimization."""
        self.loop = False
        self.generate_plots()
        self.fdtd.close()       # Disconnect from Lumerical

    def run(self):
        """Run the optimization"""
        self._pre_run() # Connects to Lumerical if it's not already connected.
        #! Warning - starts a new FDTD instance if already connected!

        # If an error is encountered while running the optimization,
        # still want to clean up afterwards.

        self._inner_evaluation()

        #! 20240721 ian - DEBUG COMMENT THIS BLOCK - UNCOMMENT FOR GIT PUSH
        # try:
        #     self._inner_optimization_loop()
        # except RuntimeError as e:
        #     vipdopt.logger.exception(
        #         f'Encountered exception while running optimization: {e}'
        #         '\nStopping Optimization...'
        #     )
        # finally:
        #     self._post_run()
        self._post_run()

    def obtain_device(self, device_source:Any|Path, lum_obj_name='design_import'):
        '''Pull the device from input device_source.'''

        if isinstance(device_source, os.PathLike):

            #  1) From a .npy file
            if device_source.suffix in ['.npy','.npz']:
                self.device = Device.from_source(device_source)
            #  2) Load another .fsp and import the permittivity.
            elif device_source.suffix in ['.fsp']:
                # TODO: CODE TO IMPORT PERMITTIVITY FROM FSP
                pass

        elif isinstance(device_source, npt.NDArray):
            #  3) Update the device design variable directly
            self.device.set_design_variable(device_source)

        else:
            return None

    def _inner_evaluation(self):
        """The core evaluation function."""

        # # Disable device index monitor(s) to save memory
        self.base_sim.disable(self.base_sim.indexmonitor_names())
        # # Disable cross section monitor(s) to save memory
        self.base_sim.disable(self.base_sim.crosssection_monitor_names())

        #** =====THIS PART OF THE CODE SHOULD MATCH optimization._inner_optimization_loop()=====)
        self.base_sim.upload_device_index_to_sim(self.base_sim,
                                                 self.device,
                                                 self.fdtd,
                                                 self.cfg['simulator_dimension'])

        # Sync up with FDTD to properly import device.
        self.fdtd.save(self.base_sim.get_path(), self.base_sim)

        # Save statistics to do with design variable away before running simulations.
        # # Save device and design variable
        # self.device.save(self.project.current_device_path())
        # # Calculate material % and binarization level, store away
        # # todo: redo this section once you get multilevel sigmoid filters up and can start counting materials
        # cur_index = self.device.index_from_permittivity(self.device.get_permittivity())
        # tio2_pct = 100 * np.count_nonzero(self.device.get_design_variable() < 0.5) / cur_index.size
        # # todo: should wrap these as functions of Device object
        # vipdopt.logger.info(f'TiO2% is {tio2_pct}%.')		# todo: seems to be wrong?
        # self.param_hist.get('tio2_pct').append( tio2_pct )
        # # logging.info(f'Binarization is {100 * np.sum(np.abs(cur_density-0.5))/(cur_density.size*0.5)}%.')
        # binarization_fraction = self.device.compute_binarization(self.device.get_design_variable())
        # vipdopt.logger.info(f'Binarization is {100 * binarization_fraction}%.')
        # self.param_hist.get('binarization').append( binarization_fraction )
        # # todo: re-code binarization for multiple materials.

        vipdopt.logger.info('EVAL: Step 1: All Simulations Setup')

        # TODO: This should be put outside. i.e. several different LumericalEvaluation objects
        # todo: with adjusted params. Use param_config_array to generate the different config ymls,
        # todo: then run separate LumericalEvaluations for each.
		# Import sweep parameters, assemble jobs corresponding to each value of the sweep.
        # Edit the environment of each job accordingly.
		# Queue and run parallel, save out.
		# todo: ====================================================================================

        #
        # NOTE: This next block should be wrapped in a function.
        # Step 1: After importing the current epoch's permittivity value to the device;
        # We create a different optimization job for:
        # - each of the polarizations for the forward source waves
        # - one without the device present for normalization purposes.
        # We then enqueue each job and run them all in parallel.

        # Create jobs
        fwd_sims = self.fom.create_forward_sim(self.base_sim, 
                                               link_sims=self.cfg['pull_sim_files_from_debug_folder'])
        # adj_sims = self.fom.create_adjoint_sim(self.base_sim)
        nodev_sims = [sim.with_monitors(['src_transmission_monitor', 'src_spill_monitor'],
                                        name = sim.info['name'].replace('_fwd_','_nodev_'))
                      for sim in fwd_sims]
        [nds.disable(self.base_sim.import_names()) for nds in nodev_sims]

        self.base_sim.run_sims(self, sim_list=list(chain(fwd_sims, nodev_sims)),
                               file_dir=self.dirs['eval_temp'],
                               add_job_to_fdtd=True)

        vipdopt.logger.info('Completed Step 1: All Simulations Run.')

        # Reformat monitor data for easy use
        self.fdtd.reformat_monitor_data(list(chain(fwd_sims, nodev_sims)))

        #** ====================================================================================

        # Compute intensity FoM and apply spectral and performance weights.
        f = self.fom.compute_fom(**self.fom_kwargs) # self.fom_args,
        self.fom_hist.get('intensity_overall').append(f)
        vipdopt.logger.debug(f'FoM: {f}')

        # Compute transmission FoM and apply spectral and performance weights.
        fom_kwargs_trans = self.fom_kwargs.copy()
        fom_kwargs_trans.update({'type': 'transmission'})
        t = np.array([ fom[0].fom_func(**self.fom_kwargs) # self.fom_args,
            for fom in self.fom.foms
        ])
        [ self.fom_hist.get(f'transmission_{idx}').append(t_i) for idx, t_i in enumerate(t) ]
        self.fom_hist.get('transmission_overall').append( np.squeeze(np.sum(t, 0)) )
        # [plt.plot(np.squeeze(t_i)) for t_i in t]
        # todo: remove hardcode for the monitor.
        intensity = np.sum(np.square(np.abs(fwd_sims[0].monitors()[4].e)), axis=0)
        self.fom_hist['intensity_overall_xyzwl'] = intensity
        # # We need to save space for fom_history. Just save the most recent iteration's data.
        # self.fom_hist.get('intensity_overall_xyzwl').append(intensity)

        # # Here is where we would start plotting the loss landscape. Probably should be accessed by a separate class...
        # # Or we could move it to the device step part
        # loss_landscape_mapper = LossLandscapeMapper.LossLandscapeMapper(simulations, devices)

        print(3)

    def call_callbacks(self):
        """Call all of the callback functions."""
        for fun in self._callbacks:
            fun(self)


if __name__ == '__main__':
    from vipdopt.optimization.filter import Scale, Sigmoid
    from vipdopt.optimization.optimizer import GradientAscentOptimizer
    from vipdopt.simulation.monitor import Power, Profile
    from vipdopt.simulation.source import DipoleSource, GaussianSource
    from vipdopt.utils import import_lumapi, setup_logger

    vipdopt.logger = setup_logger('logger', 0)
    vipdopt.lumapi = import_lumapi(
        'C:\\Program Files\\Lumerical\\v221\\api\\python\\lumapi.py'
    )
    fom = BayerFilterFoM(
        'TE',
        [GaussianSource('forward_src_x')],
        [GaussianSource('forward_src_x'), DipoleSource('adj_src_0x')],
        [
            Power('focal_monitor_0'),
            Power('transmission_monitor_0'),
            Profile('design_efield_monitor'),
        ],
        [Profile('design_efield_monitor')],
        list(range(60)),
        [],
        spectral_weights=np.ones(60),
    )
    sim = LumericalSimulation('runs\\test_run\\sim.json')
    dev = Device(
        (41, 41, 42),
        (0.0, 1.0),
        {
            'x': 1e-6 * np.linspace(0, 1, 41),
            'y': 1e-6 * np.linspace(0, 1, 41),
            'z': 1e-6 * np.linspace(0, 1, 42),
        },
        name='Bayer Filter',
        randomize=True,
        init_seed=0,
        filters=[Sigmoid(0.5, 0.05), Scale((0, 1))],
    )
    opt = LumericalEvaluation(
        sim,
        dev,
        GradientAscentOptimizer(step_size=1e-4),
        fom,
        epoch_list=[1, 2],
    )
    opt.run()
