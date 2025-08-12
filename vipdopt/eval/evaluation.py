"""Class for handling evaluation setup, running, and saving."""

from __future__ import annotations

import os
import pickle
import copy
from collections.abc import Callable
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any

import contextlib
import shutil
import glob

import numpy as np
import numpy.typing as npt
#import nlopt

import vipdopt
# from vipdopt import GDS, STL
from vipdopt.configuration import Config
from vipdopt.eval import plotter # plotter_v2, plotter_v3
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import FoM, get_mode_coefficient #, # BayerFilterFoM,
from vipdopt.optimization.optimizer import GradientOptimizer, NLOptOptimizer, _load_optimizer
from vipdopt.simulation import ISimulation, Simulation, LumericalFDTD #, LumericalSimulation
from vipdopt.utils import glob_first, rmtree, real_part_complex_product #, replace_border

DEFAULT_EVAL_FOLDERS = {
    'temp': Path('./evaluation/temp'),
    'eval_info': Path('./evaluation'),
    'eval_plots': Path('./evaluation/plots'),
}

TI02_THRESHOLD = 0.5


class Evaluation:
    """Class for orchestrating all the pieces of an evaluation."""

    def __init__(
        self,
        base_sim: ISimulation,
        # sims: list[ISimulation, ...],
        device: Device,
        # optimizer: GradientOptimizer,
        fom: FoM,
        fom_args: tuple[Any, ...] = tuple(),
        fom_kwargs: dict = {},
        # grad_args: tuple[Any, ...] = tuple(),
        # grad_kwargs: dict = {},
        cfg: Config = Config(),
        # epoch_list: list[int] = [100],
        # true_iteration: int = 0,
        dirs: dict[str, Path] = DEFAULT_EVAL_FOLDERS,
        env_vars: dict = {},
        project: Any = None
    ):
        """Initialize Evaluation object."""

        # Ensure directories exist
        for directory in dirs.values():
            directory.mkdir(exist_ok=True, mode=0o777, parents=True)
        self.dirs = dirs

        # self.sim_files = [dirs['temp'] / f'sim_{i}.fsp' for i in range(self.nsims)]
        self.cfg = cfg

        #self.epoch_list = epoch_list
        self.loop = True
        # self.iteration = true_iteration

        self.spectral_weights = np.array(1)
        self.performance_weights = np.array(1)

        # Set up simulation connections
        self.base_sim = copy.deepcopy(base_sim)
        match self.base_sim.solver:
            case 'LumericalFDTD':
                # Setup Lumerical Hook - the FDTD hook needs to be one level above the simulations unfortunately.
                self.solver = LumericalFDTD()
                self.solver.promise_env_setup(**LumericalFDTD.get_env_vars(cfg,
                                                        nsims=len(list(self.base_sim.source_names()))
                                                ))
                vipdopt.solver = self.solver
            case _:
                pass

        # Setup device
        self.device = copy.deepcopy(device)

        # # Setup FoMs
        # if isinstance(fom, FoM):
        #     self.fom = SuperFoM([(fom,)], [1])
        # else:
        #     self.fom = fom
        self.fom = fom
        self.fom_args = fom_args
        self.fom_kwargs = fom_kwargs
        #self.grad_args = grad_args
        #self.grad_kwargs = grad_kwargs

        # # Setup Optimizer
        #self.optimizer = optimizer


        # Setup histories
        self.fom_hist: dict[
            str, list[npt.NDArray]
        ] = {}  # History of FoM-related parameters with each iteration
        self.param_hist: dict[
            str, list[npt.NDArray]
        ] = {}  # History of all other parameters with each iteration
        #! These will be fed directly into plotter.py so this is the place to be changing labels / variable names and somesuch.
        for metric in ['fom']: # ['transmission', 'intensity']:
            self.fom_hist.update({f'{metric}_overall': []})
            for i, f in enumerate(self.fom.foms):
                self.fom_hist.update( {f'{metric}_{i}': []} )
        # self.param_hist.update({'design': []})

        # Set up parent project
        self.project = project

        # Setup callback functions
        self._callbacks: list[Callable[[Evaluation], None]] = []

        # TODO: 20250721 Maybe don't need this at all. ===========================
        # Pre-declare inner evaluation function
        self.eval_func = self.evaluate()

        # Turn off structures that should not exist in periodic BCs.
        # right now that is every dict in simulation_template.j2 that relies on data.boundary_conditions
        # and isn't overwritten by an equivalent in eval_objects
        # PEC_screen; source_aperture; substrate(s)
        self.base_sim.disable(['PEC_screen', 'source_aperture', 'substrate'])
        # TODO: 20250721 Maybe don't need this at all. ===========================

        #! TODO: OVERWRITE OBJECTS THAT NEED TO APPEAR IN AN EVALUATION.


    def __eq__(self, other):
        if self.base_sim == other.base_sim and self.device == other.device:
            return True
        else:   return False

    def add_callback(self, func: Callable[[Evaluation], None]):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)
    
    def call_callbacks(self):
        """Call all of the callback functions."""
        for fun in self._callbacks:
            fun(self)





    def run(self):
        """Run the evaluation"""
        self._pre_run() # Connects to Lumerical if it's not already connected.
        #! Warning - starts a new FDTD instance if already connected!

        self.eval_func()

        # Planning 20241220.
        # NLOpt - does the loop by itself.
        # Mainly takes arguments algorithm, f, x, grad,
        # Trial Attempt: Choose x = 4-vector (x,y,z,t), and navigate a potential field where
        # the source potentials are arbitrarily located and oscillating
        # optional args lb, ub, maxeval
        # Returns opt_val, result

        # #! 20240721 ian - DEBUG COMMENT THIS BLOCK - UNCOMMENT FOR GIT PUSH
        # try:
        #     self._inner_evaluation_loop()
        # except RuntimeError as e:
        #     vipdopt.logger.exception(
        #         f'Encountered exception while running evaluation: {e}'
        #         '\nStopping Evaluation...'
        #     )
        # finally:
        #     # If an error is encountered while running the evaluation still want to
        #     # clean up afterwards
        #     self._post_run()
        self._post_run()

    def _pre_run(self):
        """Final pre-processing before running the evaluation."""
        self.loop = True

        # Connect to simulator
        match self.base_sim.solver:
            case 'LumericalFDTD':
                # Connect to Lumerical. #! Warning - starts a new project if already connected
                self.solver.connect(hide=False)
            case _:
                pass

        # if isinstance(self.optimizer, NLOptOptimizer):
        #     self.eval_func = partial(self.NLOpt_evaluation, func=self.fom.compute_fom, gradient=self.fom.compute_grad,
        #                                       min=True)
        # else:
        #     self.eval_func = self.evaluate()


    def _post_run(self):
        """Final post-processing after running the evaluation."""
        self.loop = False
        self.save_histories()
        self.generate_plots()

        match self.base_sim.solver:
            case 'LumericalFDTD':
                self.solver.close()       # Disconnect from Lumerical
            case _:
                pass

    #! TODO: Just link this to the function of the same name in optimization.py
    def import_device(self, x, iteration_in_epoch):
        '''Function to pack together the steps needed to update a device and save it to simulation.'''
        self.device.set_design_variable(x.reshape(self.device.size))

        # Each epoch the device filters are changed (usually getting stronger).
        self.device.update_filters(
                epoch = np.max( np.where( np.array(self.epoch_list)<=self.iteration ) ), # NOTE: separate from epoch
                epoch_list = self.epoch_list,
                num_layers_per_epoch = self.cfg['num_layers_per_epoch']     # Added to test layering changes during evaluation
            )
        # Pass the permittivity through the new filters
        self.device.update_density()

        if self.base_sim.solver is not None:
            cur_density, cur_permittivity = self.import_device_to_sim(
                self.device, self.base_sim,
                reinterpolation_factors=(1,1,1),
                reset_field_shape=(iteration_in_epoch==0), # Just grab field shape from solver once per epoch
            )

        # Sync up with solver to properly import device.
        vipdopt.solver.save(self.base_sim.get_path(), self.base_sim)

        # Save device and design variable
        self.device.save(self.current_device_path())

    def get_attribute(self, sim, monitor_name, attribute):
        try:
            monitor = sim.monitors_by_name(monitor_name)[0]
        except Exception as ex:
            vipdopt.logger.warning(f'Monitor {monitor_name} not active for sim {sim.info["name"]}.')
            return None
        try:
            return getattr(monitor, attribute)
        except Exception as ex:
            vipdopt.logger.warning(f'Attribute does not exist for monitor {monitor_name}.')
            return None
    
    def _inner_evaluation_loop(self):
        """The core evaluation loop."""

        self.do_background = True
        for epoch, max_iter in enumerate(self.epoch_list):


            if max_iter < self.iteration:
                vipdopt.logger.debug(f'Skipping Iteration {self.iteration}. Current epoch has max. iteration {max_iter}.')
                continue


            vipdopt.logger.info(
                f'=============== Starting Epoch {epoch} ===============\n'
            )
            iters_in_epoch = max_iter - self.iteration

            for i in range(iters_in_epoch):
                vipdopt.logger.info(
                    f'===== Epoch {epoch}, Iteration {i} / {iters_in_epoch}, Total Iteration {self.iteration} =====\n'
                )
                if not self.loop:
                    break

                # Clean scratch directory to save storage space
                if not self.cfg['pull_sim_files_from_debug_folder']:
                    rmtree(self.dirs['temp'], keep_dir=True)

                def calculate_device_fom(x:npt.NDArray, grad:npt.NDArray):

                    self.import_device(x, i)

                    # Extract statistics about device and store before running simulations.
                        # # Calculate material % and binarization level, store away
                        # # todo: redo this section once you get sigmoid filters up and can start counting materials
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

                    self.base_sim.misc_processes()       # todo: or it could be an internal function _pre_run() and _post_run() ?
                    # Disable device index monitor(s) to save memory
                    self.base_sim.disable(self.base_sim.indexmonitor_names())
                    # Disable cross section monitor(s) to save memory
                    self.base_sim.disable(self.base_sim.crosssection_monitor_names())



                    vipdopt.logger.info('Beginning Step 1: Setup All Evaluations and their Respective Simulations')
                    #
                    # Step 1: After importing the current epoch's permittivity value to the device;
                    # We create a different evaluation job for:
                    # - each of the polarizations for the forward source waves
                    # - each of the polarizations for each of the adjoint sources
                    # Since here, each adjoint source is corresponding to a focal location for a target color band, we have
                    # <num_wavelength_bands> x <num_polarizations> adjoint sources.
                    # We then enqueue each job and run them all in parallel.

                    # Create jobs
                    fwd_sims = self.fom.create_forward_sim(self.base_sim)
                    # adj_sims = self.fom.create_adjoint_sim(self.base_sim)
                    nodev_sims = [sim.with_monitors(['src_transmission_monitor', 'src_spill_monitor'],
                                                    name = sim.info['name'].replace('_fwd_','_nodev_'))
                                for sim in fwd_sims]
                    [nds.disable(self.base_sim.import_names()) for nds in nodev_sims]

                    if self.do_background:
                        # Perform background simulation for source intensity, but only once per running of code
                        vipdopt.logger.info('Adding background simulations to this iteration.')
                        fwd_sims.append(self.base_sim.with_enabled([self.base_sim.objects['bg_fwd_src']], 'bg_fwd'))
                        # adj_sims.append(self.base_sim.with_enabled([self.base_sim.objects['bg_adj_src']], 'bg_adj'))
                        fwd_sims[-1].disable(['design_import'])
                        # adj_sims[-1].disable(['design_import'])

                    self.base_sim.run_sims(self,
                                # sim_list=chain(fwd_sims, adj_sims),
                                sim_list=list(chain(fwd_sims, nodev_sims)),
                                file_dir=self.dirs['eval_temp'],
                                add_job_to_fdtd=True)

                    vipdopt.logger.info('Completed Step 1: All Simulations Run.')

                    # Reformat monitor data for easy use
                    self.solver.reformat_monitor_data(list(chain(fwd_sims, nodev_sims)))

                    self.fom_kwargs.update({'dimension': self.cfg['simulator_dimension'],
                                            'dx': self.cfg['mesh_spacing_um'],
                                            'dy': self.cfg['mesh_spacing_um'],
                                        })
                    if self.do_background:
                        # Perform background simulation for source intensity, but only once per running of code
                        # fwd_prop_fields_bg = {'E': propagation_monitor.e, 'H': propagation_monitor.h}
                        # adj_prop_fields_bg = {'E': propagation_monitor.e, 'H': propagation_monitor.h}

                        fwd_prop_E = fwd_sims[-1].monitors()[0].e       # See config - monitor 0 is propagation monitor
                        fwd_prop_H = fwd_sims[-1].monitors()[0].h
                        adj_prop_E = adj_sims[-1].monitors()[0].e
                        adj_prop_H = adj_sims[-1].monitors()[0].h
                        norm_coeff = get_mode_coefficient(fwd_prop_E, fwd_prop_H, adj_prop_E, adj_prop_H, **self.fom_kwargs)
                        self.fom.norm_intensity = np.real(np.conj(norm_coeff)*norm_coeff)
                        if np.any(self.fom.norm_intensity == 0):
                            vipdopt.logger.info('Background simulations did not converge correctly.')
                            self.fom.norm_intensity = 1e-5*np.ones(self.fom.norm_intensity.shape)
                        self.do_background = False

                    # Compute mode overlap FoM and apply spectral and performance weights.
                    self.fom_kwargs.update({'source_intensity': self.fom.norm_intensity})
                    f = self.fom.compute_fom(*self.fom_args, **self.fom_kwargs)
                    self.fom_hist.get('fom_overall').append(f)
                    self.fom_hist.get('intensity_overall').append(f)
                    for fom_cnt in range(len(self.fom.foms)):
                        self.fom_hist.get(f'fom_{fom_cnt}').append(self.fom.foms[fom_cnt][0].compute_fom(*self.fom_args, **self.fom_kwargs))
                    vipdopt.logger.debug(f'FoM: {f}')

                    self.fom_hist.get('intensity_overall').append(f)

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
                    intensity = fwd_sims[0].monitors()[4].intensity
                    self.fom_hist['intensity_overall_xyzwl'] = intensity
                    # # We need to save space for fom_history. Just save the most recent iteration's data.
                    # self.fom_hist.get('intensity_overall_xyzwl').append(intensity)

                    # # Here is where we would start plotting the loss landscape. Probably should be accessed by a separate class...
                    # # Or we could move it to the device step part
                    # loss_landscape_mapper = LossLandscapeMapper.LossLandscapeMapper(simulations, devices)

                    # TODO: Wrap the below in an evaluation function ===============

                    # Power Quantities
                    attribute = 'power'
                    sourcepower = fwd_sims[0].monitors()[0].sp
                    wavelength = np.array(self.cfg['lambda_values_um'])

                    P_incident = self.get_attribute(fwd_sims[0], 'incident_aperture_monitor', attribute)

                    P_reflected = self.get_attribute(nodev_sims[0], 'src_spill_monitor', attribute) -\
                                    self.get_attribute(fwd_sims[0], 'src_spill_monitor', attribute)
                                # getattr(fwd_sims[0].monitors_by_name('src_spill_monitor')[0], attribute) -\
                                    
                    P_sides = {}
                    for mntr in fwd_sims[0].monitors_by_name('side'):
                        P_sides[mntr.name] = getattr(mntr, attribute, None)
                    P_exit = self.get_attribute(fwd_sims[0], 'exit_aperture_monitor', attribute)

                    P_focal = self.get_attribute(fwd_sims[0], 'transmission_focal_monitor_', attribute)
                    
                    #!! TODO: 20241218: WOW EVERYTHING IS WRONG.
                    
                    # TODO: Wrap the above in an evaluation function ===============




                    print(3)

                    # Generate Plots and call callback functions
                    self.save_histories()
                    self.generate_plots()   #! TODO:
                    self.call_callbacks()

                    self.iteration += 1
                    # Save Project
                    #! TODO:
                    self.project.save_as(self.project.subdirectories['checkpoints'])

    # TODO: ADJUST THIS
    def performance_weighting(self, fom_values: npt.NDArray):
        """Recompute the weights based on the performance of the evaluation.

        All gradients are combined with a weighted average in Eq.(3), with weights
        chosen according to Eq.(2) such that all figures of merit seek the same
        efficiency. In these equations, FoM represents the current value of a figure of
        merit, N is the total number of figures of merit, and wi represents the weight
        applied to its respective merit function's gradient. The maximum operator is
        used to ensure the weights are never negative, thus ignoring the gradient of
        high-performing figures of merit rather than forcing the figure of merit to
        decrease. The 2/N factor is used to ensure all weights conveniently sum to 1
        unless some weights were negative before the maximum operation. Although the
        maximum operator is non-differentiable, this function is used only to apply the
        gradient rather than to compute it. Therefore, it does not affect the
        applicability of the adjoint method. Taken from: https://doi.org/10.1038/s41598-021-88785-5

        Arguments:
            fom_values (npt.NDArray): The values of the computed FoMs to determine the
                new weights from. Should have shape 1 x N where N is the number of
                FoMs.
        """
        weights = (2.0 / len(fom_values)) - fom_values**2 / np.sum(fom_values**2)

        # # Zero-shift and renormalize
        # if np.min(weights) < 0:
        #     weights -= np.min(weights)
        #     weights /= np.sum(weights)

        # Max(x,0) according to Eq. (2), https://www.nature.com/articles/s41598-021-88785-5
        weights = np.fmax(weights, 0)

        self.performance_weights = weights

    def calc_normalize_power(monitors_data, normalize_option):
            baseline_power = 0

            if normalize_option == 'input_power':
                baseline_power = monitors_data['incident_aperture_monitor']['P']

            elif normalize_option == 'sourcepower':
                baseline_power = monitors_data['sourcepower']

            elif normalize_option in ['power_sum', 'uniform_cube']:
                sourcepower = monitors_data['sourcepower']
                input_power = monitors_data['incident_aperture_monitor']['P']

                R_coeff_4 = monitors_data['incident_aperture_monitor']['R_power']		# this is actually power not R-coefficient
                R_coeff_5 = monitors_data['src_spill_monitor']['P'] - input_power		# this is actually power not R-coefficient
                R_power = (np.abs(R_coeff_4)) # + R_coeff_5) * 0.5
                side_powers = []
                side_directions = ['E','N','W','S']
                sides_power = 0
                for idx in range(0, 4):
                    side_powers.append(monitors_data['side_monitor_'+str(idx)]['P'])
                    sides_power = sides_power + monitors_data['side_monitor_'+str(idx)]['P']
                focal_power = monitors_data['transmission_focal_monitor_']['P']
                scatter_power = 0
                for idx in range(0,4):
                    scatter_power = scatter_power + monitors_data['vertical_scatter_monitor_'+str(idx)]['P']

                power_sum = R_power + focal_power + scatter_power
                sides_power_2 = sides_power#  - (sourcepower - input_power)
                power_sum += sides_power_2

                baseline_power = power_sum

            elif normalize_option == 'unity':
                baseline_power = 1


            return baseline_power

    def save_histories(self, folder=None):
        """Save the fom and parameter histories to file."""
        if folder is None:
            folder = self.dirs['eval_info']
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
            folder = self.dirs['eval_info']

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

    def current_device_path(self) -> Path:
        """Get the current device path for saving."""
        return self.dirs['device'] / f'i_{self.iteration}.npy'

    def import_device_to_sim(self,
                             device, base_sim,
                             import_idx=0,
                             reinterpolation_factors=(1,1,1), # For 2D the last entry of the tuple must always be 1.
                             reset_field_shape=True):

        if base_sim.solver in ['LumericalFDTD',]: # todo: all other supported EM solvers
            # Set device field shape - only necessary for EM solvers, where the field mesh might not match the index voxels
                # #! THE ORDER of the following matters because device.field_shape must be set properly
                # #! before calling device.import_cur_index()
            if reset_field_shape:
                # Sync up base sim LumericalSimObject with FDTD in order to get device index monitor shape.
                vipdopt.solver.save(base_sim.get_path(), base_sim)
                # Reassign field shape now that the device has been properly imported into Lumerical.
                device.set_field_shape(base_sim.import_field_shape())
                # Handle 2D exception
                if self.cfg['simulator_dimension']=='2D' and len(device.field_shape) == 2:
                    device.field_shape += tuple([3])

        # Import device index now into base simulation and reinterpolate if necessary
        # Hard-code reinterpolation size as this seems to be what works for accurate Lumerical imports.
        reinterpolation_size = (300,306,3) if self.cfg['simulator_dimension']=='2D' else (300,300,306)
        cur_density, cur_permittivity = device.import_cur_index(
            base_sim.imports()[import_idx],
            reinterpolation_factors=reinterpolation_factors,    # For 2D the last entry of the tuple must always be 1.
            reinterpolation_size=reinterpolation_size,   # For 2D the last entry of the tuple must be 3.
            binarize=False,
            )
        #! cur_density and cur_permittivity are not the values in device.w but rather the
        #! EM solver values after reinterpolation.

        return cur_density, cur_permittivity

    def generate_plots(self):
        """Generate the plots and save to file."""
        folder = self.dirs['eval_info']
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

        #!! TODO:  generate_plots() should also be a function that is passed in, btw

        fom_fig = plotter.plot_fom_trace(
            np.array(self.fom_hist['fom_overall']),
            folder)

        quads_to_plot = [0,1] if self.cfg['simulator_dimension']=='2D' else [0,1,2,3]
        quad_trans_fig = plotter.plot_bayer_quadrant_transmission_trace(
            np.array([self.fom_hist[f'transmission_{x}'] for x in quads_to_plot]).swapaxes(0,1),
            folder,
        )

        overall_trans_fig = plotter.plot_bayer_quadrant_transmission_trace(
            np.expand_dims(np.array(self.fom_hist['transmission_overall']), axis=1),
            folder,
            filename='overall_trans_trace',
        )

        if self.cfg['simulator_dimension'] == '2D':
            intensity_f = np.squeeze(self.fom_hist.get('intensity_overall_xyzwl')) #[-1]) only if we're recording more than the most recent one
            spatial_x = np.linspace(self.device.coords['x'][0], self.device.coords['x'][-1], intensity_f.shape[0])
            intensity_figs = plotter.plot_Enorm_2d(
                spatial_x,
                intensity_f,
                self.cfg['lambda_values_um'],
                folder,
                filename = 'Enorm', #f'Enorm_wl{wl_str}_i{iteration}'
                wl_idxs=[7, 22],
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
        indiv_trans_fig = plotter.plot_bayer_quadrant_transmission_spectra(
                                self.cfg['lambda_values_um'],
                                np.array([self.fom_hist[f'transmission_{x}'][-1] for x in trans_quadrants]),
                                folder,
                                # filename='trans_spec',
                                f'trans_i{iteration}',
                                line_labels=['Q0', 'Q1', 'Q2', 'Q3'],
                                plot_colors=['blue', 'green', 'red', 'xkcd:fuchsia'],
                            ) # continuously produces only one plot per epoch to save space


        cur_index = self.device.index_from_permittivity(self.device.get_permittivity())
        final_device_layer_fig, _ = plotter.visualize_device(
                                            self.device.coords['x'], self.device.coords['y'], cur_index,
                                            # self.device.coords['x'], self.device.coords['z'],
                                            # np.rot90(cur_index),         # 20241003: Want to see the side view for layering.
                                            folder,
                                            filename=f'_{iteration}'
                                        )

    #     # # plotter.plot_moments(adam_moments, OPTIMIZATION_PLOTS_FOLDER)
    #     # # plotter.plot_step_size(adam_moments, OPTIMIZATION_PLOTS_FOLDER)

        # Evaluation Plots


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

        plotter.close_all()

    def update_histories():
        pass

    @classmethod
    def create_opt_folder_structure(cls, root_dir: Path, pull_files_debug_mode=False):
        """Create the subdirectories of the evaluation folder.
        pull_files_debug_mode: if True, sets the folder for pulling completed jobs to the debug folder which should contain already-run sims.
        This is for avoiding simulation runtime during debug testing.
        """
        directories: dict[str, Path] = {'main': root_dir}

        # * Output / Save Paths
        data_folder = root_dir / 'data'
        summary_folder = root_dir / 'summary'
        temp_folder = root_dir / '.tmp'
        device_folder = root_dir / 'device'
        checkpoint_folder = data_folder / 'checkpoints'
        saved_scripts_folder = data_folder / 'saved_scripts'
        evaluation_info_folder = data_folder / 'eval_info'
        evaluation_plots_folder = evaluation_info_folder / 'plots'
        debug_completed_jobs_folder = root_dir / 'ares_test_dev'
        pull_completed_jobs_folder = temp_folder
        if pull_files_debug_mode:
            pull_completed_jobs_folder = debug_completed_jobs_folder

        evaluation_folder = root_dir / 'eval'
        evaluation_config_folder = evaluation_folder / 'configs'
        evaluation_info_folder = evaluation_folder / 'eval_info'
        evaluation_utils_folder = evaluation_folder / 'utils'
        evaluation_temp_folder = evaluation_folder / '.tmp'

        # parameters['MODEL_PATH'] = DATA_FOLDER / 'model.pth'
        # parameters['OPTIMIZER_PATH'] = DATA_FOLDER / 'optimizer.pth'

        # Save out the various files that exist right before the evaluation runs for
        # debugging purposes. If these files have changed significantly, the evaluation
        # should be re-run to compare to anything new.

        with contextlib.suppress(Exception):
            for file in list(glob.glob('*.sh')):
                shutil.copy2(root_dir/file, saved_scripts_folder/file)

        # shutil.copy2(
        # cfg.python_src_directory + "/SonyBayerFilterEvaluation.py",
        # SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterEvaluation.py" )
        # # TODO: et cetera... might have to save out various scripts from each folder

        # shutil.copy2(
        #     os.path.abspath(python_src_directory + "/evaluation/plotter.py"),
        #     EVALUATION_UTILS_FOLDER + "/plotter.py" )

        directories = {
            'root': root_dir,
            'data': data_folder,
            'summary': summary_folder,
            'saved_scripts': saved_scripts_folder,
            'eval_info': evaluation_info_folder,
            'eval_plots': evaluation_plots_folder,
            'pull_completed_jobs': pull_completed_jobs_folder,
            'debug_completed_jobs': debug_completed_jobs_folder,
            'device': device_folder,
            'evaluation': evaluation_folder,
            'eval_config': evaluation_config_folder,
            'eval_info': evaluation_info_folder,
            'eval_utils': evaluation_utils_folder,
            'eval_temp': evaluation_temp_folder,
            'checkpoints': checkpoint_folder,
            'temp': temp_folder,
        }
        # Create missing directories with full permissions
        for d in directories.values():
            d.mkdir(exist_ok=True, mode=0o777, parents=True)
        return directories

    def as_dict(self):
        assert self.optimizer is not None
        assert self.base_sim is not None

        opt_dict = {}

        # Miscellaneous Settings
        opt_dict['current_iteration'] = self.iteration

        # Optimizer
        opt_dict.update(self.optimizer.as_dict())

        # FoMs
        opt_dict['figures_of_merit'] = self.fom.as_dict()

        # Device
        self.device.save(self.current_device_path())
        opt_dict['device'] = self.current_device_path()

        # Base Simulation
        opt_dict['base_simulation'] = self.base_sim.as_dict()

        return opt_dict

    @classmethod
    def from_dict(cls, opt_dict, *args, **kwargs):
        device = Device.from_source(opt_dict['device'])
        base_sim = Simulation.load(opt_dict['base_simulation'])
        optimizer = _load_optimizer(opt_dict)
        fom = FoM.from_dict(opt_dict['figures_of_merit'])

        return Evaluation(
                    base_sim, device, optimizer, fom, true_iteration=opt_dict['current_iteration'],
                    # fom_args, fom_kwargs, grad_args, grad_kwargs,
                    # cfg, epoch_list, dirs, project will be assigned in Project.load_project()
                    *args, **kwargs
        )

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

# TODO: If ever this is needed, have a multi-objective Evaluation class.
# class MultiEvaluation(Evaluation):
#     def __init__(self, *args, **kwargs):
#         # instead of self.fom we have self.foms