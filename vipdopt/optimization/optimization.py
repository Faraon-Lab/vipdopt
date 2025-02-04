"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

import os
import pickle
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
import nlopt

import vipdopt
# from vipdopt import GDS, STL
from vipdopt.configuration import Config
# from vipdopt.eval import plotter_v2, plotter_v3
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import FoM #, # BayerFilterFoM,
from vipdopt.optimization.optimizer import GradientOptimizer, NLOptOptimizer
from vipdopt.simulation import ISimulation, LumericalFDTD #, LumericalSimulation
from vipdopt.utils import glob_first, rmtree#, real_part_complex_product, replace_border

DEFAULT_OPT_FOLDERS = {
    'temp': Path('./optimization/temp'),
    'opt_info': Path('./optimization'),
    'opt_plots': Path('./optimization/plots'),
}

class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
        self,
        base_sim: ISimulation,
        # sims: list[ISimulation, ...],
        device: Device,
        optimizer: GradientOptimizer,
        fom: FoM,
        fom_args: tuple[Any, ...] = tuple(),
        fom_kwargs: dict = {},
        grad_args: tuple[Any, ...] = tuple(),
        grad_kwargs: dict = {},
        cfg: Config = Config(),
        epoch_list: list[int] = [100],
        true_iteration: int = 0,
        dirs: dict[str, Path] = DEFAULT_OPT_FOLDERS,
        # env_vars: dict = {},
        project: Any = None
    ):
        """Initialize Optimization object."""

        # Ensure directories exist
        for directory in dirs.values():
            directory.mkdir(exist_ok=True, mode=0o777, parents=True)
        self.dirs = dirs

        # self.sim_files = [dirs['temp'] / f'sim_{i}.fsp' for i in range(self.nsims)]
        self.cfg = cfg

        self.epoch_list = epoch_list
        self.loop = True
        self.iteration = true_iteration

        self.spectral_weights = np.array(1)
        self.performance_weights = np.array(1)

        # Set up simulation connections
        self.base_sim = base_sim
        match self.base_sim.solver:
            case 'LumericalFDTD':
                # Setup Lumerical Hook - the FDTD hook needs to be one level above the simulations unfortunately.
                self.solver = LumericalFDTD()
                self.solver.promise_env_setup(**LumericalFDTD.get_env_vars(cfg,
                                                        nsims=len(list(self.base_sim.source_names()))
                                                ))
            case _:
                pass

        # Setup device
        self.device = device

        # # Setup FoMs
        # if isinstance(fom, FoM):
        #     self.fom = SuperFoM([(fom,)], [1])
        # else:
        #     self.fom = fom
        self.fom = fom
        self.fom_args = fom_args
        self.fom_kwargs = fom_kwargs
        self.grad_args = grad_args
        self.grad_kwargs = grad_kwargs

        # Setup Optimizer
        self.optimizer = optimizer


        # Setup histories
        self.fom_hist: dict[
            str, list[npt.NDArray]
        ] = {}  # History of FoM-related parameters with each iteration
        self.param_hist: dict[
            str, list[npt.NDArray]
        ] = {}  # History of all other parameters with each iteration
        #! These will be fed directly into plotter.py so this is the place to be changing labels / variable names and somesuch.
        for metric in ['transmission', 'intensity']:
            self.fom_hist.update({f'{metric}_overall': []})
            # TODO: 20241227 Uncomment when you start working on FOMs
            # for i, f in enumerate(self.fom.foms):
            #     self.fom_hist.update( {f'{metric}_{i}': []} )
        self.fom_hist.update( {'intensity_overall_xyzwl': []} )
        # self.param_hist.update({'design': []})

        # Set up parent project
        self.project = project

        # Setup callback functions
        self._callbacks: list[Callable[[Optimization], None]] = []

        # Pre-declare inner optimization function
        self._optimize_fom_func = self._inner_optimization_loop



    def __eq__(self, other):
        if self.base_sim == other.base_sim and self.device == other.device:
            return True
        else:   return False

    def add_callback(self, func: Callable[[Optimization], None]):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)

    def call_callbacks(self):
        """Call all of the callback functions."""
        for fun in self._callbacks:
            fun(self)



    @classmethod
    def create_opt_folder_structure(cls, root_dir: Path, pull_files_debug_mode=False):
        """Create the subdirectories of the optimization folder.
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
        optimization_info_folder = data_folder / 'opt_info'
        optimization_plots_folder = optimization_info_folder / 'plots'
        debug_completed_jobs_folder = root_dir / 'ares_test_dev'
        pull_completed_jobs_folder = temp_folder
        if pull_files_debug_mode:
            pull_completed_jobs_folder = debug_completed_jobs_folder

        evaluation_folder = root_dir / 'eval'
        evaluation_config_folder = evaluation_folder / 'configs'
        evaluation_info_folder = evaluation_folder / 'opt_info'
        evaluation_utils_folder = evaluation_folder / 'utils'
        evaluation_temp_folder = evaluation_folder / '.tmp'

        # parameters['MODEL_PATH'] = DATA_FOLDER / 'model.pth'
        # parameters['OPTIMIZER_PATH'] = DATA_FOLDER / 'optimizer.pth'

        # Save out the various files that exist right before the optimization runs for
        # debugging purposes. If these files have changed significantly, the optimization
        # should be re-run to compare to anything new.

        with contextlib.suppress(Exception):
            for file in list(glob.glob('*.sh')):
                shutil.copy2(root_dir/file, saved_scripts_folder/file)

        # shutil.copy2(
        # cfg.python_src_directory + "/SonyBayerFilterOptimization.py",
        # SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterOptimization.py" )
        # # TODO: et cetera... might have to save out various scripts from each folder

        # shutil.copy2(
        #     os.path.abspath(python_src_directory + "/evaluation/plotter.py"),
        #     EVALUATION_UTILS_FOLDER + "/plotter.py" )

        directories = {
            'root': root_dir,
            'data': data_folder,
            'summary': summary_folder,
            'saved_scripts': saved_scripts_folder,
            'opt_info': optimization_info_folder,
            'opt_plots': optimization_plots_folder,
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


    def run(self):
        """Run the optimization"""
        self._pre_run() # Connects to Lumerical if it's not already connected.
        #! Warning - starts a new FDTD instance if already connected!

        self._optimize_fom_func()

        # Planning 20241220.
        # NLOpt - does the loop by itself.
        # Mainly takes arguments algorithm, f, x, grad,
        # Trial Attempt: Choose x = 4-vector (x,y,z,t), and navigate a potential field where
        # the source potentials are arbitrarily located and oscillating
        # optional args lb, ub, maxeval
        # Returns opt_val, result

        # #! 20240721 ian - DEBUG COMMENT THIS BLOCK - UNCOMMENT FOR GIT PUSH
        # try:
        #     self._inner_optimization_loop()
        # except RuntimeError as e:
        #     vipdopt.logger.exception(
        #         f'Encountered exception while running optimization: {e}'
        #         '\nStopping Optimization...'
        #     )
        # finally:
        #     # If an error is encountered while running the optimization still want to
        #     # clean up afterwards
        #     self._post_run()
        # self._post_run()

    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        self.loop = True

        # Connect to simulator
        match self.base_sim.solver:
            case 'LumericalFDTD':
                # Connect to Lumerical. #! Warning - starts a new project if already connected
                self.solver.connect(hide=True)
            case _:
                pass

        if isinstance(self.optimizer, NLOptOptimizer):
            self._optimize_fom_func = partial(self.NLOpt_optimization, func=self.fom.compute_fom, gradient=self.fom.compute_grad,
                                              min=True)
        else:
            self._optimize_fom_func = self._inner_optimization_loop


    def _post_run(self):
        """Final post-processing after running the optimization."""
        self.loop = False
        self.save_histories()
        self.generate_plots()

        match self.base_sim.solver:
            case 'LumericalFDTD':
                self.solver.close()       # Disconnect from Lumerical
            case _:
                pass

    def NLOpt_optimization(self, func, gradient, *args, **kwargs):
        """Wrapper for calling nlopt"""

        # Needs to access self.device and be passed grad
        # self.device is "x" in the example below
        shape = self.device.w[...,0].shape                  # Keep note of the original shape for later reshaping
        x = np.real(np.ravel(self.device.w[...,0]))         # Ravel it into a 1D array
        grad = np.zeros(shape)                              # Create grad

        def f(x:npt.NDArray, grad:npt.NDArray):
            self.iteration += 1
            
            self.device.set_design_variable(x.reshape(shape))
            # Each epoch the device filters are changed (usually getting stronger).
            self.device.update_filters(
                        epoch = np.max( np.where( np.array(self.epoch_list)<=self.iteration ) ), # NOTE: separate from epoch
                        epoch_list = self.epoch_list,
                        num_layers_per_epoch = self.cfg['num_layers_per_epoch']     # Added to test layering changes during optimization
                    )
             # Pass the permittivity through the new filters
            self.device.update_density()
            
            # Remember to set grad in-place, i.e. grad[:] = ...
            grad[:] = np.ravel( self.device.backpropagate(
                                                gradient(x=self.device.get_permittivity())
                                            ) )
            return func(x=self.device.get_permittivity())
            
            x_orig = np.real(self.device.get_permittivity())
            # # Remember to set grad in-place, i.e. grad[:] = ...
            # grad[:] = np.ravel( gradient(x=x_orig) )
            return func(x=x_orig)
            # return func(x=self.device.get_permittivity())

        opt = nlopt.opt(kwargs.get('algorithm', nlopt.LD_MMA),
                        int(np.prod(shape)))
        if kwargs.get('max', True) and not kwargs.get('min', False):
            opt.set_max_objective(f)
        elif kwargs.get('min', True) and not kwargs.get('max', False):
            opt.set_min_objective(f)

        opt.set_xtol_abs(1e-4)
        # opt.set_maxeval(300)

        xopt:npt.NDArray = opt.optimize(x)
        self.device.set_design_variable(xopt.reshape(shape))


    def _inner_optimization_loop(self):
        """The core optimization loop."""

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
                rmtree(self.dirs['temp'], keep_dir=True)

                # Each epoch the device filters are changed (usually getting stronger).
                self.device.update_filters(
                        epoch = np.max( np.where( np.array(self.epoch_list)<=self.iteration ) ), # NOTE: separate from epoch
                        epoch_list = self.epoch_list,
                        num_layers_per_epoch = self.cfg['num_layers_per_epoch']     # Added to test layering changes during optimization
                    )
                # Pass the permittivity through the new filters
                self.device.update_density()

                # Set device field shape - only necessary for EM solvers where the field mesh might not match the index voxels
                self.device.set_field_shape()
                    # #! THE ORDER of the following matters because device.field_shape must be set properly
                    # #! before calling device.import_cur_index()
                    # if i == 0:  # Just do it once per epoch
                    #     # Sync up base sim LumericalSimObject with FDTD in order to get device index monitor shape.
                    #     self.fdtd.save(self.base_sim.get_path(), self.base_sim)
                    #     # Reassign field shape now that the device has been properly imported into Lumerical.
                    #     self.device.field_shape = self.base_sim.import_field_shape()
                    #     # Handle 2D exception
                    #     if self.cfg['simulator_dimension']=='2D' and len(self.device.field_shape) == 2:
                    #         self.device.field_shape += tuple([3])

                # Import device index now into base simulation and reinterpolate if necessary
                import_primitive = self.base_sim.imports()[0]
                    # # Hard-code reinterpolation size as this seems to be what works for accurate Lumerical imports.
                    # reinterpolation_size = (300,306,3) if self.cfg['simulator_dimension']=='2D' else (300,300,306)

                    # cur_density, cur_permittivity = self.device.import_cur_index(
                    #     import_primitive,
                    #     reinterpolation_factors=(1,1,1),    # For 2D the last entry of the tuple must always be 1.
                    #     reinterpolation_size=reinterpolation_size,   # For 2D the last entry of the tuple must be 3.
                    #     binarize=False,
                    # )
                    # # Sync up with FDTD to properly import device.
                    # self.fdtd.save(self.base_sim.get_path(), self.base_sim)

                # Save device and design variable
                self.device.save(self.project.current_device_path())

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

                self.base_sim.misc_processes()       # or it could be an internal function _pre_run() and _post_run() ?
                    # # Disable device index monitor(s) to save memory
                    # self.base_sim.disable(self.base_sim.indexmonitor_names())



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
                adj_sims = self.fom.create_adjoint_sim(self.base_sim)

                self.base_sim.run_sims(self,
                                sim_list=chain(fwd_sims, adj_sims),
                               file_dir=self.dirs['temp'],
                               add_job_to_fdtd=True)

                vipdopt.logger.info('Completed Step 1: All Simulations Run.')

                # Reformat monitor data for easy use
                self.fdtd.reformat_monitor_data(list(chain(fwd_sims, adj_sims)))



            if not self.loop:
                break

    # TODO: ADJUST THIS
    def performance_weighting(self, fom_values: npt.NDArray):
        """Recompute the weights based on the performance of the optimization.

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
