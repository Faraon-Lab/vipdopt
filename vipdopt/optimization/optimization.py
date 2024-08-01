"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt import GDS, STL
from vipdopt.configuration import Config
from vipdopt.eval import plotter
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import BayerFilterFoM, FoM, SuperFoM
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import LumericalFDTD, LumericalSimulation
from vipdopt.utils import real_part_complex_product, rmtree

DEFAULT_OPT_FOLDERS = {
    'temp': Path('./optimization/temp'),
    'opt_info': Path('./optimization'),
    'opt_plots': Path('./optimization/plots'),
}

TI02_THRESHOLD = 0.5


class LumericalOptimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
        self,
        base_sim: LumericalSimulation,
        device: Device,
        optimizer: GradientOptimizer,
        fom: SuperFoM,
        fom_args: tuple[Any, ...] = tuple(),
        fom_kwargs: dict = {},
        grad_args: tuple[Any, ...] = tuple(),
        grad_kwargs: dict = {},
        cfg: Config = Config(),
        epoch_list: list[int] = [100],
        true_iteration: int = 0,
        dirs: dict[str, Path] = DEFAULT_OPT_FOLDERS,
        env_vars: dict = {},
    ):
        """Initialize Optimization object."""
        self.base_sim = base_sim
        self.device = device
        self.optimizer = optimizer
        if isinstance(fom, FoM):
            self.fom = SuperFoM([(fom,)], [1])
        else:
            self.fom = fom
        self.fom_args = fom_args
        self.fom_kwargs = fom_kwargs
        self.grad_args = grad_args
        self.grad_kwargs = grad_kwargs
        self.dirs = dirs
        # Ensure directories exist
        for directory in dirs.values():
            directory.mkdir(exist_ok=True, mode=0o777, parents=True)

        # self.sim_files = [dirs['temp'] / f'sim_{i}.fsp' for i in range(self.nsims)]
        self.cfg = (
            cfg  # ! 20240228 Ian - Optimization needs to call a few config variables
        )

        self.epoch_list = epoch_list
        self.loop = True
        self.iteration = true_iteration

        self.spectral_weights = np.array(1)
        self.performance_weights = np.array(1)

        # Setup Lumerical Hook
        self.fdtd = LumericalFDTD()
        # # TODO: Are we running it locally or on SLURM or on AWS or?
        self.fdtd.promise_env_setup(**env_vars)

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
            for i, f in enumerate(self.fom.foms):
                self.fom_hist.update({f'{metric}_{i}': []})
        self.param_hist.update({'design': []})

        # Setup callback functions
        self._callbacks: list[Callable[[LumericalOptimization], None]] = []

    def add_callback(self, func: Callable[[LumericalOptimization], None]):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)

    def set_env_vars(self, env_vars: dict):
        """Set env vars that are passed to the runner sim of this Optimization."""
        self.env_vars = env_vars

    def create_history(self, fom_types, max_iter, num_design_frequency_points):
        """Set up numpy arrays to track the FoMs and progress of the simulation."""
        # # By iteration	# TODO: store in a block of lists?
        self.figure_of_merit_evolution = np.zeros(max_iter)
        # pdaf_transmission_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
        # step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
        self.binarization_evolution = np.zeros(max_iter)
        # average_design_variable_change_evolution = np.zeros((
        #     num_epochs,
        #     num_iterations_per_epoch,
        # ))
        # max_design_variable_change_evolution = np.zeros((
        #     num_epochs,
        #     num_iterations_per_epoch,
        # ))

        # By iteration, focal area, polarization, wavelength:
        # Create a dictionary of FoMs. Each key corresponds to a an array storing data
        # of a specific FoM; for each epoch, iteration, quadrant, polarization,
        # and wavelength.
        # TODO: Consider using xarray instead of numpy arrays in order to have dimension labels for each axis.

        self.fom_evolution = {}
        for fom_type in fom_types:
            self.fom_evolution[fom_type] = np.zeros((
                max_iter,
                len(self.foms),
                num_design_frequency_points,
            ))
        # for fom_type in fom_types:	# TODO: delete this part when all is finished
        # 	fom_evolution[fom_type] = np.zeros(
        #         (
        #             num_epochs,
        #             num_iterations_per_epoch,
        # 			num_focal_spots,
        #             len(xy_names),
        #             num_design_frequency_points,
        #         )
        #     )
        # The main FoM being used for optimization is indicated in the config.

        # NOTE: Turned this off because the memory and savefile space of gradient_evolution are too large.
        # Repeat the definition of field_shape here because it's not loaded in if we
        # start from a debug point
        # if start_from_step != 0:
        # 	field_shape = [
        #         device_voxels_simulation_mesh_lateral,
        #         device_voxels_simulation_mesh_lateral,
        #         device_voxels_simulation_mesh_vertical,
        #     ]
        # gradient_evolution = np.zeros(
        #     (
        #         num_epochs,
        #         num_iterations_per_epoch,
        #         *field_shape,
        #         num_focal_spots,
        #         len(xy_names),
        #         num_design_frequency_points,
        #     )
        # )

    def save_histories(self):
        """Save the fom and parameter histories to file."""
        folder = self.dirs['opt_info']
        foms = np.array(self.fom_hist)
        with (folder / 'fom_history.npy').open('wb') as f:
            np.save(f, foms)
        params = np.array(self.param_hist)
        with (folder / 'parameter_history.npy').open('wb') as f:
            np.save(f, params)

    def generate_plots(self):
        """Generate the plots and save to file."""
        folder = self.dirs['opt_plots']

        # ! 20240229 Ian - Best to be specifying functions for 2D and for 3D.

        # TODO: Plot key information such as Figure of Merit evolution for easy visualization and checking in the middle of optimizations
        fom_fig = plotter.plot_fom_trace(
            self.figure_of_merit_evolution, folder, self.epoch_list
        )
        quad_trans_fig = plotter.plot_quadrant_transmission_trace(
            self.fom_evolution['transmission'], folder, self.epoch_list
        )
        overall_trans_fig = plotter.plot_quadrant_transmission_trace(
            self.fom_evolution['overall_transmission'],
            folder,
            self.epoch_list,
            filename='overall_trans_trace',
        )

        # TODO: code this to pull the forward sim for FoM_0 i.e. forward_src_x in this case but not EVERY case.
        # self.runner_sim.fdtd.load(
        #     self.foms[0].fom_monitors[0].sim.info['path']
        # )  # untested
        self.runner_sim.fdtd.load(self.sim_files[0].name)
        e_focal = self.runner_sim.getresult('transmission_focal_monitor_', 'E')
        if self.cfg['simulator_dimension'] == '2D':
            intensity_fig = plotter.plot_Enorm_focal_2d(
                np.sum(np.abs(np.squeeze(e_focal['E'])) ** 2, axis=-1),
                e_focal['x'],
                e_focal['lambda'],
                folder,
                self.iteration,
                wl_idxs=[7, 22],
            )
        elif self.cfg['simulator_dimension'] == '3D':
            intensity_fig = plotter.plot_Enorm_focal_3d(
                np.sum(np.abs(np.squeeze(e_focal['E'])) ** 2, axis=-1),
                e_focal['x'],
                e_focal['y'],
                e_focal['lambda'],
                folder,
                self.iteration,
                wl_idxs=[9, 29, 49],
            )

            indiv_trans_fig = plotter.plot_individual_quadrant_transmission(
                self.fom_evolution['transmission'],
                self.cfg['lambda_values_um'],
                folder,
                self.iteration,
            )  # continuously produces only one plot per epoch to save space

        cur_index = self.device.index_from_permittivity(self.device.get_permittivity())
        final_device_layer_fig, _ = plotter.visualize_device(
            cur_index, folder, iteration=self.iteration
        )

        # # plotter.plot_moments(adam_moments, OPTIMIZATION_PLOTS_FOLDER)
        # # plotter.plot_step_size(adam_moments, OPTIMIZATION_PLOTS_FOLDER)

        # Create plot pickle files for GUI visualization
        with (folder / 'fom.pkl').open('wb') as f:
            pickle.dump(fom_fig, f)
        with (folder / 'quad_trans.pkl').open('wb') as f:
            pickle.dump(quad_trans_fig, f)
        with (folder / 'overall_trans.pkl').open('wb') as f:
            pickle.dump(overall_trans_fig, f)
        with (folder / 'enorm.pkl').open('wb') as f:
            pickle.dump(intensity_fig, f)
        with (folder / 'indiv_trans.pkl').open('wb') as f:
            pickle.dump(indiv_trans_fig, f)
        with (folder / 'final_device_layer.pkl').open('wb') as f:
            pickle.dump(final_device_layer_fig, f)
        # TODO: rest of the plots

    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        # Connect to Lumerical. #! Warning - starts a new project if already connected
        self.loop = True
        self.fdtd.connect(hide=True)

    def _post_run(self):
        """Final post-processing after running the optimization."""
        # Disconnect from Lumerical
        self.loop = False
        self.save_histories()
        self.generate_plots()
        self.fdtd.close()

        # STL Export requires density to be fully binarized
        full_density = self.device.binarize(self.device.get_density())
        stl_generator = STL.STL(full_density)
        stl_generator.generate_stl()
        stl_generator.save_stl(self.dirs['data'] / 'final_device.stl')

        # GDS Export must be done in layers. We split the full design into individual
        # layers, (not related to design layers). Express each layer as polygons in an
        # STL Mesh object and write that layer into its own cell of the GDS file
        layer_mesh_array = []
        for z_layer in range(full_density.shape[2]):
            stl_generator = STL.STL(full_density[..., z_layer][..., np.newaxis])
            stl_generator.generate_stl()
            layer_mesh_array.append(stl_generator.stl_mesh)

        # Create a GDS object containing a Library with Cells corresponding to each 2D
        # layer in the 3D device.
        gds_generator = GDS.GDS().set_layers(
            full_density.shape[2],
            unit=1e-6
            * np.abs(self.device.coords['x'][-1] - self.device.coords['x'][0])
            / self.device.size[0],
        )
        gds_generator.assemble_device(layer_mesh_array, listed=False)

        # Directory for export
        gds_layer_dir = os.path.join(self.dirs['data'] / 'gds')
        if not os.path.exists(gds_layer_dir):
            os.makedirs(gds_layer_dir)
        # Export both GDS and SVG for easy visualization.
        gds_generator.export_device(gds_layer_dir, filetype='gds')
        gds_generator.export_device(gds_layer_dir, filetype='svg')
        # for layer_idx in range(
        #     full_density.shape[2]
        # ):  # Individual layer as GDS file export
        #     gds_generator.export_device(
        #         gds_layer_dir, filetype='gds', layer_idx=layer_idx
        #     )
        #     gds_generator.export_device(
        #         gds_layer_dir, filetype='svg', layer_idx=layer_idx
        #     )

        # # Here's a function for GDS device import - takes maybe 3-5 minutes per layer.
        # def import_gds(sim, device, gds_layer_dir):
        #     import time

        #     sim.fdtd.load(sim.info['name'])
        #     sim.fdtd.switchtolayout()
        #     device_name = device.import_names()[0]
        #     sim.disable([device_name])

        #     z_vals = device.coords['z']
        #     for z_idx in range(len(z_vals)):
        #         t = time.time()
        #         fdtd.gdsimport(
        #             os.path.join(gds_layer_dir, 'device.gds'), f'L{layer_idx}', 0
        #         )
        #         sim.fdtd.gdsimport(
        #             os.path.join(gds_layer_dir, 'device.gds'),
        #             f'L{z_idx}',
        #             0,
        #             'Si (Silicon) - Palik',
        #             z_vals[z_idx],
        #             z_vals[z_idx + 1],
        #         )
        #         sim.fdtd.set({
        #             'x': device.coords['x'][0],
        #             'y': device.coords['y'][0],
        #         })  # Be careful of units - 1e-6
        #         vipdopt.logger.info(
        #             f'Layer {z_idx} imported in {time.time() - t} seconds.'
        #         )

    def run(self):
        """Run the optimization"""
        self._pre_run()  # Connects to Lumerical if it's not already connected. #! Warning - starts a new project if already connected!

        # If an error is encountered while running the optimization still want to
        # clean up afterwards

        self._inner_optimization_loop()

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

    def _inner_optimization_loop(self):
        """The core optimization loop."""
        # self.iteration = 0
        for epoch, max_iter in enumerate(self.epoch_list):
            if max_iter < self.iteration:
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

                # # Disable device index monitor(s) to save memory
                self.base_sim.disable(self.base_sim.indexmonitor_names())

                #! THE ORDER of the following matters because device.field_shape must be set properly
                #! before calling device.import_cur_index()
                if i == 0:  # Just do it once per epoch
                    # Sync up base sim LumericalSimObject with FDTD in order to get device index monitor shape.
                    self.fdtd.save(self.base_sim.get_path(), self.base_sim)
                    # Reassign field shape now that the device has been properly imported into Lumerical.
                    self.device.field_shape = self.base_sim.import_field_shape()

                # Each epoch the filters get stronger and so the permittivity must be passed through the new filters
                self.device.update_filters(
                    epoch=np.max(np.where(np.array(self.epoch_list) <= self.iteration))
                )
                self.device.update_density()
                # Import device index now into base simulation
                import_primitive = self.base_sim.imports()[0]
                cur_density, cur_permittivity = self.device.import_cur_index(
                    import_primitive,
                    reinterpolation_factor=1,
                    binarize=False,
                )
                # Sync up with FDTD to properly import device.
                self.fdtd.save(self.base_sim.get_path(), self.base_sim)

                # # Save statistics to do with design variable away before running simulations.
                # Save current design before iteration
                self.param_hist.get('design').append(self.device.get_design_variable())
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

                vipdopt.logger.info('Beginning Step 1: All Simulations Setup')
                #
                # Step 1: After importing the current epoch's permittivity value to the device;
                # We create a different optimization job for:
                # - each of the polarizations for the forward source waves
                # - each of the polarizations for each of the adjoint sources
                # Since here, each adjoint source is corresponding to a focal location for a target color band, we have
                # <num_wavelength_bands> x <num_polarizations> adjoint sources.
                # We then enqueue each job and run them all in parallel.

                # Create jobs
                fwd_sims = self.fom.create_forward_sim(self.base_sim)
                adj_sims = self.fom.create_adjoint_sim(self.base_sim)
                for sim in chain(fwd_sims, adj_sims):
                    sim_file = self.dirs['temp'] / f'{sim.info["name"]}.fsp'
                    # sim.link_monitors()

                    self.fdtd.save(sim_file, sim)  # Saving also sets the path
                    self.fdtd.addjob(sim_file)
                vipdopt.logger.info(
                    'In-Progress Step 1: All Simulations Setup and Jobs Added'
                )

                # If true, we're in debugging mode and it means no simulations are run.
                # Data is instead pulled from finished simulation files in the debug folder.
                # If false, run jobs and check that they all ran to completion.
                if True:  # self.cfg['pull_sim_files_from_debug_folder']:
                    for sim in chain(fwd_sims, adj_sims):
                        sim_file = (
                            self.dirs['debug_completed_jobs']
                            / f'{sim.info["name"]}.fsp'
                        )
                        sim.set_path(sim_file)
                else:
                    while self.fdtd.fdtd.listjobs(
                        'FDTD'
                    ):  # Existing job list still occupied
                        # Run simulations from existing job list
                        self.fdtd.runjobs()

                        # Check if there are any jobs that didn't run
                        for sim in chain(fwd_sims, adj_sims):
                            sim_file = sim.get_path()
                            # self.fdtd.load(sim_file)
                            # if self.fdtd.layoutmode():
                            if (
                                sim_file.stat().st_size <= 2e7
                            ):  # Arbitrary 20MB filesize for simulations that didn't run completely
                                self.fdtd.addjob(sim_file)
                                vipdopt.logger.info(
                                    f'Failed to run: {sim_file.name}. Re-adding ...'
                                )
                vipdopt.logger.info('Completed Step 1: All Simulations Run.')

                # Reformat monitor data for easy use
                self.fdtd.reformat_monitor_data(list(chain(fwd_sims, adj_sims)))

                # Compute intensity FoM and apply spectral and performance weights.
                f = self.fom.compute_fom(*self.fom_args, **self.fom_kwargs)
                # Scale by max_intensity_by_wavelength weighting (any intensity FoM needs this)
                f /= np.array(self.cfg['max_intensity_by_wavelength'])
                self.fom_hist.get('intensity_overall').append(f)
                vipdopt.logger.debug(f'FoM: {f}')

                # Compute transmission FoM and apply spectral and performance weights.
                fom_kwargs_trans = self.fom_kwargs.copy()
                fom_kwargs_trans.update({'type': 'transmission'})
                t = np.array([
                    fom[0].fom_func(*self.fom_args, **fom_kwargs_trans)
                    for fom in self.fom.foms
                ])
                [
                    self.fom_hist.get(f'transmission_{idx}').append(t_i)
                    for idx, t_i in enumerate(t)
                ]
                self.fom_hist.get('transmission_overall').append(
                    np.squeeze(np.sum(t, 0))
                )
                # [plt.plot(np.squeeze(t_i)) for t_i in t]

                # # Here is where we would start plotting the loss landscape. Probably should be accessed by a separate class...
                # # Or we could move it to the device step part
                # loss_landscape_mapper = LossLandscapeMapper.LossLandscapeMapper(simulations, devices)

                # Compute gradient and apply spectral and performance weights.
                g = self.fom.compute_grad(
                    *self.grad_args,
                    apply_performance_weights=True,
                    **self.grad_kwargs,
                )
                # Scale by max_intensity_by_wavelength weighting (any intensity FoM needs this)
                g /= np.array(self.cfg['max_intensity_by_wavelength'])
                # vipdopt.logger.debug(f'Gradient: {g}')

                # Process gradient accordingly for application to device through optimizer.

                g = np.sum(g, -1)  # Sum over wavelength
                vipdopt.logger.info(
                    f'Design_gradient has average {np.mean(g)}, max {np.max(g)}'
                )

                # Permittivity factor in amplitude of electric dipole at x_0:
                # We need to properly account here for the current real and imaginary index
                # because they both contribute in the end to the real part of the gradient ΔFoM/Δε_r
                # in Eq. 5 of Lalau-Keraly paper https://doi.org/10.1364/OE.21.021693
                # todo: if dispersion is considered, this needs to be assembled spectrally --------------------------------------------------
                # dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ lookup_dispersive_range_idx ] )
                dispersive_max_permittivity = self.device.permittivity_constraints[1]
                delta_permittivity = (
                    dispersive_max_permittivity
                    - self.device.permittivity_constraints[0]
                )
                # todo: -----------------------------------------------------------------------------------------------------
                get_grad_density = real_part_complex_product(delta_permittivity, g)
                #! This is where the gradient picks up a permittivity factor i.e. becomes larger than 1!
                #! Mitigated by backpropagating through the Scale filter.
                get_grad_density = (
                    2 * get_grad_density
                )  # Factor of 2 after taking the real part according to algorithm

                # # Get the full design gradient by summing the x,y polarization components
                # # todo: Do we need to consider polarization in the same way with the new implementation??
                # design_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )

                # Project / interpolate the design_gradient, the values of which we have at each (mesh) voxel point, and obtain it at each (geometry) voxel point
                design_gradient_interpolated = self.device.interpolate_gradient(
                    get_grad_density, dimension=self.cfg['simulator_dimension']
                )

                # Each device needs to remember its gradient!
                # todo: refine this
                if self.cfg['enforce_xy_gradient_symmetry']:
                    if self.cfg['simulator_dimension'] in '2D':
                        transpose_design_gradient_interpolated = np.flip(
                            design_gradient_interpolated, 1
                        )
                    if self.cfg['simulator_dimension'] in '3D':
                        transpose_design_gradient_interpolated = np.swapaxes(
                            design_gradient_interpolated, 0, 1
                        )
                    design_gradient_interpolated = 0.5 * (
                        design_gradient_interpolated
                        + transpose_design_gradient_interpolated
                    )
                # self.device.gradient = design_gradient_interpolated.copy()	# This is BEFORE backpropagation

                # Step the device with the gradient
                vipdopt.logger.debug('Stepping device along gradient.')
                self.optimizer.step(
                    self.device, design_gradient_interpolated.copy(), self.iteration
                )

                # Generate Plots and call callback functions
                self.save_histories()
                self.generate_plots()
                self.call_callbacks()

                self.iteration += 1
            if not self.loop:
                break

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
    opt = LumericalOptimization(
        sim,
        dev,
        GradientAscentOptimizer(step_size=1e-4),
        fom,
        epoch_list=[1, 2],
    )
    opt.run()
