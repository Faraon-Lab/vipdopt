"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable, Iterable
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import interpolate

import vipdopt
from vipdopt import GDS, STL
from vipdopt.configuration import Config
from vipdopt.eval import plotter
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import FoM
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import LumericalSimulation

DEFAULT_OPT_FOLDERS = {'temp': Path('.'), 'opt_info': Path('.'), 'opt_plots': Path('.')}
class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sims: Iterable[LumericalSimulation],
            device: Device,
            optimizer: GradientOptimizer,
            fom: FoM | None,
            cfg: Config,
            start_epoch: int=0,
            start_iter: int=0,
            max_epochs: int=1,
            iter_per_epoch: int=100,
            dirs: dict[str, Path]=DEFAULT_OPT_FOLDERS,
            env_vars: dict | None = None
    ):
        """Initialize Optimization object."""
        if env_vars is None:
            env_vars = {}
        self.sims = list(sims)
        self.nsims = len(sims)
        self.device = device
        self.optimizer = optimizer
        self.fom = fom
        self.dirs = dirs
        self.sim_files = [dirs['temp'] / f'{sim.info["name"]}.fsp' for sim in self.sims]
        # self.sim_files = [dirs['temp'] / f'sim_{i}.fsp' for i in range(self.nsims)]
        self.cfg = cfg                  #! 20240228 Ian - Optimization needs to call a few config variables
        self.foms = []
        self.weights = []
        self.env_vars = env_vars
        self.loop = True

        self.runner_sim = LumericalSimulation()  # Dummy sim for running in parallel
        self.runner_sim.fdtd = vipdopt.fdtd
        self.runner_sim.promise_env_setup(**env_vars)


        self.fom_hist: list[npt.NDArray] = []
        self.param_hist: list[npt.NDArray] = []
        self._callbacks: list[Callable[[Optimization], None]] = []

        self.epoch = start_epoch
        self.iteration = start_iter
        self.max_epochs = max_epochs
        self.iter_per_epoch = iter_per_epoch
        self.epoch_list = list(range(0,max_epochs*iter_per_epoch+1,iter_per_epoch))

        self.stats = {}

    def add_callback(self, func: Callable):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)

    def create_history(self, fom_types, max_iter, num_design_frequency_points):
        """#* Set up some numpy arrays to handle all the data used to track the FoMs and progress of the simulation."""
        # # By iteration	# todo: store in a block of lists?
        self.figure_of_merit_evolution = np.zeros(max_iter)
        # pdaf_transmission_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
        # step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
        self.binarization_evolution = np.zeros(max_iter)
        # average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
        # max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

        # By iteration, focal area, polarization, wavelength:
        # Create a dictionary of FoMs. Each key corresponds to a value: an array storing data of a specific FoM,
        # for each epoch, iteration, quadrant, polarization, and wavelength.
        # TODO: Consider using xarray instead of numpy arrays in order to have dimension labels for each axis.

        self.fom_evolution = {}
        for fom_type in fom_types:
            self.fom_evolution[fom_type] = np.zeros((max_iter, len(self.foms), num_design_frequency_points))
        # for fom_type in fom_types:	# todo: delete this part when all is finished
        # 	fom_evolution[fom_type] = np.zeros((num_epochs, num_iterations_per_epoch,
        # 										num_focal_spots, len(xy_names), num_design_frequency_points))
        # The main FoM being used for optimization is indicated in the config.

        # # NOTE: Turned this off because the memory and savefile space of gradient_evolution are too large.
        # # if start_from_step != 0:	# Repeat the definition of field_shape here just because it's not loaded in if we start from a debug point
        # # 	field_shape = [device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_vertical]
        # # gradient_evolution = np.zeros((num_epochs, num_iterations_per_epoch, *field_shape,
        # # 								num_focal_spots, len(xy_names), num_design_frequency_points))


    def save_sims(self):
        """Save all simulation files."""
        with ThreadPool() as pool:
            pool.starmap(
                LumericalSimulation.save_lumapi,
                zip(self.sims, self.sim_files),
            )

    def load_sims(self):
        """Load all simulation files."""
        with ThreadPool() as pool:
            pool.starmap(
                LumericalSimulation.load,
                zip(self.sims, [file.name for file in self.sim_files])
                # zip(self.sims, self.sim_files)
            )

    def save_histories(self):
        """Save the fom and parameter histories to file."""
        folder = self.dirs['opt_info']
        foms = np.array(self.fom_hist)
        with (folder / 'fom_history.npy').open('wb') as f:
            np.save(f, foms)
        params = np.array(self.param_hist)
        with (folder / 'paramater_history.npy').open('wb') as f:
            np.save(f, params)

    def generate_plots(self):
        """Generate the plots and save to file."""
        folder = self.dirs['opt_plots']

        #! 20240229 Ian - Best to be specifying functions for 2D and for 3D.

        # TODO: Plot key information such as Figure of Merit evolution for easy visualization and checking in the middle of optimizations
        fom_fig = plotter.plot_fom_trace(self.figure_of_merit_evolution, folder, self.epoch_list)
        quad_trans_fig = plotter.plot_quadrant_transmission_trace(self.fom_evolution['transmission'], folder, self.epoch_list)
        overall_trans_fig = plotter.plot_quadrant_transmission_trace(self.fom_evolution['overall_transmission'], folder, self.epoch_list,
                                                                filename='overall_trans_trace')

        # todo: code this to pull the forward sim for FoM_0 i.e. forward_src_x in this case but not EVERY case.
        # self.runner_sim.fdtd.load(self.foms[0].fom_monitors[0].sim.info['path'])  # untested
        self.runner_sim.fdtd.load(self.sim_files[0].name)
        E_focal = self.runner_sim.getresult('transmission_focal_monitor_','E')
        if self.cfg['simulator_dimension'] == '2D':
            intensity_fig = plotter.plot_Enorm_focal_2d(np.sum(np.abs(np.squeeze(E_focal['E']))**2, axis=-1),
                                                    E_focal['x'], E_focal['lambda'],
                                                    folder, self.true_iteration, wl_idxs=[7,22])
        elif self.cfg['simulator_dimension'] == '3D':
            intensity_fig = plotter.plot_Enorm_focal_3d(np.sum(np.abs(np.squeeze(E_focal['E']))**2, axis=-1),
                                                    E_focal['x'], E_focal['y'], E_focal['lambda'],
                                                    folder, self.true_iteration, wl_idxs=[9,29,49])

            indiv_trans_fig = plotter.plot_individual_quadrant_transmission(self.fom_evolution['transmission'],
                                self.cfg['lambda_values_um'], folder, self.true_iteration)	# continuously produces only one plot per epoch to save space

        cur_index = self.device.index_from_permittivity(self.device.get_permittivity())
        final_device_layer_fig, _ = plotter.visualize_device(cur_index, folder, iteration=self.true_iteration)

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
        # todo: rest of the plots


    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        # Connect to Lumerical
        self.loop = True
        return
        with ThreadPool() as pool:
            pool.apply(LumericalSimulation.connect, (self.runner_sim,))
            pool.map(LumericalSimulation.connect, self.sims)

    def _post_run(self):
        """Final post-processing after running the optimization."""
        # Disconnect from Lumerical
        self.loop = False
        self.save_histories()
        self.generate_plots()
        with ThreadPool() as pool:
            pool.map(LumericalSimulation.close, self.sims)

    def run_simulations(self):
        """Run all of the simulations in parallel."""
        # jobs = [
        #     (i, self.dir, sim.as_dict(), sim.get_env_vars())
        #     for i, sim in enumerate(self.sims)
        # ]

        # with Pool() as pool:
        #     sim_files = pool.starmap(_simulation_dispatch, jobs)

        # vipdopt.logger.debug('Creating new fdtd...')
        if self.nsims == 0:
            return

        vipdopt.logger.debug(
            f'Epoch {self.epoch}, iter {self.iteration}: Running simulations...'
        )

        # self.save_sims()      #! 20240227 Ian - Seems to be overwriting with blanks.

        # Loading device
        # Each epoch the filters get stronger and so the permittivity must be passed through the new filters
        # todo: test once you get a filter online
        self.device.update_filters(epoch=self.epoch)
        self.device.update_density()


        # Calculate material % and binarization level, store away
        # todo: redo this section once you get sigmoid filters up and can start counting materials
        cur_index = self.device.index_from_permittivity(self.device.get_permittivity())
        vipdopt.logger.info(f'TiO2% is {100 * np.count_nonzero(self.device.get_design_variable() < 0.5) / cur_index.size}%.')		# todo: seems to be wrong?
        # logging.info(f'Binarization is {100 * np.sum(np.abs(cur_density-0.5))/(cur_density.size*0.5)}%.')
        binarization_fraction = self.device.compute_binarization(self.device.get_design_variable())
        vipdopt.logger.info(f'Binarization is {100 * binarization_fraction}%.')
        # 	self.binarization_evolution[self.iteration] = 100 * utility.compute_binarization(cur_density)

        # 	# Save template copy out
        # 	self.simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])

        # Save out current design/density profile
        np.save(os.path.abspath(self.dirs['opt_info'] / 'cur_design.npy'), self.device.get_design_variable())
        if self.true_iteration in self.epoch_list:	# i.e. new epoch
            np.save(os.path.abspath(self.dirs['opt_info'] /  f'cur_design_e{self.epoch_list.index(self.true_iteration)}.npy'), self.device.get_design_variable())


        #
        # Step 3: Import the current epoch's permittivity value to the device. We create a different optimization job for:
        # - each of the polarizations for the forward source waves
        # - each of the polarizations for each of the adjoint sources
        # Since here, each adjoint source is corresponding to a focal location for a target color band, we have
        # <num_wavelength_bands> x <num_polarizations> adjoint sources.
        # We then enqueue each job and run them all in parallel.
        # NOTE: Polymer slab / substrate, if included, must change index when the cube index changes.
        #

        vipdopt.logger.info('Beginning Step 3: Setting up Forward and Adjoint Jobs')

        for sim_idx, sim in enumerate(self.sims):
            self.runner_sim.fdtd.load(sim.info['name'])
            # self.runner_sim.fdtd.load(self.sim_files[sim_idx].name)
            # Switch to Layout
            self.runner_sim.fdtd.switchtolayout()

            cur_density, cur_permittivity = sim.import_cur_index(self.device,
                                                reinterpolate_permittivity = False,     # cfg.cv.reinterpolate_permittivity,
                                                reinterpolate_permittivity_factor = 1   # cfg.cv.reinterpolate_permittivity_factor
                                            )
            cur_index = self.device.index_from_permittivity(cur_permittivity)

            self.runner_sim.save_lumapi(self.sim_files[sim_idx].absolute(), debug_mode=False) #! Debug using test_dev folder

        # Use dummy simulation to run all of them at once using lumapi
        # self.runner_sim.promise_env_setup(self.sims[0]._env_vars)
        if os.getenv('SLURM_JOB_NODELIST') is None: # True
            self.runner_sim.fdtd.setresource('FDTD', 1, 'Job launching preset', 'Local Computer')
        for fname in self.sim_files:
            vipdopt.logger.debug(f'Adding {fname.name} to Lumerical job queue.')
            self.runner_sim.fdtd.addjob(fname.name, 'FDTD')     #! Has to just add the filename

        vipdopt.logger.debug('Running all simulations...')
        self.runner_sim.fdtd.runjobs('FDTD') #! Debug using test_dev folder
        # todo: still running the linux mpi files. Fix this
        # todo: also check sims 0-6 to see if they're running correctly
        vipdopt.logger.debug('Done running simulations')

        # self.load_sims()          #! 20240227 Ian - Not Working

        # self.sims[0].connect()
        # vipdopt.logger.info(f'field_shape before: {self.sims[0].get_field_shape()}')
        # self.sims[0].load(self.dir / 'sim_0.fsp')
        # vipdopt.logger.info(f'field_shape after: {self.sims[0].get_field_shape()}')
        # Use a thread pool for this part so that state is shared

        # self.sims[0].connect()
        # self.sims[0].load(sim_files[0])
        # vipdopt.logger.debug(self.sims[0].fdtd.getresult('design_index_monitor'))

    def run(self):
        """Run the optimization."""
        self._pre_run()
        vipdopt.logger.info(f'Initial Device: {self.device.get_design_variable()}')
        while self.epoch < self.max_epochs:
            while self.iteration < self.iter_per_epoch:
                if not self.loop:
                    break
                for callback in self._callbacks:
                    callback(self)

                vipdopt.logger.info(
                    f'=========\nEpoch {self.epoch}, iter {self.iteration}: Running simulations...'
                )
                self.true_iteration = self.epoch * self.iter_per_epoch + self.true_iteration

                vipdopt.logger.debug(
                    f'\tDesign Variable: {self.device.get_design_variable()}'
                )

                # Run all the simulations
                self.runner_sim.fdtd = vipdopt.fdtd
                self.run_simulations()

                # #! 20240227 Ian - Math Helper is not working too well. Need to do this manually for now I'm afraid.
                # # Compute FoM and Gradient
                # fom = self.fom.compute_fom()
                # self.fom_hist.append(fom)

                # gradient = self.fom.compute_gradient()
                # #! 20240227 Ian - Gradient seems to be returned in tuple form, which is weird
                # # e.g. tuple of len (41), each element being an array (42, 3) - instead of np.array((41,42,3))
                # # We need to reshape the gradient

                # vipdopt.logger.debug(
                #     f'FoM at epoch {self.epoch}, iter {self.iteration}: {fom}\n'
                #     f'Gradient {self.epoch}, iter {self.iteration}: {gradient}'
                # )

                #! ESSENTIAL STEP - MUST CLEAR THE E-FIELDS OTHERWISE THEY WON'T UPDATE (see monitor.e function)
                for f in self.foms:
                    for m in f.fom_monitors:
                        m._e = None
                        m._h = None
                        m._trans_mag = None
                    for m in f.grad_monitors:
                        m._e = None
                        m._h = None
                        m._trans_mag = None

                # #! Debug using test_dev folder
                # self.sim_files = [self.dirs['debug_completed_jobs'] / f'{sim.info["name"]}.fsp' for sim in self.sims]
                # for sim in self.sims:
                #     sim.info['path'] = self.dirs['debug_completed_jobs'] / Path(sim.info['path']).name

                vipdopt.logger.info('Beginning Step 4: Computing Figure of Merit')

                tempfile_fwd_name_arr = [file.name for file in self.sim_files if any(x in file.name for x in ['forward','fwd'])]
                tempfile_adj_name_arr = [file.name for file in self.sim_files if any(x in file.name for x in ['adjoint','adj'])]

                for tempfile_fwd_name in tempfile_fwd_name_arr:
                    # self.runner_sim.fdtd.load(os.path.abspath(self.dirs['debug_completed_jobs'] / tempfile_fwd_name))   #! Debug using test_dev folder
                    self.runner_sim.fdtd.load(tempfile_fwd_name)
                    # efield_test = self.runner_sim.get_efield('design_efield_monitor')
                    # vipdopt.logger.debug(f'Test value of design_efield_monitor in file {tempfile_fwd_name} is {efield_test[1,2,2,0,15]}')

                    # print(3)

                    for f_idx, f in enumerate(self.foms):
                        if f.fom_monitors[-1].sim.info['name'] in tempfile_fwd_name:      # todo: might need to account for file extensions
                            # todo: rewrite this to hook to whichever device the FoM is tied to
                            # print(f'FoM property has length {len(f.fom)}')

                            f.design_fwd_fields = f.grad_monitors[0].e
                            vipdopt.logger.info(f'Accessed design E-field monitor. NOTE: Field shape obtained is: {f.grad_monitors[0].fshape}')
                            # vipdopt.logger.debug(f'Test value of design_efield_monitor in file {tempfile_fwd_name} is alternately {f.design_fwd_fields[1,2,2,0,15]}')
                            # Store FoM value and Restricted FoM value (possibly not the true FoM being used for gradient)
                            # f.compute_fom(self.simulator)

                            vipdopt.logger.info(f'Accessing FoM {self.foms.index(f)} in list.')
                            vipdopt.logger.info(f'File being accessed: {tempfile_fwd_name} should match FoM tempfile {f.fom_monitors[-1].sim.info["name"]}')
                            # vipdopt.logger.debug(f'TrueFoM before has shape{f.true_fom.shape}')
                            f.fom, f.true_fom = f.compute_fom()
                            # This also creates and assigns source_weight for the FoM
                            vipdopt.logger.info(f'TrueFoM after has shape{f.true_fom.shape}')
                            vipdopt.logger.info(f'Source weight after has shape{f.source_weight.shape}')

                            # Scale by max_intensity_by_wavelength weighting (any intensity FoM needs this)
                            f.true_fom /= np.array(self.cfg['max_intensity_by_wavelength'])[list(f.opt_ids)]


                            # Store fom, restricted fom, true fom
                            self.fom_evolution['transmission'][self.true_iteration, f_idx, :] = np.squeeze(f.fom)
                            self.fom_evolution['overall_transmission'][self.true_iteration, f_idx, :] = np.squeeze(np.abs(
                                        f.fom_monitors[0].sim.getresult('transmission_focal_monitor_', 'T', 'T')
                                    ))
                            # todo: add line for restricted fom
                            self.fom_evolution[self.cfg['optimization_fom']][self.true_iteration, f_idx, :] = np.squeeze(f.true_fom)

                            # Compute adjoint amplitude or source weight (for mode overlap)

                            # Conner:
                            # if f.polarization == 'TE':
                            # 	adj_amp = np.conj(np.squeeze(f.adj_src.mon.results.Ez[:,:,:,f.freq_index_opt + f.freq_index_restricted_opt]))
                            # else:	adj_amp = np.conj(np.squeeze(f.adj_src.mon.results.Ex[:,:,:,f.freq_index_opt + f.freq_index_restricted_opt]))

                            # Greg:
                            # # This is going to be the amplitude of the dipole-adjoint source driven at the focal plane
                            # source_weight = np.squeeze( np.conj(
                            # 	get_focal_data[adj_src_idx][xy_idx, 0, 0, 0, spectral_indices[0]:spectral_indices[1]:1]
                            # ) )
                        # gc.collect()

                for tempfile_adj_name in tempfile_adj_name_arr:
                    self.runner_sim.fdtd.load(tempfile_adj_name)
                    # efield_test = self.runner_sim.get_efield('design_efield_monitor')
                    # vipdopt.logger.debug(f'Test value of design_efield_monitor in file {tempfile_adj_name} is {efield_test[1,2,2,0,15]}')

                    #* Process the various figures of merit for performance weighting.
                    # Copy everything so as not to disturb the original data
                    # process_fom_evolution = copy.deepcopy(self.fom_evolution)

                    for f_idx, f in enumerate(self.foms):
                        if f.grad_monitors[-1].sim.info['name'] in tempfile_adj_name:      # todo: might need to account for file extensions
                            # todo: rewrite this to hook to whichever device the FoM is tied to
                            # print(f'FoM property has length {len(f.fom)}')

                            f.design_adj_fields = f.grad_monitors[-1].e
                            vipdopt.logger.info('Accessed design E-field monitor.')
                            # vipdopt.logger.debug(f'Test value of design_efield_monitor in file {tempfile_adj_name} is alternately {f.design_adj_fields[1,2,2,0,15]}')

                            #* Compute gradient
                            # f.compute_gradient()

                            vipdopt.logger.info(f'Accessing FoM {self.foms.index(f)} in list.')
                            vipdopt.logger.info(f'File being accessed: {tempfile_adj_name} should match FoM tempfile {f.grad_monitors[-1].sim.info["name"]}')
                            f.gradient = f.compute_gradient()
                            # vipdopt.logger.info(f'TrueFoM before has shape{f.gradient.shape}')
                            vipdopt.logger.info(f'freq opt indices are {f.opt_ids}')
                            vipdopt.logger.info(f'Gradient after has shape{f.gradient.shape}')

                            # Scale by max_intensity_by_wavelength weighting (any intensity FoM needs this)
                            f.gradient /= np.array(self.cfg['max_intensity_by_wavelength'])[list(f.opt_ids)]

                        # gc.collect()

                #! TODO: sort this out
                # NOTE: Reminder that fom_evolution is a dictionary with keys: fom_types
                # and values being numpy arrays of shape (epochs, iterations, focal_areas, polarizations, wavelengths)
                list_overall_figure_of_merit = self.fom_func(self.foms, self.weights)
                # list_overall_figure_of_merit = env_constr.overall_figure_of_merit(indiv_foms, weights)		# Gives a wavelength vector
                # # todo: plots and histories
                self.figure_of_merit_evolution[self.true_iteration] = np.sum(list_overall_figure_of_merit, -1)			# Sum over wavelength
                vipdopt.logger.info(f'Figure of merit is now: {self.figure_of_merit_evolution[self.true_iteration]}')

#                 # Compute FoM and Gradient
#                 fom = self.fom.compute(self.device.get_design_variable())
#                 self.fom_hist.append(fom)

#                 gradient = self.fom.gradient(self.device.get_design_variable())

#                 vipdopt.logger.debug(
#                     f'\tFoM: {fom}\n'
#                     f'\tGradient: {gradient}'
#                 )


                # # # TODO: Save out variables that need saving for restart and debug purposes
                np.save(os.path.abspath(self.dirs['opt_info'] / 'figure_of_merit.npy'), self.figure_of_merit_evolution)
                np.save(os.path.abspath(self.dirs['opt_info'] / 'fom_by_focal_pol_wavelength.npy'), self.fom_evolution[self.cfg['optimization_fom']])
                for fom_type in self.cfg['fom_types']:
                    np.save(os.path.abspath(self.dirs['opt_info'] /  f'{fom_type}_by_focal_pol_wavelength.npy'), self.fom_evolution[fom_type])
                np.save(os.path.abspath(self.dirs['opt_info'] / 'binarization.npy'), self.binarization_evolution)

                self.param_hist.append(self.device.get_design_variable())

                # #* Here is where we would start plotting the loss landscape. Probably should be accessed by a separate class...
                # # Or we could move it to the device step part
                # loss_landscape_mapper = LossLandscapeMapper.LossLandscapeMapper(simulations, devices)

                vipdopt.logger.info('Completed Step 4: Computed Figure of Merit.')

                vipdopt.logger.info('Beginning Step 5: Adjoint Optimization Jobs and Gradient Computation.')
                #! TODO: sort this out
                list_overall_adjoint_gradient = self.grad_func(self.foms, self.weights)
                # list_overall_adjoint_gradient = env_constr.overall_adjoint_gradient( indiv_foms, weights )	# Gives a wavelength vector
                design_gradient = np.sum(list_overall_adjoint_gradient, -1)									# Sum over wavelength
                vipdopt.logger.info(f'Design_gradient has average {np.mean(design_gradient)}, max {np.max(design_gradient)}')

                # Permittivity factor in amplitude of electric dipole at x_0
                # Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)

                # todo: might need to assemble this spectrally -------------------------------------------------------------
                # We need to properly account here for the current real and imaginary index
                # because they both contribute in the end to the real part of the gradient ΔFoM/Δε_r
                # in Eq. 5 of Lalau-Keraly paper https://doi.org/10.1364/OE.21.021693
                #
                # dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ lookup_dispersive_range_idx ] )
                dispersive_max_permittivity = self.cfg['max_device_permittivity']
                delta_real_permittivity = np.real( dispersive_max_permittivity - self.cfg['min_device_permittivity'] )
                delta_imag_permittivity = np.imag( dispersive_max_permittivity - self.cfg['min_device_permittivity'] )
                # todo: -----------------------------------------------------------------------------------------------------

                # Permittivity factor in amplitude of electric dipole at x_0
                # Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)
                real_part_gradient = np.real( design_gradient )
                imag_part_gradient = -np.imag( design_gradient )
                get_grad_density = delta_real_permittivity * real_part_gradient + delta_imag_permittivity * imag_part_gradient
                #! This is where the gradient picks up a permittivity factor i.e. becomes larger than 1!
                #! Mitigated by backpropagating through the Scale filter.
                get_grad_density = 2 * get_grad_density		# Factor of 2 after taking the real part

                assert self.device.field_shape == get_grad_density.shape, f'The sizes of field_shape {self.device.field_shape} and gradient_density {get_grad_density.shape} are not the same.'
                # todo: which design? This stops working when there are multiple designs
                # xy_polarized_gradients[pol_name_to_idx] += get_grad_density
                # # TODO: The shape of this should be (epoch, iteration, nx, ny, nz, n_adj, n_pol, nλ) and right now it's missing nx,ny,nz
                # # todo: But also if we put that in, then the size will rapidly be too big
                # # gradient_evolution[epoch, iteration, :,:,:,
                # # 					adj_src_idx, xy_idx,
                # # 					spectral_indices[0] + spectral_idx] += get_grad_density

                # Get the full design gradient by summing the x,y polarization components # todo: Do we need to consider polarization??
                # todo: design_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
                # Project / interpolate the design_gradient, the values of which we have at each (mesh) voxel point, and obtain it at each (geometry) voxel point
                gradient_region_x = 1e-6 * np.linspace(self.device.coords['x'][0], self.device.coords['x'][-1], self.device.field_shape[0]) #cfg.pv.device_voxels_simulation_mesh_lateral_bordered)
                gradient_region_y = 1e-6 * np.linspace(self.device.coords['y'][0], self.device.coords['y'][-1], self.device.field_shape[1])
                try:
                    gradient_region_z = 1e-6 * np.linspace(self.device.coords['z'][0], self.device.coords['z'][-1], self.device.field_shape[2])
                except IndexError:	# 2D case
                    gradient_region_z = 1e-6 * np.linspace(self.device.coords['z'][0], self.device.coords['z'][-1], 3)
                design_region_geometry = np.array(np.meshgrid(1e-6*self.device.coords['x'], 1e-6*self.device.coords['y'], 1e-6*self.device.coords['z'], indexing='ij')
                                    ).transpose((1,2,3,0))

                if self.cfg['simulator_dimension'] in '2D':
                    design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ),
                                                        np.repeat(get_grad_density[..., np.newaxis], 3, axis=2), 	# Repeats 2D array in 3rd dimension, 3 times
                                                        design_region_geometry, method='linear' )
                elif self.cfg['simulator_dimension'] in '3D':
                    design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ),
                                                       get_grad_density,
                                                       design_region_geometry, method='linear' )
                # design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), device_gradient, bayer_filter_region, method='linear' )

                # Each device needs to remember its gradient!
                # todo: refine this
                if self.cfg['enforce_xy_gradient_symmetry']:
                    if self.cfg['simulator_dimension'] in '2D':
                        transpose_design_gradient_interpolated = np.flip(design_gradient_interpolated, 1)
                    if self.cfg['simulator_dimension'] in '3D':
                        transpose_design_gradient_interpolated = np.swapaxes( design_gradient_interpolated, 0, 1 )
                    design_gradient_interpolated = 0.5 * ( design_gradient_interpolated + transpose_design_gradient_interpolated )
                self.device.gradient = design_gradient_interpolated.copy()	# This is BEFORE backpropagation

                vipdopt.logger.info('Completed Step 5: Adjoint Optimization Jobs and Gradient Computation.')

                vipdopt.logger.info('Beginning Step 6: Stepping Design Variable.')
                #* DEBUG: Device+Optimizer Working===================
                # self.device.gradient = np.zeros(self.device.get_permittivity().shape)

                # def function_4( optimization_variable_, grad):
                #     '''f(x) = Σ(abs(y-x))'''
                #     y_const = 4.00
                #     figure_of_merit_total = 0.0

                #     figure_of_merit_total = np.abs(y_const- optimization_variable_)
                #     grad[:] = -1 * (y_const - optimization_variable_) / np.abs(y_const - optimization_variable_)

                #     vipdopt.logger.info(f'N={self.iteration} - Figure of Merit: {np.sum(figure_of_merit_total)}')
                #     return np.sum(figure_of_merit_total)

                # self.figure_of_merit_evolution[self.iteration] = function_4(self.device.get_permittivity(), self.device.gradient)

                # plt.imshow(np.real(self.device.get_permittivity()[..., 1]))
                # plt.colorbar()
                # plt.clim(vmin=2.25, vmax=5.76)
                # plt.savefig(self.dir / f'opt_info/device_layers/Device_N{self.iteration}.png')
                # plt.close()
                #* DEBUG: Device+Optimizer Working===================

                # Backpropagate the design_gradient through the various filters to pick up (per the chain rule) gradients from each filter
                # so the final product can be applied to the (0-1) scaled permittivity
                design_gradient_unfiltered = self.device.backpropagate( self.device.gradient )

                # Get current values of design variable before perturbation, for later comparison.
                last_design_variable = self.device.get_design_variable().copy()

                # Step with the gradient
                # design_gradient_unfiltered *= -1             #! minimize
                # w_hat = self.device.get_design_variable() + 0.05 * design_gradient_unfiltered
                # w_hat = np.random.random(self.device.get_design_variable().shape)
                # self.device.set_design_variable(np.maximum(np.minimum(w_hat, 1), 0))
                self.optimizer.step(self.device, design_gradient_unfiltered, self.true_iteration)


                #* Save out overall savefile, save out all save information as separate files as well.
                difference_design = np.abs(self.device.get_design_variable() - last_design_variable)
                vipdopt.logger.info(f'Device variable stepped by maximum of {np.max(difference_design)} and minimum of {np.min(difference_design)}.')

                # todo: saving out
                # self.optimizer.save_opt()

                self.generate_plots()
                self.iteration += 1


            if not self.loop:
                break
            self.iteration = 0
            self.epoch += 1

        #! TODO: Turn fom_hist into figure_of_merit_evolution
        final_fom = self.figure_of_merit_evolution[-1]
        vipdopt.logger.info(f'Final FoM: {final_fom}')
        final_params = self.param_hist[-1]
        vipdopt.logger.info(f'Final Parameters: {final_params}')
        self._post_run()

        # STL Export requires density to be fully binarized
        full_density = self.device.binarize(self.device.get_density())
        stl_generator = STL.STL(full_density)
        stl_generator.generate_stl()
        stl_generator.save_stl( self.dirs['data'] / 'final_device.stl' )

        # GDS Export must be done in layers. We split the full design into individual layers, (not related to design layers)
        # Express each layer as polygons in an STL Mesh object and write that layer into its own cell of the GDS file
        layer_mesh_array = []
        for z_layer in range(full_density.shape[2]):
            stl_generator = STL.STL(full_density[..., z_layer][..., np.newaxis])
            stl_generator.generate_stl()
            layer_mesh_array.append(stl_generator.stl_mesh)

        # Create a GDS object that contains a Library with Cells corresponding to each 2D layer in the 3D device.
        gds_generator = GDS.GDS().set_layers(full_density.shape[2],
                unit = 1e-6 * np.abs(self.device.coords['x'][-1] - self.device.coords['x'][0])/self.device.size[0])
        gds_generator.assemble_device(layer_mesh_array, listed=False)

        # Directory for export
        gds_layer_dir = os.path.join(self.dirs['data'] / 'gds')
        if not os.path.exists(gds_layer_dir):
            os.makedirs(gds_layer_dir)
        # Export both GDS and SVG for easy visualization.
        gds_generator.export_device(gds_layer_dir, filetype='gds')
        gds_generator.export_device(gds_layer_dir, filetype='svg')
        # for layer_idx in range(0, full_density.shape[2]):         # Individual layer as GDS file export
        #     gds_generator.export_device(gds_layer_dir, filetype='gds', layer_idx=layer_idx)
        #     gds_generator.export_device(gds_layer_dir, filetype='svg', layer_idx=layer_idx)

        # # Here is a function for GDS device import - be warned this takes maybe 3-5 minutes per layer.
        # def import_gds(sim, device, gds_layer_dir):
        #     import time

        #     sim.fdtd.load(sim.info['name'])
        #     sim.fdtd.switchtolayout()
        #     device_name = device.import_names()[0]
        #     sim.disable([device_name])

        #     z_vals = device.coords['z']
        #     for z_idx in range(len(z_vals)):
        #         t = time.time()
        #         # fdtd.gdsimport(os.path.join(gds_layer_dir, 'device.gds'), f'L{layer_idx}', 0)
        #         sim.fdtd.gdsimport(os.path.join(gds_layer_dir, 'device.gds'), f'L{z_idx}', 0, "Si (Silicon) - Palik",
        #                     z_vals[z_idx], z_vals[z_idx+1])
        #         sim.fdtd.set({'x': device.coords['x'][0], 'y': device.coords['y'][0]})      # Be careful of units - 1e-6
        #         vipdopt.logger.info(f'Layer {z_idx} imported in {time.time()-t} seconds.')
