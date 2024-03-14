"""Class for handling optimization setup, running, and saving."""

import sys
import os
import gc
import copy
import logging
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Custom Classes and Imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import configs.yaml_to_params as cfg
# Gets all parameters from config file - store all those variables within the namespace. Editing cfg edits it for all modules accessing it
# See https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
# from utils import Device
# from utils import LumericalUtils
# from utils import FoM
from utils import manager
from utils import utility
from evaluation import plotter




class Optimization:
	"""Class for orchestrating all the pieces of an optimization."""

	def __init__( self, sims, foms, designs, design_objs, optimizer, fom_func, grad_func, weights, simulator ):
		"""Initialize Optimization object."""
		self.name = 'OptimizationState'

		self.simulations = sims          # Simulation object(s) passed to the Optimizer
		self.foms = foms            	 # List of FoM objects
		self.designs = designs       	 # Device object
		self.design_objs = design_objs	 # Dictionaries corresponding to Devices
		self.optimizer = optimizer 		 # Optimizer (see below for class definition)
		self.fom_func = fom_func		 # FoM function
		self.grad_func = grad_func		 # Gradient function
		self.weights = weights			 # Weights
		self.simulator = simulator		 # Simulator (LumericalUtils.py)

	def setup(self, max_iter, current_iter=0, epoch_list=None):
		"""Set up the optimization for running with control parameters."""
		self.max_iter = max_iter
		self.iteration = current_iter
		if epoch_list is None:
			self.epoch_list = list(np.linspace(current_iter, max_iter, num=10))
		else:	self.epoch_list = epoch_list
		# self.max_epoch = utility.partition_iterations(self.max_iter,self.epoch_list)

	def create_history(self, fom_types, max_iter, num_design_frequency_points):
		"""#* Set up some numpy arrays to handle all the data used to track the FoMs and progress of the simulation."""

		# # By iteration	# todo: store in a block of lists?
		self.figure_of_merit_evolution = np.zeros((max_iter))
		# pdaf_transmission_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
		# step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
		self.binarization_evolution = np.zeros((max_iter))
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


	def run(self):
		"""Run this optimization."""
		logging.info(f'Running {self.optimizer.__class__.__name__} optimization from Iteration {self.iteration} to {self.max_iter}. ')

		while self.iteration < self.max_iter:
			for callback in self.optimizer._callbacks:
				self.optimizer.callback()
   
			# Get current iteration and determines which epoch we're in, according to the epoch list in config.
			current_epoch = utility.partition_iterations(self.iteration, self.epoch_list)
			# current_subepoch = iteration - cfg.cv.epoch_list[current_epoch]		# We don't count things like this anymore
			logging.info(f'Working on Iteration {self.iteration} [Epoch {current_epoch}].')
			self.optimizer.iteration = self.iteration

			# Each epoch the filters get stronger and so the permittivity must be passed through the new filters
			# todo: test once you get a filter online
			for num_design, design in enumerate(self.designs):
				design.update_filters(current_epoch)
				design.update_density()

			#
			# Step 3: Import the current epoch's permittivity value to the device. We create a different optimization job for:
			# - each of the polarizations for the forward source waves
			# - each of the polarizations for each of the adjoint sources
			# Since here, each adjoint source is corresponding to a focal location for a target color band, we have 
			# <num_wavelength_bands> x <num_polarizations> adjoint sources.
			# We then enqueue each job and run them all in parallel.
			# NOTE: Polymer slab / substrate, if included, must change index when the cube index changes.
			#

			logging.info("Beginning Step 3: Setting up Forward and Adjoint Jobs")
		
			for num_design, design in enumerate(self.designs):
				design_obj = self.design_objs[num_design]								# Access corresponding dictionary
				simulation = self.simulations[design_obj['DEVICE_INFO']['simulation']]	# Access simulation region containing design region.
				self.simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))

				# Disable all sources
				self.simulator.switch_to_layout()
				self.simulator.disable_all_sources(simulation)		# disabling all sources sets it up for creating forward and adjoint-sourced sims later

				# Import cur_index into design regions
				cur_density, cur_permittivity = utility.import_cur_index(design, design_obj, self.simulator,
							reinterpolate_permittivity = False, # cfg.cv.reinterpolate_permittivity,
							reinterpolate_permittivity_factor = 1 # cfg.cv.reinterpolate_permittivity_factor
						)
				cur_index = utility.index_from_permittivity(cur_permittivity)
				
				# Calculate material % and binarization level, store away
				# todo: redo this section once you get sigmoid filters up and can start counting materials
				logging.info(f'TiO2% is {100 * np.count_nonzero(cur_index > cfg.cv.min_device_index) / cur_index.size}%.')		# todo: seems to be wrong?		
				# logging.info(f'Binarization is {100 * np.sum(np.abs(cur_density-0.5))/(cur_density.size*0.5)}%.')
				binarization_fraction = utility.compute_binarization(cur_density)
				logging.info(f'Binarization is {100 * binarization_fraction}%.')
				self.binarization_evolution[self.iteration] = 100 * utility.compute_binarization(cur_density)

				# Save template copy out
				self.simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])

				# Save out current design/density profile
				np.save(os.path.abspath(cfg.OPTIMIZATION_INFO_FOLDER + "/cur_design.npy"), cur_density)
				if self.iteration in self.epoch_list:	# i.e. new epoch
					np.save(os.path.abspath(cfg.OPTIMIZATION_INFO_FOLDER + f"/cur_design_e{self.epoch_list.index(self.iteration)}.npy"), cur_density)


			#* Start creating array of jobs with different sources
			# Run all simulations in parallel

			for simulation in self.simulations:
				# todo: store fwd_src_arr, adj_src_arr, tempfile_fwd, tempfile_adj, tempfile_to_source_map in each simulation object

				# FORWARD SIMS
				fwd_src_arr, tempfile_fwd_name_arr = manager.create_tempfiles_fwd_parallel(self.foms, self.simulator, simulation)
				# NOTE: indiv_foms, simulation are also amended accordingly
				# NOTE: indiv_foms have tempfile names as indices at this point

				# All forward simulations are queued. Return to the original file (just to be safe).
				self.simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))
				self.simulator.switch_to_layout()
				self.simulator.disable_all_sources(simulation)

				# ADJOINT SIMS
				active_adj_src_arr, inactive_adj_src_arr, tempfile_adj_name_arr = \
									manager.create_tempfiles_adj_parallel(self.foms, self.simulator, simulation)
				# NOTE: indiv_foms, simulation are also amended accordingly
				# NOTE: indiv_foms have tempfile names as indices at this point

				# All adjoint simulations are queued. Return to the original file (just to be safe).
				self.simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))
				self.simulator.switch_to_layout()
				self.simulator.disable_all_sources(simulation)


				# Now that the tempfile_fwd_name_arr is filled, and same for adj,
				# for each FoM object, we store the corresponding filename for easy access later.
				# We also create a map of the activated sources each file has.
				tempfile_to_source_map = {}
				for f in self.foms:
					tempfile_to_source_map[tempfile_fwd_name_arr[f.tempfile_fwd_name]] = f.tempfile_fwd_name
					tempfile_to_source_map[tempfile_adj_name_arr[f.tempfile_adj_name]] = f.tempfile_adj_name
					f.tempfile_fwd_name = tempfile_fwd_name_arr[f.tempfile_fwd_name]	# Replace the index with the actual string.
					f.tempfile_adj_name = tempfile_adj_name_arr[f.tempfile_adj_name]	# Replace the index with the actual string.
				

				# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
				utility.write_dict_to_file(tempfile_to_source_map, 'tempfile_to_source_map', folder=cfg.OPTIMIZATION_INFO_FOLDER, serialization='npy')

				#* Run all jobs in parallel
				job_names = tempfile_fwd_name_arr + tempfile_adj_name_arr
				manager.run_jobs(job_names, cfg.PULL_COMPLETED_JOBS_FOLDER, cfg.workload_manager, self.simulator)

			logging.info("Completed Step 3: Forward and Adjoint Simulations complete.")
			# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
			utility.write_dict_to_file(tempfile_to_source_map, 'tempfile_to_source_map', folder=cfg.OPTIMIZATION_INFO_FOLDER, serialization='npy')


			logging.info("Beginning Step 4: Computing Figure of Merit")

			for simulation in self.simulations:
				# design_list = simulation_to_device_map[simulation['SIM_INFO']['sim_id']]	# this is a list of indices: designs present in the sim
				# todo: make this work when there are multiple simulations and multiple devices
				design_list = [0]
	
				for tempfile_fwd_name in tempfile_fwd_name_arr:
					self.simulator.open(os.path.join(cfg.PULL_COMPLETED_JOBS_FOLDER, tempfile_fwd_name))

					for num_design in design_list:
						design = self.designs[num_design]
						# forward_e_fields = np.zeros([len(tempfile_fwd_name_arr)] + \
						# 							[3] + \
						# 							list(field_shapes[num_design].shape) + [len(cfg.pv.lambda_values_um)]
						# 						)	# Each tempfile stores E-fields of shape (3, nx, ny, nz, nλ)
		
						for f_idx, f in enumerate(self.foms):
							if tempfile_fwd_name in f.tempfile_fwd_name:	# todo: might need to account for file extensions
								# todo: rewrite this to hook to whichever device the FoM is tied to
								print(f'FoM property has length {len(f.fom)}')	
  
								f.design_fwd_fields = \
									self.simulator.get_efield('design_efield_monitor') #[..., f.freq_index_opt + f.freq_index_restricted_opt]
								logging.info(f'Accessed design E-field monitor. NOTE: Field shape obtained is: {f.design_fwd_fields.shape}')
								# Store FoM value and Restricted FoM value (possibly not the true FoM being used for gradient)
								# f.compute_fom(self.simulator)
								
								logging.debug(f'Accessing FoM {self.foms.index(f)} in list.')
								logging.debug(f'File being accessed: {tempfile_fwd_name} should match FoM tempfile {f.tempfile_fwd_name}')
								logging.debug(f'TrueFoM before has shape{f.true_fom.shape}')
								f.compute_fom(self.simulator)
								logging.debug(f'TrueFoM after has shape{f.true_fom.shape}')
								
								# Scale by max_intensity_by_wavelength weighting (any intensity FoM needs this)
								f.true_fom /= cfg.pv.max_intensity_by_wavelength
								
								# Store fom, restricted fom, true fom
								self.fom_evolution['transmission'][self.iteration, f_idx, :] = np.squeeze(f.fom)
								# todo: add line for restricted fom
								self.fom_evolution[cfg.cv.optimization_fom][self.iteration, f_idx, :] = np.squeeze(f.true_fom)
		
								# # Compute adjoint amplitude or source weight (for mode overlap)
		
								# Conner:
								# if f.polarization == 'TE':
								# 	adj_amp = np.conj(np.squeeze(f.adj_src.mon.results.Ez[:,:,:,f.freq_index_opt + f.freq_index_restricted_opt]))
								# else:	adj_amp = np.conj(np.squeeze(f.adj_src.mon.results.Ex[:,:,:,f.freq_index_opt + f.freq_index_restricted_opt]))
		
								# Greg:
								# # This is going to be the amplitude of the dipole-adjoint source driven at the focal plane
								# source_weight = np.squeeze( np.conj(
								# 	get_focal_data[adj_src_idx][xy_idx, 0, 0, 0, spectral_indices[0]:spectral_indices[1]:1]
								# ) )
							gc.collect()

				for tempfile_adj_name in tempfile_adj_name_arr:
					self.simulator.open(os.path.join(cfg.PULL_COMPLETED_JOBS_FOLDER, tempfile_adj_name))

					for num_design in design_list:
						design = self.designs[num_design]
		
						# Initialize arrays to store the gradients of each polarization, each the shape of the voxel mesh
						# This array therefore has shape (polarizations, mesh_lateral, mesh_lateral, mesh_vertical)
						xy_polarized_gradients = [ np.zeros(design.field_shape, dtype=np.complex128), np.zeros(design.field_shape, dtype=np.complex128) ]

						
						#* Process the various figures of merit for performance weighting.
						# Copy everything so as not to disturb the original data
						process_fom_evolution = copy.deepcopy(self.fom_evolution)
		
						for f in self.foms:
							logging.debug(f'tempfile_adj_name is {tempfile_adj_name}; FoM has the tempfile {f.tempfile_adj_name}')
							if tempfile_adj_name in f.tempfile_adj_name:	# todo: might need to account for file extensions
								# todo: rewrite this to hook to whichever device the FoM is tied to
								f.design_adj_fields = \
									self.simulator.get_efield('design_efield_monitor') #[..., f.freq_index_opt + f.freq_index_restricted_opt]
								logging.info('Accessed design E-field monitor.')

								#* Compute gradient
								# f.compute_gradient()
		
								logging.debug(f'Accessing FoM {self.foms.index(f)} in list.')
								logging.debug(f'File being accessed: {tempfile_adj_name} should match FoM tempfile {f.tempfile_adj_name}')
								logging.debug(f'Gradient before has shape{f.gradient.shape}')
								logging.debug(f'freq_index_opt is {f.freq_index_opt}')
								f.compute_gradient()
								logging.debug(f'Gradient after has shape{f.gradient.shape}')
	
								# Scale by max_intensity_by_wavelength weighting (any intensity FoM needs this)
								f.gradient /= cfg.pv.max_intensity_by_wavelength #[f.freq_index_opt]
		
							gc.collect()

				# todo: sort this out
				# NOTE: Reminder that fom_evolution is a dictionary with keys: fom_types
				# and values being numpy arrays of shape (epochs, iterations, focal_areas, polarizations, wavelengths)
				list_overall_figure_of_merit = self.fom_func(self.foms, self.weights)
				# list_overall_figure_of_merit = env_constr.overall_figure_of_merit(indiv_foms, weights)		# Gives a wavelength vector
				self.figure_of_merit_evolution[self.iteration] = np.sum(list_overall_figure_of_merit)			# Sum over wavelength
				logging.info(f'Figure of merit is now: {self.figure_of_merit_evolution[self.iteration]}')

				
				# # TODO: Save out variables that need saving for restart and debug purposes
				np.save(os.path.abspath(cfg.OPTIMIZATION_INFO_FOLDER + "/figure_of_merit.npy"), self.figure_of_merit_evolution)
				np.save(os.path.abspath(cfg.OPTIMIZATION_INFO_FOLDER + "/fom_by_focal_pol_wavelength.npy"), self.fom_evolution[cfg.cv.optimization_fom])
				for fom_type in cfg.cv.fom_types:
					np.save(os.path.abspath(cfg.OPTIMIZATION_INFO_FOLDER + f"/{fom_type}_by_focal_pol_wavelength.npy"), self.fom_evolution[fom_type])
				np.save(os.path.abspath(cfg.OPTIMIZATION_INFO_FOLDER + "/binarization.npy"), self.binarization_evolution)

				# # TODO: Plot key information such as Figure of Merit evolution for easy visualization and checking in the middle of optimizations
				plotter.plot_fom_trace(self.figure_of_merit_evolution, cfg.OPTIMIZATION_PLOTS_FOLDER, cfg.cv.epoch_list)
				plotter.plot_quadrant_transmission_trace(self.fom_evolution['transmission'], cfg.OPTIMIZATION_PLOTS_FOLDER, cfg.cv.epoch_list)
				# plotter.plot_individual_quadrant_transmission(quadrant_transmission_data, lambda_values_um, OPTIMIZATION_PLOTS_FOLDER,
				# 										epoch, iteration=30)	# continuously produces only one plot per epoch to save space
				plotter.plot_overall_transmission_trace(self.fom_evolution['transmission'], cfg.OPTIMIZATION_PLOTS_FOLDER, cfg.cv.epoch_list)
				plotter.visualize_device(cur_index, cfg.OPTIMIZATION_PLOTS_FOLDER, iteration=self.iteration)
				# # plotter.plot_moments(adam_moments, OPTIMIZATION_PLOTS_FOLDER)
				# # plotter.plot_step_size(adam_moments, OPTIMIZATION_PLOTS_FOLDER)

				# #* Here is where we would start plotting the loss landscape. Probably should be accessed by a separate class...
				# # Or we could move it to the device step part
				# loss_landscape_mapper = LossLandscapeMapper.LossLandscapeMapper(simulations, devices)

				logging.info("Completed Step 4: Computed Figure of Merit.")
				# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)

				logging.info("Beginning Step 5: Adjoint Optimization Jobs and Gradient Computation.")
				list_overall_adjoint_gradient = self.grad_func(self.foms, self.weights)
				# list_overall_adjoint_gradient = env_constr.overall_adjoint_gradient( indiv_foms, weights )	# Gives a wavelength vector
				design_gradient = np.sum(list_overall_adjoint_gradient, -1)									# Sum over wavelength

				# Permittivity factor in amplitude of electric dipole at x_0
				# Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)

				# todo: might need to assemble this spectrally -------------------------------------------------------------
				# We need to properly account here for the current real and imaginary index
				# because they both contribute in the end to the real part of the gradient ΔFoM/Δε_r
				# in Eq. 5 of Lalau-Keraly paper https://doi.org/10.1364/OE.21.021693
				#
				# dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ lookup_dispersive_range_idx ] )
				dispersive_max_permittivity = cfg.pv.max_device_permittivity
				delta_real_permittivity = np.real( dispersive_max_permittivity - cfg.pv.min_device_permittivity )
				delta_imag_permittivity = np.imag( dispersive_max_permittivity - cfg.pv.min_device_permittivity )
				# todo: -----------------------------------------------------------------------------------------------------

				# Permittivity factor in amplitude of electric dipole at x_0
				# Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)
				real_part_gradient = np.real( design_gradient )
				imag_part_gradient = -np.imag( design_gradient )
				get_grad_density = delta_real_permittivity * real_part_gradient + delta_imag_permittivity * imag_part_gradient
				#! This is where the gradient picks up a permittivity factor i.e. becomes larger than 1!
				#! Mitigated by backpropagating through the Scale filter.
				get_grad_density = 2 * get_grad_density		# Factor of 2 after taking the real part

				assert design.field_shape == get_grad_density.shape, f"The sizes of field_shape {design.field_shape} and gradient_density {get_grad_density.shape} are not the same."
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
				gradient_region_x = 1e-6 * np.linspace(design.coords['x'][0], design.coords['x'][-1], design.field_shape[0]) #cfg.pv.device_voxels_simulation_mesh_lateral_bordered)
				gradient_region_y = 1e-6 * np.linspace(design.coords['y'][0], design.coords['y'][-1], design.field_shape[1])
				try:
					gradient_region_z = 1e-6 * np.linspace(design.coords['z'][0], design.coords['z'][-1], design.field_shape[2])
				except IndexError as err:	# 2D case
					gradient_region_z = 1e-6 * np.linspace(design.coords['z'][0], design.coords['z'][-1], 3)
				design_region_geometry = np.array(np.meshgrid(1e-6*design.coords['x'], 1e-6*design.coords['y'], 1e-6*design.coords['z'], indexing='ij')
									).transpose((1,2,3,0))
	
				design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ),
														# get_grad_density											# 3D case
                										np.repeat(get_grad_density[..., np.newaxis], 3, axis=2), 	# Repeats 2D array in 3rd dimension, 3 times
													   design_region_geometry, method='linear' )
				# design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), device_gradient, bayer_filter_region, method='linear' )

				# Each device needs to remember its gradient!
				# todo: refine this
				if cfg.cv.enforce_xy_gradient_symmetry:
					if simulation['FDTD']['dimension'] in '2D':
						transpose_design_gradient_interpolated = np.flip(design_gradient_interpolated, 1)
					if simulation['FDTD']['dimension'] in '3D':
						transpose_design_gradient_interpolated = np.swapaxes( design_gradient_interpolated, 0, 1 )
					design_gradient_interpolated = 0.5 * ( design_gradient_interpolated + transpose_design_gradient_interpolated )
				design.gradient = copy.deepcopy(design_gradient_interpolated)	# This is BEFORE backpropagation
				
				logging.info("Completed Step 5: Adjoint Optimization Jobs and Gradient Computation.")
				# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


				logging.info("Beginning Step 6: Stepping Design Variable.")

				# Backpropagate the design_gradient through the various filters to pick up (per the chain rule) gradients from each filter
				# so the final product can be applied to the (0-1) scaled permittivity
				design_gradient_unfiltered = design.backpropagate( design.gradient )

				# Get current values of design variable before perturbation, for later comparison.
				last_design_variable = copy.deepcopy(design.get_design_variable())

				# Use the optimizer to determine what the next design variable should be
				self.optimizer.step(design, design_gradient_unfiltered)


				# if cfg.cv.optimizer_algorithm.lower() not in ['gradient descent']:
				# 	use_fixed_step_size = True
				# if use_fixed_step_size:
				# 	step_size = cfg.cv.fixed_step_size
				# else:
				# 	step_size = 0.01 # step_size_start


				# def scale_step_size(epoch_start_design_change_min, epoch_start_design_change_max, 
				# 					epoch_end_design_change_min, epoch_end_design_change_max):
				# 	'''Begin scaling of step size so that the design change stays within epoch_design_change limits in config.'''
				# 	return 3

				# if False: # cfg.cv.optimizer_algorithm.lower() in ['adam']:
				# 	adam_moments = design.step_adam(current_iter, -design_gradient,
				# 									adam_moments, step_size, cfg.cv.adam_betas)
				# else:
				# 	design.step(-design.gradient, step_size)
				# 	# self.optimizer.step(design, -design.gradient)
				

				#* Save out overall savefile, save out all save information as separate files as well.
				difference_design = np.abs(design.get_design_variable() - last_design_variable)
				logging.info(f'Device variable stepped by maximum of {np.max(difference_design)} and minimum of {np.min(difference_design)}.')
			
			
			# self.optimizer.run(self.device, self.sim, self.foms)
			self.optimizer.save_opt()
			self.iteration += 1


	# #! ---------------------------------------------------------------------------------------------------------

		

	# 		# TODO:
	# 		# For sim in sims:
	# 		#     call sim.run()
	# 		sim.save(f'test_sim_step_{self.iteration}')
	# 		sim.run()
			
	# 		# device should have access to it's necessary monitors in the simulations
	# 		fom = self.fom_func(foms)
	# 		self.fom_hist.append(fom)

	# 		# Likewise for gradient
	# 		gradient = self.grad_func(foms)

	# 		# Step with the gradient
	# 		self.param_hist.append(device.get_design_variable())
	# 		self.optimizer.step(self.device, gradient)
	## 		self.step(device, gradient)
	# 		self.iteration += 1

	# 		break


	# #! ---------------------------------------------------------------------------------------------------------

		# final_fom = self.optimizer.fom_hist[-1]
		# logging.info(f'Final FoM: {final_fom}')
		# final_params = self.optimizer.param_hist[-1]
		# logging.info(f'Final Parameters: {final_params}')
		# return final_fom, final_params
		

class Optimizer:
	"""Abstraction class for all optimizers."""
	# TODO: SHOULD NOT BE JUST GRADIENT BASED (e.g. genetic algorithms)

	def __init__( self, fom_func, grad_func, max_iter=100, start_from=0, folder='' ):
		"""Initialize an Optimizer object."""
		self.name = 'Optimizer'
		self.fom_hist = []
		self.param_hist = []
		self._callbacks = []
		self.iteration = start_from
		self.max_iter = max_iter
		self.fom_func = fom_func
		self.grad_func = grad_func
		self.folder = folder


	def add_callback(self, func):
		"""Register a callback function to call after each iteration."""
		self._callbacks.apend(func)

	def save_opt(self):
		'''Shelves vars() as a dictionary for pickling.'''
		utility.backup_all_vars(vars(self), os.path.join(self.folder, self.name))

	def load_opt(self, filename=None):
		'''Overwrites class variables with a previously shelved dictionary vars()
			Note: A toy device must be created first before it can then be overwritten.'''
		if filename is None:
			vars(self).update( utility.load_backup_vars( os.path.join(self.folder, self.name) ) )
		else:
			vars(self).update( utility.load_backup_vars( os.path.join(self.folder, filename) ) )

	def step(self, device, *args, **kwargs):
		"""Step forward one iteration in the optimization process."""
		grad = device.backpropagate(gradient)

		# x = some_func(x)

		# self.device.set_design_variable(x)

		# self.save_opt()
	
	def run(self, device):
		"""Run an optimization."""
		for callback in self._callbacks:
			callback()

class GDOptimizer(Optimizer):
	
	def __init__( self, step_size, **kwargs ):
		super().__init__(**kwargs)
		self.name = 'GradientDescentOptimizer'
		self.step_size = step_size

	def set_callback(self, func):
		return super().set_callback(func)

	def proposed_step(self, design, gradient):
		# Eq. S2 of Optica Paper Supplement, https://doi.org/10.1364/OPTICA.384228
		# [DEPRECATED] minus sign here because it was fed the negative of the gradient so (-1)*(-1) = +1
		#! Watch out for minus signs here and there
		proposed_design_variable = design.w[0] + np.multiply(self.step_size, gradient)
		proposed_design_variable = np.maximum(                             # Has to be between minimum_design_value and maximum_design_value
												np.minimum(proposed_design_variable, design.maximum_design_value),
												design.minimum_design_value)

		return proposed_design_variable

	def step(self, design, gradient):
	
		# Get current values of design variable before perturbation, for later comparison.
		last_design_variable = copy.deepcopy(design.get_design_variable())
		
		# Using the current value of the step size, step the design forward and examine the change in density.
		proposed_design_variable = self.proposed_step(design, gradient)
		difference = np.abs(proposed_design_variable - last_design_variable)

		# For the gradient-based optimization to work, the design cannot change by too much per iteration.
		# We have preset limits on the device set in config, but these limits depend heavily on the device dimensions and function.
		design_change_limits = utility.design_change_limits(self.iteration, cfg.cv.epoch_list)
		while np.max(difference) > design_change_limits[1]: # or np.min(difference) < design_change_limits[0]:
			logging.info('Modifying step size (Gradient Descent)...')
			if np.max(difference) < design_change_limits[0]:
				self.step_size *= 2
			if np.max(difference) > design_change_limits[1]:
				self.step_size /= 2

			# Using the NEW value of the step size, step the design forward and examine the change in density.
			proposed_design_variable = self.proposed_step(design, gradient)
			difference = np.abs(proposed_design_variable - last_design_variable)
			# If it is correct, it will break the while loop.

		# The step size is now finalized.
		design.step(self.proposed_step(design, gradient))
			
		

class AdamOptimizer(Optimizer):

	def __init__( self, step_size, betas=(0.9,0.999), eps=1e-8, moments=(0,0), **kwargs ):
		super().__init__(**kwargs)
		self.name = 'ADAMOptimizer'
		self.moments = moments
		self.step_size = step_size
		self.betas = betas
		self.eps = eps

	def set_callback(self, func):
		return super().set_callback(func)

	def step(self, device, gradient, *args, **kwargs):
		# super().step(device, gradient, *args, **kwargs)	#! Might need to comment this out ?

		# Get current values of design variable before perturbation, for later comparison.
		last_design_variable = copy.deepcopy(device.get_design_variable())

		# Take gradient step using Adam algorithm
		grad = device.backpropagate(gradient)

		b1, b2 = self.betas
		m = self.moments[0, ...]
		v = self.moments[1, ...]

		m = b1 * m + (1 - b1) * gradient
		v = b2 * v + (1 - b2) * gradient ** 2
		self.moments = np.array([m, v])

		m_hat = m / (1 - b1 ** (self.iteration + 1))
		v_hat = v / (1 - b2 ** (self.iteration + 1))
		w_hat = device.w[0] + self.step_size * m_hat / np.sqrt(v_hat + self.eps)

		# Apply changes
		device.set_design_variable(np.maximum(np.minimum(w_hat, 1), 0))


	def run(self, device, sim, foms):
		while self.iteration < self.max_iter:
			for callback in self._callbacks:
				callback()

			# TODO:
			# For sim in sims:
			#     call sim.run()
			sim.save(f'test_sim_step_{self.iteration}')
			sim.run()
			
			# device should have access to it's necessary monitors in the simulations
			fom = self.fom_func(foms)
			self.fom_hist.append(fom)

			# Likewise for gradient
			gradient = self.grad_func(foms)

			# Step with the gradient
			self.param_hist.append(device.get_design_variable())
			self.step(device, gradient)
			self.iteration += 1

			break