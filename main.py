# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Main file for the adjoint source optimization.
# Accesses config file via: configs/xxx.py
# Sets up instances of all the classes in utils
# Sets up Lumerical simulation
# etc.

#* Import all modules

# Math and Autograd
import numpy as np
import scipy
# from scipy.ndimage import gaussian_filter
# from scipy import optimize as scipy_optimize
from scipy import interpolate
# import pandas as pd

# System, Plotting, etc.
import os
import shelve
import gc
import sys
import logging
import time
from datetime import datetime
import copy
import shutil
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sn

# Custom Classes and Imports
import configs.yaml_to_params as cfg
# Gets all parameters from config file - store all those variables within the namespace. Editing cfg edits it for all modules accessing it
# See https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
from utils import utility
from utils import Device
from utils import FoM
from utils import Source
from utils import manager

#* Are we running it locally or on SLURM or on AWS or?
workload_manager = cfg.workload_manager
# Are we running using Lumerical or ceviche or fdtd-z or SPINS or?
simulator = cfg.simulator
# Depending on which one we choose here - different converter files should be run, 
# e.g. lum_utils.py for Lumerical and slurm_handler.py for SLURM

#* Start logging - Logger is already initialized through the call to yaml_to_params.py
logging.info(f'Starting up optimization file: {os.path.basename(__file__)}')
logging.info("All modules loaded.")
logging.debug(f'Current working directory: {os.getcwd()}')

def create_folder_structure(python_src_directory, projects_directory_location, running_on_local_machine):
	global DATA_FOLDER, SAVED_SCRIPTS_FOLDER, OPTIMIZATION_INFO_FOLDER, OPTIMIZATION_PLOTS_FOLDER
	global DEBUG_COMPLETED_JOBS_FOLDER, PULL_COMPLETED_JOBS_FOLDER, EVALUATION_FOLDER, EVALUATION_CONFIG_FOLDER, EVALUATION_UTILS_FOLDER
	
	#* Output / Save Paths 
	DATA_FOLDER = projects_directory_location
	SAVED_SCRIPTS_FOLDER = os.path.join(DATA_FOLDER, 'saved_scripts')
	OPTIMIZATION_INFO_FOLDER = os.path.join(DATA_FOLDER, 'opt_info')
	OPTIMIZATION_PLOTS_FOLDER = os.path.join(OPTIMIZATION_INFO_FOLDER, 'plots')
	DEBUG_COMPLETED_JOBS_FOLDER = 'ares_test_dev'
	PULL_COMPLETED_JOBS_FOLDER = DATA_FOLDER
	if running_on_local_machine:
		PULL_COMPLETED_JOBS_FOLDER = DEBUG_COMPLETED_JOBS_FOLDER

	EVALUATION_FOLDER = os.path.join(DATA_FOLDER, 'lumproc_' + os.path.basename(python_src_directory))
	EVALUATION_CONFIG_FOLDER = os.path.join(EVALUATION_FOLDER, 'configs')
	EVALUATION_UTILS_FOLDER = os.path.join(EVALUATION_FOLDER, 'utils')

	# parameters['MODEL_PATH'] = os.path.join(DATA_FOLDER, 'model.pth')
	# parameters['OPTIMIZER_PATH'] = os.path.join(DATA_FOLDER, 'optimizer.pth')


	# #* Save out the various files that exist right before the optimization runs for debugging purposes.
	# # If these files have changed significantly, the optimization should be re-run to compare to anything new.

	if not os.path.isdir( SAVED_SCRIPTS_FOLDER ):
		os.mkdir( SAVED_SCRIPTS_FOLDER )
	if not os.path.isdir( OPTIMIZATION_INFO_FOLDER ):
		os.mkdir( OPTIMIZATION_INFO_FOLDER )
	if not os.path.isdir( OPTIMIZATION_PLOTS_FOLDER ):
		os.mkdir( OPTIMIZATION_PLOTS_FOLDER )

	try:
		shutil.copy2( python_src_directory + "/slurm_vis10lyr.sh", SAVED_SCRIPTS_FOLDER + "/slurm_vis10lyr.sh" )
	except Exception as ex:
		pass
	# shutil.copy2( cfg.python_src_directory + "/SonyBayerFilterOptimization.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterOptimization.py" )
	# shutil.copy2( os.path.join(python_src_directory, yaml_filename), 
	# 			 os.path.join(SAVED_SCRIPTS_FOLDER, os.path.basename(yaml_filename)) )
	# shutil.copy2( python_src_directory + "/configs/SonyBayerFilterParameters.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterParameters.py" )
	# # TODO: et cetera... might have to save out various scripts from each folder

	#  Create convenient folder for evaluation code
	if os.path.exists(EVALUATION_CONFIG_FOLDER):
		shutil.rmtree(EVALUATION_CONFIG_FOLDER)
	shutil.copytree(os.path.join(python_src_directory, "configs"), EVALUATION_CONFIG_FOLDER)

	if os.path.exists(EVALUATION_UTILS_FOLDER):
		shutil.rmtree(EVALUATION_UTILS_FOLDER)
	shutil.copytree(os.path.join(python_src_directory, "utils"), EVALUATION_UTILS_FOLDER)

	#shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/plotter.py"), EVALUATION_UTILS_FOLDER + "/plotter.py" )

create_folder_structure(cfg.python_src_directory, cfg.projects_directory_location, cfg.running_on_local_machine)

#* Utility Functions that must be located in this module




#* Step 0: Set up simulation conditions and environment.
# i.e. current sources, boundary conditions, supporting structures, surrounding regions.
# Also any other necessary editing of the Lumerical environment and objects.
logging.info("Beginning Step 0: Lumerical Environment Setup")

simulator.connect()
# simulation_environment = cfg.simulation_environment		# Pointer to a JSON/YAML file describing each simulation object
from utils import EnvironmentConstructor as env_constr		# todo: use Python script for now to replace the above. Replace EnvironmentConstructor with whichever file is appropriate.
# Get simulator module to partition the simulation region accordingly
entire_simulation = env_constr.construct_sim_objects()
simulations = simulator.partition_objs(entire_simulation, (1,))
# The function partition_objs() can also take a similar dictionary as a "savefile" argument to reconstruct.
# simulations is a list of Simulations objects. 
# Each Simulation object should contain a dictionary from which the corresponding simulation file can be regenerated
# Multiple simulations may be created here due to the need for large-area simulation segmentation, or genetic optimizations

# Construct each partitioned simulation region according to instructions
for simulation in simulations:
	simulator.open()
	simulation = simulator.import_obj(simulation, create_object_=True)
	simulation['SIM_INFO']['path'] = DATA_FOLDER
	simulator.disable_all_sources(simulation)		# disabling all sources sets it up for creating forward and adjoint-sourced sims later
	simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])

logging.info("Completed Step 0: Lumerical environment setup complete.")
# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
utility.write_dict_to_file(simulations, 'simulations', folder=OPTIMIZATION_INFO_FOLDER)


#* Step 1: Design Region(s) + Constraints
# e.g. feature sizes, bridging/voids, permittivity constraints, corresponding E-field monitor regions
logging.info("Beginning Step 1: Design Regions Setup")

device_objs = env_constr.construct_device_regions()
# todo: should we keep devices and subdevices separate? Maybe we can just deal with subdevices as the things we assign gradients to.
# Since it's more or less general, it should be simple to swap things around. Or combine things back later on.
design_objs = simulator.assign_devices_to_simulations(simulations, device_objs)
#! Must keep track of which devices correspond to which regions and vice versa.
# Check coordinates of device region. Does it fit in this simulation? If yes, split into subdevices (designs) and store in this simulation.
#? What is the best method to keep track of maps? Try: https://stackoverflow.com/a/21894086
design_to_device_map, design_to_simulation_map = simulator.construct_sim_device_hashtable(simulations, design_objs)
device_to_simulation_map = simulator.get_device_to_simulation_map(design_to_device_map, design_to_simulation_map)
simulation_to_device_map = simulator.get_simulation_to_device_map(design_to_device_map, design_to_simulation_map)

# For each subdevice (design) region, create a field_shape variable to store the E- and H-fields for adjoint calculation.
field_shapes = []
# Open simulation files and create subdevice (design) regions as well as attached meshes and monitors.
for design_obj in design_objs:
	simulation = simulations[design_obj['DEVICE_INFO']['simulation']]	# Access simulation region containing design region.
	simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))
	design_obj = simulator.import_obj(design_obj, create_object_=True)
	simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])

	#! Shape Check! Check the mesh for the E-field monitors and adjust each field_shape variable for each design to match.
	# Instead of a lookup table approach, probably should just directly pull from the simulator - we can always interpolate anyway.
	# Pull the array shape of the design region's corresponding index monitor, squeezing as we do so.
	field_shape = np.squeeze(simulator.fdtd_hook.getresult(design_obj['design_index_monitor']['name'], 'index preview')['index_x']).shape
	field_shapes.append(field_shape)


# Create instances of Device class objects to store permittivity values and attached filters.
designs = [None] * len(design_objs)
for num_design, design_obj in enumerate(design_objs):
	
	voxel_array_size = design_obj['DEVICE_INFO']['size_voxels']
	feature_dimensions = design_obj['DEVICE_INFO']['feature_dimensions']
	permittivity_constraints = design_obj['DEVICE_INFO']['permittivity_constraints']
	region_coordinates = design_obj['DEVICE_INFO']['coordinates']
	design_name = design_obj['DEVICE_INFO']['name']
	design_obj['DEVICE_INFO']['path'] = os.path.abspath(OPTIMIZATION_INFO_FOLDER)
	design_save_path = design_obj['DEVICE_INFO']['path']
	# other_arguments
	
	designs[num_design] = Device.Device(
		voxel_array_size,
		feature_dimensions,
		permittivity_constraints,
		region_coordinates,
		name=design_name,
		folder=design_save_path,
		init_seed='random'			#! Random is just for testing
		#other_arguments
	)


logging.info("Completed Step 1: Design regions setup complete.")
# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
utility.write_dict_to_file(device_objs, 'device_objs', folder=OPTIMIZATION_INFO_FOLDER, serialization='npy')
utility.write_dict_to_file(design_objs, 'design_objs', folder=OPTIMIZATION_INFO_FOLDER, serialization='npy')




#* Step 2: Set up figure of merit. Determine monitors from which E-fields, Poynting fields, and transmission data will be drawn.
# Handle all processing of weighting and functions that go into the figure of merit

fom_dict = env_constr.fom_dict
weights = env_constr.weights

forward_sources = []
adjoint_sources = []
for simulation in simulations:
	# We create fwd_src, adj_src as Source objects, then link them together using an "FoM" class, 
 	# which can then store E_fwd and E_adj gradients. We can also then just return a gradient by doing FoM.E_fwd * FoM.E_adj
	# todo: monitor might also need to be their own class.

	# Add the forward sources
	fwd_srcs, fwd_mons = env_constr.construct_fwd_srcs_and_monitors(simulation)
	for idx, fwd_src in enumerate(fwd_srcs):
		forward_sources.append(Source.Fwd_Src(fwd_src, monitor_object=fwd_mons[fwd_src['attached_monitor']], sim_id=simulation['SIM_INFO']['sim_id']))

	# Add dipole or adjoint sources, and their monitors.
	# Adjoint monitors and what data to obtain should be included in the FoM class / indiv_fom.
	adj_srcs, adj_mons = env_constr.construct_adj_srcs_and_monitors(simulation)
	for idx, adj_src in enumerate(adj_srcs):
		adjoint_sources.append(Source.Adj_Src(adj_src, monitor_object=adj_mons[adj_src['attached_monitor']], sim_id=simulation['SIM_INFO']['sim_id']))

positive_foms = []
restricted_foms = []	# or alternatively, negative_foms
for fom in fom_dict:
	
	#! Add dipole or adjoint sources, and their monitors.
	# Adjoint monitors and what data to obtain should be included in indiv_fom.
	# todo: alternatively, create fwd_src, adj_src, then link them together using an "FoM" class, which can then store E_fwd and E_adj gradients?
	# todo: might be a more elegant solution. We can also then just return a gradient by doing FoM.E_fwd * FoM.E_adj
	# todo: source and monitor might also need to be their own class.

	fom_fwd_srcs = [forward_sources[k] for k in fom['fwd']]
	fom_adj_srcs = [adjoint_sources[k] for k in fom['adj']]

	if not fom['freq_idx_restricted_opt']:	# empty list
		positive_foms.append(FoM.BayerFilter_FoM(fom_fwd_srcs, fom_adj_srcs, 'TE',
									   cfg.pv.lambda_values_um, fom['freq_idx_opt'], fom['freq_idx_restricted_opt']))
	else:
		restricted_foms.append(FoM.BayerFilter_FoM(fom_fwd_srcs, fom_adj_srcs, 'TE',
									   cfg.pv.lambda_values_um, fom['freq_idx_opt'], fom['freq_idx_restricted_opt']))

indiv_foms = positive_foms + restricted_foms	# It's important to keep track of which FoMs are positive and negative.


logging.info("Completed Step 2: Figure of Merit setup complete.")
# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


#*
#* AT THIS POINT THE OPTIMIZATION LOOP BEGINS
#*

#
#* Set up some numpy arrays to handle all the data used to track the FoMs and progress of the simulation.
#

# # By iteration
figure_of_merit_evolution = np.zeros((cfg.cv.total_num_iterations))
# pdaf_transmission_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
# step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
binarization_evolution = np.zeros((cfg.cv.total_num_iterations))
# average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
# max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

# By iteration, focal area, polarization, wavelength: 
# Create a dictionary of FoMs. Each key corresponds to a value: an array storing data of a specific FoM,
# for each epoch, iteration, quadrant, polarization, and wavelength.
# TODO: Consider using xarray instead of numpy arrays in order to have dimension labels for each axis.
fom_evolution = {}
for fom_type in cfg.cv.fom_types:
	fom_evolution[fom_type] = np.zeros((cfg.cv.total_num_iterations, 
										len(fom_dict), cfg.pv.num_design_frequency_points))
# for fom_type in fom_types:	# todo: delete this part when all is finished
# 	fom_evolution[fom_type] = np.zeros((num_epochs, num_iterations_per_epoch, 
# 										num_focal_spots, len(xy_names), num_design_frequency_points))
# The main FoM being used for optimization is indicated in the config.

# # NOTE: Turned this off because the memory and savefile space of gradient_evolution are too large.
# # if start_from_step != 0:	# Repeat the definition of field_shape here just because it's not loaded in if we start from a debug point
# # 	field_shape = [device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_vertical]
# # gradient_evolution = np.zeros((num_epochs, num_iterations_per_epoch, *field_shape,
# # 								num_focal_spots, len(xy_names), num_design_frequency_points))

#
#* Handle Restart
#
restart=False
if cfg.cv.restart_iter:
	restart = True

if restart:
	
	# Get current iteration and determines which epoch we're in, according to the epoch list in config.
	current_epoch = utility.partition_iterations(cfg.cv.restart_iter, cfg.cv.epoch_list)
	logging.info(f'Restarting optimization from Iteration {cfg.cv.restart_iter} [Epoch {current_epoch}].')
	
	# todo: convert the following code
	
	# cur_design_variable = np.load(OPTIMIZATION_INFO_FOLDER + "/cur_design_variable.npy" )
	# cur_design_variable = utility.binarize(cur_design_variable)
	# designs[0].set_design_variable(cur_design_variable[0:24,0:24,:])
	# designs[0].save_device()

	# reload_design_variable = cur_design_variable.copy()
	# if evaluate_bordered_extended:
	# 	design_var_reload_width = cur_design_variable.shape[ 0 ]
	# 	design_var_width = bayer_filter.w[0].shape[ 0 ]

	# 	pad_2x_width_reload = design_var_width - design_var_reload_width
	# 	assert ( pad_2x_width_reload > 0 ) and ( ( pad_2x_width_reload % 2 ) == 0 ), 'Expected the width difference to be positive and even!'
	# 	pad_width_reload = pad_2x_width_reload // 2
  
	# 	reload_design_variable = np.pad( cur_design_variable, ( ( pad_width_reload, pad_width_reload ), ( pad_width_reload, pad_width_reload ), ( 0, 0 ) ), mode='constant' )
 
	# # Re-initialize bayer_filter instance
	for num_design, design in enumerate(designs):
		design.load_device()
  
		# # Load cur_design_variable if they have it
		# cur_design_variable = np.load(OPTIMIZATION_INFO_FOLDER + "/cur_design_variable.npy" )[0:24, 0:24, :]
		# design.set_design_variable(cur_design_variable)
  
		# Regenerate permittivity based on the most recent design, assuming epoch 0		
		# NOTE: This gets corrected later in the optimization loop (Step 1)
		design.update_filters(current_epoch)
		design.update_density()

	# # # Other necessary savestate data
	# figure_of_merit_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/figure_of_merit.npy")
	# # gradient_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/gradient_by_focal_pol_wavelength.npy")		# Took out feature because it consumes too much memory
	# for fom_type in fom_types:
	# 	fom_evolution[fom_type] = np.load(OPTIMIZATION_INFO_FOLDER + f"/{fom_type}_by_focal_pol_wavelength.npy")
	# step_size_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/step_size_evolution.npy")
	# average_design_variable_change_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/average_design_change_evolution.npy")
	# max_design_variable_change_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/max_design_change_evolution.npy")
	# adam_moments = np.load(OPTIMIZATION_INFO_FOLDER + "/adam_moments.npy")


	# if restart_epoch == 0 and restart_iter == 1:
	# 	# Redeclare everything if starting from scratch (but using a certain design as the seed)
	# 	figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
	# 	pdaf_transmission_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
	# 	step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
	# 	average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
	# 	max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
	# 	fom_evolution = {}
	# 	for fom_type in fom_types:
	# 		fom_evolution[fom_type] = np.zeros((num_epochs, num_iterations_per_epoch, 
	# 											num_focal_spots, len(xy_names), num_design_frequency_points))
	# 	# The main FoM being used for optimization is indicated in the config.

	# 	# [OPTIONAL]: Use binarized density variable instead of intermediate
	# 	bayer_filter.update_filters( num_epochs - 1 )
	# 	bayer_filter.update_permittivity()
	# 	cur_density = bayer_filter.get_permittivity()
	# 	cur_design_variable = cur_density
	# 	bayer_filter.w[0] = cur_design_variable
	
	# # Set everything ahead of the restart epoch and iteration to zero.
	# for rs_epoch in range(restart_epoch, num_epochs):
	# 	if rs_epoch == restart_epoch:
	# 		rst_iter = restart_iter
	# 	else:	rst_iter = 0

	# 	for rs_iter in range(rst_iter, num_iterations_per_epoch):
	# 		figure_of_merit_evolution[rs_epoch, rs_iter] = 0
	# 		pdaf_transmission_evolution[rs_epoch, rs_iter] = 0
	# 		step_size_evolution[rs_epoch, rs_iter] = 0
	# 		average_design_variable_change_evolution[rs_epoch, rs_iter] = 0
	# 		max_design_variable_change_evolution[rs_epoch, rs_iter] = 0
	# 		for fom_type in fom_types:
	# 			fom_evolution[fom_type][rs_epoch, rs_iter] = np.zeros((num_focal_spots, len(xy_names), num_design_frequency_points))


#
#* Run the optimization
#

# Initial Optimization Parameters
step_size_start = 1.
# Choose the dispersion model for this optimization
# dispersion_model = DevicePermittivityModel()	# TODO
# ADAM Optimizer Parameters
adam_moments = np.zeros([2] + list(voxel_array_size))

# Control/Restart Parameters
start_iter = cfg.cv.restart_iter
for current_iter in range(start_iter, cfg.cv.total_num_iterations):

	# Get current iteration and determines which epoch we're in, according to the epoch list in config.
	current_epoch = utility.partition_iterations(current_iter, cfg.cv.epoch_list)
	# current_subepoch = iteration - cfg.cv.epoch_list[current_epoch]		# We don't count things like this anymore
	logging.info(f'Working on Iteration {current_iter} [Epoch {current_epoch}].')

	# Each epoch the filters get stronger and so the permittivity must be passed through the new filters
	# todo: test once you get a filter online
	for num_design, design in enumerate(designs):
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

	if cfg.cv.start_from_step == 0 or cfg.cv.start_from_step == 3:
		if cfg.cv.start_from_step == 3:
			globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))
		logging.info("Beginning Step 3: Setting up Forward and Adjoint Jobs")
		
		for num_design, design in enumerate(designs):
			design_obj = design_objs[num_design]								# Access corresponding dictionary
			simulation = simulations[design_obj['DEVICE_INFO']['simulation']]	# Access simulation region containing design region.
			simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))

			# Disable all sources
			simulator.switch_to_layout()
			simulator.disable_all_sources(simulation)		# disabling all sources sets it up for creating forward and adjoint-sourced sims later
			
			#* Import cur_index into design regions
			# Get current density of bayer filter after being passed through current extant filters.
			# TODO: Might have to rewrite this to be performed on the entire device and NOT the subdesign / device. In which case
			# todo: partitioning must happen after reinterpolating and converting to index.
			cur_density = design.get_density()							# Here cur_density will have values between 0 and 1
			# Reinterpolate
			if cfg.cv.reinterpolate_permittivity:
				cur_density_import = np.repeat( np.repeat( np.repeat( cur_density, 
															int(np.ceil(cfg.cv.reinterpolate_permittivity_factor)), axis=0 ),
														int(np.ceil(cfg.cv.reinterpolate_permittivity_factor)), axis=1 ),
													int(np.ceil(cfg.cv.reinterpolate_permittivity_factor)), axis=2 )

				# Create the axis vectors for the reinterpolated 3D density region
				design_region_reinterpolate_x = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, cfg.pv.device_voxels_lateral_bordered * cfg.cv.reinterpolate_permittivity_factor)
				design_region_reinterpolate_y = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, cfg.pv.device_voxels_lateral_bordered * cfg.cv.reinterpolate_permittivity_factor)
				design_region_reinterpolate_z = 1e-6 * np.linspace(cfg.cv.device_vertical_minimum_um, cfg.pv.device_vertical_maximum_um, cfg.pv.device_voxels_vertical * cfg.cv.reinterpolate_permittivity_factor)
				# Create the 3D array for the 3D density region for final import
				design_region_import_x = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, 300)
				design_region_import_y = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, 300)
				design_region_import_z = 1e-6 * np.linspace(cfg.cv.device_vertical_minimum_um, cfg.pv.device_vertical_maximum_um, 306)
				design_region_import = np.array( np.meshgrid(
							design_region_import_x, design_region_import_y, design_region_import_z,
						indexing='ij')
					).transpose((1,2,3,0))
	
				# Perform reinterpolation so as to fit into Lumerical mesh.
				# NOTE: The imported array does not need to have the same size as the Lumerical mesh. Lumerical will perform its own reinterpolation (params. of which are unclear.)
				cur_density_import_interp = interpolate.interpn( ( design_region_reinterpolate_x, 
																	design_region_reinterpolate_y, 
																	design_region_reinterpolate_z ), 
										cur_density_import, design_region_import, 
										method='linear' )
	
			else:	
				cur_density_import_interp = copy.deepcopy(cur_density)
				design_region_import_x = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, field_shape[0])
				design_region_import_y = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, field_shape[1])
				design_region_import_z = 1e-6 * np.linspace(cfg.cv.device_vertical_minimum_um, cfg.pv.device_vertical_maximum_um, field_shape[2])
				design_region_import = np.array( np.meshgrid(
							design_region_import_x, design_region_import_y, design_region_import_z,
						indexing='ij')
					).transpose((1,2,3,0))
   
			# # IF YOU WANT TO BINARIZE BEFORE IMPORTING:
			# cur_density_import_interp = utility.binarize(cur_density_import_interp)

			for dispersive_range_idx in range(0, 1):		# todo: Add dispersion. Account for dispersion by using the dispersion_model
				# dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
				dispersive_max_permittivity = cfg.pv.max_device_permittivity
				dispersive_max_index = utility.index_from_permittivity( dispersive_max_permittivity )

				# Convert device to permittivity and then index
				cur_permittivity = Device.density_to_permittivity(cur_density_import_interp, cfg.pv.min_device_permittivity, dispersive_max_permittivity)
				cur_index = utility.index_from_permittivity(cur_permittivity)
				# Calculate material % and binarization level, store away
				# todo: redo this section once you get sigmoid filters up and can start counting materials
				logging.info(f'TiO2% is {100 * np.count_nonzero(cur_index > cfg.cv.min_device_index) / cur_index.size}%.')		
				# logging.info(f'Binarization is {100 * np.sum(np.abs(cur_density-0.5))/(cur_density.size*0.5)}%.')
				logging.info(f'Binarization is {100 * utility.compute_binarization(cur_density)}%.')
				binarization_evolution[current_iter] = 100 * utility.compute_binarization(cur_density)
				
				#?:	maybe todo: cut cur_index by region
				# Import to simulation
				simulator.import_nk_material(design_obj, cur_index,
								design_region_import_x, design_region_import_y, design_region_import_z)
				# Disable device index monitor to save memory
				design_obj['design_index_monitor']['enabled'] = False
				design_obj['design_index_monitor'] = simulator.fdtd_update_object( 
												design_obj['design_index_monitor'], create_object=False )

				# Save template copy out
				simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])
		#### End for loop

		#* Start creating array of jobs with different sources
		# Run all simulations in parallel

		for simulation in simulations:
			# todo: store fwd_src_arr, adj_src_arr, tempfile_fwd, tempfile_adj, tempfile_to_source_map in each simulation object

			# FORWARD SIMS
			fwd_src_arr, tempfile_fwd_name_arr = manager.create_tempfiles_fwd_parallel(indiv_foms, simulator, simulation)
			# NOTE: indiv_foms, simulation are also amended accordingly
			# NOTE: indiv_foms have tempfile names as indices at this point

			# All forward simulations are queued. Return to the original file (just to be safe).
			simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))
			simulator.switch_to_layout()
			simulator.disable_all_sources(simulation)

			# ADJOINT SIMS
			active_adj_src_arr, inactive_adj_src_arr, tempfile_adj_name_arr = \
								manager.create_tempfiles_adj_parallel(indiv_foms, simulator, simulation)
			# NOTE: indiv_foms, simulation are also amended accordingly
			# NOTE: indiv_foms have tempfile names as indices at this point

			# All adjoint simulations are queued. Return to the original file (just to be safe).
			simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))
			simulator.switch_to_layout()
			simulator.disable_all_sources(simulation)


			# Now that the tempfile_fwd_name_arr is filled, and same for adj,
			# for each FoM object, we store the corresponding filename for easy access later.
			# We also create a map of the activated sources each file has.
			tempfile_to_source_map = {}
			for f in indiv_foms:
				tempfile_to_source_map[tempfile_fwd_name_arr[f.tempfile_fwd_name]] = f.tempfile_fwd_name
				tempfile_to_source_map[tempfile_adj_name_arr[f.tempfile_adj_name]] = f.tempfile_adj_name
				f.tempfile_fwd_name = tempfile_fwd_name_arr[f.tempfile_fwd_name]	# Replace the index with the actual string.
				f.tempfile_adj_name = tempfile_adj_name_arr[f.tempfile_adj_name]	# Replace the index with the actual string.
			

			utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
			utility.write_dict_to_file(tempfile_to_source_map, 'tempfile_to_source_map', folder=OPTIMIZATION_INFO_FOLDER, serialization='npy')

			#* Run all jobs in parallel
			job_names = tempfile_fwd_name_arr + tempfile_adj_name_arr
			manager.run_jobs(job_names, PULL_COMPLETED_JOBS_FOLDER, workload_manager, simulator)

		logging.info("Completed Step 3: Forward and Adjoint Simulations complete.")
		utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
		utility.write_dict_to_file(tempfile_to_source_map, 'tempfile_to_source_map', folder=OPTIMIZATION_INFO_FOLDER, serialization='npy')


	#
	# Step 4: Compute the figure of merit
	#

	if cfg.cv.start_from_step == 0 or cfg.cv.start_from_step == 4:
		if cfg.cv.start_from_step == 4:
			globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))
		logging.info("Beginning Step 4: Computing Figure of Merit")

		for simulation in simulations:
			design_list = simulation_to_device_map[simulation['SIM_INFO']['sim_id']]	# this is a list of indices: designs present in the sim

			for tempfile_fwd_name in tempfile_fwd_name_arr:
				simulator.open(os.path.join(PULL_COMPLETED_JOBS_FOLDER, tempfile_fwd_name))

				for num_design in design_list:
					design = designs[num_design]
					# forward_e_fields = np.zeros([len(tempfile_fwd_name_arr)] + \
					# 							[3] + \
					# 							list(field_shapes[num_design].shape) + [len(cfg.pv.lambda_values_um)]
					# 						)	# Each tempfile stores E-fields of shape (3, nx, ny, nz, nλ)
	 
					for f_idx, f in enumerate(indiv_foms):
						if tempfile_fwd_name in f.tempfile_fwd_name:	# todo: might need to account for file extensions
							# todo: rewrite this to hook to whichever device the FoM is tied to
							print(f'FoM property has length {len(f.fom)}')
	
							f.design_fwd_fields = \
								simulator.get_efield('design_efield_monitor')[..., f.freq_index_opt + f.freq_index_restricted_opt]
							logging.info(f'Accessed design E-field monitor. NOTE: Field shape obtained is: {f.design_fwd_fields.shape}')
							# Store FoM value and Restricted FoM value (possibly not the true FoM being used for gradient)
							f.compute_fom(simulator)
	
							# Store fom, restricted fom, true fom
							fom_evolution['transmission'] = np.squeeze(f.fom)
							# todo: add line for restricted fom
							fom_evolution[cfg.cv.optimization_fom][current_iter, f_idx, :] = np.squeeze(f.true_fom)
	
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

			# todo: sort this out
			# NOTE: Reminder that fom_evolution is a dictionary with keys: fom_types
			# and values being numpy arrays of shape (epochs, iterations, focal_areas, polarizations, wavelengths)
			list_overall_figure_of_merit = env_constr.overall_figure_of_merit(indiv_foms, weights)		# Gives a wavelength vector
			figure_of_merit_evolution[current_iter] = np.sum(list_overall_figure_of_merit)			# Sum over wavelength

			
			# # TODO: Save out variables that need saving for restart and debug purposes
			np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/figure_of_merit.npy"), figure_of_merit_evolution)
			np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/fom_by_focal_pol_wavelength.npy"), fom_evolution[cfg.cv.optimization_fom])
			for fom_type in cfg.cv.fom_types:
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + f"/{fom_type}_by_focal_pol_wavelength.npy"), fom_evolution[fom_type])


			# # TODO: Plot key information such as Figure of Merit evolution for easy visualization and checking in the middle of optimizations
			# plotter.plot_fom_trace(figure_of_merit_evolution, OPTIMIZATION_PLOTS_FOLDER)
			# plotter.plot_quadrant_transmission_trace(fom_evolution['transmission'], OPTIMIZATION_PLOTS_FOLDER)
			# plotter.plot_individual_quadrant_transmission(quadrant_transmission_data, lambda_values_um, OPTIMIZATION_PLOTS_FOLDER,
			# 										epoch, iteration=30)	# continuously produces only one plot per epoch to save space
			# plotter.plot_overall_transmission_trace(fom_evolution['transmission'], OPTIMIZATION_PLOTS_FOLDER)
			# plotter.visualize_device(cur_index, OPTIMIZATION_PLOTS_FOLDER)
			# # plotter.plot_moments(adam_moments, OPTIMIZATION_PLOTS_FOLDER)
			# # plotter.plot_step_size(adam_moments, OPTIMIZATION_PLOTS_FOLDER)

			# #* Here is where we would start plotting the loss landscape. Probably should be accessed by a separate class...
			# # Or we could move it to the device step part
			# loss_landscape_mapper = LossLandscapeMapper.LossLandscapeMapper(simulations, devices)


			logging.info("Completed Step 4: Computed Figure of Merit.")
			utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


	#
	# Step 5: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
	# gradients for x- and y-polarized forward sources.
	#

	# if cfg.cv.start_from_step == 0 or cfg.cv.start_from_step == 5:
	# 	if cfg.cv.start_from_step == 5:
			globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))
			logging.info("Beginning Step 5: Adjoint Optimization Jobs")

			for tempfile_adj_name in tempfile_adj_name_arr:
				simulator.open(os.path.join(PULL_COMPLETED_JOBS_FOLDER, tempfile_adj_name))

				for num_design in design_list:
					design = designs[num_design]
	
					# Initialize arrays to store the gradients of each polarization, each the shape of the voxel mesh
					# This array therefore has shape (polarizations, mesh_lateral, mesh_lateral, mesh_vertical)
					xy_polarized_gradients = [ np.zeros(field_shape, dtype=np.complex128), np.zeros(field_shape, dtype=np.complex128) ]

					
					#* Process the various figures of merit for performance weighting.
					# Copy everything so as not to disturb the original data
					process_fom_evolution = copy.deepcopy(fom_evolution)
	 
					for f in indiv_foms:
						logging.debug(f'tempfile_adj_name is {tempfile_adj_name}; FoM has the tempfile {f.tempfile_adj_name}')
						if tempfile_adj_name in f.tempfile_adj_name:	# todo: might need to account for file extensions
							# todo: rewrite this to hook to whichever device the FoM is tied to
							f.design_adj_fields = \
								simulator.get_efield('design_efield_monitor')[..., f.freq_index_opt + f.freq_index_restricted_opt]
							logging.info('Accessed design E-field monitor.')
 
							#* Compute gradient
							f.compute_gradient()
						gc.collect()


			list_overall_adjoint_gradient = env_constr.overall_adjoint_gradient( indiv_foms, weights )	# Gives a wavelength vector
			design_gradient = np.sum(list_overall_adjoint_gradient, -1)									# Sum over wavelength

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

			assert field_shape == get_grad_density.shape, f"The sizes of field_shape {field_shape} and gradient_density {get_grad_density.shape} are not the same."
			# xy_polarized_gradients[pol_name_to_idx] += get_grad_density
			# # TODO: The shape of this should be (epoch, iteration, nx, ny, nz, n_adj, n_pol, nλ) and right now it's missing nx,ny,nz
			# # todo: But also if we put that in, then the size will rapidly be too big
			# # gradient_evolution[epoch, iteration, :,:,:,
			# # 					adj_src_idx, xy_idx,
			# # 					spectral_indices[0] + spectral_idx] += get_grad_density

			# Get the full design gradient by summing the x,y polarization components # todo: Do we need to consider polarization??
			# todo: design_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
			# Project / interpolate the design_gradient, the values of which we have at each (mesh) voxel point, and obtain it at each (geometry) voxel point
			gradient_region_x = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, cfg.pv.device_voxels_simulation_mesh_lateral_bordered)
			gradient_region_y = 1e-6 * np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, cfg.pv.device_voxels_simulation_mesh_lateral_bordered)
			gradient_region_z = 1e-6 * np.linspace(cfg.cv.device_vertical_minimum_um, cfg.pv.device_vertical_maximum_um, cfg.pv.device_voxels_simulation_mesh_vertical)			
			design_region_geometry = np.array(np.meshgrid(1e-6*design.coords['x'], 1e-6*design.coords['y'], 1e-6*design.coords['z'], indexing='ij')
								).transpose((1,2,3,0))
			design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), design_gradient, design_region_geometry, method='linear' )
			# design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), device_gradient, bayer_filter_region, method='linear' )

			# Each device needs to remember its gradient!
			# todo: refine this
			design.gradient = design_gradient_interpolated

			logging.info("Completed Step 5: Adjoint Optimization Jobs and Gradient Computation.")
			utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


	#
	# Step 6: Step the design variable.
	#

	if cfg.cv.start_from_step == 0 or cfg.cv.start_from_step == 6:
		if cfg.cv.start_from_step == 6:
			globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))
			logging.info("Beginning Step 6: Stepping Design Variable.")
			# TODO: Rewrite this part to use optimizers / NLOpt. -----------------------------------	

			if cfg.cv.enforce_xy_gradient_symmetry:
				design_gradient_interpolated = 0.5 * ( design_gradient_interpolated + np.swapaxes( design_gradient_interpolated, 0, 1 ) )
			design_gradient = design_gradient_interpolated.copy()

			# Backpropagate the design_gradient through the various filters to pick up (per the chain rule) gradients from each filter
			# so the final product can be applied to the (0-1) scaled permittivity
			# design_gradient_unfiltered = design.backpropagate( design_gradient )

			cur_design_variable = design.get_design_variable()
			# Get current values of design variable before perturbation, for later comparison.
			last_design_variable = cur_design_variable.copy()

			if cfg.cv.optimizer_algorithm.lower() not in ['gradient descent']:
				use_fixed_step_size = True
			if use_fixed_step_size:
				step_size = cfg.cv.fixed_step_size
			else:
				step_size = step_size_start


			def scale_step_size(epoch_start_design_change_min, epoch_start_design_change_max, 
								epoch_end_design_change_min, epoch_end_design_change_max):
				'''Begin scaling of step size so that the design change stays within epoch_design_change limits in config.'''
				return 3

			# Feed proposed design gradient into bayer_filter, run it through filters and update permittivity
			# negative of the device gradient because the step() function has a - sign instead of a +
			# NOT the design gradient; we do not want to backpropagate twice!

			if False: # cfg.cv.optimizer_algorithm.lower() in ['adam']:
				adam_moments = design.step_adam(current_iter, -design_gradient,
												adam_moments, step_size, cfg.cv.adam_betas)
			else:
				design.step(-design_gradient, step_size)
			cur_design_variable = design.get_design_variable()
			logging.info(f'Current design sum: {np.sum(cur_design_variable)}, Previous design sum: {np.sum(last_design_variable)}')
			logging.info(f'Device variable stepped by total of {np.sum(cur_design_variable - last_design_variable)}')
			# Get permittivity after passing through all the filters of the device
			cur_design = design.get_permittivity()

			#* Save out overall savefile, save out all save information as separate files as well.

			logging.info("Completed Step 6: Stepped Design Variable.")
			# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
# globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))

logging.info('Reached end of code.')