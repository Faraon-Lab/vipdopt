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
from utils import Optimization

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
	simulation['SIM_INFO']['path'] = cfg.DATA_FOLDER
	simulator.disable_all_sources(simulation)		# disabling all sources sets it up for creating forward and adjoint-sourced sims later
	simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])

logging.info("Completed Step 0: Lumerical environment setup complete.")
# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
utility.write_dict_to_file(simulations, 'simulations', folder=cfg.OPTIMIZATION_INFO_FOLDER)


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
	# # Is this field shape equal to the number of simulation mesh voxels?
	correct_field_shape = (cfg.pv.device_voxels_simulation_mesh_lateral_bordered, cfg.pv.device_voxels_simulation_mesh_vertical)
	# assert field_shape == correct_field_shape, f"The sizes of field_shape {field_shape} should instead be {correct_field_shape}."
	field_shapes.append(field_shape)


# Create instances of Device class objects to store permittivity values and attached filters.
designs = [None] * len(design_objs)
for num_design, design_obj in enumerate(design_objs):
	
	voxel_array_size = design_obj['DEVICE_INFO']['size_voxels']
	feature_dimensions = design_obj['DEVICE_INFO']['feature_dimensions']
	permittivity_constraints = design_obj['DEVICE_INFO']['permittivity_constraints']
	region_coordinates = design_obj['DEVICE_INFO']['coordinates']
	design_name = design_obj['DEVICE_INFO']['name']
	design_obj['DEVICE_INFO']['path'] = os.path.abspath(cfg.OPTIMIZATION_INFO_FOLDER)
	design_save_path = design_obj['DEVICE_INFO']['path']
	# other_arguments
	
	designs[num_design] = Device.Device(
		voxel_array_size,
		feature_dimensions,
		permittivity_constraints,
		region_coordinates,
		name=design_name,
		folder=design_save_path,
		init_seed='uniform'			#! Random is just for testing
		#other_arguments
	)
	# todo: Consider putting filters in __init__ arguments? e.g. filters=[Sigmoid(0.5, 1.0), Sigmoid(0.5, 1.0)]

	designs[num_design].field_shape = field_shapes[num_design]

logging.info("Completed Step 1: Design regions setup complete.")
# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
utility.write_dict_to_file(device_objs, 'device_objs', folder=cfg.OPTIMIZATION_INFO_FOLDER, serialization='npy')
utility.write_dict_to_file(design_objs, 'design_objs', folder=cfg.OPTIMIZATION_INFO_FOLDER, serialization='npy')


#* Step 2: Set up figure of merit. Determine monitors from which E-fields, Poynting fields, and transmission data will be drawn.
# Handle all processing of weighting and functions that go into the figure of merit

fom_dict = env_constr.fom_dict
weights = env_constr.weights[..., np.newaxis] * env_constr.spectral_weights_by_fom
#! env_constr.overall_figure_of_merit() details the instructions for constructing the final FoM number

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



#* Step 3: Set up optimization and define Optimizer.
# Handle all hyperparameters and algorithms that are going into the Optimizer

# Initial Optimization Parameters
step_size_start = 1.
# Choose the dispersion model for this optimization
# dispersion_model = DevicePermittivityModel()	# TODO

# Define the Optimizer
adam_moments = np.zeros((2,) + tuple(voxel_array_size))		# ADAM Optimizer Parameters
optimizer = Optimization.AdamOptimizer(
	step_size=cfg.cv.fixed_step_size,
	betas=cfg.cv.adam_betas,
	moments=adam_moments,
	fom_func=env_constr.overall_figure_of_merit,
	grad_func=env_constr.overall_adjoint_gradient,
	max_iter=cfg.cv.total_num_iterations,
	folder=cfg.OPTIMIZATION_INFO_FOLDER
)
# optimizer = Optimization.GDOptimizer(
# 	step_size=cfg.cv.fixed_step_size,
# 	fom_func=env_constr.overall_figure_of_merit,
# 	grad_func=env_constr.overall_adjoint_gradient,
# 	max_iter=cfg.cv.total_num_iterations,
# 	folder=cfg.OPTIMIZATION_INFO_FOLDER
# )

# Define the Optimization
opt = Optimization.Optimization( simulations, indiv_foms, designs, design_objs, 
                                optimizer, env_constr.overall_figure_of_merit, env_constr.overall_adjoint_gradient,
                                weights, simulator )
#! NOTE: All edits made to these input objects within the Optimization class' methods, are also made to the same objects in this module.

#* Set up some numpy arrays to handle all the data used to track the FoMs and progress of the simulation.
# todo: store in Optimization or Optimizer?
opt.create_history(cfg.cv.fom_types, cfg.cv.total_num_iterations, cfg.pv.num_design_frequency_points)

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

	# Reload simulations and device / design dictionaries - must have same serialization as when utility.write_dict_to_file() was called
	simulations = utility.load_dict_from_file('simulations', folder=cfg.OPTIMIZATION_INFO_FOLDER)
	device_objs = utility.load_dict_from_file('device_objs', folder=cfg.OPTIMIZATION_INFO_FOLDER, serialization='npy')
	design_objs = utility.load_dict_from_file('design_objs', folder=cfg.OPTIMIZATION_INFO_FOLDER, serialization='npy')

	# # Re-initialize bayer_filter instance
	for num_design, design in enumerate(designs):
		design.load_device()
		# Only if we need to overwrite the designs:
		cur_design = np.load(cfg.OPTIMIZATION_INFO_FOLDER + "/cur_design.npy" )
		design.set_design_variable(cur_design)
		
		# Regenerate permittivity based on the most recent design
		# NOTE: This gets corrected later in the optimization loop (Step 1)
		design.update_filters(current_epoch)
		design.update_density()

	# Re-load optimizer history including parameters like moments
	opt.optimizer.load_opt()

	# # Other necessary savestate data - FoM history
	opt.figure_of_merit_evolution = np.load(cfg.OPTIMIZATION_INFO_FOLDER + "/figure_of_merit.npy")
	opt.binarization_evolution = np.load(cfg.OPTIMIZATION_INFO_FOLDER + "/binarization.npy")
	# # gradient_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/gradient_by_focal_pol_wavelength.npy")		# Took out feature because it consumes too much memory
	for fom_type in cfg.cv.fom_types:
		opt.fom_evolution[fom_type] = np.load(cfg.OPTIMIZATION_INFO_FOLDER + f"/{fom_type}_by_focal_pol_wavelength.npy")
	# step_size_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/step_size_evolution.npy")
	# average_design_variable_change_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/average_design_change_evolution.npy")
	# max_design_variable_change_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/max_design_change_evolution.npy")
	# adam_moments = np.load(cfg.OPTIMIZATION_INFO_FOLDER + "/adam_moments.npy")


	# TODO:!!!!!!
 
	
	# todo: convert the following code

	# reload_design_variable = cur_design_variable.copy()
	# if evaluate_bordered_extended:
	# 	design_var_reload_width = cur_design_variable.shape[ 0 ]
	# 	design_var_width = bayer_filter.w[0].shape[ 0 ]

	# 	pad_2x_width_reload = design_var_width - design_var_reload_width
	# 	assert ( pad_2x_width_reload > 0 ) and ( ( pad_2x_width_reload % 2 ) == 0 ), 'Expected the width difference to be positive and even!'
	# 	pad_width_reload = pad_2x_width_reload // 2
  
	# 	reload_design_variable = np.pad( cur_design_variable, ( ( pad_width_reload, pad_width_reload ), ( pad_width_reload, pad_width_reload ), ( 0, 0 ) ), mode='constant' )


  


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

# Control/Restart Parameters
start_iter = cfg.cv.restart_iter
opt.setup(cfg.cv.total_num_iterations, current_iter=start_iter, epoch_list = cfg.cv.epoch_list)

logging.info("Completed Step 3: Optimization and Optimizer setup complete.")
# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)

#*
#* AT THIS POINT THE OPTIMIZATION LOOP BEGINS
#*
opt.run()
# todo: need to make sure that all the max iterations and whatever are fed in correctly. Follow the Sugarcube.py example.

#! -----------------------------------------------------------------------------------------------------------------------------------

# # logger.info('Beginning Step 3: Run Optimization')

# # max_epochs = cfg['num_epochs']
# # max_iter = cfg['num_iterations_per_epoch'] * max_epochs
# # optimization.run(max_iter, max_epochs)
# - Run Optimization

# # logger.info('Completed Step 3: Run Optimization')

#! -----------------------------------------------------------------------------------------------------------------------------------

# sim.save('test_sim_after_opt')
# sim.close()

# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
# globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))

# Finalize the binarized device
cur_index = np.real(utility.index_from_permittivity(
						utility.binarize(designs[0].get_density())
					))
from evaluation import plotter
plotter.visualize_device(cur_index, cfg.OPTIMIZATION_PLOTS_FOLDER, iteration=cfg.cv.total_num_iterations)

logging.info('Reached end of code.')