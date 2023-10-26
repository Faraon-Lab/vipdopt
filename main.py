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
simulations = simulator.partition_objs(env_constr.construct_sim_objects(), (1,))
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
utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
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
		folder=design_save_path
		#other_arguments
	)


logging.info("Completed Step 1: Design regions setup complete.")
utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


#* Step 2: Set up figure of merit. Determine monitors from which E-fields, Poynting fields, and transmission data will be drawn.
# Handle all processing of weighting and functions that go into the figure of merit

def overall_figure_of_merit( indiv_foms, weights ):
	return 3

weights = configs.weights
indiv_foms = configs.indiv_foms

# # Add the forward sources
# for i in range(0, len(angle_phi)):
#     opt.add_tsfs_input('north', angle_theta = imaging_angle, angle_phi = angle_phi[i], polarization_angle=-angle_phi[i])
# opt.add_tsfs_input('north', angle_theta = 0, angle_phi = 0)

# # Define the overall output monitor (i.e. overall detector plane)
# output_y = -focal_length
# output_monitor = opt.add_2D_power_monitor('output_monitor', 0, output_y, 0, opt.simset_obj.sim_x, opt.simset_obj.sim_z)

fom_monitors = []
for indiv_fom in indiv_foms:
	
	#! Add dipole or adjoint sources, and their monitors.
	# Adjoint monitors and what data to obtain should be included in indiv_fom.
	# todo: alternatively, create fwd_src, adj_src, then link them together using an "FoM" class, which can then store E_fwd and E_adj gradients?
	# todo: might be a more elegant solution. We can also then just return a gradient by doing FoM.E_fwd * FoM.E_adj
	fom_monitor = create_monitor_dict(indiv_fom)
	adj_srcs = create_adjoint_sources(indiv_fom)

	for simulation in simulations:
		# check coordinates of monitor region. Does it fit in this simulation?
		# If yes, construct all or part of region and mesh in simulation
		simulation.create_monitors(fom_monitor)
		simulation.add_adj_srcs(adj_srcs)
	
fom_to_monitor_map = bidict( indiv_foms, fom_monitors)
# During the optimization we would disable and enable monitors depending on which adjoint simulation


logging.info("Completed Step 2: Figure of Merit setup complete.")
utility.backup_all_vars(globals(), cfg.cv.shelf_fn)

# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
# globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))

logging.info('Reached end of code.')