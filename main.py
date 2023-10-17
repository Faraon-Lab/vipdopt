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

# monitor_to_region_map = [  [2,3,5,6],   [3,6],    [4,5,6],    [7,8]  ]
# region_to_monitor_map = [ blah blah inverse of that ]

bd = utility.bidict({'a': 1, 'b': 2})
bd = utility.bidict({'a': 1, 'b': 1, 'c': 2})
bd = utility.bidict({1: 'a', 2: 'c'})
print(3)


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
	simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])

## disable_all_sources()

logging.info("Completed Step 0: Lumerical environment setup complete.")
utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


designs = simulator.partition_objs(env_constr.construct_design_regions(), (1,))
for design in designs:
    simulation = simulations[0] # design['simulation']	# Map design region to simulation region. Probably using a tuple is the way to go?
    # todo: ok, need to figure out the way to map these things ASAP.
    simulator.open(os.path.join(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name']))
    design = simulator.import_obj(design, create_object_=True)
    simulator.save(simulation['SIM_INFO']['path'], simulation['SIM_INFO']['name'])

#* Step 1: Design Region(s) + Constraints
# e.g. feature sizes, bridging/voids, permittivity constraints, corresponding E-field monitor regions

# # Create instances of devices to store permittivity values
# devices = []
# for num_region, region in enumerate(region_list):
# 	devices[num_region] = Device.Device(
# 		voxel_array_size,
# 		feature_dimensions,
# 		permittivity_constraints,
# 		region_coordinates,
# 		other_arguments
# 	)
	
# 	for simulation in simulations:
# 		# check coordinates of device region. Does it fit in this simulation?
# 		# If yes, construct all or part of region and mesh in simulation
# 		simulation.create_device_region(region)
# 		simulation.create_device_region_mesh(region)

		
		
#! Must keep track of which devices correspond to which regions and vice versa.
#? What is the best method to keep track of maps? Try: https://stackoverflow.com/a/21894086
device_to_region_map = bidict(devices)
# For each region, create a field_shape variable to store the E- and H-fields for adjoint calculation.

#! Shape Check! Check the mesh for the E-field monitors and adjust each field_shape variable for each device to match.
# Instead of a lookup table approach, probably should just directly pull from the simulator - we can always interpolate anyway.


logging.info("Completed Step 1: Design regions setup complete.")
utility.backup_all_vars(globals(), cfg.cv.shelf_fn)


# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)
# globals().update(utility.load_backup_vars(cfg.cv.shelf_fn))

logging.info('Reached end of code.')