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
import time
from datetime import datetime
import copy
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sn

# HPC Multithreading
import queue
import subprocess

# Add filepath to default path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
#! Gets all parameters from config file - explicitly calls it so that the variables pass into this module's globals()
from configs.global_params import *
globals().update(config_vars)
globals().update(processed_vars)

# Custom Classes and Imports
from configs import *
from evaluation import *
# from executor import *
from utils import *
from utils import lum_utils as lm
from utils import SonyBayerFilter

# Photonics
import importlib.util as imp	# Lumerical hook Python library API
spec_lumapi = imp.spec_from_file_location('lumapi', lumapi_filepath)
lumapi = imp.module_from_spec(spec_lumapi)
spec_lumapi.loader.exec_module(lumapi)

# Start logging - Logger is already initialized through the call to cfg_to_params_sony.py
logging.info(f'Starting up optimization file: {os.path.basename(__file__)}')
logging.info("All modules loaded.")
logging.debug(f'Current working directory: {os.getcwd()}')

#* Handle Input Arguments and Environment Variables
parser = argparse.ArgumentParser(description=\
					"Optimization file for Bayer Filter (SONY 2022-2023). Produces permittivity design through adjoint optimization.")
for argument in utility.argparse_universal_args:
	parser.add_argument(argument[0], argument[1], **(argument[2]))
args = parser.parse_args()

# Determine run mode from bash script input arg
run_mode = args.mode
if not ( ( run_mode == 'eval' ) or ( run_mode == 'opt' ) ):
	logging.critical( 'Unrecognized mode!' )
	sys.exit( 1 )
logging.debug("Input arguments processed.")

#* Output / Save Paths
DATA_FOLDER = projects_directory_location
SAVED_SCRIPTS_FOLDER = os.path.join(projects_directory_location, 'saved_scripts')
OPTIMIZATION_INFO_FOLDER = os.path.join(projects_directory_location, 'opt_info')
OPTIMIZATION_PLOTS_FOLDER = os.path.join(OPTIMIZATION_INFO_FOLDER, 'plots')
DEBUG_COMPLETED_JOBS_FOLDER = 'ares_test_dev'
PULL_COMPLETED_JOBS_FOLDER = projects_directory_location
if running_on_local_machine:
	PULL_COMPLETED_JOBS_FOLDER = DEBUG_COMPLETED_JOBS_FOLDER

EVALUATION_FOLDER = os.path.join(DATA_FOLDER, 'lumproc_' + os.path.basename(python_src_directory))
EVALUATION_CONFIG_FOLDER = os.path.join(EVALUATION_FOLDER, 'configs')
EVALUATION_UTILS_FOLDER = os.path.join(EVALUATION_FOLDER, 'utils')

# parameters['MODEL_PATH'] = os.path.join(projects_directory_location, 'model.pth')
# parameters['OPTIMIZER_PATH'] = os.path.join(projects_directory_location, 'optimizer.pth')


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
shutil.copy2( python_src_directory + "/SonyBayerFilterOptimization.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterOptimization.py" )
shutil.copy2( os.path.join(python_src_directory, yaml_filename), 
			 os.path.join(SAVED_SCRIPTS_FOLDER, os.path.basename(yaml_filename)) )
shutil.copy2( python_src_directory + "/configs/SonyBayerFilterParameters.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterParameters.py" )
# TODO: et cetera... might have to save out various scripts from each folder

#  Create convenient folder for evaluation code
if os.path.exists(EVALUATION_CONFIG_FOLDER):
	shutil.rmtree(EVALUATION_CONFIG_FOLDER)
shutil.copytree(os.path.join(python_src_directory, "configs"), EVALUATION_CONFIG_FOLDER)

if os.path.exists(EVALUATION_UTILS_FOLDER):
	shutil.rmtree(EVALUATION_UTILS_FOLDER)
shutil.copytree(os.path.join(python_src_directory, "utils"), EVALUATION_UTILS_FOLDER)

shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/LumProcSweep.py"), EVALUATION_FOLDER + "/LumProcSweep.py" )
shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/slurm_lumproc.sh"), EVALUATION_FOLDER + "/slurm_lumproc.sh" )
shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/LumProcParameters.py"), EVALUATION_CONFIG_FOLDER + "/LumProcParameters.py" )
shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/sweep_settings.json"), EVALUATION_CONFIG_FOLDER + "/sweep_settings.json" )
shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/plotter.py"), EVALUATION_UTILS_FOLDER + "/plotter.py" )


# SLURM Environment
if not running_on_local_machine:
	cluster_hostnames = utility.get_slurm_node_list()
	num_nodes_available = int(os.getenv('SLURM_JOB_NUM_NODES'))	# should equal --nodes in batch script. Could also get this from len(cluster_hostnames)
	num_cpus_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))	# should equal 8 or --ntasks-per-node in batch script. Could also get this SLURM_CPUS_ON_NODE or SLURM_JOB_CPUS_PER_NODE?
	# logging.info(f'There are {len(cluster_hostnames)} nodes available.')
	logging.info(f'There are {num_nodes_available} nodes available, with {num_cpus_per_node} CPUs per node. Cluster hostnames are: {cluster_hostnames}')


#* Backup

def backup_var(variable_name, savefile_name=None):
	'''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
	
	if savefile_name is None:
		savefile_name = shelf_fn

	global my_shelf
	my_shelf = shelve.open(savefile_name,'n')
	if variable_name != "my_shelf":
		try:
			my_shelf[variable_name] = globals()[variable_name]
			# logging.debug(variable_name)
		except Exception as ex:
			# # __builtins__, my_shelf, and imported modules can not be shelved.
			# logging.exception('ERROR shelving: {0}'.format(variable_name))
			# template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			# message = template.format(type(ex).__name__, ex.args)
			# logging.error(message)
			pass
	my_shelf.close()
	
	logging.debug("Successfully backed up variable: " + variable_name)

def backup_all_vars(savefile_name=None, global_enabled=True):
	'''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
	
	if savefile_name is None:
		savefile_name = shelf_fn

	if global_enabled:
		global my_shelf
		my_shelf = shelve.open(savefile_name,'n')
		for key in sorted(globals()):
			if key not in ['my_shelf']: #, 'forward_e_fields']:
				try:
					my_shelf[key] = globals()[key]
					logging.debug(f'{key} shelved.')
				except Exception as ex:
					# # __builtins__, my_shelf, and imported modules can not be shelved.
					# logging.error('ERROR shelving: {0}'.format(key))
					# template = "An exception of type {0} occurred. Arguments:\n{1!r}"
					# message = template.format(type(ex).__name__, ex.args)
					# logging.error(message)
					pass
		my_shelf.close()
		
		
		logging.info("Backup complete.")
	else:
		logging.info("Backup disabled.")
  
def load_backup_vars(loadfile_name):
	'''Restores session variables from a save_state file. Debug purposes.'''
	#! This will overwrite all current session variables!
	# except for the following, which are environment variables:
	do_not_overwrite = ['running_on_local_machine',
						'lumapi_filepath',
						'start_from_step',
						'start_epoch', 'num_epochs', 'start_iter',
						'restart_epoch', 'restart_iter',
						'num_iterations_per_epoch']
	
	global my_shelf
	my_shelf = shelve.open(loadfile_name, flag='r')
	for key in my_shelf:
		if key not in do_not_overwrite:
			try:
				globals()[key]=my_shelf[key]
			except Exception as ex:
				logging.exception('ERROR retrieving shelf: {0}'.format(key))
				logging.error("An exception of type {0} occurred. Arguments:\n{1!r}".format(type(ex).__name__, ex.args))
				# pass
	my_shelf.close()
	
	# # Restore lumapi SimObjects as dictionaries
	# for key,val in fdtd_objects.items():
	# 	globals()[key]=val
	
	logging.info("Successfully loaded backup.")


#* Utility Functions that must be located in this module

def create_dict_nx(dict_name):
	'''Checks if dictionary already exists in globals(). If not, creates it as empty.'''
	if dict_name in globals():
			logging.warning(dict_name +' already exists in globals().')
	else:
		globals()[dict_name] = {}


 
# TODO: Put into some dispersion module -----------------------------
class DevicePermittivityModel():
	'''Simple dispersion model where dispersion is ignored.'''
	# If used in multiple places, it could be called something like "NoDispersion" and be initialized with a give permittivity to return
	# Used to provide a dispersion model object that supports the function average_permittivity() that computes permittivity based on a 
	# wavelength range, when there is an actual dispersion model to consider

	def average_permittivity(self, wl_range_um_):
		'''Computes permittivity based on a wavelength range and the dispersion model of this object.'''
		return max_device_permittivity
# TODO: -------------------------------------------------------------

#
#* Create FDTD hook
#
fdtd_hook = lumapi.FDTD()
fdtd_hook.newproject()
if start_from_step != 0:
	fdtd_hook.load(os.path.abspath(projects_directory_location + "/optimization"))
fdtd_hook.save(os.path.abspath(projects_directory_location + "/optimization"))

#*---- ADD TO FDTD ENVIRONMENT AND OBJECTS ----

#! Step 0 Functions

def fdtd_get_simobject(fdtd_hook_, object_name):
	fdtd_hook_.select(object_name)
	#! Warning: This will only return the first object of this name. You must make sure all objects are named differently.
	return fdtd_hook_.getObjectBySelection()

def fdtd_get_all_simobjects(fdtd_hook_):
	fdtd_hook_.selectall()
	return fdtd_hook_.getAllSelectedObjects()
	
def fdtd_simobject_to_dict( simObj ):
	''' Duplicates a Lumapi SimObject (that cannot be shelved) to a Python dictionary equivalent.
		 Works for dictionary objects and lists of dictionaries.'''
   
	if isinstance(simObj, lumapi.SimObject):
			duplicate_dict = {}
			for key in simObj._nameMap:
				duplicate_dict[key] = simObj[key]
			return duplicate_dict
		
	elif isinstance(simObj, list):	# list of SimObjects
		duplicate_list = []
		for object in simObj:
			duplicate_list.append(fdtd_simobject_to_dict(object))
		return duplicate_list
	
	else:   
		raise ValueError('Tried to convert something that is not a Lumapi SimObject or list thereof.')

def fdtd_update_object( fdtd_hook_, simDict, create_object=False ):
	'''Accepts a dictionary that represents the FDTD Object, and applies all its parameters to the corresponding object.
	Returns the updated dictionary (since certain parameters may have changed in FDTD due to the applied changes.)
	If the create_object flag is set to True, creates an Object according to the dictionary. Checks if Object exists first.'''
	
	try:
		# Use fdtd_hook.getnamed() to raise an Exception if the object doesn't exist. Other methods don't raise exceptions.
		objExists = fdtd_hook_.getnamed(simDict['name'],'name')
	except Exception as ex:
		if create_object:
			# Determine which add() method to call based on what type of FDTD object it is
   
			type_string = simDict['type'].lower()
			if any(ele.lower() in type_string for ele in ['FDTD']):
				fdtd_hook_.addfdtd()
			elif any(ele.lower() in type_string for ele in ['Mesh']):
				fdtd_hook_.addmesh(name=simDict['name'])
			elif any(ele.lower() in type_string for ele in ['GaussianSource']):
				fdtd_hook_.addgaussian(name=simDict['name'])
			elif any(ele.lower() in type_string for ele in ['TFSFSource']):
				fdtd_hook_.addtfsf(name=simDict['name'])
			elif any(ele.lower() in type_string for ele in ['DipoleSource']):
				fdtd_hook_.adddipole(name=simDict['name'])
			elif any(ele.lower() in type_string for ele in ['DFTMonitor']):
				if simDict['power_monitor']:
					fdtd_hook_.addpower(name=simDict['name'])
				else:
					fdtd_hook_.addprofile(name=simDict['name'])
			elif any(ele.lower() in type_string for ele in ['Import']):
				fdtd_hook_.addimport(name=simDict['name'])
			elif any(ele.lower() in type_string for ele in ['Rectangle']):
				fdtd_hook_.addrect(name=simDict['name'])
	
			# Add more as needed!
			else:
				logging.error(f'FDTD Object {simDict["name"]}: type not specified.')
   
		else:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
			message = template.format(type(ex).__name__, ex.args)
			logging.error(message + f' FDTD Object {simDict["name"]}: does not exist.')
			return 0
	
	for key, val in simDict.items():
		# Update every single property in the FDTD object to sync up with the dictionary.
		if key not in ['name', 'type', 'power_monitor']:	#NOTE: These either should never be updated, or don't exist in FDTD as properties.
			# logging.debug(f'Monitor: {monitor["name"]}, Key: {key}, and Value: {val}')
			try:
				fdtd_hook_.setnamed(simDict['name'], key, val)
			except Exception as ex:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
				message = template.format(type(ex).__name__, ex.args)
				logging.debug(message + f' Property {key} in FDTD Object {simDict["name"]} is inactive / cannot be changed.')
	
	# Update the record of the dictionary in globals()
	updated_simobject = fdtd_get_simobject( fdtd_hook_, simDict['name'] )
	updated_simdict = fdtd_simobject_to_dict( updated_simobject )
	simDict.update(updated_simdict)
	return simDict

def update_object(object_path, key, val):
	'''Updates selected objects in the FDTD simulation.'''
	global fdtd_objects
	
	utility.set_by_path(fdtd_objects, object_path + [key], val)
	new_dict = fdtd_update_object(fdtd_hook, utility.get_by_path(fdtd_objects, object_path))
	utility.set_by_path(fdtd_objects, object_path, new_dict)

def disable_objects(object_list):
	'''Disable selected objects in the simulation, to save on memory and runtime'''
	global fdtd_objects
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
	
	for object_path in object_list:
		update_object(object_path, 'enabled', 0)

def disable_all_sources():
	'''Disable all sources in the simulation, so that we can selectively turn single sources on at a time'''
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	object_list = []

	for key, val in fdtd_objects.items():
		if isinstance(val, list):
			# type_string = val[0]['type'].lower()
			pass
		else:
			type_string = val['type'].lower()
  
			if any(ele.lower() in type_string for ele in ['GaussianSource', 'TFSFSource', 'DipoleSource']):
				object_list.append([val['name']])

	disable_objects(object_list)

#
#* Step 0: Set up structures, regions and monitors from which E-fields, Poynting fields, and transmission data will be drawn.
# Also any other necessary editing of the Lumerical environment and objects.
#

# Create an instance of the device to store permittivity values
bayer_filter_size_voxels = np.array(
	[device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
logging.info(f'Creating Bayer Filter with voxel size {bayer_filter_size_voxels}.')
bayer_filter = SonyBayerFilter.SonyBayerFilter(
	bayer_filter_size_voxels,
	[ 0.0, 1.0 ],
	init_permittivity_0_1_scale,
	num_vertical_layers,
	vertical_layer_height_voxels)

if start_from_step == 0:
 
	# # Initialize the seed differently
	# bayer_filter.set_random_init(0.1, 0.2)
	# c = plt.imshow(bayer_filter.w[0][:,:,20], vmin=0, vmax=1)
	# plt.colorbar(c)
	# plt.show()

	bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
	bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
	bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)
 
	# Populate a 3+1-D array at each xyz-position with the values of x, y, and z
	bayer_filter_region = np.zeros( 
		( device_voxels_lateral, device_voxels_lateral, device_voxels_vertical, 3 ) )
	for x_idx in range( 0, device_voxels_lateral ):
		for y_idx in range( 0, device_voxels_lateral ):
			for z_idx in range( 0, device_voxels_vertical ):
				bayer_filter_region[ x_idx, y_idx, z_idx, : ] = [
						bayer_filter_region_x[ x_idx ], bayer_filter_region_y[ y_idx ], bayer_filter_region_z[ z_idx ] ]
	logging.debug(f'Bayer Filter Region has array shape {np.shape(bayer_filter_region)}.')

	gradient_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_simulation_mesh_lateral)
	gradient_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_simulation_mesh_lateral)
	gradient_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_simulation_mesh_vertical)

	# Get the device's dimensions in terms of (MESH) voxels
	field_shape = [device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_vertical]
	logging.info(f'NOTE: Simulation mesh should yield field shape in design_Efield_monitor of {field_shape}.')
	
	
	logging.info("Beginning Step 0: Lumerical Environment Setup")
	sys.stdout.flush()

	# Create dictionary to duplicate FDTD SimObjects as dictionaries for saving out
	# This dictionary captures the states of all FDTD objects in the simulation, and has the advantage that it can be saved out or exported.
	# The dictionary will be our master record, and is continually updated for every change in FDTD.
	fdtd_objects = {}

	# Set up the FDTD region and mesh.
	# We are using dict-like access: see https://support.lumerical.com/hc/en-us/articles/360041873053-Session-management-Python-API
	# The line fdtd_objects['FDTD'] = fdtd means that both sides of the equation point to the same object.
	# See https://stackoverflow.com/a/10205681 and https://nedbatchelder.com/text/names.html

	fdtd = {}
	fdtd['name'] = 'FDTD'
	fdtd['type'] = 'FDTD'
	fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
	fdtd['y span'] = fdtd_region_size_lateral_um * 1e-6
	fdtd['z max'] = fdtd_region_maximum_vertical_um * 1e-6
	fdtd['z min'] = fdtd_region_minimum_vertical_um * 1e-6
	fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
	fdtd['index'] = background_index
	fdtd = fdtd_update_object(fdtd_hook, fdtd, create_object=True)
	fdtd_objects['FDTD'] = fdtd

	device_mesh = {}
	device_mesh['name'] = 'device_mesh'
	device_mesh['type'] = 'Mesh'
	device_mesh['x'] = 0
	device_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
	device_mesh['y'] = 0
	device_mesh['y span'] = fdtd_region_size_lateral_um * 1e-6
	device_mesh['z max'] = ( device_vertical_maximum_um + 0.5 ) * 1e-6
	device_mesh['z min'] = ( device_vertical_minimum_um - 0.5 ) * 1e-6
	device_mesh['dx'] = mesh_spacing_um * 1e-6
	device_mesh['dy'] = mesh_spacing_um * 1e-6
	device_mesh['dz'] = mesh_spacing_um * 1e-6
	device_mesh = fdtd_update_object(fdtd_hook, device_mesh, create_object=True)
	fdtd_objects['device_mesh'] = device_mesh
 
	# Add a Gaussian wave forward source at angled incidence
	# (Deprecated) Add a TFSF plane wave forward source at normal incidence
	forward_sources = []
 
	for xy_idx in range(0, 2):
		if use_gaussian_sources:
	  
			def snellRefrOffset(src_angle_incidence, object_list):
				''' Takes a list of objects in between source and device, retrieves heights and indices,
				and calculates injection angle so as to achieve the desired angle of incidence upon entering the device.'''
				# structured such that n0 is the input interface and n_end is the last above the device
				# height_0 should be 0
				
				input_theta = 0
				r_offset = 0

				heights = [0.]
				indices = [1.]
				total_height = device_vertical_maximum_um
				max_height = src_maximum_vertical_um + src_hgt_Si
				for object_name in object_list:
					hgt = fdtd_hook.getnamed(object_name, 'z span')/1e-6
					total_height += hgt
					indices.append(float(fdtd_hook.getnamed(object_name, 'index')))
					
					if total_height >= max_height:
						heights.append(hgt-(total_height - max_height))
						break
					else:
						heights.append(hgt)

				# print(f'Device ends at z = {device_vertical_maximum_um}; Source is located at z = {max_height}')
				# print(f'Heights are {heights} with corresponding indices {indices}')
				# sys.stdout.flush()

				src_angle_incidence = np.radians(src_angle_incidence)
				snellConst = float(indices[-1])*np.sin(src_angle_incidence)
				input_theta = np.degrees(np.arcsin(snellConst/float(indices[0])))

				r_offset = 0
				for cnt, h_i in enumerate(heights):
					s_i = snellConst/indices[cnt]
					r_offset = r_offset + h_i*s_i/np.sqrt(1-s_i**2)

				return [input_theta, r_offset]

			forward_src = {}
			forward_src['name'] = 'forward_src_' + xy_names[xy_idx]
			forward_src['type'] = 'GaussianSource'
			forward_src['angle theta'] = source_angle_theta_deg
			forward_src['angle phi'] = source_angle_phi_deg
			forward_src['polarization angle'] = xy_phi_rotations[xy_idx]
			forward_src['direction'] = 'Backward'
			forward_src['x span'] = 2 * fdtd_region_size_lateral_um * 1e-6
			forward_src['y span'] = 2 * fdtd_region_size_lateral_um * 1e-6
			forward_src['z'] = src_maximum_vertical_um * 1e-6

			# forward_src['angle theta'], shift_x_center = snellRefrOffset(source_angle_theta_deg, objects_above_device)	# this code takes substrates and objects above device into account
			shift_x_center = np.abs( device_vertical_maximum_um - src_maximum_vertical_um ) * np.tan( source_angle_theta_rad )
			forward_src['x'] = shift_x_center * 1e-6
			# TODO: Edit this as per previous ian edits in order to take substrates or whatever into account

			forward_src['wavelength start'] = lambda_min_um * 1e-6
			forward_src['wavelength stop'] = lambda_max_um * 1e-6

			forward_src['waist radius w0'] = gaussian_waist_radius_um * 1e-6
			forward_src['distance from waist'] = ( device_vertical_maximum_um - src_maximum_vertical_um ) * 1e-6
	
		else:
			forward_src = {}
			forward_src['name'] = 'forward_src_' + xy_names[xy_idx]
			forward_src['type'] = 'TFSFSource'
			# forward_src['angle phi'] = 0
			forward_src['polarization angle'] = xy_phi_rotations[xy_idx]
			forward_src['direction'] = 'Backward'
			forward_src['x span'] = lateral_aperture_um * 1e-6
			forward_src['y span'] = lateral_aperture_um * 1e-6
			forward_src['z max'] = src_maximum_vertical_um * 1e-6
			forward_src['z min'] = src_minimum_vertical_um * 1e-6
			forward_src['wavelength start'] = lambda_min_um * 1e-6
			forward_src['wavelength stop'] = lambda_max_um * 1e-6
	
		forward_src = fdtd_update_object(fdtd_hook, forward_src, create_object=True)
		forward_sources.append(forward_src)
		fdtd_objects[forward_src['name']] = forward_src
	fdtd_objects['forward_sources'] = forward_sources

	# Place dipole adjoint sources at the focal plane that can ring in both
	# x-axis and y-axis
	adjoint_sources = []

	for adj_src_idx in range(0, num_adjoint_sources):
		adjoint_sources.append([])
		for xy_idx in range(0, 2):
			adj_src = {}
			adj_src['name'] = 'adj_src_' + str(adj_src_idx) + xy_names[xy_idx]
			adj_src['type'] = 'DipoleSource'
			adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
			adj_src['y'] = adjoint_y_positions_um[adj_src_idx] * 1e-6
			adj_src['z'] = adjoint_vertical_um * 1e-6
			adj_src['theta'] = 90
			adj_src['phi'] = xy_phi_rotations[xy_idx]
			adj_src['wavelength start'] = lambda_min_um * 1e-6
			adj_src['wavelength stop'] = lambda_max_um * 1e-6
			adj_src = fdtd_update_object(fdtd_hook, adj_src, create_object=True)
			fdtd_objects[adj_src['name']] = adj_src
			adjoint_sources[adj_src_idx].append(adj_src)
	fdtd_objects['adjoint_sources'] = adjoint_sources

	if add_pdaf:
		# Here, we are going to neglect angular dispersion with wavelength and assume
		# it is in a narrow enough range
		#
		# Eventually, you will probably want to enforce filtering on certain colors
		# as well, but for now let's just limit to one color!
		#
		# Need to have monitors tagged to use the source limits, which is the case right now
		#
		
		pdaf_sources = []
		pdaf_adjoint_sources = []

		for xy_idx in range(0, 2):
			pdaf_source = {}
			pdaf_source['name'] = 'pdaf_src_' + xy_names[xy_idx]
			pdaf_source['type'] = 'TFSFSource'
			pdaf_source['angle phi'] = pdaf_angle_phi_degrees
			pdaf_source['angle theta'] = pdaf_angle_theta_degrees
			pdaf_source['polarization angle'] = xy_phi_rotations[xy_idx]
			pdaf_source['direction'] = 'Backward'
			pdaf_source['x span'] = lateral_aperture_um * 1e-6
			pdaf_source['y span'] = lateral_aperture_um * 1e-6
			pdaf_source['z max'] = src_maximum_vertical_um * 1e-6
			pdaf_source['z min'] = src_minimum_vertical_um * 1e-6
			pdaf_source['wavelength start'] = pdaf_lambda_min_um * 1e-6
			pdaf_source['wavelength stop'] = pdaf_lambda_max_um * 1e-6
			pdaf_source = fdtd_update_object(fdtd_hook, pdaf_source, create_object=True)
			fdtd_objects[pdaf_source['name']] = pdaf_source
			pdaf_sources.append(pdaf_source)

		for xy_idx in range(0, 2):
			pdaf_adjoint_source = {}
			pdaf_adjoint_source['name'] = 'pdaf_adj_src_' + xy_names[xy_idx]
			pdaf_adjoint_source['type'] = 'DipoleSource'
			pdaf_adjoint_source['x'] = pdaf_focal_spot_x_um * 1e-6
			pdaf_adjoint_source['y'] = pdaf_focal_spot_x_um * 1e-6
			pdaf_adjoint_source['z'] = adjoint_vertical_um * 1e-6
			pdaf_adjoint_source['theta'] = 90
			pdaf_adjoint_source['phi'] = xy_phi_rotations[xy_idx]
			pdaf_adjoint_source['wavelength start'] = pdaf_lambda_min_um * 1e-6
			pdaf_adjoint_source['wavelength stop'] = pdaf_lambda_max_um * 1e-6
			pdaf_adjoint_source = fdtd_update_object(fdtd_hook, pdaf_adjoint_source, create_object=True)
			fdtd_objects[pdaf_adjoint_source['name']] = pdaf_adjoint_source
			pdaf_adjoint_sources.append( pdaf_adjoint_source )

		fdtd_objects[pdaf_sources] = pdaf_sources
		fdtd_objects[pdaf_adjoint_sources] = pdaf_adjoint_sources


	# Set up the volumetric electric field monitor inside the design region.  We will need this to compute
	# the adjoint gradient
	design_efield_monitor = {}
	design_efield_monitor['name'] = 'design_efield_monitor'
	design_efield_monitor['type'] = 'DFTMonitor'
	design_efield_monitor['power_monitor'] = False
	design_efield_monitor['monitor type'] = '3D'
	design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
	design_efield_monitor['y span'] = device_size_lateral_um * 1e-6
	design_efield_monitor['z max'] = device_vertical_maximum_um * 1e-6
	design_efield_monitor['z min'] = device_vertical_minimum_um * 1e-6
	design_efield_monitor['override global monitor settings'] = 1
	design_efield_monitor['use wavelength spacing'] = 1
	design_efield_monitor['use source limits'] = 1
	design_efield_monitor['frequency points'] = num_design_frequency_points
	design_efield_monitor['output Hx'] = 0
	design_efield_monitor['output Hy'] = 0
	design_efield_monitor['output Hz'] = 0
	design_efield_monitor = fdtd_update_object(fdtd_hook, design_efield_monitor, create_object=True)
	fdtd_objects['design_efield_monitor'] = design_efield_monitor

	# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
	# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
	# gradient.
	focal_monitors = []

	for adj_src in range(0, num_adjoint_sources):
		focal_monitor = {}
		focal_monitor['name'] = 'focal_monitor_' + str(adj_src)
		focal_monitor['type'] = 'DFTMonitor'
		focal_monitor['power_monitor'] = True
		focal_monitor['monitor type'] = 'point'
		focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
		focal_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
		focal_monitor['z'] = adjoint_vertical_um * 1e-6
		focal_monitor['override global monitor settings'] = 1
		focal_monitor['use wavelength spacing'] = 1
		focal_monitor['use source limits'] = 1
		focal_monitor['frequency points'] = num_design_frequency_points
		focal_monitor = fdtd_update_object(fdtd_hook, focal_monitor, create_object=True)
		fdtd_objects[focal_monitor['name']] = focal_monitor
		focal_monitors.append(focal_monitor)
	fdtd_objects['focal_monitors'] = focal_monitors

	transmission_monitors = []

	for adj_src in range(0, num_adjoint_sources):
		transmission_monitor = {}
		transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
		transmission_monitor['type'] = 'DFTMonitor'
		transmission_monitor['power_monitor'] = True
		transmission_monitor['monitor type'] = '2D Z-normal'
		transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
		transmission_monitor['x span'] = 0.5 * device_size_lateral_um * 1e-6
		transmission_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
		transmission_monitor['y span'] = 0.5 * device_size_lateral_um * 1e-6
		transmission_monitor['z'] = adjoint_vertical_um * 1e-6
		transmission_monitor['override global monitor settings'] = 1
		transmission_monitor['use wavelength spacing'] = 1
		transmission_monitor['use source limits'] = 1
		transmission_monitor['frequency points'] = num_design_frequency_points
		transmission_monitor = fdtd_update_object(fdtd_hook, transmission_monitor, create_object=True)
		fdtd_objects[transmission_monitor['name']] = transmission_monitor
		transmission_monitors.append(transmission_monitor)
	fdtd_objects['transmission_monitors'] = transmission_monitors

	transmission_monitor_focal = {}
	transmission_monitor_focal['name'] = 'transmission_focal_monitor_'
	transmission_monitor_focal['type'] = 'DFTMonitor'
	transmission_monitor_focal['power_monitor'] = True
	transmission_monitor_focal['monitor type'] = '2D Z-normal'
	transmission_monitor_focal['x'] = 0 * 1e-6
	transmission_monitor_focal['x span'] = device_size_lateral_um * 1e-6
	transmission_monitor_focal['y'] = 0 * 1e-6
	transmission_monitor_focal['y span'] = device_size_lateral_um * 1e-6
	transmission_monitor_focal['z'] = adjoint_vertical_um * 1e-6
	transmission_monitor_focal['override global monitor settings'] = 1
	transmission_monitor_focal['use wavelength spacing'] = 1
	transmission_monitor_focal['use source limits'] = 1
	transmission_monitor_focal['frequency points'] = num_design_frequency_points
	transmission_monitor_focal = fdtd_update_object(fdtd_hook, transmission_monitor_focal, create_object=True)
	fdtd_objects['transmission_monitor_focal'] = transmission_monitor_focal

	# Add device region and create device permittivity
	design_import = {}
	design_import['name'] = 'design_import'
	design_import['type'] = 'Import'
	design_import['x span'] = device_size_lateral_um * 1e-6
	design_import['y span'] = device_size_lateral_um * 1e-6
	design_import['z max'] = device_vertical_maximum_um * 1e-6
	design_import['z min'] = device_vertical_minimum_um * 1e-6
	design_import = fdtd_update_object(fdtd_hook, design_import, create_object=True)
	fdtd_objects['design_import'] = design_import

	# Install Aperture that blocks off source
	if use_source_aperture:
		source_block = {}
		source_block['name'] = 'PEC_screen'
		source_block['type'] = 'Rectangle'
		source_block['x'] = 0 * 1e-6
		source_block['x span'] = 1.1*4/3*1.2 * device_size_lateral_um * 1e-6
		source_block['y'] = 0 * 1e-6
		source_block['y span'] = 1.1*4/3*1.2 * device_size_lateral_um * 1e-6
		source_block['z min'] = (device_vertical_maximum_um + 3 * mesh_spacing_um) * 1e-6
		source_block['z max'] = (device_vertical_maximum_um + 6 * mesh_spacing_um) * 1e-6
		source_block['material'] = 'PEC (Perfect Electrical Conductor)'
		# TODO: set enable true or false?
		source_block = fdtd_update_object(fdtd_hook, source_block, create_object=True)
		fdtd_objects['source_block'] = source_block
		
		source_aperture = {}
		source_aperture['name'] = 'source_aperture'
		source_aperture['type'] = 'Rectangle'
		source_aperture['x'] = 0 * 1e-6
		source_aperture['x span'] = device_size_lateral_um * 1e-6
		source_aperture['y'] = 0 * 1e-6
		source_aperture['y span'] = device_size_lateral_um * 1e-6
		source_aperture['z min'] = (device_vertical_maximum_um + 3 * mesh_spacing_um) * 1e-6
		source_aperture['z max'] = (device_vertical_maximum_um + 6 * mesh_spacing_um) * 1e-6
		source_aperture['material'] = 'etch'
		# TODO: set enable true or false?
		source_aperture = fdtd_update_object(fdtd_hook, source_aperture, create_object=True)
		fdtd_objects['source_aperture'] = source_aperture

	# Set up sidewalls on the side to try and attenuate crosstalk
	device_sidewalls = []

	for dev_sidewall_idx in range(0, num_sidewalls):
		device_sidewall = {}
		device_sidewall['name'] = 'sidewall_' + str(dev_sidewall_idx)
		device_sidewall['type'] = 'Rectangle'
		device_sidewall['x'] = sidewall_x_positions_um[dev_sidewall_idx] * 1e-6
		device_sidewall['x span'] = sidewall_xspan_positions_um[dev_sidewall_idx] * 1e-6
		device_sidewall['y'] = sidewall_y_positions_um[dev_sidewall_idx] * 1e-6
		device_sidewall['y span'] = sidewall_yspan_positions_um[dev_sidewall_idx] * 1e-6
		if sidewall_extend_focalplane:
			device_sidewall['z min'] = adjoint_vertical_um * 1e-6
		else:   
			device_sidewall['z min'] = sidewall_vertical_minimum_um * 1e-6
		device_sidewall['z max'] = device_vertical_maximum_um * 1e-6
		device_sidewall['material'] = sidewall_material
		device_sidewall = fdtd_update_object(fdtd_hook, device_sidewall, create_object=True)
		fdtd_objects[device_sidewall['name']] = device_sidewall
		device_sidewalls.append(device_sidewall)
	fdtd_objects['device_sidewalls'] = device_sidewalls


	# Apply finer mesh regions restricted to sidewalls
	device_sidewall_meshes = []

	for dev_sidewall_idx in range(0, num_sidewalls):
		device_sidewall_mesh = {}
		device_sidewall_mesh['name'] = 'mesh_sidewall_' + str(dev_sidewall_idx)
		device_sidewall_mesh['type'] = 'Mesh'
		device_sidewall_mesh['set maximum mesh step'] = 1
		device_sidewall_mesh['override x mesh'] = 0
		device_sidewall_mesh['override y mesh'] = 0
		device_sidewall_mesh['override z mesh'] = 0
		device_sidewall_mesh['based on a structure'] = 1
		device_sidewall_mesh['structure'] = device_sidewalls[dev_sidewall_idx]['name']
		if device_sidewalls[dev_sidewall_idx]['x span'] < device_sidewalls[dev_sidewall_idx]['y span']:
			device_sidewall_mesh['override x mesh'] = 1
			device_sidewall_mesh['dx'] = mesh_spacing_um * 1e-6 #(sidewall_thickness_um / 5) * 1e-6
		else:
			device_sidewall_mesh['override y mesh'] = 1
			device_sidewall_mesh['dy'] = mesh_spacing_um * 1e-6 #(sidewall_thickness_um / 5) * 1e-6
		device_sidewall_mesh = fdtd_update_object(fdtd_hook, device_sidewall_mesh, create_object=True)
		fdtd_objects[device_sidewall_mesh['name']] = device_sidewall_mesh
		device_sidewall_meshes.append(device_sidewall_mesh)
	fdtd_objects['device_sidewall_meshes'] = device_sidewall_meshes


	disable_all_sources()

	logging.info("Completed Step 0: Lumerical environment setup complete.")
	backup_all_vars()


#! Step 2 Functions

#
#* Set up queue for parallel jobs
#
jobs_queue = queue.Queue()

def add_job( job_name, queue_in ):
	full_name = projects_directory_location + "/" + job_name
	fdtd_hook.save( os.path.abspath(full_name) )
	queue_in.put( full_name )

	return full_name

def run_jobs( queue_in ):
	small_queue = queue.Queue()

	while not queue_in.empty():
		for node_idx in range( 0, num_nodes_available ):
			# this scheduler schedules jobs in blocks of num_nodes_available, waits for them ALL to finish, and then schedules the next block
			# TODO: write a scheduler that enqueues a new job as soon as one finishes
			# not urgent because for the most part, all jobs take roughly the same time
			if queue_in.qsize() > 0:
				small_queue.put( queue_in.get() )

		run_jobs_inner( small_queue )

def run_jobs_inner( queue_in ):
	processes = []
	job_idx = 0
	while not queue_in.empty():
		get_job_path = queue_in.get()

		logging.info(f'Node {cluster_hostnames[job_idx]} is processing file at {get_job_path}.')

		process = subprocess.Popen(
			[
				#! Make sure this points to the right place
				# os.path.abspath(python_src_directory + "/utils/run_proc.sh"),		# TODO: There's some permissions issue??
				'/home/gdrobert/Develompent/adjoint_lumerical/inverse_design/run_proc.sh',
				cluster_hostnames[ job_idx ],
				get_job_path
			]
		)
		processes.append( process )

		job_idx += 1
	sys.stdout.flush()

	job_queue_time = time.time()
	completed_jobs = [ 0 for i in range( 0, len( processes ) ) ]
	while np.sum( completed_jobs ) < len( processes ):
		for job_idx in range( 0, len( processes ) ):
			sys.stdout.flush()
   
			if completed_jobs[ job_idx ] == 0:

				poll_result = processes[ job_idx ].poll()
				if not( poll_result is None ):
					completed_jobs[ job_idx ] = 1

		if time.time()-job_queue_time > 3600:
			logging.CRITICAL(r"Warning! The following jobs are taking more than an hour and might be stalling:")
			logging.CRITICAL(completed_jobs)

		time.sleep( 5 )


#* Functions for obtaining information from Lumerical monitors
# See utils/lum_utils.py

#
#* These numpy arrays store all the data we will pull out of the simulation.
#
forward_e_fields = {}                       # Records E-fields throughout device volume
focal_data = {}                             # Records E-fields at focal plane spots (locations of adjoint sources)
quadrant_transmission_data = {}             # Records transmission through each quadrant

pdaf_forward_e_fields = {}
pdaf_focal_data = {}
pdaf_quadrant_transmission_data = {}

#
#* Set up some numpy arrays to handle all the data used to track the FoMs and progress of the simulation.
#

# By iteration
figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
pdaf_transmission_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

# By iteration, focal area, polarization, wavelength: 
# Create a dictionary of FoMs. Each key corresponds to a value: an array storing data of a specific FoM,
# for each epoch, iteration, quadrant, polarization, and wavelength.
# TODO: Consider using xarray instead of numpy arrays in order to have dimension labels for each axis.
fom_evolution = {}
for fom_type in fom_types:
	fom_evolution[fom_type] = np.zeros((num_epochs, num_iterations_per_epoch, 
										num_focal_spots, len(xy_names), num_design_frequency_points))
# The main FoM being used for optimization is indicated in the config.

# TODO: Turned this off because the memory and savefile space of gradient_evolution are too large.
# if start_from_step != 0:	# Repeat the definition of field_shape here just because it's not loaded in if we start from a debug point
# 	field_shape = [device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_vertical]
# gradient_evolution = np.zeros((num_epochs, num_iterations_per_epoch, *field_shape,
# 								num_focal_spots, len(xy_names), num_design_frequency_points))


def calculate_performance_weighting(fom_list):
	'''All gradients are combined with a weighted average in Eq. (3), with weights chosen according to Eq. (2) such that
	all figures of merit seek the same efficiency. In these equations, FoM represents the current value of a figure of 
	merit, N is the total number of figures of merit, and wi represents the weight applied to its respective merit function’s
	gradient. The maximum operator is used to ensure the weights are never negative, thus ignoring the gradient of 
	high-performing figures of merit rather than forcing the figure of merit to decrease. The 2/N factor is used to ensure all
	weights conveniently sum to 1 unless some weights were negative before the maximum operation. Although the maximum operator
	is non-differentiable, this function is used only to apply the gradient rather than to compute it. Therefore, it does not 
	affect the applicability of the adjoint method.
	
	Taken from: https://doi.org/10.1038/s41598-021-88785-5'''

	performance_weighting = (2. / len(fom_list)) - fom_list**2 / np.sum(fom_list**2)

	# Zero-shift and renormalize
	if np.min( performance_weighting ) < 0:
		performance_weighting -= np.min(performance_weighting)
		performance_weighting /= np.sum(performance_weighting)

	return performance_weighting

#* Optimization Parameters
step_size_start = 1.
# Choose the dispersion model for this optimization
dispersion_model = DevicePermittivityModel()
# ADAM Optimizer Parameters
adam_moments = np.zeros([2] + list(bayer_filter_size_voxels))

# PDAF Settings
pdaf_scan = False	#True
pdaf_defocus_gaussian = False
pdaf_gaussian_defocus_um = 6#-4
num_defocus_points = 7
pdaf_defocus_amounts_um = np.linspace( -pdaf_gaussian_defocus_um, pdaf_gaussian_defocus_um, num_defocus_points )
pdaf_angular = True
pdaf_angular_range = -14
assert ( not( pdaf_defocus_gaussian and pdaf_angular ) ), 'These options are mutually exclusive for the pdaf scan!'

# Control Parameters
restart = False                 #!! Set to true if restarting from a particular iteration - be sure to change start_epoch and start_iter
evaluate = False                # Set to true if evaluating device performance at this specific iteration - exports out transmission at focal plane and at focal spots
if restart_epoch or restart_iter:
	restart = True
if run_mode == 'eval':
	evaluate = True

if restart or evaluate:
	logging.info(f'Restarting optimization from epoch {restart_epoch}, iteration {restart_iter}.')
	
	# Load in the most recent Design density variable (before filters of device), values between 0 and 1
	cur_design_variable = np.load(
			OPTIMIZATION_INFO_FOLDER + "/cur_design_variable.npy" )
	# Re-initialize bayer_filter instance
	bayer_filter.w[0] = cur_design_variable
	# Regenerate permittivity based on the most recent design, assuming epoch 0		
 	# NOTE: This gets corrected later in the optimization loop (Step 1)
	bayer_filter.update_permittivity()

	# # Other necessary savestate data
	figure_of_merit_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/figure_of_merit.npy")
	# gradient_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/gradient_by_focal_pol_wavelength.npy")		# Took out feature because it consumes too much memory
	for fom_type in fom_types:
		fom_evolution[fom_type] = np.load(OPTIMIZATION_INFO_FOLDER + f"/{fom_type}_by_focal_pol_wavelength.npy")
	step_size_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/step_size_evolution.npy")
	average_design_variable_change_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/average_design_change_evolution.npy")
	max_design_variable_change_evolution = np.load(OPTIMIZATION_INFO_FOLDER + "/max_design_change_evolution.npy")


	if restart_epoch == 0 and restart_iter == 1:
		# Redeclare everything if starting from scratch (but using a certain design as the seed)
		figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
		pdaf_transmission_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
		step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
		average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
		max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
		fom_evolution = {}
		for fom_type in fom_types:
			fom_evolution[fom_type] = np.zeros((num_epochs, num_iterations_per_epoch, 
												num_focal_spots, len(xy_names), num_design_frequency_points))
		# The main FoM being used for optimization is indicated in the config.

		# Use binarized density variable instead of intermediate
		bayer_filter.update_filters( num_epochs - 1 )
		bayer_filter.update_permittivity()
		cur_density = bayer_filter.get_permittivity()
		cur_design_variable = cur_density
		bayer_filter.w[0] = cur_design_variable
	
	# Set everything ahead of the restart epoch and iteration to zero.
	for rs_epoch in range(restart_epoch, num_epochs):
		if rs_epoch == restart_epoch:
			rst_iter = restart_iter
		else:	rst_iter = 0

		for rs_iter in range(rst_iter, num_iterations_per_epoch):
			figure_of_merit_evolution[rs_epoch, rs_iter] = 0
			pdaf_transmission_evolution[rs_epoch, rs_iter] = 0
			step_size_evolution[rs_epoch, rs_iter] = 0
			average_design_variable_change_evolution[rs_epoch, rs_iter] = 0
			max_design_variable_change_evolution[rs_epoch, rs_iter] = 0
			for fom_type in fom_types:
				fom_evolution[fom_type][rs_epoch, rs_iter] = np.zeros((num_focal_spots, len(xy_names), num_design_frequency_points))


if evaluate:
	
	# Set sigmoid strength to most recent epoch and get (filtered) permittivity
	bayer_filter.update_filters( num_epochs - 1 )
	bayer_filter.update_permittivity()
	cur_density = bayer_filter.get_permittivity()

	fdtd_hook.switchtolayout()
	
	# TODO: Evaluate block
 
 
else:		# optimization
	if start_from_step != 0:
		load_backup_vars(shelf_fn)

	# 20230220 Greg - Fixing a certain geometry for quartering and gradient accumulation, but a good place to start I think!
	def quartered_and_rotate_index( index_in_, quarter_x, quarter_y ):
		assert ( ( index_in_.shape[ 0 ] ) % 2 ) == 0, "Expected this to be even for convenience"
		assert ( ( index_in_.shape[ 1 ] ) % 2 ) == 0, "Expected this to be even for convenience"

		quartered_size_x = index_in_.shape[ 0 ] // 2
		quartered_size_y = index_in_.shape[ 1 ] // 2

		offset_x = quarter_x * quartered_size_x
		offset_y = quarter_y * quartered_size_y

		index_out = np.zeros( index_in_.shape, dtype=index_in_.dtype )

		for z_idx in range( 0, index_in_.shape[ 2 ] ):
			get_index_in_slice = np.squeeze( index_in_[ :, :, z_idx ] )
			extract_quarter = get_index_in_slice[ offset_x : ( offset_x + quartered_size_x ), offset_y : ( offset_y + quartered_size_y ) ]

			get_index_out_slice = np.zeros( get_index_in_slice.shape, dtype=get_index_in_slice.dtype )

			get_index_out_slice[ 0 : quartered_size_x, 0 : quartered_size_y ] = extract_quarter
			get_index_out_slice[ quartered_size_x : , 0 : quartered_size_y ] = np.rot90( extract_quarter, k=1 )
			get_index_out_slice[ quartered_size_x : , quartered_size_y : ] = np.rot90( extract_quarter, k=2 )
			get_index_out_slice[ 0 : quartered_size_x, quartered_size_y : ] = np.rot90( extract_quarter, k=3 )

			index_out[ :, :, z_idx ] = get_index_out_slice

		return index_out

	def quarted_gradient_accumulate( gradient_in_, quarter_x, quarter_y ):
		assert ( ( gradient_in_.shape[ 0 ] ) % 2 ) == 0, "Expected this to be even for convenience"
		assert ( ( gradient_in_.shape[ 1 ] ) % 2 ) == 0, "Expected this to be even for convenience"

		quartered_size_x = gradient_in_.shape[ 0 ] // 2
		quartered_size_y = gradient_in_.shape[ 1 ] // 2

		offset_x = quarter_x * quartered_size_x
		offset_y = quarter_y * quartered_size_y

		gradient_out = np.zeros( gradient_in_.shape, dtype=gradient_in_.dtype )

		for z_idx in range( 0, gradient_in_.shape[ 2 ] ):
			get_gradient_in_slice = np.squeeze( gradient_in_[ :, :, z_idx ] )

			extract_qstart = get_gradient_in_slice[ 0 * quartered_size_x : ( 0 * quartered_size_x + quartered_size_x ), 0 * quartered_size_y : ( 0 * quartered_size_y + quartered_size_y ) ]
			extract_q90 = get_gradient_in_slice[ 1 * quartered_size_x : ( 1 * quartered_size_x + quartered_size_x ), 0 * quartered_size_y : ( 0 * quartered_size_y + quartered_size_y ) ]
			extract_q180 = get_gradient_in_slice[ 1 * quartered_size_x : ( 1 * quartered_size_x + quartered_size_x ), 1 * quartered_size_y : ( 1 * quartered_size_y + quartered_size_y ) ]
			extract_q270 = get_gradient_in_slice[ 0 * quartered_size_x : ( 0 * quartered_size_x + quartered_size_x ), 1 * quartered_size_y : ( 1 * quartered_size_y + quartered_size_y ) ]

			get_gradient_out_slice = np.zeros( get_gradient_in_slice.shape, dtype=get_gradient_in_slice.dtype )

			get_gradient_out_slice[ offset_x : ( offset_x + quartered_size_x ), offset_y : ( offset_y + quartered_size_y ) ] = extract_qstart
			get_gradient_out_slice[ offset_x : ( offset_x + quartered_size_x ), offset_y : ( offset_y + quartered_size_y ) ] = np.rot90( extract_q90, k=-1 )
			get_gradient_out_slice[ offset_x : ( offset_x + quartered_size_x ), offset_y : ( offset_y + quartered_size_y ) ] = np.rot90( extract_q180, k=-1 )
			get_gradient_out_slice[ offset_x : ( offset_x + quartered_size_x ), offset_y : ( offset_y + quartered_size_y ) ] = np.rot90( extract_q270, k=-3 )

			gradient_out[ :, :, z_idx ] = get_gradient_out_slice

		return gradient_out

	#
	#* Run the optimization
	#
	start_epoch = restart_epoch
	for epoch in range(start_epoch, num_epochs):
		
		# Each epoch the filters get stronger and so the permittivity must be passed through the new filters
		bayer_filter.update_filters(epoch)
		bayer_filter.update_permittivity()
  
		start_iter = 0

		if epoch == start_epoch:
			start_iter = restart_iter
			bayer_filter.current_iteration = start_epoch * num_iterations_per_epoch + start_iter

		for iteration in range(start_iter, num_iterations_per_epoch):
			logging.info("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

			job_names = {}

			fdtd_hook.switchtolayout()
			# Get current density of bayer filter after being passed through current extant filters.
			cur_density = bayer_filter.get_permittivity()		# Despite the name, here cur_density will have values between 0 and 1

			#
			# Step 1: Import the current epoch's permittivity value to the device. We create a different optimization job for:
			# - each of the polarizations for the forward source waves
			# - each of the polarizations for each of the adjoint sources
			# Since here, each adjoint source is corresponding to a focal location for a target color band, we have 
   			# <num_wavelength_bands> x <num_polarizations> adjoint sources.
			# We then enqueue each job and run them all in parallel.
			# NOTE: Polymer slab / substrate, if included, must change index when the cube index changes.
			#

			if start_from_step == 0 or start_from_step == 1:
				if start_from_step == 1:
					load_backup_vars(shelf_fn)
			
				logging.info("Beginning Step 1: Forward and Adjoint Optimizations")
				
				for dispersive_range_idx in range( 0, num_dispersive_ranges ):
					# [DEPRECATED] Account for dispersion by using the dispersion_model.
					dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
					dispersive_max_index = utility.index_from_permittivity( dispersive_max_permittivity )
	 
					# Scale from [0, 1] back to [min_device_permittivity, max_device_permittivity]
					#! This is the only cur_whatever_variable that will ever be anything other than [0, 1].
					cur_permittivity = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
					cur_index = utility.index_from_permittivity( cur_permittivity )
					logging.info(f'TiO2% is {100 * np.count_nonzero(cur_index > min_device_index) / cur_index.size}%.')
                    logging.info(f'Binarization is {100 * np.sum(np.abs(cur_density-0.5))/(cur_density.size*0.5)}%.')
    
					# Update bayer_filter data for actual permittivity (not just density 0 to 1)
					bayer_filter.update_actual_permittivity_values(min_device_permittivity, dispersive_max_permittivity)

					fdtd_hook.switchtolayout()
					fdtd_hook.select( design_import['name'] )
					fdtd_hook.importnk2( cur_index, bayer_filter_region_x, 
										bayer_filter_region_y, bayer_filter_region_z )

					for xy_idx in range(0, 2):
						disable_all_sources()	# Create a job with only the source of interest active.

						forward_sources[xy_idx]['enabled'] = 1
						forward_sources[xy_idx] = fdtd_update_object(fdtd_hook, forward_sources[xy_idx])

						job_name = f'forward_job_{xy_idx}_{dispersive_range_idx}.fsp'
						fdtd_hook.save( os.path.abspath( projects_directory_location + "/optimization.fsp") )
						job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )
	
					for adj_src_idx in range(0, num_adjoint_sources):
						for xy_idx in range(0, 2):
							disable_all_sources()	# Create a job with only the source of interest active.
							
							adjoint_sources[adj_src_idx][xy_idx]['enabled'] = 1
							adjoint_sources[adj_src_idx][xy_idx] = fdtd_update_object(fdtd_hook, adjoint_sources[adj_src_idx][xy_idx])

							job_name = f'adjoint_job_{adj_src_idx}_{xy_idx}_{dispersive_range_idx}.fsp'
							fdtd_hook.save( os.path.abspath(projects_directory_location + "/optimization.fsp") )
							job_names[ ( 'adjoint', adj_src_idx, xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )


					if add_pdaf:

						pdaf_index = quartered_and_rotate_index( cur_index, 1, 1 )
						fdtd_hook.select( design_import['name'] )
						fdtd_hook.importnk2( pdaf_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )

						for xy_idx in range(0, 2):
							disable_all_sources()	# Create a job with only the source of interest active.

							pdaf_sources[xy_idx]['enabled'] = 1
							pdaf_sources[xy_idx] = fdtd_update_object(fdtd_hook, pdaf_sources[xy_idx])

							job_name = f'pdaf_forward_job_{xy_idx}_{dispersive_range_idx}.fsp'
							fdtd_hook.save( os.path.abspath(projects_directory_location + "/optimization.fsp") )
							job_names[ ( 'pdaf_fwd', xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )


						for xy_idx in range(0, 2):
							disable_all_sources()	# Create a job with only the source of interest active.

							pdaf_adjoint_sources[xy_idx]['enabled'] = 1
							pdaf_adjoint_sources[xy_idx] = fdtd_update_object(fdtd_hook, pdaf_adjoint_sources[xy_idx])

							job_name = f'pdaf_adjoint_job_{xy_idx}_{dispersive_range_idx}.fsp'
							fdtd_hook.save( os.path.abspath(projects_directory_location + "/optimization.fsp") )
							job_names[ ( 'pdaf_adj', xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )

					# Save a copy out for Lumerical processing
					fdtd_hook.save( os.path.abspath(EVALUATION_FOLDER + "/optimization.fsp" ))

				backup_all_vars()
				if not running_on_local_machine:
					run_jobs( jobs_queue )
	
				logging.info("Completed Step 1: Forward and Adjoint Optimization Jobs Completed.")
				backup_all_vars()

			#
			# Step 2: Compute the figure of merit
			#

			if start_from_step == 0 or start_from_step == 2:
				if start_from_step == 2:
					load_backup_vars(shelf_fn)
			
				logging.info("Beginning Step 2: Computing Figure of Merit")
				
				for xy_idx in range(0, 2):
					focal_data[xy_names[xy_idx]] = [
							None for idx in range( 0, num_focal_spots ) ]
					quadrant_transmission_data[xy_names[xy_idx]] = [
							None for idx in range( 0, num_focal_spots ) ]

				for dispersive_range_idx in range( 0, num_dispersive_ranges ):
					for xy_idx in range(0, 2):
						completed_job_filepath = utility.convert_root_folder(
											job_names[ ( 'forward', xy_idx, dispersive_range_idx )], PULL_COMPLETED_JOBS_FOLDER)
						start_loadtime = time.time()
						fdtd_hook.load(os.path.abspath(completed_job_filepath))
						logging.info(f'Loading: {utility.isolate_filename(completed_job_filepath)}. Time taken: {time.time()-start_loadtime} seconds.')
	  
						fwd_e_fields = lm.get_efield(fdtd_hook, design_efield_monitor['name'])
						logging.info(f'Accessed design E-field monitor. NOTE: Field shape obtained is: {fwd_e_fields.shape}')
	
						#* Populate forward_e_fields with device E-field info
						if ( dispersive_range_idx == 0 ):
							forward_e_fields[xy_names[xy_idx]] = fwd_e_fields
						else:
							get_dispersive_range_buckets = dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]

							for bucket_idx in range( 0, len( get_dispersive_range_buckets ) ):
								# Each wavelength bucket says which wavelength (indices) correspond to which dispersive range,
								# which itself contains a map for how the wavelength buckets correspond to which specific adjoint 
								# sources on the focal plane.
								get_wavelength_bucket = spectral_focal_plane_map[ get_dispersive_range_buckets[ bucket_idx ] ]

								for spectral_idx in range( get_wavelength_bucket[ 0 ], get_wavelength_bucket[ 1 ] ):
									# fwd_e_fields has shape (3, nx, ny, nz, nλ)
									forward_e_fields[xy_names[xy_idx]][ :, :, :, :, spectral_idx ] = fwd_e_fields[ :, :, :, :, spectral_idx ]
						gc.collect()

						#* Populate arrays with adjoint source E-field and quadrant transmission info.
						# NOTE: Only these will be needed to calculate the FoM.
						for adj_src_idx in dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]:					
							focal_data[xy_names[xy_idx]][ adj_src_idx ] = lm.get_efield(fdtd_hook, focal_monitors[adj_src_idx]['name'])
							logging.info(f'Accessed adjoint E-field data at focal spot {adj_src_idx}.')

							quadrant_transmission_data[xy_names[xy_idx]][ adj_src_idx ] = lm.get_transmission_magnitude(fdtd_hook, 
																								   		transmission_monitors[adj_src_idx]['name'])
							logging.info(f'Accessed transmission data for quadrant {adj_src_idx}.')

				# Save out fields for debug purposes
				for xy_idx in range( 0, 2 ):
					for adj_src_idx in range( 0, 4 ):
						np.save( os.path.abspath(OPTIMIZATION_INFO_FOLDER + f'/debug_quadrant_transmission_data_{xy_idx}_{adj_src_idx}.npy'),
									quadrant_transmission_data[xy_names[xy_idx]][ adj_src_idx ] 
		  						)
				
				if add_pdaf:

					for xy_idx in range(0, 2):
						pdaf_focal_data[xy_names[xy_idx]] = [ None for idx in range( 0, num_focal_spots ) ]
						pdaf_quadrant_transmission_data[xy_names[xy_idx]] = [ None for idx in range( 0, num_focal_spots ) ]

					assert ( num_dispersive_ranges == 1 ), "We have implicitly assumed we are currently using only one dispersive range!"

					for dispersive_range_idx in range( 0, num_dispersive_ranges ):
						for xy_idx in range(0, 2):
							fdtd_hook.load( job_names[ ( 'pdaf_fwd', xy_idx, dispersive_range_idx ) ] )

							fwd_e_fields = lm.get_efield(fdtd_hook, design_efield_monitor['name'])

							if ( dispersive_range_idx == 0 ):

								pdaf_forward_e_fields[xy_names[xy_idx]] = fwd_e_fields
							else:
								get_dispersive_range_buckets = dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]

								for bucket_idx in range( 0, len( get_dispersive_range_buckets ) ):

									get_wavelength_bucket = spectral_focal_plane_map[ get_dispersive_range_buckets[ bucket_idx ] ]

									for spectral_idx in range( get_wavelength_bucket[ 0 ], get_wavelength_bucket[ 1 ] ):
										pdaf_forward_e_fields[xy_names[xy_idx]][ :, :, :, :, spectral_idx ] = fwd_e_fields[ :, :, :, :, spectral_idx ]


							for adj_src_idx in dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]:					
								pdaf_focal_data[xy_names[xy_idx]][ adj_src_idx ] = lm.get_efield(fdtd_hook, focal_monitors[adj_src_idx]['name'])

								pdaf_quadrant_transmission_data[xy_names[xy_idx]][ adj_src_idx ] = lm.get_transmission_magnitude(fdtd_hook, 
																										 		transmission_monitors[adj_src_idx]['name'])

				#* These arrays record intensity and transmission FoMs for all focal spots
				transmission_green_blue_xpol = 0
				transmission_green_blue_ypol = 0
				transmission_green_red_xpol = 0
				transmission_green_red_ypol = 0

				for focal_idx in range(0, num_focal_spots):             # For each focal spot:

					# choose which polarization(s) are directed to this focal spot
					polarizations = polarizations_focal_plane_map[focal_idx]
	
					
					for polarization_idx in range(0, len(polarizations)):			# For each polarization:
						# Access a subset of the data pulled from Lumerical.
						get_focal_data = focal_data[polarizations[polarization_idx]]                                    # shape: (#focalspots, 3, nx, ny, nz, nλ)
						get_quad_transmission_data = quadrant_transmission_data[polarizations[polarization_idx]]        # shape: (#focalspots, nλ)
	
						# TODO: Try and offload these parts to another wavelength modular thing.
						max_intensity_weighting = max_intensity_by_wavelength[spectral_focal_plane_map[focal_idx][0] : spectral_focal_plane_map[focal_idx][1] : 1]
						# Use the max_intensity_weighting to renormalize the intensity distribution used to calculate the FoM
						# and insert the custom weighting schemes here as well.
						total_weighting = max_intensity_weighting / weight_focal_plane_map_performance_weighting[focal_idx]
						one_over_weight_focal_plane_map_performance_weighting = 1. / weight_focal_plane_map_performance_weighting[focal_idx]
	
						for spectral_idx in range(0, total_weighting.shape[0]):			# For each wavelength:

							# weight_spectral_rejection = 1.0
							weight_spectral_rejection = weight_individual_wavelengths_by_quad[ focal_idx, spectral_focal_plane_map[ focal_idx ][ 0 ] + spectral_idx  ] 
							if do_rejection:
								weight_spectral_rejection = spectral_focal_plane_map_directional_weights[ focal_idx ][ spectral_idx ]
		
							#* Transmission FoM Calculation: FoM = T
							# \text{FoM}_{\text{transmission}} = \sum_i^{} \left{ \sum_\lambda^{} w_{\text{rejection}}(\lambda) w_{f_i} T_{f_i}(\lambda) \right}
							# https://imgur.com/7qefinH.png
							transmission_addition = weight_spectral_rejection * 																					\
															get_quad_transmission_data[focal_idx][spectral_focal_plane_map[focal_idx][0] + spectral_idx] / 			\
															one_over_weight_focal_plane_map_performance_weighting
							fom_evolution['transmission'][epoch, iteration,
														focal_idx, polarization_idx,
														spectral_focal_plane_map[focal_idx][0] + spectral_idx] += transmission_addition
			
							#* Intensity FoM Calculation: FoM = |E(x_0)|^2                 
							# \text{FoM}_{\text{intensity}} = \sum_i^{} \left{ \sum_\lambda^{} w_{\text{rejection}}(\lambda) w_{f_i}
							# \frac{\left|E_{f_i}(\lambda)\right|^2}{I_{\text{norm, Airy}}}
							# https://i.imgur.com/5jcN01k.png
							intensity_addition = weight_spectral_rejection * 																								\
														np.sum(( np.abs(get_focal_data[focal_idx][:, 0, 0, 0, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 /	\
																total_weighting[spectral_idx] ))
							fom_evolution['intensity'][epoch, iteration,
														focal_idx, polarization_idx,
														spectral_focal_plane_map[focal_idx][0] + spectral_idx] += intensity_addition
			   
			   
							if focal_idx == 1:
								if polarization_idx == 0:
									transmission_green_blue_xpol += transmission_addition
								else:
									transmission_green_blue_ypol += transmission_addition
							if focal_idx == 3:
								if polarization_idx == 0:
									transmission_green_red_xpol += transmission_addition
								else:
									transmission_green_red_ypol += transmission_addition
				
	
				# NOTE: Reminder that fom_evolution is a dictionary with keys: fom_types
				# and values being numpy arrays of shape (epochs, iterations, focal_areas, polarizations, wavelengths)
				figure_of_merit = np.sum(fom_evolution[optimization_fom][epoch, iteration])
				figure_of_merit_evolution[epoch, iteration] = figure_of_merit
	
				# Save out variables that need saving for restart and debug purposes
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/figure_of_merit.npy"), np.sum(fom_evolution[optimization_fom], (2,3,4)))
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/fom_by_focal_pol_wavelength.npy"), fom_evolution[optimization_fom])
				for fom_type in fom_types:
					np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + f"/{fom_type}_by_focal_pol_wavelength.npy"), fom_evolution[fom_type])

				logging.info("Figure of Merit So Far:")
				logging.info(f"{np.sum(fom_evolution[optimization_fom], (2,3,4))}")
				logging.info("Saved out figures of merit debug info.")

				# Plot key information such as Figure of Merit evolution for easy visualization and checking in the middle of optimizations
				plotter.plot_fom_trace(figure_of_merit_evolution, OPTIMIZATION_PLOTS_FOLDER)
				plotter.plot_quadrant_transmission_trace(fom_evolution['transmission'], OPTIMIZATION_PLOTS_FOLDER)
				plotter.plot_individual_quadrant_transmission(quadrant_transmission_data, lambda_values_um, OPTIMIZATION_PLOTS_FOLDER,
												  		epoch, iteration=30)	# continuously produces only one plot per epoch to save space
				plotter.plot_overall_transmission_trace(fom_evolution['transmission'], OPTIMIZATION_PLOTS_FOLDER)
				plotter.visualize_device(cur_index, OPTIMIZATION_PLOTS_FOLDER)



				#* Process the various figures of merit for performance weighting.
				# Copy everything so as not to disturb the original data
				process_fom_evolution = copy.deepcopy(fom_evolution)

				process_fom_by_focal_spot = np.sum(process_fom_evolution[optimization_fom][epoch, iteration], (1,2))
				process_transmission_by_quadrant = np.sum(process_fom_evolution['transmission'][epoch, iteration], (1,2))
				process_fom_by_wavelength = np.sum(process_fom_evolution[optimization_fom][epoch, iteration], (0,1))
				process_transmission_by_wavelength = np.sum(process_fom_evolution['transmission'][epoch, iteration], (0,1))
				if do_rejection:        # softplus each array
					process_fom_by_focal_spot = utility.softplus(process_fom_by_focal_spot, softplus_kappa)
					process_transmission_by_quadrant = utility.softplus(process_transmission_by_quadrant, softplus_kappa)
					process_fom_by_wavelength = utility.softplus(process_fom_by_wavelength, softplus_kappa)
					process_transmission_by_wavelength = utility.softplus(process_transmission_by_wavelength, softplus_kappa)
	
				# Performance weighting is an adjustment factor to the gradient computation so that each focal area's FoM
				# seeks the same efficiency. See Eq. 2, https://doi.org/10.1038/s41598-021-88785-5
				performance_weighting = calculate_performance_weighting(process_fom_by_focal_spot)
				if weight_by_quadrant_transmission:
					performance_weighting = calculate_performance_weighting(process_transmission_by_quadrant)
				if weight_by_individual_wavelength:
					performance_weighting = calculate_performance_weighting(process_fom_by_wavelength)
					if weight_by_quadrant_transmission:
						performance_weighting = calculate_performance_weighting(process_transmission_by_wavelength)

				
				weight_greens = np.ones( 4 )
				stack_green_T = np.array( [ transmission_green_blue_xpol, transmission_green_blue_ypol, transmission_green_red_xpol, transmission_green_red_ypol ] )
				if do_green_balance:
					weight_greens = calculate_performance_weighting(stack_green_T)

				logging.info( f"Green balance = {stack_green_T}")
				logging.info( f"Green weights = {weight_greens}")


				if add_pdaf:
					for focal_idx in range(0, num_focal_spots):
						compute_pdaf_transmission_fom = 0

						polarizations = [ 'x', 'y' ]

						for polarization_idx in range(0, len(polarizations)):
							get_pdaf_quad_transmission_data = pdaf_quadrant_transmission_data[polarizations[polarization_idx]]
							compute_pdaf_transmission_fom += np.sum( get_pdaf_quad_transmission_data[ pdaf_adj_src_idx ] )

						pdaf_transmission_evolution[ epoch, iteration ] = compute_pdaf_transmission_fom

					np.save( os.path.abspath(OPTIMIZATION_INFO_FOLDER + f"/pdaf_transmission.npy"), pdaf_transmission_evolution )
	
				
				logging.info("Completed Step 2: Computed Figures of Merit and performance weightings.")
				backup_all_vars()
				gc.collect()


			#
			# Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
			# gradients for x- and y-polarized forward sources.
			#

			if start_from_step == 0 or start_from_step == 3:
				if start_from_step == 3:
					load_backup_vars(shelf_fn)

				logging.info("Beginning Step 3: Adjoint Optimization Jobs")
	
				## Get the device / current permittivity's dimensions in terms of (geometry) voxels      
				# cur_permittivity_shape = cur_permittivity.shape				 # NOTE: UNUSED.
	
				# Initialize arrays to store the gradients of each polarization, each the shape of the voxel mesh
				# This array therefore has shape (polarizations, mesh_lateral, mesh_lateral, mesh_vertical)
				xy_polarized_gradients = [ np.zeros(field_shape, dtype=np.complex), np.zeros(field_shape, dtype=np.complex) ]
				

				for adj_src_idx in range(0, num_adjoint_sources):
					# Get the polarizations and wavelengths that will be directed to that focal area
					polarizations = polarizations_focal_plane_map[adj_src_idx]
					spectral_indices = spectral_focal_plane_map[adj_src_idx]
	
					# Derive the gradient of the performance weighting from the performance weighting functions in Step 2. 
	 				# NOTE: Bear in mind that the performance weighting might have different shapes.
					# TODO: Maybe move this to Step 2?
					gradient_performance_weight = 0
					if not weight_by_individual_wavelength:
						# TODO: Don't we need to take the gradient of calculate_performance_weighting()?
						# And that was an identity function so doesn't this become 0 like above...
						gradient_performance_weight = performance_weighting[adj_src_idx]

						if do_rejection:
							if weight_by_quadrant_transmission:
								gradient_performance_weight *= utility.softplus_prime( process_transmission_by_quadrant[ adj_src_idx ], softplus_kappa )
							else:
								gradient_performance_weight *= utility.softplus_prime( process_fom_by_focal_spot[ adj_src_idx ], softplus_kappa )
		
					# Check which dispersive range goes to this adjoint source
					lookup_dispersive_range_idx = adjoint_src_to_dispersive_range_map[ adj_src_idx ]
	 
					# Store the design E-field for the adjoint simulations
					adjoint_e_fields = []
					for xy_idx in range(0, 2):
						completed_job_filepath = utility.convert_root_folder(
											job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx )], PULL_COMPLETED_JOBS_FOLDER)
						start_loadtime = time.time()
						fdtd_hook.load(os.path.abspath(completed_job_filepath))
						logging.info(f'Loading: {utility.isolate_filename(completed_job_filepath)}. Time taken: {time.time()-start_loadtime} seconds.')

						adjoint_e_fields.append(
							lm.get_efield(fdtd_hook, design_efield_monitor['name']))
						logging.info('Accessed design E-field monitor.')
					
					for pol_idx in range(0, len(polarizations)):		# For each polarization:
						pol_name = polarizations[pol_idx]
						get_focal_data = focal_data[pol_name]
						pol_name_to_idx = polarization_name_to_idx[pol_name]
	
						# TODO: Why are we doing polarizations and xy_idx twice? Doesn't this double the for-loop?
						for xy_idx in range(0, 2):
							#* Conjugate of E_{old}(x_0) -field at the adjoint source of interest, with direction along the polarization
							# This is going to be the amplitude of the dipole-adjoint source driven at the focal plane
							source_weight = np.squeeze( np.conj(
								get_focal_data[adj_src_idx][xy_idx, 0, 0, 0, spectral_indices[0]:spectral_indices[1]:1]
							) )

							# TODO: Try and offload these parts to another wavelength modular thing.
							# todo: do you need to change this if you weight by transmission? the figure of merit still has this weight, the transmission is just
							# todo: guiding additional weights on top of this
							# NOTE: The weighting calculations here are the same as in Step 2 except that the spectral indices and focal plane weightings are determined differently.

							max_intensity_weighting = max_intensity_by_wavelength[spectral_indices[0] : spectral_indices[1] : 1]
							total_weighting = max_intensity_weighting / weight_focal_plane_map[adj_src_idx]
		
							#
							# We need to properly account here for the current real and imaginary index
							# because they both contribute in the end to the real part of the gradient ΔFoM/Δε_r
							# in Eq. 5 of Lalau-Keraly paper https://doi.org/10.1364/OE.21.021693
							#
							dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ lookup_dispersive_range_idx ] )
							delta_real_permittivity = np.real( dispersive_max_permittivity - min_device_permittivity )
							delta_imag_permittivity = np.imag( dispersive_max_permittivity - min_device_permittivity )
	   
	   
							for spectral_idx in range(0, len( source_weight ) ):		# For each wavelength:
								
								# TODO: Gradient performance weight is set above already - could probably merge the two blocks or make it a function
								if weight_by_individual_wavelength:
									gradient_performance_weight = (
										weight_individual_wavelengths_by_quad[ adj_src_idx, spectral_indices[ 0 ] + spectral_idx  ] *
										weight_individual_wavelengths[ spectral_indices[ 0 ] + spectral_idx ] *
										performance_weighting[ spectral_indices[0] + spectral_idx ] )
									# gradient_performance_weight = weight_individual_wavelengths[ spectral_indices[ 0 ] + spectral_idx ] * performance_weighting[ spectral_indices[0] + spectral_idx ]
									# gradient_performance_weight = performance_weighting[ spectral_indices[0] + spectral_idx ]

									if do_rejection:
										if weight_by_quadrant_transmission:
											gradient_performance_weight *= softplus_prime( transmission_by_wavelength[ spectral_idx ] )
										else:
											gradient_performance_weight *= softplus_prime( intensity_fom_by_wavelength[ spectral_idx ] )
										
										# gradient_performance_weight *= softplus_prime( intensity_fom_by_wavelength[ spectral_idx ] )
		  
								if do_rejection:
									gradient_performance_weight *= spectral_focal_plane_map_directional_weights[ adj_src_idx ][ spectral_idx ]

								# Green weight balances
								green_weight = 1.0
								if do_green_balance:
									if adj_src_idx == 1:
										if pol_idx == 0:
											green_weight = weight_greens[ 0 ]
										else:
											green_weight = weight_greens[ 1 ]
									if adj_src_idx == 3:
										if pol_idx == 0:
											green_weight = weight_greens[ 2 ]
										else:
											green_weight = weight_greens[ 3 ]
		   
								# See Eqs. (5), (6) of Lalau-Keraly et al: https://doi.org/10.1364/OE.21.021693
								gradient_component = np.sum(
									green_weight * 																		# Green balance weighting
		 							(source_weight[spectral_idx] * 														# Amplitude of electric dipole at x_0
						 			gradient_performance_weight / total_weighting[spectral_idx]) *						# Performance weighting, Manual override weighting, and intensity normalization factors
									adjoint_e_fields[xy_idx][:, :, :, :, spectral_indices[0] + spectral_idx] *			# E-field induced at x from electric dipole at x_0
									forward_e_fields[pol_name][:, :, :, :, spectral_indices[0] + spectral_idx],			# E_old(x)
									axis=0)
	
								# Permittivity factor in amplitude of electric dipole at x_0
								# Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)
								real_part_gradient = np.real( gradient_component )
								imag_part_gradient = -np.imag( gradient_component )
								get_grad_density = delta_real_permittivity * real_part_gradient + delta_imag_permittivity * imag_part_gradient

								assert xy_polarized_gradients[0].shape == get_grad_density.shape, f"The sizes of field_shape {field_shape} and gradient_component {gradient_component.shape} are not the same."
								xy_polarized_gradients[pol_name_to_idx] += get_grad_density
								# TODO: The shape of this should be (epoch, iteration, nx, ny, nz, n_adj, n_pol, nλ) and right now it's missing nx,ny,nz
								# todo: But also if we put that in, then the size will rapidly be too big
								# gradient_evolution[epoch, iteration, :,:,:,
								# 					adj_src_idx, xy_idx,
								# 					spectral_indices[0] + spectral_idx] += get_grad_density

				if add_pdaf:

					xy_polarized_gradients_pdaf = [ np.zeros(field_shape, dtype=np.complex), np.zeros(field_shape, dtype=np.complex) ]

					polarizations = [ 'x', 'y' ]

					pdaf_adjoint_e_fields = []
					for xy_idx in range(0, 2):
						#
						# This is ok because we are only going to be using the spectral idx corresponding to this range in the summation below
						#
						fdtd_hook.load( job_names[ ( 'pdaf_adj', xy_idx, lookup_dispersive_range_idx ) ] )

						pdaf_adjoint_e_fields.append(
							lm.get_efield(fdtd_hook, design_efield_monitor['name']))


					for pol_idx in range(0, len(polarizations)):
						pol_name = polarizations[pol_idx]
						get_pdaf_focal_data = pdaf_focal_data[pol_name]
						pol_name_to_idx = polarization_name_to_idx[pol_name]

						for xy_idx in range(0, 2):
							source_weight = np.squeeze( np.conj(
								get_pdaf_focal_data[pdaf_adj_src_idx][xy_idx, 0, 0, 0, :]) )

							#
							# todo: do you need to change this if you weight by transmission? the figure of merit still has this weight, the transmission is just
							# guiding additional weights on top of this
							#
							max_intensity_weighting = pdaf_max_intensity_by_wavelength
							total_weighting = max_intensity_weighting

							#
							# We need to properly account here for the current real and imaginary index
							#
							dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ 0 ] )
							delta_real_permittivity = np.real( dispersive_max_permittivity - min_device_permittivity )
							delta_imag_permittivity = np.imag( dispersive_max_permittivity - min_device_permittivity )

							for spectral_idx in range(0, len( source_weight ) ):

								#
								# No performance weighting in here for now
								#

								pdaf_gradient_component = np.sum(
									(source_weight[spectral_idx] * 1.0 / total_weighting[spectral_idx]) *
									pdaf_adjoint_e_fields[xy_idx][:, :, :, :, spectral_idx] *
									pdaf_forward_e_fields[pol_name][:, :, :, :, spectral_idx],
									axis=0)

								pdaf_real_part_gradient = np.real( pdaf_gradient_component )
								pdaf_imag_part_gradient = -np.imag( pdaf_gradient_component )

								pdaf_get_grad_density = delta_real_permittivity * pdaf_real_part_gradient + delta_imag_permittivity * pdaf_imag_part_gradient

								xy_polarized_gradients_pdaf[pol_name_to_idx] += pdaf_get_grad_density

				##  Turned this off because savefile space is too large.
				# np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + f"/gradient_by_focal_pol_wavelength.npy"), gradient_evolution)

				logging.info("Completed Step 3: Retrieved Adjoint Optimization results and calculated gradient.")
				gc.collect()
				backup_all_vars()


			#
			# Step 4: Step the design variable.
			#

			if start_from_step == 0 or start_from_step == 4:
				if start_from_step == 4:
					load_backup_vars(shelf_fn)
			
				logging.info("Beginning Step 4: Stepping Design Variable.")
				# TODO: Rewrite this part to use optimizers / NLOpt. -----------------------------------
	
				# Get the full device gradient by summing the x,y polarization components 
				device_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
				# Project / interpolate the device_gradient, the values of which we have at each (mesh) voxel point, and obtain it at each (geometry) voxel point
				device_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), device_gradient, bayer_filter_region, method='linear' )
	
				if add_pdaf:
					device_gradient_pdaf = 2 * ( xy_polarized_gradients_pdaf[ 0 ] + xy_polarized_gradients_pdaf[ 1 ] )

					device_gradient_interpolated_pdaf = interpolate.interpn(
						( gradient_region_x, gradient_region_y, gradient_region_z ), device_gradient_pdaf, bayer_filter_region, method='linear' )

					device_gradient_interpolated_pdaf = quarted_gradient_accumulate(
						device_gradient_interpolated_pdaf,
						0, 0 )

					unshuffle_pdaf_grad = np.zeros( device_gradient_interpolated_pdaf.shape, dtype=device_gradient_interpolated_pdaf.dtype )
					unshuffle_pdaf_grad[
						device_gradient_interpolated_pdaf.shape[ 0 ] // 2 : ,
						device_gradient_interpolated_pdaf.shape[ 1 ] // 2 : ,
						: ] = device_gradient_interpolated_pdaf[
						0 : ( device_gradient_interpolated_pdaf.shape[ 0 ] // 2 ),
						0 : ( device_gradient_interpolated_pdaf.shape[ 1 ] // 2 ),
						:
					]

					pdaf_weight_relative = 0.5

					device_gradient_interpolated = (
						( device_gradient_interpolated / np.sqrt( np.sum( np.abs( device_gradient_interpolated )**2 ) ) ) +
						pdaf_weight_relative * ( unshuffle_pdaf_grad / np.sqrt( np.sum( np.abs( unshuffle_pdaf_grad )**2 ) ) ) )


					# device_gradient_interpolated = (
					# 	( unshuffle_pdaf_grad / np.sqrt( np.sum( np.abs( unshuffle_pdaf_grad )**2 ) ) ) )
	 
				if enforce_xy_gradient_symmetry:
					device_gradient_interpolated = 0.5 * ( device_gradient_interpolated + np.swapaxes( device_gradient_interpolated, 0, 1 ) )
				device_gradient = device_gradient_interpolated.copy()
	
				# Backpropagate the design_gradient through the various filters to pick up (per the chain rule) gradients from each filter
				# so the final product can be applied to the (0-1) scaled permittivity
				design_gradient = bayer_filter.backpropagate( device_gradient )

				#
				#* Begin scaling of step size so that the design change stays within epoch_design_change limits in config
				#
	
				# Control the maximum that the design is allowed to change per epoch. This will increase according to the limits set, for each iteration
				max_change_design = epoch_start_design_change_max
				min_change_design = epoch_start_design_change_min

				if num_iterations_per_epoch > 1:
					max_change_design = (
						epoch_end_design_change_max +
						(num_iterations_per_epoch - 1 - iteration) * (epoch_range_design_change_max / (num_iterations_per_epoch - 1))
					)
					min_change_design = (
						epoch_end_design_change_min +
						(num_iterations_per_epoch - 1 - iteration) * (epoch_range_design_change_min / (num_iterations_per_epoch - 1))
					)
	
				# Step size
	 
				cur_design_variable = bayer_filter.get_design_variable()

				if optimizer_algorithm.lower() not in ['gradient descent']:
					use_fixed_step_size = True
				if use_fixed_step_size:
					step_size = fixed_step_size
				else:
					step_size = step_size_start
					check_last = False
					last = 0

					while True:
						# Gets proposed design variable according to Eq. S2, OPTICA Paper Supplement: https://doi.org/10.1364/OPTICA.384228
						# Divides step size by 2 until the difference in the design variable is within the ranges set.
						
						proposed_design_variable = cur_design_variable + step_size * design_gradient
						proposed_design_variable = np.maximum(                       # Makes sure that it's between 0 and 1
																np.minimum(proposed_design_variable, 1.0),
																0.0)

						difference = np.abs(proposed_design_variable - cur_design_variable)
						max_difference = np.max(difference)

						if (max_difference <= max_change_design) and (max_difference >= min_change_design):
							break										# max_difference in [min_change_design, max_change_design]
						elif (max_difference <= max_change_design):		# max_difference < min_change_design, by definition
							step_size *= 2
							if (last ^ 1) and check_last:	# For a Boolean, last ^ 1 = !last
								break						# skips the next two lines only if last=0 and check_last=True
							check_last = True
							last = 1
						else:											# max_difference > max_change_design
							step_size /= 2
							if (last ^ 0) and check_last:	# For a Boolean, last ^ 0 = last
								break						# skips the next two lines only if last=1 and check_last=True
							check_last = True
							last = 0

					# Sets the starting step size - this will be used as the step size in the next iteration.
					step_size_start = step_size
	
				#
				#* Now that we have obtained the right step size, and the (backpropagated) design gradient,
				#* we will finalize everything and save it out.
				#
	
				# Get current values of design variable before perturbation, for later comparison.
				last_design_variable = cur_design_variable.copy()
	
				# Feed proposed design gradient into bayer_filter, run it through filters and update permittivity
				# negative of the device gradient because the step() function has a - sign instead of a +
				# NOT the design gradient; we do not want to backpropagate twice!

				if optimizer_algorithm.lower() in ['adam']:
					adam_moments = bayer_filter.step_adam(epoch*num_iterations_per_epoch+iteration, -device_gradient,
													adam_moments, step_size, adam_betas)
				else:
					bayer_filter.step(-device_gradient, step_size)
				cur_design_variable = bayer_filter.get_design_variable()
				# Get permittivity after passing through all the filters of the device
				cur_design = bayer_filter.get_permittivity()
	
				average_design_variable_change = np.mean( np.abs(cur_design_variable - last_design_variable) )
				max_design_variable_change = np.max( np.abs(cur_design_variable - last_design_variable) )

				step_size_evolution[epoch][iteration] = step_size
				average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
				max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change
	
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + '/device_gradient.npy'), device_gradient)			# Gradient of FoM obtained from adjoint method
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + '/design_gradient.npy'), design_gradient)			# device_gradient, propagated through chain-rule filters
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/step_size_evolution.npy"), step_size_evolution)
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/cur_design_variable.npy"), cur_design_variable)	# Current Design variable i.e. density between 0 and 1
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/cur_design.npy"), cur_design)						# Current Design variable after filters
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/cur_design_variable_epoch_" + str( epoch ) + ".npy"), cur_design_variable)
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/average_design_change_evolution.npy"), average_design_variable_change_evolution)
				np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/max_design_change_evolution.npy"), max_design_variable_change_evolution)
				
				logging.info("Completed Step 4: Saved out cur_design_variable.npy and other files.")
				backup_all_vars()


shutil.copy2(utility.loggingArgs['filename'], SAVED_SCRIPTS_FOLDER + "/" + utility.loggingArgs['filename'])
logging.info("Reached end of file. Program completed.")