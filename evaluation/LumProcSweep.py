import os
import sys
import time
import shelve
import gc
import copy
import re

# Add filepath to default path
sys.path.append(os.getcwd())
#! Gets all parameters from config file - explicitly calls it so that the variables pass into this module's globals()
from configs.LumProcParameters import *

# Lumerical hook Python library API
import imp
imp.load_source( "lumapi", lumapi_filepath)
import lumapi

import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
							   AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

import queue
import subprocess

# Start logging - Logger is already initialized through the call to cfg_to_params_sony.py
logging.info(f'Starting up Lumerical processing file: {os.path.basename(__file__)}')
logging.info("All modules loaded.")
logging.debug(f'Current working directory: {os.getcwd()}')

#* Handle Input Arguments and Environment Variables
parser = argparse.ArgumentParser(description=\
					"Post-processing file for Bayer Filter (SONY 2022-2023). Obtains device metrics and produces plots for each.")
for argument in utility.argparse_universal_args:
	parser.add_argument(argument[0], argument[1], **(argument[2]))
args = parser.parse_args()

#* Output / Save Paths
DATA_FOLDER = projects_directory_location
PLOTS_FOLDER = os.path.join(projects_directory_location, 'plots')

#* SLURM Environment
if not running_on_local_machine:
	cluster_hostnames = utility.get_slurm_node_list()
	num_nodes_available = int(os.getenv('SLURM_JOB_NUM_NODES'))	# should equal --nodes in batch script. Could also get this from len(cluster_hostnames)
	num_cpus_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))	# should equal 8 or --ntasks-per-node in batch script. Could also get this SLURM_CPUS_ON_NODE or SLURM_JOB_CPUS_PER_NODE?
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


#
#* Create FDTD hook
#
fdtd_hook = lumapi.FDTD()
# Load saved session from adjoint optimization
fdtd_hook.newproject()
fdtd_hook.load(os.path.abspath(os.path.join(python_src_directory, device_filename)))
fdtd_hook.save(os.path.abspath(os.path.join(projects_directory_location, device_filename)))

# Grab positions and dimensions from the .fsp itself. Replaces values from LumProcParameters.py
#! No checks are done to see if there are discrepancies.
device_size_lateral_um = fdtd_hook.getnamed('design_import','x span') / 1e-6
device_vertical_maximum_um = fdtd_hook.getnamed('design_import','z max') / 1e-6
device_vertical_minimum_um = fdtd_hook.getnamed('design_import','z min') / 1e-6
# fdtd_region_size_lateral_um = fdtd_hook.getnamed('FDTD','x span') / 1e-6
fdtd_region_maximum_vertical_um = fdtd_hook.getnamed('FDTD','z max') / 1e-6
fdtd_region_minimum_vertical_um = fdtd_hook.getnamed('FDTD','z min') / 1e-6
fdtd_region_size_vertical_um = fdtd_region_maximum_vertical_um + abs(fdtd_region_minimum_vertical_um)
monitorbox_size_lateral_um = device_size_lateral_um + 2 * sidewall_thickness_um + (mesh_spacing_um)*5
monitorbox_vertical_maximum_um = device_vertical_maximum_um + (mesh_spacing_um*5)
monitorbox_vertical_minimum_um = device_vertical_minimum_um - (mesh_spacing_um*5)
adjoint_vertical_um = fdtd_hook.getnamed('transmission_focal_monitor_', 'z') / 1e-6

# TODO: Think we need to wrap this part in a function or class or another module honestly --------------------------------------------------------------------------------------------
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
	
	for i in range(2):	# Repeat the following twice because the order that the properties are changed matters. 
						# (Sometimes certain properties are grayed out depending on the value of others)
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

	for xy_idx in range(0, 2):
		# object_list.append( forward_sources[xy_idx]['name'] )
		object_list.append(['forward_sources', xy_idx])

	for adj_src_idx in range(0, num_adjoint_sources):
		for xy_idx in range(0, 2):
			# object_list.append( adjoint_sources[adj_src_idx][xy_idx]['name'] )
			object_list.append(['adjoint_sources', adj_src_idx, xy_idx])
   
	if add_pdaf:
		for xy_idx in range(0, 2):
			object_list.append(['pdaf_sources', xy_idx])

		for xy_idx in range(0, 2):
			object_list.append(['pdaf_adjoint_sources', xy_idx])

	disable_objects(object_list)

#
#* Step 0: Set up structures, regions and monitors from which E-fields, Poynting fields, and transmission data will be drawn.
# Also any other necessary editing of the Lumerical environment and objects.
#

if start_from_step == 0:
	logging.info("Beginning Step 0: Monitor Setup")

	# Create dictionary to duplicate FDTD SimObjects as dictionaries for saving out
	# This dictionary captures the states of all FDTD objects in the simulation, and has the advantage that it can be saved out or exported.
	# The dictionary will be our master record, and is continually updated for every change in FDTD.
	fdtd_objects = {}
	for simObj in fdtd_get_all_simobjects(fdtd_hook):
		fdtd_objects[simObj['name']] = fdtd_simobject_to_dict(simObj)
	# TODO: Probably have to do some processing because this way of loading things means 
 	# todo: fdtd_objects will have dictionaries for individual transmission monitors i.e. fdtd_objects['transmission_monitor_2'] = dict{}
	# todo: instead of fdtd_objects['transmission_monitors'] =  list(dict{})

	#
	# Set up the FDTD monitors that weren't present during adjoint optimization
	# We are using dict-like access: see https://support.lumerical.com/hc/en-us/articles/360041873053-Session-management-Python-API
	# The line fdtd_objects['FDTD'] = fdtd means that both sides of the equation point to the same object.
	# See https://stackoverflow.com/a/10205681 and https://nedbatchelder.com/text/names.html

	# Monitor right below source to measure overall transmission / reflection
	src_transmission_monitor = {}
	src_transmission_monitor['name'] = 'src_transmission_monitor'
	src_transmission_monitor['type'] = 'DFTMonitor'
	src_transmission_monitor['power_monitor'] = True
	src_transmission_monitor.update(monitors_keyed[src_transmission_monitor['name']])
	src_transmission_monitor['x'] = 0 * 1e-6
	src_transmission_monitor['x span'] = (monitorbox_size_lateral_um*1.1) * 1e-6
	# src_transmission_monitor['x span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','x span') /2 * 6) / 1e-6) * 1e-6  
	src_transmission_monitor['y'] = 0 * 1e-6
	src_transmission_monitor['y span'] = (monitorbox_size_lateral_um*1.1) * 1e-6
	# src_transmission_monitor['y span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','x span') /2 * 6) / 1e-6) * 1e-6
	src_transmission_monitor['z'] = monitorbox_vertical_maximum_um * 1e-6 #fdtd_hook.getnamed('forward_src_x','z') - (mesh_spacing_um*5)*1e-6
	src_transmission_monitor['override global monitor settings'] = 1
	src_transmission_monitor['use wavelength spacing'] = 1
	src_transmission_monitor['use source limits'] = 1
	src_transmission_monitor['frequency points'] = num_design_frequency_points
	src_transmission_monitor = fdtd_update_object(fdtd_hook, src_transmission_monitor, create_object=True)
	fdtd_objects['src_transmission_monitor'] = src_transmission_monitor

	# Monitor at top of device across FDTD to measure source power that misses the device
	src_spill_monitor = {}
	src_spill_monitor['name'] = 'src_spill_monitor'
	src_spill_monitor['type'] = 'DFTMonitor'
	src_spill_monitor['power_monitor'] = True
	src_spill_monitor.update(monitors_keyed[src_spill_monitor['name']])
	src_spill_monitor['x'] = 0 * 1e-6
	src_spill_monitor['x span'] = fdtd_region_size_lateral_um * 1e-6
	src_spill_monitor['y'] = 0 * 1e-6
	src_spill_monitor['y span'] = fdtd_region_size_lateral_um * 1e-6
	src_spill_monitor['z'] = monitorbox_vertical_maximum_um * 1e-6
	src_spill_monitor['override global monitor settings'] = 1
	src_spill_monitor['use wavelength spacing'] = 1
	src_spill_monitor['use source limits'] = 1
	src_spill_monitor['frequency points'] = num_design_frequency_points
	src_spill_monitor = fdtd_update_object(fdtd_hook, src_spill_monitor, create_object=True)
	fdtd_objects['src_spill_monitor'] = src_spill_monitor

	# Monitor right above device to measure input power
	incident_aperture_monitor = {}
	incident_aperture_monitor['name'] = 'incident_aperture_monitor'
	incident_aperture_monitor['type'] = 'DFTMonitor'
	incident_aperture_monitor['power_monitor'] = True
	incident_aperture_monitor.update(monitors_keyed[incident_aperture_monitor['name']])
	incident_aperture_monitor['x'] = 0 * 1e-6
	incident_aperture_monitor['x span'] = monitorbox_size_lateral_um * 1e-6
	incident_aperture_monitor['y'] = 0 * 1e-6
	incident_aperture_monitor['y span'] = monitorbox_size_lateral_um * 1e-6
	incident_aperture_monitor['z'] = monitorbox_vertical_maximum_um * 1e-6
	incident_aperture_monitor['override global monitor settings'] = 1
	incident_aperture_monitor['use wavelength spacing'] = 1
	incident_aperture_monitor['use source limits'] = 1
	incident_aperture_monitor['frequency points'] = num_design_frequency_points
	incident_aperture_monitor = fdtd_update_object(fdtd_hook, incident_aperture_monitor, create_object=True)
	fdtd_objects['incident_aperture_monitor'] = incident_aperture_monitor

	# Monitor right below device to measure exit aperture power
	exit_aperture_monitor = {}
	exit_aperture_monitor['name'] = 'exit_aperture_monitor'
	exit_aperture_monitor['type'] = 'DFTMonitor'
	exit_aperture_monitor['power_monitor'] = True
	exit_aperture_monitor.update(monitors_keyed[exit_aperture_monitor['name']])
	exit_aperture_monitor['x'] = 0 * 1e-6
	exit_aperture_monitor['x span'] = monitorbox_size_lateral_um * 1e-6
	exit_aperture_monitor['y'] = 0 * 1e-6
	exit_aperture_monitor['y span'] = monitorbox_size_lateral_um * 1e-6
	exit_aperture_monitor['z'] = monitorbox_vertical_minimum_um * 1e-6
	exit_aperture_monitor['override global monitor settings'] = 1
	exit_aperture_monitor['use wavelength spacing'] = 1
	exit_aperture_monitor['use source limits'] = 1
	exit_aperture_monitor['frequency points'] = num_design_frequency_points
	exit_aperture_monitor = fdtd_update_object(fdtd_hook, exit_aperture_monitor, create_object=True)
	fdtd_objects['exit_aperture_monitor'] = exit_aperture_monitor
	
	# Sidewall monitors closing off the monitor box around the splitter cube device, to measure side scattering
	side_monitors = []
	sm_x_pos = [monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2, 0]
	sm_y_pos = [0, monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2]
	sm_xspan_pos = [0, monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um]
	sm_yspan_pos = [monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um, 0]
	# sm_monitor_type = ['2D X-normal', '2D Y-normal', '2D X-normal', '2D Y-normal']
	
	for sm_idx in range(0, 4):      # We are not using num_sidewalls because these monitors MUST be created
		side_monitor = {}
		side_monitor['name'] = 'side_monitor_' + str(sm_idx)
		side_monitor['type'] = 'DFTMonitor'
		side_monitor['power_monitor'] = True
		side_monitor.update(monitors_keyed[side_monitor['name']])
		#side_monitor['x'] = sm_x_pos[sm_idx] * 1e-6
		side_monitor['x max'] =  (sm_x_pos[sm_idx] + sm_xspan_pos[sm_idx]/2) * 1e-6   # must bypass setting 'x span' because this property is invalid if the monitor is X-normal
		side_monitor['x min'] =  (sm_x_pos[sm_idx] - sm_xspan_pos[sm_idx]/2) * 1e-6
		#side_monitor['y'] = sm_y_pos[sm_idx] * 1e-6
		side_monitor['y max'] =  (sm_y_pos[sm_idx] + sm_yspan_pos[sm_idx]/2) * 1e-6   # must bypass setting 'y span' because this property is invalid if the monitor is Y-normal
		side_monitor['y min'] =  (sm_y_pos[sm_idx] - sm_yspan_pos[sm_idx]/2) * 1e-6
		side_monitor['z max'] = monitorbox_vertical_maximum_um * 1e-6
		side_monitor['z min'] = monitorbox_vertical_minimum_um * 1e-6
		side_monitor['override global monitor settings'] = 1
		side_monitor['use wavelength spacing'] = 1
		side_monitor['use source limits'] = 1
		side_monitor['frequency points'] = num_design_frequency_points
		side_monitor = fdtd_update_object(fdtd_hook, side_monitor, create_object=True)
		side_monitors.append(side_monitor)
	fdtd_objects['side_monitors'] = side_monitors

	scatter_plane_monitor = {}
	scatter_plane_monitor['name'] = 'scatter_plane_monitor'
	scatter_plane_monitor['type'] = 'DFTMonitor'
	scatter_plane_monitor['power_monitor'] = True
	scatter_plane_monitor.update(monitors_keyed[scatter_plane_monitor['name']])
	scatter_plane_monitor['x'] = 0 * 1e-6
	# Need to consider: if the scatter plane monitor only includes one quadrant from the adjacent corner focal-plane-quadrants, 
	# the average power captured by the scatter plane monitor is only 83% of the power coming from the exit aperture.
	# It will also miss anything that falls outside of the boundary which is very probable.
	# At the same time the time to run FDTD sim will also go up by a factor squared.
	scatter_plane_monitor['x span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','x span') /2 * 6) / 1e-6) * 1e-6
	scatter_plane_monitor['y'] = 0 * 1e-6
	scatter_plane_monitor['y span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','y span') /2 * 6) / 1e-6) * 1e-6
	scatter_plane_monitor['z'] = adjoint_vertical_um * 1e-6
	scatter_plane_monitor['override global monitor settings'] = 1
	scatter_plane_monitor['use wavelength spacing'] = 1
	scatter_plane_monitor['use source limits'] = 1
	scatter_plane_monitor['frequency points'] = num_design_frequency_points
	scatter_plane_monitor = fdtd_update_object(fdtd_hook, scatter_plane_monitor, create_object=True)
	fdtd_objects['scatter_plane_monitor'] = scatter_plane_monitor

	# Side monitors enclosing the space between the device's exit aperture and the focal plane.
	# Purpose is to capture the sum power of oblique scattering.
	vertical_scatter_monitors = []
	vsm_x_pos = [device_size_lateral_um/2, 0, -device_size_lateral_um/2, 0]
	vsm_y_pos = [0, device_size_lateral_um/2, 0, -device_size_lateral_um/2]
	vsm_xspan_pos = [0, device_size_lateral_um, 0, device_size_lateral_um]
	vsm_yspan_pos = [device_size_lateral_um, 0, device_size_lateral_um, 0]
	
	for vsm_idx in range(0, 4):      # We are not using num_sidewalls because these monitors MUST be created
		vertical_scatter_monitor = {}
		vertical_scatter_monitor['name'] = 'vertical_scatter_monitor_'  + str(vsm_idx)
		vertical_scatter_monitor['type'] = 'DFTMonitor'
		vertical_scatter_monitor['power_monitor'] = True
		vertical_scatter_monitor.update(monitors_keyed[vertical_scatter_monitor['name']])
		#vertical_scatter_monitor['x'] = vsm_x_pos[vsm_idx] * 1e-6
		vertical_scatter_monitor['x max'] =  (vsm_x_pos[vsm_idx] + vsm_xspan_pos[vsm_idx]/2) * 1e-6   # must bypass setting 'x span' because this property is invalid if the monitor is X-normal
		vertical_scatter_monitor['x min'] =  (vsm_x_pos[vsm_idx] - vsm_xspan_pos[vsm_idx]/2) * 1e-6
		#vertical_scatter_monitor['y'] = vsm_y_pos[vsm_idx] * 1e-6
		vertical_scatter_monitor['y max'] =  (vsm_y_pos[vsm_idx] + vsm_yspan_pos[vsm_idx]/2) * 1e-6   # must bypass setting 'y span' because this property is invalid if the monitor is Y-normal
		vertical_scatter_monitor['y min'] =  (vsm_y_pos[vsm_idx] - vsm_yspan_pos[vsm_idx]/2) * 1e-6
		vertical_scatter_monitor['z max'] = device_vertical_minimum_um * 1e-6
		vertical_scatter_monitor['z min'] = adjoint_vertical_um * 1e-6
		vertical_scatter_monitor['override global monitor settings'] = 1
		vertical_scatter_monitor['use wavelength spacing'] = 1
		vertical_scatter_monitor['use source limits'] = 1
		vertical_scatter_monitor['frequency points'] = num_design_frequency_points
		vertical_scatter_monitor = fdtd_update_object(fdtd_hook, vertical_scatter_monitor, create_object=True)
		vertical_scatter_monitors.append(vertical_scatter_monitor)
	fdtd_objects['vertical_scatter_monitors'] = vertical_scatter_monitors

	# Device cross-section monitors that take the field inside the device, and extend all the way down to the focal plane. We have 6 slices, 2 sets of 3 for both x and y.
	# Each direction-set has slices at x(y) = -device_size_lateral_um/4, 0, +device_size_lateral_um/4, i.e. intersecting where the focal spot is supposed to be.
	# We also take care to output plots for both E_norm and Re(E), the latter of which captures the wave-like behaviour much more accurately.
	device_xsection_monitors = []
	dxm_pos = [-device_size_lateral_um/4, 0, device_size_lateral_um/4]
 
	for dev_cross_idx in range(0,6):
		device_cross_monitor = {}
		device_cross_monitor['name'] = 'device_xsection_monitor_' + str(dev_cross_idx)
		device_cross_monitor['type'] = 'DFTMonitor'
		device_cross_monitor['power_monitor'] = True
		device_cross_monitor.update(monitors_keyed[device_cross_monitor['name']])
		device_cross_monitor['z max'] = device_vertical_maximum_um * 1e-6
		device_cross_monitor['z min'] = adjoint_vertical_um * 1e-6
		
		if dev_cross_idx < 3:
			device_cross_monitor['y'] = dxm_pos[dev_cross_idx] * 1e-6
			device_cross_monitor['x max'] = (device_size_lateral_um / 2) * 1e-6
			device_cross_monitor['x min'] = -(device_size_lateral_um / 2) * 1e-6
		else:
			device_cross_monitor['x'] = dxm_pos[dev_cross_idx-3] * 1e-6
			device_cross_monitor['y max'] = (device_size_lateral_um / 2) * 1e-6
			device_cross_monitor['y min'] = -(device_size_lateral_um / 2) * 1e-6
		
		device_cross_monitor['override global monitor settings'] = 1
		device_cross_monitor['use wavelength spacing'] = 1
		device_cross_monitor['use source limits'] = 1
		device_cross_monitor['frequency points'] = num_design_frequency_points
		
		device_xsection_monitor = fdtd_update_object(fdtd_hook, device_cross_monitor, create_object=True)
		device_xsection_monitors.append(device_cross_monitor)
	fdtd_objects['device_xsection_monitors'] = device_xsection_monitors
	
	# TODO: Wrap this into a function in a LumericalPlotter.py somewhere - since the exact same code is called,-----------------------------------
 	# todo: both in SonyBayerFilterOptimization.py and this module.   ----------------------------------------------------------------------------
	try:
		objExists = fdtd_hook.getnamed('sidewall_0', 'name')
	except Exception as ex:
		logging.warning('FDTD Object sidewall_0 does not exist - but should. Creating...')
 	
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
			device_sidewall_meshes.append(device_sidewall_mesh)
		fdtd_objects['device_sidewall_meshes'] = device_sidewall_meshes
	# todo: ------------------------------------------------------------------------------------------------------------------------------------
 
	# Implement finer sidewall mesh
	# Sidewalls may not match index with sidewall monitors...
	sidewall_meshes = []
	sidewall_mesh_override_x_mesh = [1,0,1,0]

	for sidewall_idx in range(0, num_sidewalls):
		sidewall_mesh = {}
		sidewall_mesh['name'] = 'mesh_sidewall_' + str(sidewall_idx)
		sidewall_mesh['type'] = 'Mesh'
		try:
			sidewall_mesh['override x mesh'] = fdtd_hook.getnamed(sidewall_mesh['name'], 'override x mesh')
		except Exception as err:
			sidewall_mesh['override x mesh'] = sidewall_mesh_override_x_mesh[sidewall_idx]
			
		if sidewall_mesh['override x mesh']:
			sidewall_mesh['dx'] = fdtd_hook.getnamed('device_mesh','dx') / 1.5
		else:	# then override y mesh is active
			sidewall_mesh['dy'] = fdtd_hook.getnamed('device_mesh','dy') / 1.5
		sidewall_mesh = fdtd_update_object(fdtd_hook, sidewall_mesh, create_object=True)
		sidewall_meshes.append(sidewall_mesh)
	fdtd_objects['sidewall_meshes'] = sidewall_meshes
 
	# Adjust lateral region size of FDTD region and device mesh
	fdtd = fdtd_objects['FDTD']
	fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
	fdtd['y span'] = fdtd_region_size_lateral_um * 1e-6
	fdtd = fdtd_update_object(fdtd_hook, fdtd)

	# TODO: Change this to adjust wavelength ranges for every single monitor, and not just sources
	if change_wavelength_range:
		forward_src_x = fdtd_objects['forward_src_x']
		forward_src_x['wavelength start'] = lambda_min_um * 1e-6
		forward_src_x['wavelength stop'] = lambda_max_um * 1e-6
		# TODO: This bit should be put under if use_gaussian == True
		forward_src_x['beam parameters'] = 'Waist size and position'
		forward_src_x['waist radius w0'] = 3e8/np.mean(3e8/np.array([lambda_min_um, lambda_max_um])) / (np.pi*background_index*np.radians(src_div_angle)) * 1e-6
		forward_src_x['distance from waist'] = -(src_maximum_vertical_um - device_size_vertical_um) * 1e-6
		forward_src_x = fdtd_update_object(fdtd_hook, forward_src_x)

	# Install Aperture that blocks off source
	source_block = {}
	source_block['name'] = 'PEC_screen'
	source_block['type'] = 'Rectangle'
	source_block['x'] = 0 * 1e-6
	source_block['x span'] = 1.1*4/3*1.2 * device_size_lateral_um * 1e-6
	source_block['y'] = 0 * 1e-6
	source_block['y span'] = 1.1*4/3*1.2 * monitorbox_size_lateral_um * 1e-6
	source_block['z min'] = (monitorbox_vertical_maximum_um + 3 * mesh_spacing_um) * 1e-6
	source_block['z max'] = (monitorbox_vertical_maximum_um + 6 * mesh_spacing_um) * 1e-6
	source_block['material'] = 'PEC (Perfect Electrical Conductor)'
	# TODO: set enable true or false?
	source_block = fdtd_update_object(fdtd_hook, source_block, create_object=True)
	fdtd_objects['source_block'] = source_block
	
	source_aperture = {}
	source_aperture['name'] = 'source_aperture'
	source_aperture['type'] = 'Rectangle'
	source_aperture['x'] = 0 * 1e-6
	source_aperture['x span'] = monitorbox_size_lateral_um * 1e-6
	source_aperture['y'] = 0 * 1e-6
	source_aperture['y span'] = monitorbox_size_lateral_um * 1e-6
	source_aperture['z min'] = (monitorbox_vertical_maximum_um + 3 * mesh_spacing_um) * 1e-6
	source_aperture['z max'] = (monitorbox_vertical_maximum_um + 6 * mesh_spacing_um) * 1e-6
	source_aperture['material'] = 'etch'
	# TODO: set enable true or false?
	source_aperture = fdtd_update_object(fdtd_hook, source_aperture, create_object=True)
	fdtd_objects['source_aperture'] = source_aperture

	# Duplicate devices and set in an array
	if np.prod(device_array_shape) > 1:		
		# Extend device mesh to cover the entire FDTD region
		device_mesh = fdtd_objects['device_mesh']
		device_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
		device_mesh['y span'] = fdtd_region_size_lateral_um * 1e-6
		device_mesh = fdtd_update_object(fdtd_hook, device_mesh)

		# Set the central positions of each device in the array.
		# TODO: This is for a device_array_num tuple (3,3). Change this to accommodate any size
		device_arr_x_positions_um = (device_size_lateral_um + sidewall_thickness_um)*1e-6*np.array([1,1,0,-1,-1,-1,0,1])
		device_arr_y_positions_um = (device_size_lateral_um + sidewall_thickness_um)*1e-6*np.array([0,1,1,1,0,-1,-1,-1])

		# Copy the central design around in an array defined by device_array_shape.
		design_copies = []
		design_import = fdtd_objects['design_import']
		for dev_arr_idx in range(0, np.prod(device_array_shape)-1):
			fdtd_hook.select('design_import')
			fdtd_hook.copy(device_arr_x_positions_um[dev_arr_idx], device_arr_y_positions_um[dev_arr_idx])
			fdtd_hook.set('name', 'design_import_' + str(dev_arr_idx))
			design_copy = fdtd_simobject_to_dict( fdtd_get_simobject(fdtd_hook, 'design_import_' + str(dev_arr_idx)) )
			#! Doing it the below way does not also copy the internal (n,k) data. We would need to call importnk2 every time if so.
				# design_copy = copy.deepcopy(design_import)
				# design_copy['name'] = 'design_import_' + str(dev_arr_idx)
				# design_copy['x'] = device_arr_x_positions_um[dev_arr_idx]
				# design_copy['y'] = device_arr_y_positions_um[dev_arr_idx]
				# design_copy = fdtd_update_object(fdtd_hook, design_copy, create_object=True)
			design_copies.append(design_copy)
		fdtd_objects['design_copies'] = design_copies

		# Duplicate sidewalls AND sidewall meshes.
		for dev_arr_idx in range(0, np.prod(device_array_shape)-1):
			for idx in range(0, num_sidewalls):
					sm_name = 'sidewall_' + str(idx)
					sm_mesh_name = 'mesh_sidewall_' + str(idx)
					
					# Sidewalls
					try:
						objExists = fdtd_hook.getnamed(sm_name, 'name')
						fdtd_hook.select(sm_name)
						fdtd_hook.copy(device_arr_x_positions_um[dev_arr_idx], device_arr_y_positions_um[dev_arr_idx])
						fdtd_hook.set('name', 'sidewall_misc')
						# We are not going to bother copying the sidewall_misc into fdtd_objects memory, but we can do this later if necessary.
					
					except Exception as ex:
						template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
						message = template.format(type(ex).__name__, ex.args)
						logging.warning(message+'FDTD Object Sidewall does not exist.')
		
					# Sidewall meshes
					try:
						objExists = fdtd_hook.getnamed(sm_mesh_name, 'name')
						fdtd_hook.select(sm_mesh_name)
						fdtd_hook.copy(device_arr_x_positions_um[dev_arr_idx], device_arr_y_positions_um[dev_arr_idx])
						fdtd_hook.set('name', 'mesh_sidewall_misc')
					except Exception as ex:
						template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
						message = template.format(type(ex).__name__, ex.args)
						logging.warning(message+'FDTD Object Sidewall Mesh does not exist.')
	
	disable_all_sources()
	disable_objects(disable_object_list)
	fdtd_hook.save()
 
	logging.info("Completed Step 0: Lumerical environment setup complete.")
	# Save out all variables to save state file
	backup_all_vars()		



logging.info('Reached end of file. Code completed.')

