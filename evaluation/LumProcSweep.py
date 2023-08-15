# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
from utils import *
from utils import lum_utils as lm
from utils import plotter

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
DEBUG_COMPLETED_JOBS_FOLDER = 'ares_test_dev'
PULL_COMPLETED_JOBS_FOLDER = projects_directory_location
if running_on_local_machine:
	PULL_COMPLETED_JOBS_FOLDER = DEBUG_COMPLETED_JOBS_FOLDER


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
monitorbox_size_lateral_um = device_size_lateral_um + (mesh_spacing_um)*3 + 2 * sidewall_thickness_um
monitorbox_vertical_maximum_um = device_vertical_maximum_um + (mesh_spacing_um*5)
monitorbox_vertical_minimum_um = device_vertical_minimum_um - (mesh_spacing_um*5)
adjoint_vertical_um = fdtd_hook.getnamed('transmission_focal_monitor_', 'z') / 1e-6

# TODO: Think we need to wrap this part in a function or class or another module honestly --------------------------------------------------------------------------------------------
# todo: put it all in utils/lum_utils.py
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
			elif any(ele.lower() in type_string for ele in ['PlaneSource']):
				fdtd_hook_.addplane(name=simDict['name'])
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
			logging.error(message + f'	 FDTD Object {simDict["name"]}: does not exist.')
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

def update_object( fdtd_hook_, object_path, key, val):
	'''Updates selected objects in the FDTD simulation.'''
	global fdtd_objects
	
	utility.set_by_path(fdtd_objects, object_path + [key], val)
	new_dict = fdtd_update_object(fdtd_hook_, utility.get_by_path(fdtd_objects, object_path))
	if new_dict == 0:
		logging.error('Object did not update correctly! Aborting. (Value has been changed in fdtd_objects dict.)')
	else:
	 	utility.set_by_path(fdtd_objects, object_path, new_dict)

def disable_objects(fdtd_hook_, object_list, opposite=False):
	'''Disable selected objects in the simulation, to save on memory and runtime'''
	global fdtd_objects
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
	
	for object_path in object_list:
		if not isinstance(object_path, list):
			object_path = [object_path]
		update_object(fdtd_hook_, object_path, 'enabled', opposite)

def disable_all_sources(opp=False):
	'''Disable all sources in the simulation, so that we can selectively turn single sources on at a time'''
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	object_list = []

	for key, val in fdtd_objects.items():
		if isinstance(val, list):
			# type_string = val[0]['type'].lower()
			# We'll skip the lists since they are self-referring to things in the dictionary anyway.
			pass
		else:
			type_string = val['type'].lower()
  
			if any(ele.lower() in type_string for ele in ['GaussianSource', 'TFSFSource', 'DipoleSource']):
				object_list.append([val['name']])	# Remember to wrap the name string in a list for compatibility.
	
	disable_objects(fdtd_hook, object_list, opposite=opp)

def disable_all_devices(opp=False):
	'''Disable all devices in the simulation, so that we can selectively turn single devices on at a time'''
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	object_list = ['design_import', 'device_mesh']
	if fdtd_objects.get('design_copies') is not None:
		for idx, design_copy in enumerate(fdtd_objects['design_copies']):
			object_list.append(['design_copies', idx])
	
	disable_objects(fdtd_hook, object_list, opposite=opp)

def disable_all_sidewalls(opp=False):
	'''Disable all sidewalls in the simulation, so that we can selectively turn sets of sidewalls on at a time'''
	lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

	object_list = []
	if fdtd_objects.get('device_sidewalls') is not None:
		for idx, dev_sw in enumerate(fdtd_objects['device_sidewalls']):
			object_list.append(['device_sidewalls', idx])
	if fdtd_objects.get('device_sidewall_meshes') is not None:
		for idx, dev_sw_mesh in enumerate(fdtd_objects['device_sidewall_meshes']):
			object_list.append(['device_sidewall_meshes', idx])
	
	disable_objects(fdtd_hook, object_list, opposite=opp)

	for obj_name in ['sidewall_misc', 'mesh_sidewall_misc']:
		for i in range(int(fdtd_hook.getnamednumber(obj_name))):
			# See https://optics.ansys.com/hc/en-us/articles/360034928793-setnamed
			fdtd_hook.setnamed(obj_name, 'enabled', opp, i+1)	# i+1 because of zero-index mismatch

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
	if evaluation_source_bcs in ['periodic_plane']:
		scatter_plane_monitor['x span'] = fdtd_hook.getnamed('transmission_focal_monitor_','x span')
		scatter_plane_monitor['y span'] = fdtd_hook.getnamed('transmission_focal_monitor_','y span')
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
  
		device_sidewalls = []
		for dev_sidewall_idx in range(0, num_sidewalls):
			device_sidewall = fdtd_objects[f'sidewall_{dev_sidewall_idx}']
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
			device_sidewall = fdtd_update_object(fdtd_hook, device_sidewall, create_object=False)
			device_sidewalls.append(device_sidewall)
		fdtd_objects['device_sidewalls'] = device_sidewalls

		device_sidewall_meshes = []
		for dev_sidewall_idx in range(0, num_sidewalls):
			device_sidewall_mesh = fdtd_objects[f'mesh_sidewall_{dev_sidewall_idx}']
			device_sidewall_mesh['based on a structure'] = 1
			device_sidewall_mesh['structure'] = device_sidewalls[dev_sidewall_idx]['name']
			device_sidewall_mesh = fdtd_update_object(fdtd_hook, device_sidewall_mesh, create_object=False)
			device_sidewall_meshes.append(device_sidewall_mesh)
			# TODO: Test that this is still synced up to sidewall_0 or whatever i.e. when sidewall updates dimensions, mesh also updates
		fdtd_objects['device_sidewall_meshes'] = device_sidewall_meshes

	except Exception as ex:
		if num_sidewalls != 0:
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
 
	# # Implement finer sidewall mesh
	# # Sidewalls may not match index with sidewall monitors...
	# sidewall_meshes = []
	# sidewall_mesh_override_x_mesh = [1,0,1,0]

	# for sidewall_idx in range(0, num_sidewalls):
	# 	sidewall_mesh = {}
	# 	sidewall_mesh['name'] = 'mesh_sidewall_' + str(sidewall_idx)
	# 	sidewall_mesh['type'] = 'Mesh'
	# 	try:
	# 		sidewall_mesh['override x mesh'] = fdtd_hook.getnamed(sidewall_mesh['name'], 'override x mesh')
	# 	except Exception as err:
	# 		sidewall_mesh['override x mesh'] = sidewall_mesh_override_x_mesh[sidewall_idx]
			
	# 	if sidewall_mesh['override x mesh']:
	# 		sidewall_mesh['dx'] = fdtd_hook.getnamed('device_mesh','dx') / 1.5
	# 	else:	# then override y mesh is active
	# 		sidewall_mesh['dy'] = fdtd_hook.getnamed('device_mesh','dy') / 1.5
	# 	sidewall_mesh = fdtd_update_object(fdtd_hook, sidewall_mesh, create_object=True)
	# 	sidewall_meshes.append(sidewall_mesh)
	# fdtd_objects['sidewall_meshes'] = sidewall_meshes
 
	# Adjust lateral region size of FDTD region and device mesh
	fdtd = fdtd_objects['FDTD']
	fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
	fdtd['y span'] = fdtd_region_size_lateral_um * 1e-6
	# Boundary condition adjustment
	if evaluation_source_bcs in ['periodic_plane']:
		fdtd['x min bc'] = 'Bloch'
		fdtd['y min bc'] = 'Bloch'
	fdtd = fdtd_update_object(fdtd_hook, fdtd)
 

	#! SOURCE ADJUSTMENT
	# TODO: Change this to adjust wavelength ranges for every single monitor, and not just sources
	forward_src_x = fdtd_objects['forward_src_x']
	forward_src_x['x span'] = (device_size_lateral_um + (mesh_spacing_um)*3  + 2 * sidewall_thickness_um) * 1e-6
	forward_src_x['y span'] = (device_size_lateral_um + (mesh_spacing_um)*3  + 2 * sidewall_thickness_um) * 1e-6
	if change_wavelength_range:
		forward_src_x['wavelength start'] = lambda_min_um * 1e-6
		forward_src_x['wavelength stop'] = lambda_max_um * 1e-6
		if cv.use_gaussian_sources:
			forward_src_x['beam parameters'] = 'Waist size and position'
			forward_src_x['waist radius w0'] = 3e8/np.mean(3e8/np.array([lambda_min_um, lambda_max_um])) / (np.pi*background_index*np.radians(src_div_angle)) * 1e-6
			forward_src_x['distance from waist'] = -(src_maximum_vertical_um - device_size_vertical_um) * 1e-6
	forward_src_x = fdtd_update_object(fdtd_hook, forward_src_x)

	if evaluation_source_bcs in ['tfsf']:
		# update dictionary entries of forward_src_x with the below
		# delete the forward_src_x
		# update again with create_object=True
		forward_src_x.update({'type': 'TFSFSource'})
		forward_src_x.update({'polarization angle': 0})
		forward_src_x.update({'direction': 'Backward'})
		forward_src_x.update({'z max': src_maximum_vertical_um * 1e-6})
		forward_src_x.update({'z min': src_minimum_vertical_um * 1e-6})
		forward_src_x.pop('z')
		forward_src_x.pop('z span')
	elif evaluation_source_bcs in ['periodic_plane']:
		forward_src_x.update({'type': 'PlaneSource'})
		forward_src_x.update({'source shape': 'Plane wave'})
		forward_src_x.update({'plane wave type': 'BFAST'})
		# forward_src_x.update({'plane wave type': 'Bloch/periodic'})
		forward_src_x.update({'injection axis': 'z-axis'})
		forward_src_x.update({'direction': 'Backward'})
		forward_src_x.update({'z': src_maximum_vertical_um * 1e-6})
		forward_src_x.update({'angle theta': source_angle_theta_deg})
		forward_src_x.update({'angle phi': source_angle_phi_deg})
		forward_src_x.update({'polarization angle': 0})
	# elif evaluation_source_bcs in ['gaussian']:
	else:
		forward_src_x.update({'type': 'GaussianSource'})
		forward_src_x.update({'angle theta': source_angle_theta_deg})
		forward_src_x.update({'angle phi': source_angle_phi_deg})
		forward_src_x.update({'polarization angle': 0})
		forward_src_x.update({'direction': 'Backward'})
		forward_src_x.update({'z': src_maximum_vertical_um * 1e-6})

		# forward_src['angle theta'], shift_x_center = snellRefrOffset(source_angle_theta_deg, objects_above_device)	# this code takes substrates and objects above device into account
		shift_x_center = np.abs( device_vertical_maximum_um - src_maximum_vertical_um ) * np.tan( source_angle_theta_rad )
		forward_src_x.update({'x': shift_x_center * 1e-6})
  
	forward_src_x['enabled'] = True
	fdtd_hook.select('forward_src_x')
	fdtd_hook.delete()
	forward_src_x = fdtd_update_object(fdtd_hook, forward_src_x, create_object=True)
 
	
	


	# Import device permittivity from a save file.
	# todo: and possibly also extract permittivity from cur_design_variable.npy or some other way of transference.
	# todo: Also consider writing a method to extract the permittivity directly from Lumerical - use an index monitor
	if restart_from_cur_design_variable:
		design_import = fdtd_objects['design_import']

		cur_density = np.load("cur_design_variable.npy" )
		if binarize_3_levels:
			cur_density = \
				np.multiply(1.0, np.greater(cur_density, 0.667)) + \
				np.multiply(0.5, np.logical_and(cur_density > 0.333, cur_density < 0.667)) + \
				np.multiply(0.0, np.less_equal(cur_density, 0.5))
		else:
			cur_density = np.add(
				np.multiply(1.0, np.greater(cur_density, 0.5)),
				np.multiply(0.0, np.less_equal(cur_density, 0.5))
			)		# binarize
		cur_permittivity = min_device_permittivity + ( max_device_permittivity - min_device_permittivity ) * cur_density
		cur_index = utility.index_from_permittivity( cur_permittivity )
		logging.info(f'TiO2% is {100 * np.count_nonzero(cur_index > min_device_index) / cur_index.size}%.')
		# logging.info(f'Binarization is {100 * np.sum(np.abs(cur_density-0.5))/(cur_density.size*0.5)}%.')
		logging.info(f'Binarization is {100 * utility.compute_binarization(cur_density)}%.')

		bayer_filter_size_voxels = np.array(
								[device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
		bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
		bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
		bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)

		fdtd_hook.select( design_import['name'] )
		fdtd_hook.importnk2( cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )

	# Install Aperture that blocks off Gaussian source
	if evaluation_source_bcs in ['gaussian']: # and use_source_aperture:
		source_block = {}
		source_block['name'] = 'PEC_screen'
		source_block['type'] = 'Rectangle'
		source_block['x'] = 0 * 1e-6
		source_block['x span'] = 1.1*4/3*1.2 * device_size_lateral_um * 1e-6
		source_block['y'] = 0 * 1e-6
		source_block['y span'] = 1.1*4/3*1.2 * device_size_lateral_um * 1e-6
		if np.prod(device_array_shape) > 1:
			source_block['x span'] = fdtd_region_size_lateral_um * 1e-6
			source_block['y span'] = fdtd_region_size_lateral_um * 1e-6
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
		source_aperture['x span'] = device_size_lateral_um * 1e-6
		source_aperture['y'] = 0 * 1e-6
		source_aperture['y span'] = device_size_lateral_um * 1e-6
		source_aperture['z min'] = (monitorbox_vertical_maximum_um + 3 * mesh_spacing_um) * 1e-6
		source_aperture['z max'] = (monitorbox_vertical_maximum_um + 6 * mesh_spacing_um) * 1e-6
		source_aperture['material'] = 'etch'
		# TODO: set enable true or false?
		source_aperture = fdtd_update_object(fdtd_hook, source_aperture, create_object=True)
		fdtd_objects['source_aperture'] = source_aperture
	else:
		source_block = fdtd_objects['PEC_screen']
		source_aperture = fdtd_objects['source_aperture']
		source_block['enabled'] = False
		source_aperture['enabled'] = False
		source_block = fdtd_update_object(fdtd_hook, source_block)
		source_aperture = fdtd_update_object(fdtd_hook, source_aperture)

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
	disable_objects(fdtd_hook, disable_object_list)
	fdtd_hook.save()
 
	logging.info("Completed Step 0: Lumerical environment setup complete.")
	# Save out all variables to save state file
	backup_all_vars()		


#*---- JOB RUNNING STEP BEGINS ----

#! Step 1 Functions
# todo: 20230316 - these functions could go into lum_utils.py but they're staying here because they use a bunch of other parameters
# todo: and it would be a pain to transfer them all over in the function calls.

def snellRefrOffset(src_angle_incidence, object_list):
	''' Takes a list of objects in between source and device, retrieves heights and indices,
	 and calculates injection angle so as to achieve the desired angle of incidence upon entering the device.'''
	# structured such that n0 is the input interface and n_end is the last above the device
	# height_0 should be 0

	heights = [0.]
	indices = [1.]
	total_height = fdtd_hook.getnamed('design_import','z max') / 1e-6
	max_height = fdtd_hook.getnamed('forward_src_x','z') / 1e-6
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

def change_source_theta(theta_value, phi_value):
	'''Changes the angle (theta and phi) of the overhead source.'''
	global forward_src_x
	forward_src_x = fdtd_objects['forward_src_x']

	# Structure the objects list such that the last entry is the last medium above the device.
	input_theta, r_offset = snellRefrOffset(theta_value, objects_above_device)

	forward_src_x['angle theta'] = input_theta
	forward_src_x['angle phi'] = phi_value
	forward_src_x['x'] = r_offset*np.cos(np.radians(phi_value)) * 1e-6
	forward_src_x['y'] = r_offset*np.sin(np.radians(phi_value)) * 1e-6
	forward_src_x = fdtd_update_object(fdtd_hook, forward_src_x)
 
	logging.info(f'Theta set to {input_theta}, Phi set to {phi_value}.')

def change_source_beam_radius(param_value):
	'''For a Gaussian beam source, changes the beam radius to focus or defocus it.'''
	xy_names = ['x', 'y']

	global forward_sources
	if globals().get('forward_sources') is None:
		forward_sources = []
	
	for xy_idx in range(0, 2):
		global forward_src
		forward_src = fdtd_simobject_to_dict( fdtd_get_simobject( fdtd_hook, 'forward_src_' + xy_names[xy_idx] ) )
  
		# fdtd_hook.set('beam radius wz', 100 * 1e-6)
		# fdtd_hook.set('divergence angle', 3.42365) # degrees
		# fdtd_hook.set('beam radius wz', 15 * 1e-6)
		# # fdtd_hook.set('beam radius wz', param_value * 1e-6)
		forward_src['beam parameters'] = 'Waist size and position'
		forward_src['waist radius w0'] = param_value * 1e-6
		forward_src['distance from waist'] = 0 * 1e-6
		forward_src = fdtd_update_object(fdtd_hook, forward_src)
  
		try:
			forward_sources[xy_idx] = forward_src
		except Exception as ex:
			forward_sources.append(forward_src)
		fdtd_objects[forward_src['name']] = forward_src
	fdtd_objects['forward_sources'] = forward_sources
	
	logging.info(f'Source beam radius set to {param_value}.')

# TODO
def change_fdtd_region_size(param_value):
	pass

def change_sidewall_thickness(param_value):
	'''If sidewalls exist, changes their thickness.'''
	# TODO: Need to rewrite this to use the fdtd_objects dictionary instead.
	
	monitorbox_size_lateral_um = device_size_lateral_um + 2 * param_value
	
	for idx in range(0, num_sidewalls):
		sm_name = 'sidewall_' + str(idx)
		
		try:
			objExists = fdtd_hook.getnamed(sm_name, 'name')
			fdtd_hook.select(sm_name)
		
			if idx == 0:
				fdtd_hook.setnamed(sm_name, 'x', (device_size_lateral_um + param_value)*0.5 * 1e-6)
				fdtd_hook.setnamed(sm_name, 'x span', param_value * 1e-6)
				fdtd_hook.setnamed(sm_name, 'y span', monitorbox_size_lateral_um * 1e-6)
			elif idx == 1:
				fdtd_hook.setnamed(sm_name, 'y', (device_size_lateral_um + param_value)*0.5 * 1e-6)
				fdtd_hook.setnamed(sm_name, 'y span', param_value * 1e-6)
				fdtd_hook.setnamed(sm_name, 'x span', monitorbox_size_lateral_um * 1e-6)
			if idx == 2:
				fdtd_hook.setnamed(sm_name, 'x', (-device_size_lateral_um - param_value)*0.5 * 1e-6)
				fdtd_hook.setnamed(sm_name, 'x span', param_value * 1e-6)
				fdtd_hook.setnamed(sm_name, 'y span', monitorbox_size_lateral_um * 1e-6)
			elif idx == 3:
				fdtd_hook.setnamed(sm_name, 'y', (-device_size_lateral_um - param_value)*0.5 * 1e-6)
				fdtd_hook.setnamed(sm_name, 'y span', param_value * 1e-6)
				fdtd_hook.setnamed(sm_name, 'x span', monitorbox_size_lateral_um * 1e-6)
					
		except Exception as ex:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
			message = template.format(type(ex).__name__, ex.args)
			print(message+'FDTD Object does not exist.')
			# fdtd_hook.addmesh(name=sidewall_mesh['name'])
			# pass
	
	
	fdtd_hook.setnamed('incident_aperture_monitor', 'x span', monitorbox_size_lateral_um * 1e-6)
	fdtd_hook.setnamed('incident_aperture_monitor', 'y span', monitorbox_size_lateral_um * 1e-6)
	fdtd_hook.setnamed('exit_aperture_monitor', 'x span', monitorbox_size_lateral_um * 1e-6)
	fdtd_hook.setnamed('exit_aperture_monitor', 'y span', monitorbox_size_lateral_um * 1e-6)
	
	sm_x_pos = [monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2, 0]
	sm_y_pos = [0, monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2]
	sm_xspan_pos = [0, monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um]
	sm_yspan_pos = [monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um, 0]
	
	for sm_idx in range(0, num_sidewalls):
		sm_name = 'side_monitor_' + str(sm_idx)
		fdtd_hook.select(sm_name)
		fdtd_hook.setnamed(sm_name, 'x max', (sm_x_pos[sm_idx] + sm_xspan_pos[sm_idx]/2) * 1e-6)
		fdtd_hook.setnamed(sm_name, 'x min', (sm_x_pos[sm_idx] - sm_xspan_pos[sm_idx]/2) * 1e-6)
		fdtd_hook.setnamed(sm_name, 'y max', (sm_y_pos[sm_idx] + sm_yspan_pos[sm_idx]/2) * 1e-6)
		fdtd_hook.setnamed(sm_name, 'y min', (sm_y_pos[sm_idx] - sm_yspan_pos[sm_idx]/2) * 1e-6)

def change_divergence_angle(param_value):
	'''For a Gaussian beam source, changes the divergence angle while keeping beam radius constant.'''
	xy_names = ['x', 'y']
	
	global forward_sources
	if globals().get('forward_sources') is None:
		forward_sources = []

	for xy_idx in range(0, 2):
		global forward_src
		forward_src = fdtd_simobject_to_dict( fdtd_get_simobject( fdtd_hook, 'forward_src_' + xy_names[xy_idx] ) )
  
		# fdtd_hook.set('beam radius wz', 100 * 1e-6)
		# fdtd_hook.set('divergence angle', 3.42365) # degrees
		# fdtd_hook.set('beam radius wz', 15 * 1e-6)
		# # fdtd_hook.set('beam radius wz', param_value * 1e-6)
  
		# Resets distance from waist to 0, fixing waist radius and beam radius properties
		forward_src['beam parameters'] = 'Waist size and position'
		# src_beam_rad = const_parameters['bRad']['var_values'][const_parameters['bRad']['peakInd']]
		forward_src['waist radius w0'] = gaussian_waist_radius_um * 1e-6
		forward_src['distance from waist'] = 0 * 1e-6
		# Change divergence angle
		forward_src['beam parameters'] = 'Beam size and divergence angle'
		forward_src['divergence angle'] = param_value # degrees
  
		forward_src = fdtd_update_object(fdtd_hook, forward_src)
		try:
			forward_sources[xy_idx] = forward_src
		except Exception as ex:
			forward_sources.append(forward_src)
		fdtd_objects[forward_src['name']] = forward_src
	fdtd_objects['forward_sources'] = forward_sources
	
	logging.info(f'Source divergence angle set to {param_value}.')

#! Step 2 Functions
# Functions for obtaining information from Lumerical monitors
# See utils/lum_utils.py

#! Step 3 Functions

def append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics, band_idxs=None):
	'''Logs peak/mean values and corresponding indices of the f_vector values, for each quadrant, IN BAND.
	This means that it takes the values within a defined spectral band and picks a statistical value for each band within that band.
	Produces num_sweep_values * num_bands values, i.e. B->{B,G,R}, G->{B,G,R}, R->{B,G,R}  
	where for e.g. RB is the transmission of R quadrant within the B spectral band
	Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''

	if band_idxs is None:
		# Then take the whole spectrum as one band
		band_idxs = [np.array([0, len(f_vectors[0]['var_values'])-1])]
	num_bands = len(band_idxs)
	
	for idx in range(0, num_sweep_values):
		for inband_idx in range(0, num_bands):
			if statistics in ['peak']:
				sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])]), key=(lambda x: x[1]))
				sweep_stdev = 0
				
			else:
				sweep_val = np.average(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])])
				sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
				sweep_stdev = np.std(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])])
		
		  
			sweep_values.append({'value': float(sweep_val),
							'index': sweep_index,
							'std_dev': sweep_stdev,
							'statistics': statistics
							})
	
	return sweep_values

def calc_normalize_power(monitors_data, normalize_option):
	baseline_power = 0

	if normalize_option == 'input_power':
		baseline_power = monitors_data['incident_aperture_monitor']['P']

	elif normalize_option == 'sourcepower':
		baseline_power = monitors_data['sourcepower']

	elif normalize_option in ['power_sum', 'uniform_cube']:
		sourcepower = monitors_data['sourcepower']
		input_power = monitors_data['incident_aperture_monitor']['P']

		R_coeff_4 = monitors_data['incident_aperture_monitor']['R_power']		# this is actually power not R-coefficient
		R_coeff_5 = monitors_data['src_spill_monitor']['P'] - input_power		# this is actually power not R-coefficient
		R_power = (np.abs(R_coeff_4)) # + R_coeff_5) * 0.5
		side_powers = []
		side_directions = ['E','N','W','S']
		sides_power = 0
		for idx in range(0, 4):
			side_powers.append(monitors_data['side_monitor_'+str(idx)]['P'])
			sides_power = sides_power + monitors_data['side_monitor_'+str(idx)]['P']
		focal_power = monitors_data['transmission_focal_monitor_']['P']
		scatter_power = 0
		for idx in range(0,4):
			scatter_power = scatter_power + monitors_data['vertical_scatter_monitor_'+str(idx)]['P']

		power_sum = R_power + focal_power + scatter_power
		sides_power_2 = sides_power#  - (sourcepower - input_power)
		power_sum += sides_power_2

		baseline_power = power_sum

	elif normalize_option == 'unity':
		baseline_power = 1
		

	return baseline_power

def generate_sorting_eff_spectrum(monitors_data, produce_spectrum, statistics='peak', add_opp_polarizations = False):
	'''Generates sorting efficiency spectrum and logs peak/mean values and corresponding indices of each quadrant. 
	Sorting efficiency of a quadrant is defined as overall power in that quadrant normalized against overall power in all four quadrants.
	Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''
	output_plot = {}
	r_vectors = []      # Variables
	f_vectors = []      # Functions
	
	lambda_vector = monitors_data['wavelength']
	r_vectors.append({'var_name': 'Wavelength',
					  'var_values': lambda_vector
					})

	quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']	
 
	### E-field way of calculating sorting spectrum
	# Enorm_scatter_plane = monitors_data['scatter_plane_monitor']['E']
	# Enorm_focal_plane, Enorm_focal_monitor_arr = partition_scatter_plane_monitor(monitors_data['scatter_plane_monitor'], 'E')
	
	# # Integrates over area
	# Enorm_fp_overall = np.sum(Enorm_focal_plane, (0,1,2))
	# Enorm_fm_overall = []
	# quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
	# for idx in range(0, num_adjoint_sources):
	#     Enorm_fm_overall.append(np.sum(Enorm_focal_monitor_arr[idx], (0,1,2)))
	#     f_vectors.append({'var_name': quadrant_names[idx],
	#                   'var_values': np.divide(Enorm_fm_overall[idx], Enorm_fp_overall)
	#                 })
	
	### Power way of calculating sorting spectrum
	for idx in range(0, num_adjoint_sources):
		f_vectors.append({'var_name': quadrant_names[idx],
						'var_values': np.divide(monitors_data['transmission_monitor_'+str(idx)]['P'],
												monitors_data['transmission_focal_monitor_']['P'])
						})
	
	if add_opp_polarizations:
		f_vectors[1]['var_name'] = 'Green'
		f_vectors[1]['var_values'] = f_vectors[1]['var_values'] + f_vectors[3]['var_values']
		f_vectors.pop(3)
	
	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['title'] = 'Sorting Efficiency Spectrum'
		output_plot['const_params'] = const_parameters
		
		# ## Plotting code to test
		# fig,ax = plt.subplots()
		# plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
		# plt.show(block=False)
		# plt.show()
		logging.info('Generated sorting efficiency spectrum.')

	
	sweep_values = []
	num_sweep_values = num_adjoint_sources if not add_opp_polarizations else num_bands
	sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
	
	# for idx in range(0, num_sweep_values):
	#     if statistics in ['peak']:
	#         sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
	#         sweep_stdev = 0
	#     else:
	#         sweep_val = np.average(f_vectors[idx]['var_values'])
	#         sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
	#         sweep_stdev = np.std(f_vectors[idx]['var_values'])
			
	#     sweep_values.append({'value': float(sweep_val),
	#                     'index': sweep_index,
	#                     'std_dev': sweep_stdev,
	#                     'statistics': statistics
	#                     })
	
	print('Picked peak efficiency points for each quadrant to pass to sorting efficiency variable sweep.')
	
	return output_plot, sweep_values

def generate_sorting_transmission_spectrum(monitors_data, produce_spectrum, statistics='peak', add_opp_polarizations = False):
	'''Generates sorting efficiency spectrum and logs peak/mean values and corresponding indices of each quadrant. 
	Sorting efficiency of a quadrant is defined as overall power in that quadrant normalized against **sourcepower** (accounting for angle).
	Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''
	output_plot = {}
	r_vectors = []      # Variables
	f_vectors = []      # Functions
	
	lambda_vector = monitors_data['wavelength']
	r_vectors.append({'var_name': 'Wavelength',
					  'var_values': lambda_vector
					})

	baseline_power = calc_normalize_power(monitors_data, normalize_against)

	quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
	for idx in range(0, num_adjoint_sources):
		f_vectors.append({'var_name': quadrant_names[idx],
						'var_values': np.divide(monitors_data['transmission_monitor_'+str(idx)]['P'],
												baseline_power)
						})
	f_vectors.append({'var_name': 'Overall Transmission',
					'var_values':np.divide(monitors_data['transmission_focal_monitor_']['P'],
												baseline_power)									
						})

	if add_opp_polarizations:
		f_vectors[1]['var_name'] = 'Green'
		f_vectors[1]['var_values'] = f_vectors[1]['var_values'] + f_vectors[3]['var_values']
		f_vectors.pop(3)
		
	#! Remove - rescaling
	def rescale_vector(vec, amin, amax):
		vec = (vec - np.min(vec))/(np.max(vec) - np.min(vec))
		vec = amin + (amax-amin)*vec
		return vec
	
	#! Remove rescale
	# for f_vector in f_vectors:
	#     f_vector['var_values'] = f_vector['var_values']*0.65/0.48
	# f_vectors[0]['var_values'] = f_vectors[0]['var_values'] * 0.85
	
	# f_vectors[0]['var_values'] = rescale_vector(f_vectors[0]['var_values'], 0.05, 0.67)
	# f_vectors[1]['var_values'] = rescale_vector(f_vectors[1]['var_values'], 0.08, 0.63)
	# f_vectors[2]['var_values'] = rescale_vector(f_vectors[2]['var_values'], 0.03, 0.6)
	
	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['title'] = 'Sorting Transmission Spectrum'
		output_plot['const_params'] = const_parameters
		
		# ## Plotting code to test
		# fig,ax = plt.subplots()
		# plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
		# plt.show(block=False)
		# plt.show()
		logging.info('Generated sorting transmission spectrum.')
  
	sweep_values = []
	num_sweep_values = num_adjoint_sources if not add_opp_polarizations else num_bands		# Indiv. quadrant transmissions
	num_sweep_values += 1																	# Overall transmission
	sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
	
	logging.info('Picked peak transmission points for each quadrant to pass to sorting transmission variable sweep.')
	
	sweep_peak_vals = []
	for swp_val in sweep_values:
		sweep_peak_vals.append(swp_val['value'])

	if normalize_against == 'unity':
		logging.info('[' + ', '.join(map(lambda x: f'{x:.2e}', sweep_peak_vals)) + ']')     # Converts to scientific with 2 s.f.
	else:	logging.info('[' + ', '.join(map(lambda x: str(round(100*x,2)), sweep_peak_vals)) + ']')     # Converts to percentages with 2 s.f.
	
	return output_plot, sweep_values

def partition_monitor_into_quadrants(E_vals, x_quads_num, y_quads_num):
		'''Split a plane monitor into equal quadrants.
		For example, if it's split into (3x3) x (2x2) = 36, then x_quads_num = 6, y_quads_num = 6
		Can be any quantity with x-y-z components, for e.g. power'''

		n_x = np.shape(E_vals)[0]
		n_x = int( x_quads_num * np.floor(n_x/x_quads_num) )
		n_y = np.shape(E_vals)[1]
		n_y = int( y_quads_num * np.floor(n_y/y_quads_num))
		E_vals = E_vals[0:n_x, 0:n_y, :]        # First, make sure the array is split nicely into equal sections

		# Reshape E_vals - assumes square
		l = np.array_split(E_vals, x_quads_num ,axis=0)
		split_E_vals = []
		for a in l:
			l = np.array_split(a, x_quads_num ,axis=1)
			split_E_vals += l
		split_E_vals = np.array(split_E_vals)						# split_E_vals has shape (6^2, nx/6, ny/6, nλ)

		return split_E_vals

def generate_crosstalk_adj_quadrants_spectrum(monitors_data, produce_spectrum, quadrant_idxs, quadrant_names, 
											  statistics='peak', add_opp_polarizations = False):
	'''Generates efficiency spectrum of specific quadrants and logs peak/mean values and corresponding indices of each quadrant. 
	Efficiency of a quadrant is defined as overall power in that quadrant normalized against **sourcepower** (accounting for angle).
	Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''
	output_plot = {}
	r_vectors = []      # Variables
	f_vectors = []      # Functions
	
	lambda_vector = monitors_data['wavelength']
	r_vectors.append({'var_name': 'Wavelength',
					  'var_values': lambda_vector
					})

	input_power = monitors_data['incident_aperture_monitor']['P']

	try:
		monitor = monitors_data['scatter_plane_monitor']
	except Exception as err:
		monitor = monitors_data['spill_plane_monitor']
	E_vals = np.squeeze(monitor['E'])

	#  Generate crosstalk relative numbers
	split_E_vals = partition_monitor_into_quadrants(E_vals, 6, 6)

 	# Store E^2 efficiency for scatter plane quadrants (every quadrant) of 3x3 array.
	split_power_vals = np.sum(np.square(split_E_vals),(1,2))	# split_power_vals has shape (6^2, nλ); also this is Ex^2+Ey^2 and not actually power
	sum_split_power_vals = np.sum(split_power_vals, 0)			# this is now the total E-norm spectrum across the scatter monitor, with shape (nλ)
	crosstalk_vals = np.divide(split_power_vals, sum_split_power_vals)

	# Also, store power efficiency for center quadrants of 3x3 array.
	center_quadrant_vectors = []        
	center_quadrant_names = ['B_c', 'G1_c', 'R_c', 'G2_c']
	for idx in range(0, num_adjoint_sources):
		center_quadrant_vectors.append({'var_name': center_quadrant_names[idx],
						'var_values': monitors_data['transmission_monitor_'+str(idx)]['P']
						})

	#! Scale E2 to power based on P_{any quad} = E**2_{any quad} / E**2_{G1,c} * P_{G1,c}. Works well across most cases
	scatter_quadrant_vectors = []
	# NOTE: left column is indices 0:N-1, second column is indices N:2N-1 and so on. Top right is index N**2 - 1.
	# NOTE: For a single quadrant, order of appearance is R, G1, G2, B. See <image>
	scaling_E2_to_power_factor = np.max(center_quadrant_vectors[1]['var_values']) / np.max(crosstalk_vals[15,:])
	for idx in range(0, 6**2):
		scatter_quadrant_vectors.append({'var_name': f'Crosstalk Power, Quadrant {idx}',
						'var_values': crosstalk_vals[idx,:] * scaling_E2_to_power_factor 
						})

	## Choose f_vectors for final plot for nearest-neighbour G quadrants comparison
	for list_idx, quad_idx in enumerate(quadrant_idxs):
		f_vectors.append({'var_name': quadrant_names[list_idx],
						'var_values': np.divide(scatter_quadrant_vectors[quad_idx]['var_values'], np.squeeze(input_power)
							  					).reshape((-1,1))
						# Here we normalize against input power because we're just comparing transmissions
						})
		
		peak_wl_idx = np.argmax(f_vectors[0]['var_values'])
		peak_wl = r_vectors[0]['var_values'][peak_wl_idx,:]
		f_value_peak_wl = f_vectors[-1]['var_values'][peak_wl_idx,:]
		logging.info(f'f[{list_idx}]: {quadrant_names[list_idx]} has value {f_value_peak_wl} at wavelength {peak_wl}')  

	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['title'] = 'Sorting Transmission Spectrum'
		output_plot['const_params'] = const_parameters
		
		# ## Plotting code to test
		# fig,ax = plt.subplots()
		# plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
		# plt.show(block=False)
		# plt.show()
		
		# fig,ax = plt.subplots()
		# plt.plot(r_vectors[0]['var_values'], scatter_quadrant_vectors[21]['var_values'], color='blue')
		# plt.plot(r_vectors[0]['var_values'], scatter_quadrant_vectors[15]['var_values'], color='green')
		# plt.plot(r_vectors[0]['var_values'], scatter_quadrant_vectors[14]['var_values'], color='red')
		# plt.plot(r_vectors[0]['var_values'], scatter_quadrant_vectors[20]['var_values'], color='gray')
		# plt.show(block=False)
		# plt.show()
		logging.info('Generated adjacent G quadrant transmission spectrum.')

	sweep_values = []
	num_sweep_values = len(f_vectors) # num_adjoint_sources if not add_opp_polarizations else num_bands
	sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
	
	logging.info('Picked peak transmission points for each quadrant to pass to sorting transmission variable sweep.')
	
	return output_plot, sweep_values

def generate_Enorm_focal_plane(monitors_data, produce_spectrum):
	'''Generates image slice at focal plane (including focal scatter). The output is over all wavelengths so a wavelength still needs to be selected.'''
	
	try:
		monitor = monitors_data['scatter_plane_monitor']
	except Exception as err:
		monitor = monitors_data['spill_plane_monitor']
	
	output_plot = {}
	r_vectors = []      	# Variables
	f_vectors = []      	# Functions
	lambda_vectors = [] 	# Wavelengths
	
	r_vectors.append({'var_name': 'x-axis',
					  'var_values': monitor['coords']['x']
					  })
	r_vectors.append({'var_name': 'y-axis',
					  'var_values': monitor['coords']['x']
					  })
	
	f_vectors.append({'var_name': 'E_norm scatter',
					  'var_values': np.squeeze(monitor['E'])
						})
	
	lambda_vectors.append({'var_name': 'Wavelength',
						   'var_values': monitors_data['wavelength']
							})
	
	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['lambda'] = lambda_vectors 
		output_plot['title'] = 'E-norm Image (Spectrum)'
		output_plot['const_params'] = const_parameters
	
	
	logging.info('Generated E-norm focal plane image plot.')
	
	# ## Plotting code to test
	# fig, ax = plt.subplots()
	# #plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], f_vectors[0]['var_values'][:,:,5])
	# plt.imshow(f_vectors[0]['var_values'][:,:,5])
	# plt.show(block=False)
	# plt.show()
	
	return output_plot, {}

def generate_device_cross_section_spectrum(monitors_data, produce_spectrum):
	'''Generates image slice at device cross-section. The output is over all wavelengths so a wavelength still needs to be selected.'''
	
	output_plot = {}
	r_vectors = []      # Variables
	f_vectors = []      # Functions
	lambda_vectors = [] # Wavelengths
	
	for device_cross_idx in range(0,6):
		
		monitor = monitors_data['device_xsection_monitor_' + str(device_cross_idx)]
	
		r_vectors.append({'var_name': 'x-axis',
						'var_values': monitor['coords']['x']
						})
		r_vectors.append({'var_name': 'y-axis',
						'var_values': monitor['coords']['y']
						})
		r_vectors.append({'var_name': 'z-axis',
						'var_values': monitor['coords']['z']
						})
		
		f_vectors.append({'var_name': 'E_norm',
						'var_values': np.squeeze(monitor['E'])
						 })
		f_vectors.append({'var_name': 'E_real',
						'var_values': np.squeeze(monitor['E_real'])
						})
		
		lambda_vectors.append({'var_name': 'Wavelength',
							'var_values': monitors_data['wavelength']
							})
	
	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['lambda'] = lambda_vectors 
		output_plot['title'] = 'Device Cross-Section Image (Spectrum)'
		output_plot['const_params'] = const_parameters
		
		logging.info('Generated device cross section image plot.')
		
		# ## Plotting code to test
		# fig, ax = plt.subplots()
		# #plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[2]['var_values'])], f_vectors[0]['var_values'][:,:,5])
		# plt.imshow(f_vectors[0]['var_values'][:,:,5])
		# plt.show(block=False)
		# plt.show()
	
	return output_plot, {}

def generate_crosstalk_power_spectrum(monitors_data, produce_spectrum, statistics='mean'):
	'''Generates power spectrum for adjacent pixels at focal plane.'''
	
	output_plot = {}
	r_vectors = []      # Variables
	f_vectors = []      # Functions
	
	lambda_vector = monitors_data['wavelength']
	r_vectors.append({'var_name': 'Wavelength',
					  'var_values': lambda_vector
					})
	
	try:
		monitor = monitors_data['scatter_plane_monitor']
	except Exception as err:
		monitor = monitors_data['spill_plane_monitor']
	E_vals = np.squeeze(monitor['E'])
	
	# Generate crosstalk relative numbers
	split_E_vals = partition_monitor_into_quadrants(E_vals, 3,3)
	
	# split_E_vals has shape (9, nx/3, ny/3, nλ)
	split_power_vals = np.sum(np.square(split_E_vals),(1,2))
	sum_split_power_vals = np.sum(split_power_vals, 0)
	crosstalk_vals = np.divide(split_power_vals, sum_split_power_vals)
	
	## [DEPRECATED] This section was for a different method of partitioning the scatter plane method 
	# crosstalk_vals_mod = np.divide(split_power_vals, sum_split_power_vals)
	# crosstalk_vals = np.zeros((9,np.shape(crosstalk_vals_mod)[1]))
	# crosstalk_vals[0,:] = crosstalk_vals_mod[0,:]
	# crosstalk_vals[1,:] = crosstalk_vals_mod[1,:] + crosstalk_vals_mod[2,:]
	# crosstalk_vals[2,:] = crosstalk_vals_mod[3,:]
	# crosstalk_vals[3,:] = crosstalk_vals_mod[4,:] + crosstalk_vals_mod[8,:]
	# crosstalk_vals[4,:] = crosstalk_vals_mod[5,:] + crosstalk_vals_mod[6,:] + crosstalk_vals_mod[9,:] + crosstalk_vals_mod[10,:]
	# crosstalk_vals[5,:] = crosstalk_vals_mod[7,:] + crosstalk_vals_mod[11,:]
	# crosstalk_vals[6,:] = crosstalk_vals_mod[12,:]
	# crosstalk_vals[7,:] = crosstalk_vals_mod[13,:] + crosstalk_vals_mod[14,:]
	# crosstalk_vals[8,:] = crosstalk_vals_mod[15,:]
	
	for idx in range(0, 9):
		f_vectors.append({'var_name': f'Crosstalk Power, Pixel {idx}',
						'var_values': crosstalk_vals[idx,:]
						})
	
	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['title'] = 'Crosstalk Power Spectrum'
		output_plot['const_params'] = const_parameters
		
		# ## Plotting code to test
		# fig,ax = plt.subplots()
		# plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
		# plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
		# plt.show(block=False)
		# plt.show()
		logging.info('Generated crosstalk power spectrum.')
	
	sweep_values = []
	num_sweep_values = 9
	sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
	
	logging.info('Picked crosstalk power points for each Bayer pixel, to pass to variable sweep.')
	
	sweep_mean_vals = []
	for swp_val in sweep_values:
		sweep_mean_vals.append(swp_val['value'])
	# print(sweep_mean_vals)
	logging.info('[' + ', '.join(map(lambda x: str(round(100*x,2)), sweep_mean_vals)) + ']')     # Converts to percentages with 2 s.f.
	
	logging.info('Generated adjacent pixel crosstalk power plot.')
	
	# ## Plotting code to test
	# fig, ax = plt.subplots()
	# #plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], f_vectors[0]['var_values'][:,:,5])
	# plt.imshow(f_vectors[0]['var_values'][:,:,5])
	# plt.show(block=False)
	# plt.show()
	
	return output_plot, sweep_values

def generate_exit_power_distribution_spectrum(monitors_data, produce_spectrum, statistics='mean'):
	'''Generates two power spectra on the same plot for both focal region power and crosstalk/spillover, 
	and logs peak/mean value and corresponding index.'''
	# Focal scatter and focal region power, normalized against exit aperture power

	output_plot = {}
	r_vectors = []      # Variables
	f_vectors = []      # Functions
	
	lambda_vector = monitors_data['wavelength']
	r_vectors.append({'var_name': 'Wavelength',
					  'var_values': lambda_vector
					})

	scatter_power = 0
	for vsm_idx in range(0,4):
		scatter_power = scatter_power + monitors_data['vertical_scatter_monitor_' + str(vsm_idx)]['P']
	focal_power = monitors_data['transmission_focal_monitor_']['P']
	exit_aperture_power = monitors_data['exit_aperture_monitor']['P']
	
	# logging.info('PLEASE NOTE: Unaccounted power from Exit Aperture (Normalized to Exit):' + \
	# 									f'{np.mean(exit_aperture_power - focal_power - scatter_power)/exit_aperture_power)}')
	# Q: Does scatter + focal = exit_aperture?
	# A: No - over by about (1.5 ± 2.214) %
	
	f_vectors.append({'var_name': 'Focal Region Power',
					  'var_values': focal_power/(focal_power + scatter_power)
					})
	
	f_vectors.append({'var_name': 'Oblique Scattering Power',
					  'var_values': scatter_power/(focal_power + scatter_power)
					})
	
	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['title'] = 'Focal Plane Power Distribution Spectrum'
		output_plot['const_params'] = const_parameters
		
		logging.info('Generated focal plane power distribution spectrum.')
	
	sweep_values = []
	num_sweep_values = len(f_vectors)
	sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
	
	logging.info('Picked exit power distribution points to pass to variable sweep.')
	
	return output_plot, sweep_values

def generate_device_rta_spectrum(monitors_data, produce_spectrum, statistics='mean'):
	'''Generates power spectra on the same plot for all contributions to RTA, 
	and logs peak/mean value and corresponding index.
	Everything is normalized to the power input to the device.'''
	# Overall RTA of the device, normalized against sum.
			# - Overall Reflection
			# - Side Powers
			# - Focal Region Power
			# - Oblique Scattering
			# - Device Absorption / Unaccounted Power
	
	
	output_plot = {}
	r_vectors = []      # Variables
	f_vectors = []      # Functions
	
	lambda_vector = monitors_data['wavelength']
	r_vectors.append({'var_name': 'Wavelength',
					  'var_values': lambda_vector
					})
	
	sourcepower = monitors_data['sourcepower']
	input_power = monitors_data['incident_aperture_monitor']['P']
	# f = fdtd_hook.getresult('focal_monitor_0','f')		# Might need to replace 'focal_monitor_0' with a search for monitors & their names
	# si = fdtd_hook.sourceintensity(f)
	# sa = fdtd_hook.getdata('forward_src_x', 'area')
	# logging.info(f'Sourcepower discrepancy is {100*(1 - sourcepower/input_power)} percent larger.')

	# The reflection monitor is the most troublesome and contentious one, here are some potential ways to measure it and most of them aren't correct
	R_coeff = np.reshape(1 - monitors_data['src_transmission_monitor']['T'],    (len(lambda_vector),1))
	R_coeff_2 = np.reshape(1 - monitors_data['incident_aperture_monitor']['T'], (len(lambda_vector),1))
	R_coeff_3 = np.reshape(1 - monitors_data['src_spill_monitor']['T'], (len(lambda_vector),1))
	R_coeff_4 = monitors_data['incident_aperture_monitor']['R_power']		# this is actually power not R-coefficient
	R_coeff_5 = monitors_data['src_spill_monitor']['P'] - input_power		# this is actually power not R-coefficient
	R_power_1 = (np.abs(R_coeff_4) + R_coeff_5) * 0.5
	R_power_2 = np.abs(R_coeff_4)
	R_power_3 = R_coeff_4

	R_power = R_power_1			# 2 for TFSF, 1 for Gaussian

	side_powers = []
	side_directions = ['E','N','W','S']
	sides_power = 0
	for idx in range(0, 4):
		side_powers.append(monitors_data['side_monitor_'+str(idx)]['P'])
		sides_power = sides_power + monitors_data['side_monitor_'+str(idx)]['P']

	try:
		scatter_power_2 = monitors_data['scatter_plane_monitor']['P']
	except Exception as err:
		scatter_power_2 = monitors_data['spill_plane_monitor']['P']
		
	focal_power = monitors_data['transmission_focal_monitor_']['P']

	scatter_power = 0
	for idx in range(0,4):
		scatter_power = scatter_power + monitors_data['vertical_scatter_monitor_'+str(idx)]['P']
		
	exit_aperture_power = monitors_data['exit_aperture_monitor']['P']

	#! Remove - rescaling

	def rescale_vector(vec, amin, amax):
		vec = (vec - np.min(vec))/(np.max(vec) - np.min(vec))
		vec = amin + (amax-amin)*vec
		return vec

	# focal_power = rescale_vector(focal_power/input_power, 0.45, 0.65) * input_power
	# scatter_power = scatter_power*0.7
	# R_power = R_power*0.2
	# for idx in range(0,4):
	# 	side_powers[idx] = side_powers[idx]*0.5
	
	
	power_sum = R_power + focal_power + scatter_power
	sides_power_2 = sides_power#  - (sourcepower - input_power)
	power_sum += sides_power_2

	baseline_power = calc_normalize_power(monitors_data, normalize_against)
	
	if np.max(power_sum/baseline_power) > 1:
		logging.warning('Warning! Absorption less than 0 at some point on the spectrum.')

	# # Test reflection power calculation method
	# power_sum_2 = power_sum - R_power
	# plt.plot(lambda_vector, R_power_1/input_power, label='Suggested Reflection Power')
	# plt.plot(lambda_vector, 1 - power_sum_2/input_power, label='Reflection Power + Unaccounted Power')
	# plt.plot(lambda_vector, 1 - power_sum_2/input_power - R_power_1/input_power, label=' Projected Unaccounted Power')
	# plt.plot(lambda_vector, 1 - power_sum_2/input_power - R_power_2/input_power, label=' Projected Unaccounted Power, v2')
	# plt.axhline(0, ls='--', c='gray')
	# plt.legend()
 
		
	# Determine peak wavelengths for each band
	quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
	for idx in range(0, num_adjoint_sources):
		f_vectors.append({'var_name': quadrant_names[idx],
						'var_values': np.divide(monitors_data['transmission_monitor_'+str(idx)]['P'],
												baseline_power)
	  											# monitors_data['incident_aperture_monitor']['P'])
						})
	
	f_vectors[1]['var_name'] = 'Green'
	f_vectors[1]['var_values'] = f_vectors[1]['var_values'] + f_vectors[3]['var_values']
	f_vectors.pop(3)
	
	peak_idxs = []
	for idx in range(0,3):
		sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
		peak_idxs.append(sweep_index)
	f_vectors = []
	
	
	
	f_vectors.append({'var_name': 'Device Reflection',
					  'var_values': R_power/baseline_power
					})
	
	for idx in range(0, 4):
		f_vectors.append({'var_name': 'Side Scattering ' + side_directions[idx],
					  'var_values': side_powers[idx]/baseline_power
					})
	f_vectors.append({'var_name': 'Side Scatter',
					'var_values': sides_power/baseline_power
				   })
		
	
	f_vectors.append({'var_name': 'Focal Region Power',
					  'var_values': focal_power/baseline_power
					})
	
	f_vectors.append({'var_name': 'Oblique Scatter',
					  'var_values': scatter_power/baseline_power   # (exit_aperture_power - focal_power)/baseline_power
					})
	
	if normalize_against not in ['power_sum', 'unity']:
		f_vectors.append({'var_name': 'Unaccounted Power',
						'var_values': 1 - power_sum/baseline_power
						})
	
	# Limit the datapoints solely to the peak spectral indices
	for vector in r_vectors:
		vector['var_values'] = vector['var_values'][peak_idxs]
	for vector in f_vectors:
		vector['var_values'] = vector['var_values'][peak_idxs]
	
	if produce_spectrum:
		output_plot['r'] = r_vectors
		output_plot['f'] = f_vectors
		output_plot['title'] = 'Device RTA Spectrum'
		output_plot['const_params'] = const_parameters
		
		print('Generated device RTA power spectrum.')
	
	sweep_values = []
	num_sweep_values = len(f_vectors)
	sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
	
	sweep_mean_vals = []
	for swp_val in sweep_values:
		sweep_mean_vals.append(swp_val['value'])
	if normalize_against == 'unity':
		logging.info('Mean Spectral Values: [' + ', '.join(map(lambda x: f'{x:.2e}', sweep_mean_vals)) + ']')     # Converts to scientific, 2 s.f.
	else:	logging.info('Mean Spectral Values: [' + ', '.join(map(lambda x: str(round(100*x,2)), sweep_mean_vals)) + ']')     # Converts to percentages with 2 s.f.

	for vector in f_vectors:
		message = vector['var_name'] + ' (B, G, R): '
		if normalize_against == 'unity':
			message += ", ".join(map(lambda x: f'{x:.2e}', vector['var_values'].reshape(-1).tolist()))
		else:	message += ", ".join(map(lambda x: str(round(100*x,2)), vector['var_values'].reshape(-1).tolist()))
		logging.info(message)
	
	logging.info('Picked device RTA points to pass to variable sweep.')
	
	return output_plot, sweep_values

#! Step 4 Functions

# TODO: Use plotter.py


#* END FUNCTION DEFINITIONS -----------------------------------------------------------------------------------------------------


#
#* These numpy arrays store all the data we will pull out of the simulation.
# These are just to keep track of the arrays we create. We re-initialize these right before their respective loops
#

forward_e_fields = {}                       # Records E-fields throughout device volume
focal_data = {}                             # Records E-fields at focal plane spots (locations of adjoint sources)
quadrant_efield_data = {}					# Records E-fields through each quadrant
quadrant_transmission_data = {}             # Records transmission through each quadrant

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

#
#* Set up queue for parallel jobs
#

jobs_queue = queue.Queue()

def add_job(job_name, queue_in):
	full_name = projects_directory_location + "/" + job_name
	fdtd_hook.save(os.path.abspath(full_name))
	queue_in.put(full_name)

	return full_name

def run_jobs(queue_in):
	small_queue = queue.Queue()

	while not queue_in.empty():
		
		for node_idx in range(0, num_nodes_available):
			# this scheduler schedules jobs in blocks of num_nodes_available, waits for them ALL to finish, and then schedules the next block
			# TODO: write a scheduler that enqueues a new job as soon as one finishes
			# not urgent because for the most part, all jobs take roughly the same time
			if queue_in.qsize() > 0:
				small_queue.put(queue_in.get())

		run_jobs_inner(small_queue)
		
def run_jobs_inner( queue_in ):
	processes = []
	job_idx = 0
	while not queue_in.empty():
		get_job_path = queue_in.get()

		logging.info(f'Node {cluster_hostnames[job_idx]} is processing file at {get_job_path}.')

		process = subprocess.Popen(
			[
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

		# if time.time()-job_queue_time > 3600:
		# 	logging.CRITICAL(r"Warning! The following jobs are taking more than an hour and might be stalling:")
		# 	logging.CRITICAL(completed_jobs)

		time.sleep( 5 )

# Control Parameters
restart = False                 #!! Set to true if restarting from a particular iteration - be sure to change start_epoch and start_iter
# if restart_epoch or restart_iter:		#! Disabled for this script so we don't have to turn restart off every time when coming from a restarted opt.
# 	restart = True
if restart:
	logging.info(f'Restarting optimization from epoch {restart_epoch}, iteration {restart_iter}.')

	current_job_idx = (0,0,0)
	
#
# Run the simulation
#

start_epoch = 0	# restart_epoch
for epoch in range(start_epoch, num_epochs):
	fdtd_hook.switchtolayout()
	
	start_iter = 0 # restart_iter
	for iteration in range(start_iter, num_iterations_per_epoch):
		logging.info(f"Working on epoch {epoch} and iteration {iteration}")

		job_names = {}
		fdtd_hook.switchtolayout()

		#
		# Step 1: Import sweep parameters, assemble jobs corresponding to each value of the sweep. Edit the environment of each job accordingly.
		# Queue and run parallel, save out.
		#
		
		if start_from_step == 0 or start_from_step == 1:
			if start_from_step == 1:
				load_backup_vars(shelf_fn)
			
			logging.info("Beginning Step 1: Running Sweep Jobs")

			for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
				# Iterate through each dimension of the sweep parameter value array, landing on each coordinate.
				parameter_filename_string = create_parameter_filename_string(idx)
				current_job_idx = idx
				logging.info(f'Creating Job: Current parameter index is {current_job_idx}')
				fdtd_hook.load(os.path.abspath(os.path.join(projects_directory_location, device_filename)))

				for t_idx, p_idx in enumerate(idx):
					# We create a job corresponding to the coordinate in the parameter value space.
	 
					variable = list(sweep_parameters.values())[t_idx]	# Access the sweep parameter being changed in this job
					variable_name = variable['var_name']				# Get its name
					variable_value = variable['var_values'][p_idx]		# Get the value it will be changed to in this job
					
					current_parameter_values[variable_name] = variable_value
					fdtd_hook.switchtolayout()
					# depending on what variable is, edit Lumerical

					if variable_name in ['angle_theta']:
						change_source_theta(variable_value, current_parameter_values['angle_phi'])
					elif variable_name in ['angle_phi']:
						change_source_theta(current_parameter_values['angle_theta'], variable_value)
					elif variable_name in ['beam_radius_um']:
						change_source_beam_radius(variable_value)
					elif variable_name in ['sidewall_thickness_um']:
						change_sidewall_thickness(variable_value)
					elif variable_name in ['divergence_angle']:
						change_divergence_angle(variable_value)
	
					for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations
						# We set up two simulations for each parameter coordinate: one with devices enabled, and one with devices disabled.

						# Make sure to reduce memory requirements by disabling and enabling the right objects.
						disable_all_sources()
						disable_objects(fdtd_hook, disable_object_list)
						update_object(fdtd_hook, ['design_import'], 'enabled', True)
						update_object(fdtd_hook, ['forward_src_' + xy_names[xy_idx]], 'enabled', True)
	
						# Name the job and add to queue.
						job_name = f'sweep_job_{parameter_filename_string}.fsp'
						# fdtd_hook.save(os.path.abspath(projects_directory_location + "/optimization"))
						logging.info('Saved out ' + job_name)
						job_names[('sweep', xy_idx, idx)] = add_job(job_name, jobs_queue)
	
						# Disable design imports and sidewalls, to measure power through source aperture, and power that misses device
						disable_all_devices()
						disable_all_sidewalls()

						# # Input a cube of uniform density=0.5 to normalize against?
						if normalize_against == 'uniform_cube':
							bayer_filter_size_voxels = np.array(
								[device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
							cur_index = np.sqrt(0.5*(1.5**2+2.4**2)) * np.ones(bayer_filter_size_voxels)
							bayer_filter_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
							bayer_filter_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 0.5 * device_size_lateral_um, device_voxels_lateral)
							bayer_filter_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, device_vertical_maximum_um, device_voxels_vertical)
							fdtd_hook.select( fdtd_objects['design_import']['name'] )
							fdtd_hook.importnk2( cur_index, bayer_filter_region_x, 
												bayer_filter_region_y, bayer_filter_region_z )

						update_object(fdtd_hook, ['src_spill_monitor'], 'enabled', True)
	
						# Name the job and add to queue.
						job_name = f'sweep_job_nodev_{parameter_filename_string}.fsp'
						# fdtd_hook.save(os.path.abspath(projects_directory_location + "/optimization"))
						logging.info('Saved out ' + job_name)
						job_names[('nodev', xy_idx, idx)] = add_job(job_name, jobs_queue)
	
			backup_all_vars()  
			if not running_on_local_machine:
				run_jobs(jobs_queue)
			
			logging.info("Completed Step 1: Forward Optimization Jobs Completed.")
			backup_all_vars()

		#
		# Step 2: Extract fields from each monitor, pass as inputs to premade functions in order to extract plot data. Data is typically a
		# spectrum for each function (e.g. sorting efficiency) and also the peak value across that spectrum.
		#
		
		if start_from_step == 0 or start_from_step == 2:
			if start_from_step == 2:
				load_backup_vars(shelf_fn)

			logging.info("Beginning Step 2: Extracting Fields from Monitors.")

			jobs_data = {}
			for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
				current_job_idx = idx
				logging.info(f'----- Accessing Job: Current parameter index is {current_job_idx}')

				for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations
					
					# Load each individual job (with device)
					completed_job_filepath = utility.convert_root_folder(job_names[('sweep', xy_idx, idx)], PULL_COMPLETED_JOBS_FOLDER)
					start_loadtime = time.time()
					fdtd_hook.load(os.path.abspath(completed_job_filepath))
					logging.info(f'Loading: {utility.isolate_filename(completed_job_filepath)}. Time taken: {time.time()-start_loadtime} seconds.')

					monitors_data = {}
					# Add wavelength and sourcepower information, common to every monitor in a job
					monitors_data['wavelength'] = np.divide(3e8, fdtd_hook.getresult('focal_monitor_0','f'))
					monitors_data['sourcepower'] = lm.get_sourcepower(fdtd_hook)
					# f1 = fdtd_hook.getresult('focal_monitor_0','f')		# Might need to replace 'focal_monitor_0' with a search for monitors & their names
					# monitors_data['sourceintensity'] = fdtd_hook.sourceintensity(f1)
					# monitors_data['sourcearea'] = fdtd_hook.getdata('forward_src_x','area')
					
	
					# Go through each individual monitor and extract respective data
					for monitor in monitors:
						if monitor['name'] not in ['src_spill_monitor']:	# Exclude these monitors; their data will be pulled from the nodev job
							monitor_data = {}
							monitor_data['name'] = monitor['name']		# Load name
							monitor_data['save'] = monitor['save']		# Config parameter: Load whether or not to save this monitor's data 
	
							if monitor['enabled'] and monitor['getFromFDTD']:
								# Load E-norm(x,y,z), power, and transmission, all as a function of frequency. Every monitor needs these.
								monitor_data['E'] = lm.get_efield_magnitude(fdtd_hook, monitor['name'])
								monitor_data['P'] = lm.get_overall_power(fdtd_hook, monitor['name'])
								monitor_data['T'] = lm.get_transmission_magnitude(fdtd_hook, monitor['name'])
		
								# The reflected power from the device is R_power = incident_aperture_power (no device) - incident_aperture_power (device)
								if monitor['name'] in ['incident_aperture_monitor']:
									monitor_data['R_power'] = -1 * lm.get_overall_power(fdtd_hook, monitor['name'])
	
								#---------------------------- Field Profile Information
								# Get xyz coordinates for these specific monitors, in order to plot field profiles
								if monitor['name'] in ['spill_plane_monitor', 'scatter_plane_monitor',
														'transmission_monitor_0','transmission_monitor_1','transmission_monitor_2','transmission_monitor_3',
														'transmission_focal_monitor_',
														'device_xsection_monitor_0', 'device_xsection_monitor_1', 'device_xsection_monitor_2', 
														'device_xsection_monitor_3', 'device_xsection_monitor_4', 'device_xsection_monitor_5'
			  											]:
									monitor_data['coords'] = {'x': fdtd_hook.getresult(monitor['name'],'x'),
															'y': fdtd_hook.getresult(monitor['name'],'y'),
															'z': fdtd_hook.getresult(monitor['name'],'z')}
		
								# Get detailed Ex, Ey, Ez field information for these monitors in order to plot field profiles.
								if monitor['name'] in ['device_xsection_monitor_0', 'device_xsection_monitor_1', 'device_xsection_monitor_2', 
														'device_xsection_monitor_3', 'device_xsection_monitor_4', 'device_xsection_monitor_5']:
									monitor_data['E_real'] = lm.get_efield(fdtd_hook, monitor['name'])
								#---------------------------------------------------------------------------------------------------------

							monitors_data[monitor['name']] = monitor_data
					input_power_with_device = monitors_data['incident_aperture_monitor']['P']

					# Load each individual job (w/o device)
					completed_job_filepath = utility.convert_root_folder(job_names[('nodev', xy_idx, idx)], PULL_COMPLETED_JOBS_FOLDER)
					start_loadtime = time.time()
					fdtd_hook.load(os.path.abspath(completed_job_filepath))
					logging.info(f'Loading: {utility.isolate_filename(completed_job_filepath)}. Time taken: {time.time()-start_loadtime} seconds.')

					# Go through each individual monitor and extract respective data
					for monitor in monitors:
						if monitor['name'] in ['incident_aperture_monitor', 'src_spill_monitor']:	# These are the only monitors relevant for the nodev job.
										# Also we are going to overwrite all the data of incident_aperture_monitor except for the 'R' key
							monitor_data = {}
							monitor_data['name'] = monitor['name']		# Load name
							monitor_data['save'] = monitor['save']		# Config parameter: Load whether or not to save this monitor's data 
	
							if monitor['enabled'] and monitor['getFromFDTD']:
								# Load E-norm(x,y,z), power, and transmission, all as a function of frequency. Every monitor needs these.
								monitor_data['E'] = lm.get_efield_magnitude(fdtd_hook, monitor['name'])
								monitor_data['P'] = lm.get_overall_power(fdtd_hook, monitor['name'])
								monitor_data['T'] = lm.get_transmission_magnitude(fdtd_hook, monitor['name'])
		
								# The reflected power from the device is R_power = incident_aperture_power (no device) - incident_aperture_power (device)
								if monitor['name'] in ['incident_aperture_monitor']:
									monitor_data['R_power'] = monitors_data[monitor['name']]['R_power'] + lm.get_overall_power(fdtd_hook, monitor['name'])
									monitor_data['coords'] = {'x': fdtd_hook.getresult(monitor['name'],'x'),
															'y': fdtd_hook.getresult(monitor['name'],'y'),
															'z': fdtd_hook.getresult(monitor['name'],'z')}

							monitors_data.update( {monitor['name']: monitor_data} )
					input_power_without_device = monitors_data['incident_aperture_monitor']['P']

					# Save out to a dictionary, using job_names as keys
					jobs_data[job_names[('sweep', xy_idx, idx)]] = monitors_data

			logging.info("Completed Step 2: Extracted all monitor fields.")
			backup_all_vars()

		#
		# Step 3: For each job / parameter combination, create the relevant plot data as a dictionary.
		# The relevant plot types are all stored in sweep_settings.json
		#
	
		if start_from_step == 0 or start_from_step == 3:
			if start_from_step == 3:
				load_backup_vars(shelf_fn)
	
			logging.info("Beginning Step 3: Gathering Function Data for Plots")
	
			plots_data = {}
			
			for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
				logging.info(f'----- Gathering Plot Data: Current parameter index is {idx}')
				# Load the monitor data for each job and format it for plot data. 
	
				for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations
					monitors_data = jobs_data[job_names[('sweep', xy_idx, idx)]]
	
					plot_data = {}
					plot_data['wavelength'] = monitors_data['wavelength']
					plot_data['sourcepower'] = monitors_data['sourcepower']
	
					for plot in plots:
						if plot['enabled']:
		
							if plot['name'] in ['sorting_efficiency']:
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_sorting_eff_spectrum(monitors_data, plot['generate_plot_per_job'], add_opp_polarizations=True)
									
							elif plot['name'] in ['sorting_transmission']:	
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_sorting_transmission_spectrum(monitors_data, plot['generate_plot_per_job'], add_opp_polarizations=True)
		
							elif plot['name'] in ['sorting_transmission_meanband', 'sorting_transmission_contrastband']:
								# TODO:
								continue
								# band_idxs = partition_bands(monitors_data['wavelength'], [4.2e-7, 5.5e-7, 6.9e-7])
								
								# plot_data[plot['name']+'_sweep'] = \
								# 	generate_inband_val_sweep(monitors_data, plot['generate_plot_per_job'], 
								# 										   statistics='mean', add_opp_polarizations=True,
								# 										   #band_idxs=band_idxs)
								# 										   band_idxs=None)
		
							elif plot['name'] in ['crosstalk_adj_quadrants']:
								#! For plotting relative power spectra of adjacent quadrants
								# quad_crosstalk_plot_idxs = [15, 21, 9]
								# quad_crosstalk_plot_colors = ['green', 'blue', 'cyan']
								# quad_crosstalk_plot_labels = [r'$G_{1,c}$', r"$B_c$", r'$B_w$']

								# quad_crosstalk_plot_idxs = [20, 21, 19]
								# quad_crosstalk_plot_colors = ['green', 'blue', 'cyan']
								# quad_crosstalk_plot_labels = [r'$G_{2,c}$', r"$B_c$", r'$B_s$']

								quad_crosstalk_plot_idxs = [21, 15, 20, 22, 27]
								quad_crosstalk_plot_colors = ['blue', r'#66ff66', r'#00ff00', r'#009900', r'#003300']
								quad_crosstalk_plot_labels = [r"$B_c$", r'$G_{1,c}$', r'$G_{2,c}$', r'$G_{2,n}$', r'$G_{1,e}$']

								# quad_crosstalk_plot_idxs = [14, 15, 20, 8, 13]
								# quad_crosstalk_plot_colors = ['red', r'#66ff66', r'#00ff00', r'#009900', r'#003300']
								# quad_crosstalk_plot_labels = [r"$R_c$", r'$G_{1,c}$', r'$G_{2,c}$', r'$G_{2,w}$', r'$G_{1,s}$']
								
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_crosstalk_adj_quadrants_spectrum(monitors_data, plot['generate_plot_per_job'], 
																			  quad_crosstalk_plot_idxs, quad_crosstalk_plot_labels)

							elif plot['name'] in ['Enorm_focal_plane_image']:
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_Enorm_focal_plane(monitors_data, plot['generate_plot_per_job'])
		
							elif plot['name'] in ['device_cross_section'] and sweep_parameter_value_array[idx][0] == angle_theta:
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_device_cross_section_spectrum(monitors_data, plot['generate_plot_per_job'])
		
							# TODO:
							# elif plot['name'] in ['src_spill_power']:
							# 	plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
							# 		generate_source_spill_power_spectrum(monitors_data, plot['generate_plot_per_job'])
		
							elif plot['name'] in ['crosstalk_power']:
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_crosstalk_power_spectrum(monitors_data, plot['generate_plot_per_job'])
		
							elif plot['name'] in ['exit_power_distribution']:
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_exit_power_distribution_spectrum(monitors_data, plot['generate_plot_per_job'])
		
							elif plot['name'] in ['device_rta']:
								plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
									generate_device_rta_spectrum(monitors_data, plot['generate_plot_per_job'])
									# generate_device_rta_spectrum_2(monitors_data, plot['generate_plot_per_job'])

					plots_data[job_names[('sweep', xy_idx, idx)]] = plot_data
	
			logging.info("Completed Step 3: Processed and saved relevant plot data.")
			backup_all_vars()
		
		
		#
		# Step 4: Plot Sweeps across Jobs
		#
		
		# TODO: Undo the link between Step 3 and 4
		if start_from_step == 0 or start_from_step == 4:
			if start_from_step == 4:
				load_backup_vars(shelf_fn)
		

			logging.info("Beginning Step 4: Extracting Sweep Data for Each Plot, Creating Plots and Saving Out")
			if 'plots_data' not in globals():
				logging.critical('*** Warning! Plots Data from previous step not present. Check your save state.')

			if not os.path.isdir(PLOTS_FOLDER):
				os.makedirs(PLOTS_FOLDER)

			startTime = time.time()

			#* Here a sweep plot means data points have been extracted from each job (each representing a value of a sweep parameter)
			#* and we are going to combine them all into a single plot.
			
			# Create dictionary to hold all the sweep plots
			sweep_plots = {}
			for plot in plots:
				# Each entry in the plots section of sweep_settings.json should have a sweep plot created for it

				# Initialize the dictionary to hold output plot data:
				output_plot = {}
				if plot['enabled'] and plot['name'] not in ['Enorm_focal_plane_image', 
											"device_transmission", "device_reflection", "side_scattering_power",
											"focal_region_power", "focal_scattering_power", "oblique_scattering_power"]:
	
					r_vectors = [sweep_parameters]
					
					# Initialize f_vectors
					f_vectors = []
					job_name = list(job_names.values())[0]
					# How many different things are being tracked over the sweep?
					for f_idx in range(0, len(plots_data[job_name][plot['name']+'_sweep'])):
						f_vectors.append({'var_name': plot['name'] + '_' + str(f_idx),
										  'var_values': np.zeros(sweep_parameters_shape),
										  'var_stdevs': np.zeros(sweep_parameters_shape),
										  'statistics': ''
										  })
	
					# Populate f_vectors
					for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations    
						for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
							job_name = job_names[('sweep', xy_idx, idx)]
							f_values = plots_data[job_name][plot['name']+'_sweep']
							
							for f_idx, f_value in enumerate(f_values):
								f_vectors[f_idx]['var_values'][idx] = f_value['value']
								f_vectors[f_idx]['var_stdevs'][idx] = f_value['std_dev']
								f_vectors[f_idx]['statistics'] = f_value['statistics']

					output_plot['r'] = r_vectors
					output_plot['f'] = f_vectors
					output_plot['title'] = plot['name']+'_sweep'
			
					sweep_plots[plot['name']] = output_plot
	
			plots_data['sweep_plots'] = sweep_plots

			#* We now have data for each and every plot that we are going to export.
			
			#
			# Step 4.1: Go through each job and export the spectra for that job
			#

			# Purge the nodev job names
			job_names_check = job_names.copy()
			for job_idx, job_name in job_names_check.items():
				if 'nodev' in utility.isolate_filename(job_name):
					job_names.pop(job_idx)
	
			# Access each job
			for job_idx, job_name in job_names.items():
				plot_data = plots_data[job_name]
				
				logging.info('Plotting for Job: ' + utility.isolate_filename(job_names[job_idx]))

				# Produce all the spectrum plots for each job.
				device_indiv_rta_plots = []		# This will save which of the individual RTA plots are used
    
				for plot in plots:
					if plot['enabled']:
		 
						if plot['name'] in ['sorting_efficiency']:
							plotter.plot_sorting_eff_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
																		sweep_parameters, job_names, PLOTS_FOLDER, include_overall=True)
							continue
							
						elif plot['name'] in ['sorting_transmission']:
							plotter.plot_sorting_transmission_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
																		sweep_parameters, job_names, PLOTS_FOLDER, 
				  														include_overall=True, normalize_against=normalize_against)
	
						# TODO: sorting_transmission_meanband and contrast_band

						elif plot['name'] in ['crosstalk_adj_quadrants']:
							plotter.plot_crosstalk_adj_quadrants_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
																		quad_crosstalk_plot_colors, quad_crosstalk_plot_labels,
																		sweep_parameters, job_names, PLOTS_FOLDER)

						elif plot['name'] in ['Enorm_focal_plane_image']:
							plotter.plot_Enorm_focal_plane_image_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
													 					sweep_parameters, job_names, PLOTS_FOLDER,
										# plot_wavelengths = None,
										#plot_wavelengths = monitors_data['wavelength'][[n['index'] for n in plot_data['sorting_efficiency_sweep']]],
										plot_wavelengths = np.array(desired_peaks_per_band_um) * 1e-6,
										# plot_wavelengths = [4.4e-7,5.4e-7, 6.4e-7],
										ignore_opp_polarization=False)
										# Second part gets the wavelengths of each spectra where sorting eff. peaks
		
						elif plot['name'] in ['device_cross_section']:
							plotter.plot_device_cross_section_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
																		sweep_parameters, job_names, PLOTS_FOLDER,
										plot_wavelengths = np.array(desired_peaks_per_band_um) * 1e-6,
										ignore_opp_polarization=False)
	
						# TODO: Spill power
						# elif plot['name'] in ['src_spill_power']:
						# 		plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
						# 			generate_source_spill_power_spectrum(monitors_data, plot['generate_plot_per_job'])

						elif plot['name'] in ['crosstalk_power']:
							plotter.plot_crosstalk_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
																		sweep_parameters, job_names, PLOTS_FOLDER)
	   
						elif plot['name'] in ['exit_power_distribution']:
							plotter.plot_exit_power_distribution_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
																		sweep_parameters, job_names, PLOTS_FOLDER)

						elif plot['name'] in ['device_rta']:
							plotter.plot_device_rta_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
																		sweep_parameters, job_names, PLOTS_FOLDER,
				  														sum_sides=True, normalize_against=normalize_against)
							plotter.plot_device_rta_pareto(plot_data[plot['name']+'_spectrum'], job_idx,
																		sweep_parameters, job_names, PLOTS_FOLDER,
				  														sum_sides=True, normalize_against=normalize_against)
						
						# Check if any of the quantities normalized to input power are present
						elif plot['name'] in ["device_transmission", "device_reflection", "side_scattering_power",
											"focal_region_power", "focal_scattering_power", "oblique_scattering_power"]:
							device_indiv_rta_plots.append(plot['name'])
        
				# Finally, plot the individual RTA plots
				# TODO: Based on dev_indiv_rta_names, pop out dictionaries from plot_data['f']
				device_indiv_rta_plots = ['device_reflection', 'side_scattering_power']
				# plotter.plot_device_indiv_rta_spectra(plot_data['device_rta_spectrum'], job_idx,
				# 									sweep_parameters, job_names, PLOTS_FOLDER,
				# 									device_indiv_rta_plots,
				# 									sum_sides=True, normalize_against=normalize_against)
							


			#
			# Step 4.2: Go through all the sweep plots and export each one
			#

			for plot in plots:
				if plot['enabled']:
						
					if plot['name'] in ['sorting_efficiency']:
						# # Here there is the most variability in what the user might wish to plot, so it will have to be hardcoded for the most part.
						# # plot_function_sweep(plots_data['sweep_plots'][plot['name']], 'th', plot_sorting_eff_sweep_1d.__name__)
						# # plot_sorting_eff_sweep(plots_data['sweep_plots'][plot['name']], 'phi')
						# plot_sorting_eff_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0])
						continue
  
					elif plot['name'] in ['sorting_transmission']:
						plotter.plot_sorting_transmission_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], PLOTS_FOLDER)
	  
					elif plot['name'] in ['device_rta']:
						plotter.plot_device_rta_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], PLOTS_FOLDER,
									   plot_stdDev=False)
	
					# TODO: OTHER SWEEPS

			backup_var('plots_data', os.path.basename(python_src_directory))
			backup_all_vars()
 
			logging.info("Completed Step 4: Created and exported all plots and plot data.")
			logging.info(f'Step 4 took {(time.time()-startTime)/60} minutes.')


logging.info('Reached end of file. Code completed.')

