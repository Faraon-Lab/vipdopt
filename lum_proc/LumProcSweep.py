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
fdtd_hook.load(os.path.abspath(os.path.join(projects_directory_location, device_filename)))
fdtd_hook.save(os.path.abspath(os.path.join(projects_directory_location, device_filename)))

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

# TODO: Region stuff
# TODO: --------------------------------------------------------------------------------------------------------------------------





logging.info('Reached end of file. Code completed.')

