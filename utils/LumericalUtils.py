# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import os
import copy
import numpy as np
import logging
import time

# Custom Classes and Imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from utils import utility

def get_sourcepower( fdtd_hook_ ):
	f = fdtd_hook_.getresult('focal_monitor_0','f')		# Might need to replace 'focal_monitor_0' with a search for monitors & their names
	sp = fdtd_hook_.sourcepower(f)
	return sp

def get_afield( fdtd_hook_, monitor_name, field_indicator ):
	field_polarizations = [ field_indicator + 'x',
					   	field_indicator + 'y', field_indicator + 'z' ]
	data_xfer_size_MB = 0

	start = time.time()

	pol_idx = 0
	logging.debug(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}.')
	field_pol_0 = fdtd_hook_.getdata( monitor_name, field_polarizations[ pol_idx ] )
	logging.debug(f'Size {field_pol_0.nbytes}.')

	data_xfer_size_MB += field_pol_0.nbytes / ( 1024. * 1024. )

	total_field = np.zeros( [ len (field_polarizations ) ] +
						list( field_pol_0.shape ), dtype=np.complex128 )
	total_field[ 0 ] = field_pol_0

	logging.info(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}. Size: {data_xfer_size_MB}MB. Time taken: {time.time()-start} seconds.')

	for pol_idx in range( 1, len( field_polarizations ) ):
		logging.debug(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}.')
		field_pol = fdtd_hook_.getdata( monitor_name, field_polarizations[ pol_idx ] )
		logging.debug(f'Size {field_pol.nbytes}.')

		data_xfer_size_MB += field_pol.nbytes / ( 1024. * 1024. )

		total_field[ pol_idx ] = field_pol

		logging.info(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}. Size: {field_pol.nbytes/(1024*1024)}MB. Total time taken: {time.time()-start} seconds.')

	elapsed = time.time() - start

	date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
	logging.debug( "Transferred " + str( data_xfer_size_MB ) + " MB\n" )
	logging.debug( "Data rate = " + str( date_xfer_rate_MB_sec ) + " MB/sec\n\n" )

	return total_field

def get_hfield( fdtd_hook_, monitor_name ):
	return get_afield( fdtd_hook_, monitor_name, 'H' )

def get_efield( fdtd_hook_, monitor_name ):
	return get_afield( fdtd_hook_, monitor_name, 'E' )

def get_Pfield( fdtd_hook_, monitor_name ):
	'''Returns power as a function of space and wavelength, with three components Px, Py, Pz'''
	read_pfield = fdtd_hook_.getresult( monitor_name, 'P' )
	read_P = read_pfield[ 'P' ]
	return read_P
	# return get_afield( monitor_name, 'P' )

def get_power( fdtd_hook_, monitor_name ):
	'''Returns power as a function of space and wavelength, with three components Px, Py, Pz'''
	return get_Pfield( fdtd_hook_, monitor_name )

def get_efield_magnitude(fdtd_hook_, monitor_name):
	e_field = get_afield(fdtd_hook_, monitor_name, 'E')
	Enorm2 = np.square(np.abs(e_field[0,:,:,:,:]))
	for idx in [1,2]:
		Enorm2 += np.square(np.abs(e_field[idx,:,:,:,:]))
	return np.sqrt(Enorm2)

def get_transmission_magnitude( fdtd_hook_, monitor_name ):
	read_T = fdtd_hook_.getresult( monitor_name, 'T' )
	return np.abs( read_T[ 'T' ] )

def get_overall_power(fdtd_hook_, monitor_name):
	'''Returns power spectrum, power being the summed power over the monitor.'''
	# Lumerical defines all transmission measurements as normalized to the source power.
	source_power = get_sourcepower(fdtd_hook_)
	T = get_transmission_magnitude(fdtd_hook_, monitor_name)

	return source_power * np.transpose([T])

### DEPRECATED

def get_monitor_data(monitor_name, monitor_field):
	'''Consolidate the data transfer functionality for getting data from Lumerical FDTD process to
	python process.  This is much faster than going through Lumerical's interop library'''

	lumerical_data_name = "monitor_data_" + monitor_name + "_" + monitor_field
	extracted_data_name = lumerical_data_name + "_data"
	data_transfer_filename = projects_directory_location + "/data_transfer_" + monitor_name + "_" + monitor_field

	command_read_monitor = lumerical_data_name + " = getresult(\'" + monitor_name + "\', \'" + monitor_field + "\');"
	command_extract_data = extracted_data_name + " = " + lumerical_data_name + "." + monitor_field + ";"
	command_save_data_to_file = "matlabsave(\'" + data_transfer_filename + "\', " + extracted_data_name + ");"

	lumapi.evalScript(fdtd_hook.handle, command_read_monitor)
	lumapi.evalScript(fdtd_hook.handle, command_extract_data)

	# start_time = time.time()

	lumapi.evalScript(fdtd_hook.handle, command_save_data_to_file)
	monitor_data = {}
	load_file = h5py.File(data_transfer_filename + ".mat")

	monitor_data = np.array(load_file[extracted_data_name])

	# end_time = time.time()

	# print("\nIt took " + str(end_time - start_time) + " seconds to transfer the monitor data\n")

	return monitor_data

def get_complex_monitor_data(monitor_name, monitor_field):
	data = get_monitor_data(monitor_name, monitor_field)
	return (data['real'] + np.complex128(0, 1) * data['imag'])


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

	if isinstance(simObj, list):	# list of SimObjects
		duplicate_list = []
		for object in simObj:
			duplicate_list.append(fdtd_simobject_to_dict(object))
		return duplicate_list

	else: #if isinstance(simObj, lumapi.SimObject):
			duplicate_dict = {}
			for key in simObj._nameMap:
				duplicate_dict[key] = simObj[key]
			return duplicate_dict

	# else:
	# 	raise ValueError('Tried to convert something that is not a Lumapi SimObject or list thereof.')

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
			elif any(ele.lower() in type_string for ele in ['IndexMonitor']):
				fdtd_hook_.addindex(name=simDict['name'])
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

def update_object(object_path, key, val, fdtd_objects_, fdtd_hook_):
	'''Updates selected objects in the FDTD simulation.'''

	utility.set_by_path(fdtd_objects_, object_path + [key], val)
	new_dict = fdtd_update_object(fdtd_hook_, utility.get_by_path(fdtd_objects_, object_path))
	utility.set_by_path(fdtd_objects_, object_path, new_dict)

def disable_objects(fdtd_hook_, lumapi_, object_list, fdtd_objects_):
	'''Disable selected objects in the simulation, to save on memory and runtime'''
	global fdtd_objects
	lumapi_.evalScript(fdtd_hook_.handle, 'switchtolayout;')

	for object_path in object_list:
		update_object(object_path, 'enabled', 0, fdtd_objects_, fdtd_hook_)

def disable_all_sources(fdtd_hook_, lumapi_, fdtd_objects):
	'''Disable all sources in the simulation, so that we can selectively turn single sources on at a time'''
	lumapi_.evalScript(fdtd_hook_.handle, 'switchtolayout;')

	object_list = []

	for key, val in fdtd_objects.items():
		if key not in ['SIM_INFO','DEVICE_INFO']:
			if isinstance(val, list):
				# type_string = val[0]['type'].lower()
				pass
			else:
				type_string = val['type'].lower()

				if any(ele.lower() in type_string for ele in ['GaussianSource', 'TFSFSource', 'DipoleSource']):
					object_list.append([val['name']])

	disable_objects(fdtd_hook_, lumapi_, object_list, fdtd_objects)

class LumericalFDTD:
	'''Creates an object of class LumericalFDTD that can be used to interact with the FDTD hook determined at initialization.'''

	def __init__(self, lumapi_filepath):
		self.lumapi_filepath = lumapi_filepath

  		# Import and connect
		import importlib.util as imp	# Lumerical hook Python library API
		self.spec_lumapi = imp.spec_from_file_location('lumapi', self.lumapi_filepath)
		self.lumapi = imp.module_from_spec(self.spec_lumapi)
		self.spec_lumapi.loader.exec_module(self.lumapi)

	def connect(self):
		'''Create FDTD hook'''

		logging.info('Attempting to connect to Lumerical servers...')

		# self.fdtd_hook = lumapi.FDTD()
		while True:			# The while loop handles the problem of "Failed to start messaging, check licenses..." that Lumerical often encounters.
			try:	self.fdtd_hook = self.lumapi.FDTD()
			except Exception:
				logging.exception("Licensing server error - can be ignored.")
				continue
			break

		logging.info('Verified license with Lumerical servers.')
		self.fdtd_hook.newproject()

	def default_view(self):
		'''Reset default view'''
		selected_before = self.fdtd_hook.get('name')

		self.fdtd_hook.select('FDTD')
		self.fdtd_hook.setview("extent")
		self.fdtd_hook.setview("zoom",4)
		self.fdtd_hook.setview("theta", 30)

		self.fdtd_hook.select(selected_before)	# Restore current selected object

	def open(self, filename=None):
		if filename is None:
			self.fdtd_hook.newproject()
		else:
			self.fdtd_hook.load(os.path.abspath(filename))
			logging.info(f'Opened {os.path.abspath(filename)}.')

	def save(self, filepath, filename=None):

		self.default_view()

		if filename is None:
			self.fdtd_hook.save(os.path.abspath(filepath))
			filename = os.path.basename(filepath) + '.fsp'
		else:
			self.fdtd_hook.save(os.path.join( os.path.abspath(filepath), filename ))

		logging.info(f'Saved out {filename} to directory {os.path.abspath(filepath)}.')

	def import_obj(self, sim_objects, create_object_=False):
		'''Takes dictionary of simulation objects (already partitioned).'''

		for obj_key, obj in sim_objects.items():
			if obj_key not in ['SIM_INFO', 'DEVICE_INFO'] and isinstance(obj, dict):
				obj = fdtd_update_object(self.fdtd_hook, obj, create_object=create_object_)
				sim_objects[obj_key] = obj

		return sim_objects	# return a list because there might be multiple simulations (e.g. for E-field stitching)

	def disable_all_sources(self, fdtd_objects):
		disable_all_sources(self.fdtd_hook, self.lumapi, fdtd_objects)

	def partition_objs(self, initial_objects, split_array=(1,)):
		'''Takes dictionary of initial simulation objects (unpartitioned).
			split_array is of the form (2,3,5) or something, i.e. split a big simulation into (2 x 3 x 5) smaller simulations.
		Each Simulation object should contain a dictionary from which the corresponding simulation file can be regenerated.'''

		# todo: partitioning code
		# check coordinates values of initial_objects
		# split accordingly according to size - this gives length of the final list

		simulations = []
		# simulations is a list of Simulations objects.
		# Each Simulation object should contain a dictionary from which the corresponding simulation file can be regenerated
		# Multiple simulations may be created here due to the need for large-area simulation segmentation, or genetic optimizations

		for i in range(np.prod(split_array)):
			# todo: partitioning code
			new_sim_obj = copy.deepcopy(initial_objects)
			if np.prod(split_array) != 1:
				new_sim_obj['SIM_INFO']['name'] += f'_{i}'
			new_sim_obj['SIM_INFO']['sim_id'] = i
			simulations.append(new_sim_obj)

		return simulations

	def assign_devices_to_simulations(self, simulations, initial_devices):
		'''Takes already partitioned list of simulation dictionaries, and unpartitioned list of dictionaries of devices.
			Checks coordinates of each device and partitions into SUBDEVICES - or "designs / design regions".
			Then, assigns accordingly to simulation regions.
			Returns partitioned list of device dictionaries.'''

		designs = []
		# designs is a list of Design objects.
		# Each Design object should contain a dictionary from which the corresponding simulation file can be regenerated
		# Multiple Designs may be created here due to the need for co-optimization, or for large-area simulation segmentation, or genetic optimizations

		# todo: partitioning code. Check coordinates of initial_devices and see where it slots in in the simulation regions
		for init_device in initial_devices:
			num_designs = 1
			for i in range(num_designs):
				new_design = copy.deepcopy(init_device)
				# if np.prod(split_array) != 1:
				# 	new_sim_obj['SIM_INFO']['name'] += f'_{i}'
				new_design['DEVICE_INFO']['simulation'] = 0
				new_design['DEVICE_INFO']['parent_device'] = init_device['design_import']['dev_id']
				new_design['DEVICE_INFO']['subdev_id'] = str(init_device['design_import']['dev_id']) + str(i)
				designs.append(new_design)

		return designs

	def construct_sim_device_hashtable(self, simulations, designs):
		'''Takes partitioned list of simulation and design dictionaries.
			Returns maps for design to device and design to simulation.'''

		# # Example for generating device, subdevice, and simulation region maps. Diagram at https://imgur.com/a/cwfyICM
		#! Must keep track of which devices correspond to which regions and vice versa.
		#? What is the best method to keep track of maps? Try: https://stackoverflow.com/a/21894086
		# subdevice_to_device_map = {'00': 0, '01': 0, '02': 0, '03': 0,
		# 						'10': 1, '11': 1,
		# 						'20': 2, '21': 2, '22': 2,
		# 						'30': 3, '31': 3}
		# # device_to_subdevice_map = {0: ['00','01','02','03'], 1: ['10','11'], 2: ['20','21','22'], 3: ['30','31']}
		# device_to_subdevice_map = utility.bidict(subdevice_to_device_map).inverse
		# subdevice_to_simulation_map = {'00': 2, '01': 3, '02': 5, '03': 6,
		# 						'10': 3, '11': 6,
		# 						'20': 4, '21': 5, '22': 6,
		# 						'30': 7, '31': 8}
		# simulation_to_subdevice_map = utility.bidict(subdevice_to_simulation_map).inverse
		# device_to_simulation_map = {i: [subdevice_to_simulation_map[k] for k in device_to_subdevice_map[i]]     for i in list(device_to_subdevice_map.keys())}
		# simulation_to_device_map = {i: [subdevice_to_device_map[k] for k in simulation_to_subdevice_map[i]]     for i in list(simulation_to_subdevice_map.keys())}

		design_to_device_map = {}
		design_to_simulation_map = {}

		for design in designs:
			design_to_device_map.update({design['DEVICE_INFO']['subdev_id']: design['DEVICE_INFO']['parent_device']})
			design_to_simulation_map.update({design['DEVICE_INFO']['subdev_id']: design['DEVICE_INFO']['simulation']})

		return design_to_device_map, design_to_simulation_map


	def get_device_to_simulation_map(self, subdevice_to_device_map, subdevice_to_simulation_map):
		device_to_subdevice_map = utility.bidict(subdevice_to_device_map).inverse
		device_to_simulation_map = {i: [subdevice_to_simulation_map[k] for k in device_to_subdevice_map[i]]     for i in list(device_to_subdevice_map.keys())}
		return device_to_simulation_map

	def get_simulation_to_device_map(self, subdevice_to_device_map, subdevice_to_simulation_map):
		simulation_to_subdevice_map = utility.bidict(subdevice_to_simulation_map).inverse
		simulation_to_device_map = {i: [subdevice_to_device_map[k] for k in simulation_to_subdevice_map[i]]     for i in list(simulation_to_subdevice_map.keys())}
		return simulation_to_device_map



