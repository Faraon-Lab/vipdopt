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
import traceback
import shelve
import shutil
from datetime import datetime

from functools import reduce
import operator
import re

# todo: need to split the below into things that require config variables and things that do not.
#! because config calls this module, this module cannot call config.

#*Argparse
argparse_universal_args = [
	["-f", "--filename", {'help':"Optimization Config file (YAML) path", 'required':True}],	#, default='test_config.yaml')
	["-m", "--mode", {'help':"Optimization or Evaluation mode", "choices":['opt','eval'], "default":'opt'}],
	["-o", "--override", {'help':"Override the default folder to save files out"}]
]

#*Logging
logging_filename = f'_t{int(round(datetime.now().timestamp()))}'
slurm_job_id = os.getenv('SLURM_JOB_ID')
if slurm_job_id is not None:
	logging_filename += f'_slurm-{slurm_job_id}'

loggingArgs = {'level':logging.INFO, 
			   'filename': 'logfile'+logging_filename+'.log',
			   'filemode': 'a',
			   'format': '%(asctime)s %(levelname)s|| %(message)s',
			   'datefmt':'%d-%b-%y %H:%M:%S'}
logging.basicConfig(**loggingArgs)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Custom exception handler by re-assigning sys.excepthook() to logging module
def my_handler(type, value, tb):
	logging.getLogger().error("Uncaught exception {0}: {1}".format(str(type), str(value),
									exc_info=sys.exc_info()))
	logging.getLogger().error(traceback.format_tb(tb)[-1])
sys.excepthook = my_handler

logging.info('---Log: SONY Bayer Filter Adjoint Optimization---')
logging.info('Initializing Logger...')
logging.info(f'Current working directory: {os.getcwd()}')


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

def backup_all_vars(dict_vars, savefile_name=None):
	'''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
	
	if savefile_name is None:
		savefile_name = shelf_fn

	global my_shelf
	my_shelf = shelve.open(savefile_name,'n')
	for key in sorted(dict_vars):
		if key not in ['my_shelf']: #, 'forward_e_fields']:
			try:
				my_shelf[key] = dict_vars[key]
				logging.info(f'{key} shelved.')
			except Exception as ex:
				# # __builtins__, my_shelf, and imported modules can not be shelved.
				# logging.error('ERROR shelving: {0}'.format(key))
				# template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				# message = template.format(type(ex).__name__, ex.args)
				# logging.error(message)
				pass
	my_shelf.close()
	
	
	logging.info("Backup complete.")
  
def load_backup_vars(loadfile_name):
	'''Restores session variables from a save_state file. Debug purposes.'''
	#! This will overwrite all current session variables!
	# except for the following, which are environment variables:

	dict_vars = {}

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
				dict_vars[key]=my_shelf[key]
				logging.info(f'{key} retrieved.')
			except Exception as ex:
				logging.exception('ERROR retrieving shelf: {0}'.format(key))
				logging.error("An exception of type {0} occurred. Arguments:\n{1!r}".format(type(ex).__name__, ex.args))
				# pass
	my_shelf.close()
	
	# # Restore lumapi SimObjects as dictionaries
	# for key,val in fdtd_objects.items():
	# 	globals()[key]=val
	
	logging.info("Successfully loaded backup.")
	return dict_vars



#* HPC & Multithreading

#* Other Functions

def isolate_filename(address):
	return os.path.basename(address)

def convert_root_folder(address, root_folder_new):
	new_address = os.path.join(root_folder_new, isolate_filename(address))
	return new_address

def generate_value_array(start,end,step):
	'''Returns a conveniently formatted output list according to np.arange() that can be copy-pasted to a JSON.'''
	import json
	import pyperclip
	output_array = json.dumps(np.arange(start,end+step,step).tolist())
	pyperclip.copy(output_array)
	return output_array

# Nested dictionary handling - https://stackoverflow.com/a/14692747
def get_by_path(root, items):
	"""Access a nested object in root by item sequence."""
	return reduce(operator.getitem, items, root)

def set_by_path(root, items, value):
	"""Set a value in a nested object in root by item sequence."""
	get_by_path(root, items[:-1])[items[-1]] = value

def del_by_path(root, items):
	"""Delete a key-value in a nested object in root by item sequence."""
	del get_by_path(root, items[:-1])[items[-1]]
#######################################

class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

def rescale_vector(x, min0, max0, min1, max1):
	return (max1-min1)/(max0-min0) * (x-min0) + min1

def index_from_permittivity(permittivity_):
	'''Checks all permittivity values are real and then takes square root to give index.'''
	assert np.all(np.imag(permittivity_) == 0), 'Not expecting complex index values right now!'

	return np.sqrt(permittivity_)

def binarize(variable_in):
	'''Assumes density - if not, convert explicitly.'''
	return 1.0 * np.greater_equal(variable_in, 0.5)

def compute_binarization( input_variable, set_point=0.5 ):
	total_shape = np.prod( input_variable.shape )
	return ( 2. / total_shape ) * np.sum( np.sqrt( ( input_variable - set_point )**2 ) )


def softplus( x_in, softplus_kappa ):
	'''Softplus is a smooth approximation to the ReLu function.
	It equals the sigmoid function when kappa = 1, and approaches the ReLu function as kappa -> infinity.
	See: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html and https://pat.chormai.org/blog/2020-relu-softplus'''
	# return np.log( 1 + np.exp( x_in ) )
	return np.log( 1 + np.exp( softplus_kappa * x_in ) ) / softplus_kappa

def softplus_prime( x_in ):
	'''Derivative of the softplus function w.r.t. x, as defined in softplus()'''
	# return ( 1. / ( 1 + np.exp( -x_in ) ) )
	return ( 1. / ( 1 + np.exp( -softplus_kappa * x_in ) ) )

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