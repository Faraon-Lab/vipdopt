# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
from utils import utility

def permittivity_to_density(eps, eps_min, eps_max):
	return (eps - eps_min) / (eps_max - eps_min)

def density_to_permittivity(density, eps_min, eps_max):
	return density*(eps_max - eps_min) + eps_min


class Device:
	'''Creates an object of class Device that can be used for any type of functionality
	that can be optimized via adjoint optimization and then gradient descent.'''

	def __init__(self, size, feature_dimensions, permittivity_constraints, coords,
				 name='device', folder='', init_density=0.5, init_seed='uniform', iteration=0):
		# feature_dimensions,
		# permittivity_constraints,
		# region_coordinates,
		# other_arguments)
	
		self.size = size                                                # N-D array of device size, in voxels
		self.feature_dimensions = feature_dimensions                    # todo: to be determined, but stuff like num_z_layers
		self.permittivity_constraints = permittivity_constraints        # typically [0, 1] - later to be scaled to whichever min. and max. indices are being used
		self.coords = coords                                           # coords in space
		
		# optional args
		self.name = name
		self.folder = folder
		self.init_density = init_density                                # typically 0.5
		self.init_seed = init_seed                                      # usually 'uniform' - can be 'random'
		self.iteration = iteration										# current iteration of optimization
		
		self.num_filters = 0											# todo: could this be also an input argument? a dictionary or array of dicts, perhaps e.g. [{'sigmoid':{'strength': 3}, {'bridges':{}}]
		self.num_variables = self.num_filters + 1
		
		self.init_variables()
		self.save_device()
		

	def save_device(self):
		'''Shelves vars() as a dictionary for pickling.'''
		utility.backup_all_vars(vars(self), os.path.join(self.folder, self.name))

	def load_device(self):
		'''Overwrites class variables with a previously shelved dictionary vars()
			Note: A toy device must be created first before it can then be overwritten.'''
		vars(self).update( utility.load_backup_vars(self.name) )
  

	def init_variables(self):
		'''Creates the w variable for the device. For each variable of the device, creates an array with all values representing density.
		Each subsequent layer represents the density after being passed through the corresponding number of filters.
		The first entry / array is the design variable, and the last entry / array is the updated density.'''
		
		self.w = [np.zeros(self.size, dtype=np.complex128) for i in range(0, self.num_variables)]
		self.w[0] = np.multiply(self.init_density, np.ones(self.size, dtype=np.complex128))

	def get_design_variable(self):
		return self.w[0]

	def set_design_variable(self, new_design):
		self.w[0] = new_design.copy()
		self.update_density()

	def get_density(self):
		return self.w[-1]

	def get_permittivity(self):
		density = self.w[-1]
		eps_min = self.permittivity_constraints[0]
		eps_max = self.permittivity_constraints[1]
		
		return density_to_permittivity(density, eps_min, eps_max)
	
	def update_density(self):
		'''Passes each layer of the density through filters.'''

		for filter_idx in range(0, self.num_filters):
			variable_in = self.w[filter_idx]
			variable_out = (self.filters[filter_idx]).forward(variable_in)          # TODO: This method is defined not here, but in the classes that inherit from filter.py. Fix???
			self.w[filter_idx + 1] = variable_out

	def pass_through_pipeline(self, variable_in, binary_filters=False):
		'''Passes each layer of the density through filters, without actually updating the stored copy.'''
		# Hmm.. not sure how to make this explicit, but the filters run here do not need        # TODO: What
		# to do any padding and only need to produce a partial output of the same size

		# Need a copy or else the original variable in will be modified, which is not what is likely expected

		variable_out = np.copy(variable_in)
		for filter_idx in range(0, self.num_filters):
			if binary_filters:
				variable_out = (self.filters[filter_idx]).fabricate(variable_in)		# TODO: fabricate() is also not defined here, but in sigmoid.py
			else:
				variable_out = (self.filters[filter_idx]).forward(variable_in)          # TODO: This method is defined not here, but in the classes that inherit from filter.py. Fix???
		
		return variable_out
	
	def control_average_permittivity():
		return 3
	
	

# todo: contain reinterpolation functions