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
from utils import Filter

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
		self.symmetric = False
		self.iteration = iteration										# current iteration of optimization
		
		self.num_filters = 2											# todo: could this be also an input argument? a dictionary or array of dicts, perhaps e.g. [{'sigmoid':{'strength': 3}, {'bridges':{}}]
		self.num_variables = self.num_filters + 1

		self.minimum_design_value = 0
		self.maximum_design_value = 1	

		self.init_variables()
		self.update_filters(epoch=0)
		self.update_density()
		# self.save_device()	#! Don't save here - it'll overwrite any previous saves
		

	def save_device(self):
		'''Shelves vars() as a dictionary for pickling.'''
		utility.backup_all_vars(vars(self), os.path.join(self.folder, self.name))

	def load_device(self, filename=None):
		'''Overwrites class variables with a previously shelved dictionary vars()
			Note: A toy device must be created first before it can then be overwritten.'''
		if filename is None:
			vars(self).update( utility.load_backup_vars( os.path.join(self.folder, self.name) ) )
		else:
			vars(self).update( utility.load_backup_vars( os.path.join(self.folder, filename) ) )
  

	def init_variables(self):
		'''Creates the w variable for the device. For each variable of the device, creates an array with all values representing density.
		Each subsequent layer represents the density after being passed through the corresponding number of filters.
		The first entry / array is the design variable, and the last entry / array is the updated density.'''
		
		self.w = [np.zeros(self.size, dtype=np.complex128) for i in range(0, self.num_variables)]

		if self.init_seed == 'uniform':
			self.w[0] = np.multiply(self.init_density, np.ones(self.size, dtype=np.complex128))
		elif self.init_seed == 'random':
			'''Random initialization of permittivity.'''
			self.w[0] = np.random.normal( self.init_density, 0.27, self.w[0].shape )
			self.w[0] = np.maximum( np.minimum( self.w[0], 1.0 ), 0.0 )
			if self.symmetric:
				for layer in range(self.w[0].shape[2]):
					self.w[0][:,:,layer] = np.tril(self.w[0][:,:,layer]) + np.triu(self.w[0][:,:,layer].T, 1)
					self.w[0][:,:,layer] = np.flip(self.w[0][:,:,layer], axis=1)
					# check symmetry
					a = np.flip((self.w[0][:,:,layer]), axis=1)
					assert np.allclose(a, a.T, rtol=1e-5, atol=1e-8), f"Layer {layer} is not symmetric."

	def get_design_variable(self):
		return self.w[0]

	def set_design_variable(self, new_design):
		self.w[0] = new_design.copy()
		self.update_density()

	def get_density(self):
		return self.w[1]		# This needs to be adjusted if more layers are added before the Scale.

	def density_to_permittivity(self, density, eps_min, eps_max):
		return density_to_permittivity(density, eps_min, eps_max)

	def get_permittivity(self):
		# #---------Commented block only necessary if Scale filter not present-----------------
		# density = self.w[-1]
		# eps_min = self.permittivity_constraints[0]
		# eps_max = self.permittivity_constraints[1]
		
		# return self.density_to_permittivity(density, eps_min, eps_max)
		return self.w[-1]
	
	def update_density(self):
		'''Passes each layer of the density through filters.'''

		for filter_idx in range(0, self.num_filters):
			variable_in = self.w[filter_idx]
			variable_out = (self.filters[filter_idx]).forward(variable_in)          # TODO: This method is defined not here, but in the classes that inherit from filter.py. Fix???
			self.w[filter_idx + 1] = variable_out

	def update_filters(self, epoch):
		self.epoch = epoch
		self.sigmoid_beta = 0.0625 * (2**epoch)
		self.sigmoid_eta = 0.5
		print(f'Epoch {self.epoch}: Sigmoid strength is beta={self.sigmoid_beta:.2f}')

		# todo: FILTER UPDATING CODE
		self.identity = Filter.Identity()
		self.sigmoid_1 = Filter.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.scale_4 = Filter.Scale(self.permittivity_constraints)

		# self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		# self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		# if gp.cv.binarize_3_levels:
		# 	self.sigmoid_1 = sigmoid_3_level.Sigmoid3Level(self.sigmoid_beta )
		# 	self.sigmoid_3 = sigmoid_3_level.Sigmoid3Level(self.sigmoid_beta )
		# else:
		# 	self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		# 	self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)


		# self.filters = [ self.layering_z_0, self.sigmoid_1, self.scale_4 ]

		# if gp.cv.use_smooth_blur:
		# 	self.filters = [ self.layering_z_0, self.sigmoid_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4 ]

		self.filters = [self.sigmoid_1, self.scale_4]
		# self.filters = [self.identity]
		self.num_filters = len(self.filters)											# todo: could this be also an input argument? a dictionary or array of dicts, perhaps e.g. [{'sigmoid':{'strength': 3}, {'bridges':{}}]
		self.num_variables = self.num_filters + 1


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
	

	def backpropagate(self, gradient):
		'''Backpropagates the gradient so it can be applied to the pre-filtered design variable.'''
		for f_idx in range(0, self.num_filters):
			get_filter = self.filters[self.num_filters - f_idx - 1]
			variable_out = self.w[self.num_filters - f_idx]
			variable_in = self.w[self.num_filters - f_idx - 1]

			gradient = get_filter.chain_rule(gradient, variable_out, variable_in)

		return gradient


	def control_average_permittivity():
		return 3

	def proposed_design_step(self, optimizer, args):
		'''Takes a gradient and updates refractive index (design variable) accordingly.'''
		gradient = self.backpropagate(gradient)

		# todo: figure out optimizer
		proposed_design_variable = optimizer.step(self.w[0], gradient)
		proposed_design_variable = np.maximum(                             # Has to be between minimum_design_value and maximum_design_value
												np.minimum(proposed_design_variable, self.maximum_design_value),
												self.minimum_design_value)

		return proposed_design_variable
		

	def proposed_design_step_simple(self, gradient, step_size, change_limits):
		'''Takes a gradient and updates refractive index (design variable) accordingly.'''
		gradient = self.backpropagate(gradient)

		# We need to scale the step size such that the maximum change in the design density is between some certain limits
		# todo:

		# Eq. S2 of Optica Paper Supplement, https://doi.org/10.1364/OPTICA.384228
		# minus sign here because it was fed the negative of the gradient so (-1)*(-1) = +1
		proposed_design_variable = self.w[0] - np.multiply(step_size, gradient)
		# todo:
		# proposed_design_variable = Optimizer.step(self.w[0], gradient, step_size)
		proposed_design_variable = np.maximum(                             # Has to be between minimum_design_value and maximum_design_value
												np.minimum(proposed_design_variable, self.maximum_design_value),
												self.minimum_design_value)

		min_proposed_change = np.min(np.abs(proposed_design_variable - self.w[0]))
		max_proposed_change = np.max(np.abs(proposed_design_variable - self.w[0]))


		# todo: add scaling of the step size here



		return proposed_design_variable

	def proposed_design_step_adam(self, iteration, gradient, moments, step_size, betas=(0.9, 0.999), eps=1e-8):
		'''Takes a gradient and step size, and updates refractive index (design variable) accordingly, applying the ADAM optimizer.
		ADAM optimizer code is taken from https://machinelearningmastery.com/adam-optimization-from-scratch/
		and also explained here: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/'''

		gradient = self.backpropagate(gradient)

		# Eq. S2 of Optica Paper Supplement, https://doi.org/10.1364/OPTICA.384228
		# minus sign here because it was fed the negative of the gradient so (-1)*(-1) = +1
		# proposed_design_variable = self.w[0] - np.multiply(step_size, gradient)
  
		beta_1, beta_2 = betas[0], betas[1]

		# update 1st moment
		moments[0] = beta_1*moments[0] + (1-beta_1)*gradient
		# update 2nd moment
		moments[1] = beta_2*moments[1] + (1-beta_2)*gradient**2

		# apply bias corrections to step size
		alpha_t = step_size * np.sqrt(1-beta_2**(iteration+1))/(1-beta_1**(iteration+1))

		# update solution
		proposed_design_variable = self.w[0] - alpha_t * moments[0]/(np.sqrt(moments[1])+eps)

		proposed_design_variable = np.maximum(                             # Has to be between minimum_design_value and maximum_design_value
												np.minimum(proposed_design_variable, self.maximum_design_value),
												self.minimum_design_value)

		return proposed_design_variable, moments

	# def step(self, gradient, step_size, change_limits):
	# 	# In the step function, we should update the permittivity with update_permittivity
	# 	self.w[0] = self.proposed_design_step_simple(gradient, step_size, change_limits)
	# 	# Update the variable stack including getting the permittivity at the w[-1] position
	# 	self.update_density()
	# 	self.save_device()
  
	def step(self, new_design):
		self.w[0] = new_design
		self.update_density()
		self.save_device()

	def step_adam(self, iteration, gradient, moments, step_size, betas=(0.9, 0.999), eps=1e-8):
		self.w[0] = self.proposed_design_step_adam(iteration, gradient, moments, step_size, betas, eps)
		self.update_density()
	
	

# todo: contain reinterpolation functions