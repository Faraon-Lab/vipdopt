import os
import sys
import time

import numpy as np
from scipy import ndimage
import scipy.optimize
import skimage.morphology as skim
import skimage.draw
import networkx as nx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.getcwd()))
import device
import layering
import scale
import sigmoid
import square_blur
import square_blur_smooth
from configs import global_params as gp

def compute_binarization( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sum( np.sqrt( ( input_variable - set_point )**2 ) )

def compute_binarization_gradient( input_variable, set_point=0.5 ):
	total_shape = np.product( input_variable.shape )
	return ( 2. / total_shape ) * np.sign( input_variable - set_point )




class SonyBayerFilter(device.Device):

	def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers, design_height_voxels):
		super(SonyBayerFilter, self).__init__(size, permittivity_bounds, init_permittivity)

		self.num_z_layers = num_z_layers
		self.flip_threshold = 0.5
		self.minimum_design_value = 0
		self.maximum_design_value = 1
		self.current_iteration = 0
		self.design_height_voxels = design_height_voxels

		self.layer_height_voxels = self.design_height_voxels

		self.init_filters_and_variables()

		# set to the 0th epoch value
		self.update_filters( 0 )

		self.update_permittivity()


	def set_random_init( self, mean, std_dev ):
		'''Random initialization of permittivity.'''
		self.w[0] = np.random.normal( mean, std_dev, self.w[0].shape )
		self.w[0] = np.maximum( np.minimum( self.w[0], 1.0 ), 0.0 )
		self.update_permittivity()

	def set_fixed_init( self, init_density ):
		'''Uniform initialization of permittivity (all ones).'''
		self.w[0] = init_density * np.ones( self.w[0].shape )
		self.update_permittivity()


	def update_permittivity(self):
		'''Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions.'''
		var0 = self.w[ 0 ]

		if gp.cv.use_smooth_blur:
			var1 = self.layering_z_0.forward( var0 )
			self.w[ 1 ] = var1

			var2 = self.sigmoid_1.forward( var1 )
			self.w[ 2 ] = var2

			var3 = self.max_blur_xy_2.forward( var2 )
			self.w[ 3 ] = var3

			var4 = self.sigmoid_3.forward( var3 )
			self.w[ 4 ] = var4

			var5 = self.scale_4.forward( var4 )
			self.w[ 5 ] = var5

		else:
			var1 = self.layering_z_0.forward( var0 )
			self.w[ 1 ] = var1

			var2 = self.sigmoid_1.forward( var1 )
			self.w[ 2 ] = var2

			var3 = self.scale_4.forward( var2 )
			self.w[ 3 ] = var3


	def backpropagate(self, gradient):
		'''Override the backpropagation function of Device class, and multiplies input (per the chain rule) by gradients of all the filters.'''

		if gp.cv.use_smooth_blur:
			gradient = self.scale_4.chain_rule( gradient, self.w[ 5 ], self.w[ 4 ] )
			gradient = self.sigmoid_3.chain_rule( gradient, self.w[ 4 ], self.w[ 3 ] )
			gradient = self.max_blur_xy_2.chain_rule( gradient, self.w[ 3 ], self.w[ 2 ] )
			gradient = self.sigmoid_1.chain_rule( gradient, self.w[ 2 ], self.w[ 1 ] )
			gradient = self.layering_z_0.chain_rule( gradient, self.w[ 1 ], self.w[ 0 ] )

		else:
			gradient = self.scale_4.chain_rule( gradient, self.w[ 3 ], self.w[ 2 ] )
			gradient = self.sigmoid_1.chain_rule( gradient, self.w[ 2 ], self.w[ 1 ] )
			gradient = self.layering_z_0.chain_rule( gradient, self.w[ 1 ], self.w[ 0 ] )

		return gradient

	def update_filters(self, epoch):
		self.sigmoid_beta = 0.0625 * (2**epoch)

		self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		# self.filters = [ self.layering_z_0, self.sigmoid_1, self.scale_2 ]
  
		self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		self.filters = [ self.layering_z_0, self.sigmoid_1, self.scale_4 ]

		if gp.cv.use_smooth_blur:
			self.filters = [ self.layering_z_0, self.sigmoid_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4 ]


	def init_variables(self):
		super(SonyBayerFilter, self).init_variables()
		self.w[0] = np.multiply(self.init_permittivity, np.ones(self.size, dtype=np.complex))

	def init_filters_and_variables(self):
		self.num_filters = 3
		if gp.cv.use_smooth_blur:
			self.num_filters = 5
		self.num_variables = 1 + self.num_filters

		# Start the sigmoids at weak strengths
		self.sigmoid_beta = 0.0625
		self.sigmoid_eta = 0.5
		self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		x_dimension_idx = 0
		y_dimension_idx = 1
		z_dimension_idx = 2

		z_voxel_layers = self.size[2]

		spacer_value = 0
		self.layering_z_0 = layering.Layering( z_dimension_idx, self.num_z_layers, variable_bounds=[0, 1], spacer_height_voxels=0, spacer_voxels_value=0 )

		scale_min = self.permittivity_bounds[0]
		scale_max = self.permittivity_bounds[1]
		# self.scale_2 = scale.Scale([scale_min, scale_max])
		self.scale_4 = scale.Scale([scale_min, scale_max])

		
		if gp.cv.use_smooth_blur:
			self.max_blur_xy_2 = square_blur_smooth.SquareBlurSmooth(
				[ gp.blur_half_width_voxels, gp.blur_half_width_voxels, 0 ] )

		# Initialize the filter chain
		# self.filters = [ self.layering_z_0, self.sigmoid_1, self.scale_2 ]
		self.filters = [ self.layering_z_0, self.sigmoid_1, self.scale_4 ]

		if gp.cv.use_smooth_blur:
			self.filters = [ self.layering_z_0, self.sigmoid_1, self.max_blur_xy_2, self.sigmoid_3, self.scale_4 ]

		self.init_variables()
	

	# In the step function, we should update the permittivity with update_permittivity
	def step(self, gradient, step_size):
		self.w[0] = self.proposed_design_step(gradient, step_size)
		self.update_permittivity()

	def step_adam(self, iteration, gradient, moments, step_size, betas=(0.9, 0.999), eps=1e-8):
		self.w[0], moments = self.proposed_design_step_adam(iteration, gradient, moments, step_size, betas, eps)
		self.update_permittivity()
		return moments
    
	def convert_to_binary_map(self, variable):
		return np.greater(variable, self.mid_permittivity)



