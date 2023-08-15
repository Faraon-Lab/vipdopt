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
import json

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
import sigmoid_3_level
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
		self.design_height_voxels = design_height_voxels	# unused

		self.layer_height_voxels = self.design_height_voxels	# unused

		self.init_filters_and_variables()
		# self.set_random_init(self.init_permittivity, 0.27)

		# set to the 0th epoch value
		self.update_filters( 0 )

		self.update_permittivity()

	def set_random_init( self, mean, std_dev, symmetric=False ):
		'''Random initialization of permittivity.'''
		self.w[0] = np.random.normal( mean, std_dev, self.w[0].shape )
		self.w[0] = np.maximum( np.minimum( self.w[0], 1.0 ), 0.0 )
		if symmetric:
			for layer in range(self.w[0].shape[2]):
				self.w[0][:,:,layer] = np.tril(self.w[0][:,:,layer]) + np.triu(self.w[0][:,:,layer].T, 1)
				self.w[0][:,:,layer] = np.flip(self.w[0][:,:,layer], axis=1)
				# check symmetry
				a = np.flip((self.w[0][:,:,layer]), axis=1)
				assert np.allclose(a, a.T, rtol=1e-5, atol=1e-8), f"Layer {layer} is not symmetric."
		self.update_permittivity()

	def set_fixed_init( self, init_density ):
		'''Uniform initialization of permittivity (all ones).'''
		self.w[0] = init_density * np.ones( self.w[0].shape )
		self.update_permittivity()

	def update_actual_permittivity_values( self, min_device_actual_perm, max_device_actual_perm ):
		self.min_actual_permittivity = min_device_actual_perm
		self.max_actual_permittivity = max_device_actual_perm

	# def export( self, curr_epoch, min_device_perm, max_device_perm):
	# 	'''[NOT WORKING] Export filter variables so it can be regenerated'''
	# 	self.update_actual_permittivity_values(min_device_perm, max_device_perm)
	# 	self.current_epoch = curr_epoch

	# 	# # JSON doesn't work because the class contains ndarrays and the filters are class objects
	# 	# with open("bayerfilter_state.json", "w") as outfile:
	# 	# 	outfile.write(json.dumps(self.__dict__, indent=4))

	# 	# use shelve

	def update_permittivity(self):
		'''Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions.'''
		var0 = self.w[ 0 ]

		if gp.cv.should_fix_layers:
			for idx in range( 0, len( gp.cv.fixed_layers ) ):
				start_layer = gp.cv.fixed_layers[ idx ] * gp.pv.vertical_layer_height_voxels
				end_layer = start_layer + gp.pv.vertical_layer_height_voxels

				var0[ :, :, start_layer : end_layer ] = gp.cv.fixed_layers_density


		if gp.cv.border_optimization:
			
			# The border indices of the array will be <border_size_voxels> and <endpoint>.
			endpoint = var0.shape[ 0 ] - gp.pv.border_size_voxels
			# Take the central region of the array, the part that actually corresponds to the device.
			var0_ = var0[ gp.pv.border_size_voxels : endpoint, gp.pv.border_size_voxels : endpoint, : ]
			# Pad it back out to the full array size with copies of portions of the central region - this is achieved instantly by 'wrap' mode.
			var0 = np.pad( var0_, ( ( gp.pv.border_size_voxels, gp.pv.border_size_voxels ), ( gp.pv.border_size_voxels, gp.pv.border_size_voxels ), ( 0, 0 ) ), mode='wrap' )

			# var0.shape = (30, 30, 40)
			# border_size_voxels = 3
			# endpoint =  27
			# var0_ = var0[3:27, 3:27,:]
			# var0 = np.pad( var0_, ((3,3), (3,3), (0,0)), mode='wrap')

		if not gp.cv.layer_gradient:
			var1 = self.layering_z_0.forward( var0 )
			self.w[ 1 ] = var1
		else:
			var1 = np.zeros( var0.shape, dtype=var0.dtype )

			for z_idx in range( 0, len( gp.pv.voxels_per_layer ) ):
				start_voxel = np.sum( gp.pv.voxels_per_layer[ 0 : z_idx ] )
				end_voxel = start_voxel + gp.pv.voxels_per_layer[ z_idx ]
				get_mean = np.mean( var0[ :, :, start_voxel : end_voxel ], axis=2 )

				for internal_z_idx in range( start_voxel, end_voxel ):
					var1[ :, :, internal_z_idx ] = get_mean

			self.w[ 1 ] = var1

		if gp.cv.use_smooth_blur:

			var2 = self.sigmoid_1.forward( var1 )
			self.w[ 2 ] = var2

			var3 = self.max_blur_xy_2.forward( var2 )
			self.w[ 3 ] = var3

			var4 = self.sigmoid_3.forward( var3 )
			self.w[ 4 ] = var4

			var5 = self.scale_4.forward( var4 )
			self.w[ 5 ] = var5

		else:

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


		if not gp.cv.layer_gradient:
			gradient = self.layering_z_0.chain_rule( gradient, self.w[ 1 ], self.w[ 0 ] )
		else:
			for z_idx in range( 0, len( gp.pv.voxels_per_layer ) ):
				start_voxel = np.sum( gp.pv.voxels_per_layer[ 0 : z_idx ] )
				end_voxel = start_voxel + gp.pv.voxels_per_layer[ z_idx ]
				get_mean = np.mean( gradient[ :, :, start_voxel : end_voxel ], axis=2 )

				for internal_z_idx in range( start_voxel, end_voxel ):
					gradient[ :, :, internal_z_idx ] = get_mean


		if gp.cv.border_optimization:

			# The border indices of the array will be <border_size_voxels> and <endpoint>.
			endpoint = gradient.shape[ 0 ] - gp.pv.border_size_voxels
			# For, say,
   			# gradient.shape = (30, 30, 40)
			# border_size_voxels = 3
			# endpoint = 30 - 3 = 27
			# so that means the central region is actually (3:27, 3:27, 40)

			# Fold in the gradient on the edges (x-direction) and all across the y-direction.
			gradient[ gp.pv.border_size_voxels : ( gp.pv.border_size_voxels + gp.pv.border_size_voxels ), gp.pv.border_size_voxels : endpoint, : ] \
						+= gradient[ endpoint :, gp.pv.border_size_voxels : endpoint, : ]
			gradient[ ( endpoint - gp.pv.border_size_voxels ) : endpoint, gp.pv.border_size_voxels : endpoint, : ] \
						+= gradient[ 0 : gp.pv.border_size_voxels, gp.pv.border_size_voxels : endpoint, : ]
			# gradient[3:6, 3:27, :] += gradient[27:, 3:27,:]
			# gradient[24:27, 3:27, :] += gradient[0:3, 3:27,:]

			# Fold in the gradient on the edges (y-direction) and all across the x-direction.
			gradient[ gp.pv.border_size_voxels : endpoint, gp.pv.border_size_voxels : ( gp.pv.border_size_voxels + gp.pv.border_size_voxels ), : ] \
						+= gradient[ gp.pv.border_size_voxels : endpoint, endpoint :, : ]
			gradient[ gp.pv.border_size_voxels : endpoint, ( endpoint - gp.pv.border_size_voxels ) : endpoint, : ] \
						+= gradient[ gp.pv.border_size_voxels : endpoint, 0 : gp.pv.border_size_voxels, : ]
			# gradient[3:27, 3:6, :] += gradient[3:27, 27:, :]
			# gradient[3:27, 24:27, :] += gradient[3:27, 0:3, :]

			# Fold in the gradient for the corners as well! Each corner takes the gradient from its opposite corner (on a 2D square).
			gradient[
				gp.pv.border_size_voxels : ( gp.pv.border_size_voxels + gp.pv.border_size_voxels ),
				gp.pv.border_size_voxels : ( gp.pv.border_size_voxels + gp.pv.border_size_voxels ),
				: ] += gradient[ endpoint :, endpoint :, : ]
			# gradient[3:6, 3:6, :] += gradient[27:, 27:, :]

			gradient[
				( endpoint - gp.pv.border_size_voxels ) : endpoint,
				( endpoint - gp.pv.border_size_voxels ) : endpoint,
				: ] += gradient[ 0 : gp.pv.border_size_voxels, 0 : gp.pv.border_size_voxels, : ]
			# gradient[24:27, 24:27, :] += gradient[0:3, 0:3, :]

			gradient[
				( endpoint - gp.pv.border_size_voxels ) : endpoint,
				gp.pv.border_size_voxels : ( gp.pv.border_size_voxels + gp.pv.border_size_voxels ),
				: ] += gradient[ 0 : gp.pv.border_size_voxels, endpoint :, : ]
			# gradient[24:27, 3:6, :] += gradient[0:3, 27:, :]

			gradient[
				gp.pv.border_size_voxels : ( gp.pv.border_size_voxels + gp.pv.border_size_voxels ),
				( endpoint - gp.pv.border_size_voxels ) : endpoint,
				: ] += gradient[ endpoint :, 0 : gp.pv.border_size_voxels, : ]
			# gradient[3:6, 24:27, :] += gradient[27:, 0:3, :]

			# Finally set the gradient on the border to just be zero so that it's not stepped forward.
			# The border areas themselves will be regenerated in the next call to update_permittivity().
			gradient[ 0 : gp.pv.border_size_voxels, :, : ] = 0
			gradient[ endpoint :, :, : ] = 0
			gradient[ :, 0 : gp.pv.border_size_voxels, : ] = 0
			gradient[ :, endpoint :, : ] = 0
			# gradient[0:3, :, :] = 0
			# gradient[27:, :, :] = 0
			# gradient[:, 0:3, :] = 0
			# gradient[:, 27:, :] = 0

		if gp.cv.should_fix_layers:
			for idx in range( 0, len( gp.cv.fixed_layers ) ):
				start_layer = gp.cv.fixed_layers[ idx ] * gp.pv.vertical_layer_height_voxels
				end_layer = start_layer + gp.pv.vertical_layer_height_voxels

				gradient[ :, :, start_layer : end_layer ] = 0



		return gradient

	def update_filters(self, epoch):
		self.current_epoch = epoch
		self.sigmoid_beta = 0.0625 * (2**epoch)

		# self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
		# self.sigmoid_3 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

		if gp.cv.binarize_3_levels:
			self.sigmoid_1 = sigmoid_3_level.Sigmoid3Level(self.sigmoid_beta )
			self.sigmoid_3 = sigmoid_3_level.Sigmoid3Level(self.sigmoid_beta )
		else:
			self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
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

		if gp.cv.binarize_3_levels:
			self.sigmoid_1 = sigmoid_3_level.Sigmoid3Level(self.sigmoid_beta )
			self.sigmoid_3 = sigmoid_3_level.Sigmoid3Level(self.sigmoid_beta )
		else:
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

def visualize(cur_design_variable, visualize_epoch=10, min_device_perm=2.25, max_device_perm=5.76,
			  num_vertical_layers=10, init_permittivity_0_1_scale=0.5):
	'''Plots out the separate layers of the cur_design_variable. The visualize_epoch argument controls
	the degree of binarization.
	Note: The arguments of this function are precisely what are needed to regenerate the savestate of
	a SonyBayerFilter instance.'''
	# # This is the information that the filter needs to save
	# # cur_design_variable address to load
	# num_vertical_layers = 10				# bayer_filter.num_z_layers
	# init_permittivity_0_1_scale = 0.5		# bayer_filter.init_permittivity
	# min_device_permittivity = 2.25
	# max_device_permittivity = 5.76
	# current_epoch = 0
	# current_iter = 0
	
	import logging
	from utils import utility
	logging.info(f'Loading filter from cur_design_variable.npy.')
 
	# Create an instance of the device to store permittivity values
	bayer_filter_size_voxels = np.array(cur_design_variable.shape)
	logging.info(f'Creating Bayer Filter with voxel size {bayer_filter_size_voxels}.')

	bayer_filter = SonyBayerFilter.SonyBayerFilter(
		bayer_filter_size_voxels,
		[ 0.0, 1.0 ],
		init_permittivity_0_1_scale,
		num_vertical_layers,
		vertical_layer_height_voxels=cur_design_variable.shape[2]/num_vertical_layers)

	# Update design data
	bayer_filter.w[0] = cur_design_variable
	# Regenerate permittivity based on the most recent design, assuming epoch 0		
 	# NOTE: This gets corrected later down below
	bayer_filter.update_permittivity()

	# Use binarized density variable instead of intermediate
	bayer_filter.update_filters( visualize_epoch )
	bayer_filter.update_permittivity()

	# Get density after passing design through sigmoid filters
	cur_density = bayer_filter.get_permittivity()

	# Assume a simple dispersion model without any actual dispersion
	# dispersion_model = DevicePermittivityModel()
	dispersive_max_permittivity = max_device_perm # dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )

	# Scale from [0, 1] back to [min_device_permittivity, max_device_permittivity]
	#! This is the only cur_whatever_variable that will ever be anything other than [0, 1].
	cur_permittivity = min_device_perm + ( dispersive_max_permittivity - min_device_perm ) * cur_density
	cur_index = utility.index_from_permittivity( cur_permittivity )

	# Then plot cur_index, each layer
	from evaluation import plotter
	plotter.visualize_device(cur_index, OPTIMIZATION_PLOTS_FOLDER, num_visualize_layers=num_vertical_layers)

# If this code is run by itself, visualize the device:
if __name__ == "__main__":

	DATA_FOLDER = 'ares' + os.path.basename(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))))
	OPTIMIZATION_INFO_FOLDER = os.path.join(DATA_FOLDER, 'opt_info')
	OPTIMIZATION_PLOTS_FOLDER = os.path.join(OPTIMIZATION_INFO_FOLDER, 'plots')

	# Load in the most recent Design density variable (before filters of device), values between 0 and 1
	cur_design_variable = np.load(
			OPTIMIZATION_INFO_FOLDER + "/cur_design_variable.npy" )
	visualize(cur_design_variable, OPTIMIZATION_PLOTS_FOLDER)

#print(3)



