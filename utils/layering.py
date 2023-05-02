# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import filter as filter

import numpy as np
# import matplotlib.pyplot as plt

#
# dim: dimension to layer on
# num_layers: number of layers
#
class Layering(filter.Filter):
	'''Takes a device and splits it into layers in the dimension requested. Gives the functionality to get the indices of each layer,
	and add spacers in between each layer.
	
	Restricts optimization by averaging the calculated sensitivity in the vertical direction within EACH layer, such that voxels 
	within each layer are governed by a shared 2D profile.
	i.e. n layers, 1 layer has m voxels, all voxels have a shared 2D permittivity profile WITHIN the nth layer, but each layer differs from the rest in its 2D profile
	
	See OPTICA paper supplement Section IIC, https://doi.org/10.1364/OPTICA.384228,  for details.'''

	def __init__(self, dim, num_layers, variable_bounds=[0, 1], spacer_height_voxels=0, spacer_voxels_value=0):
		super(Layering, self).__init__(variable_bounds)

		#
		# dim: dimension to layer on
		# num_layers: number of layers
		#

		self.dim = dim
		self.num_layers = num_layers
		self.spacer_height_voxels = spacer_height_voxels
		self.spacer_voxels_value = spacer_voxels_value

	def get_layer_idxs(self, variable_shape):
		'''Gets the indices of each layer in the dimension that was declared to the Layering object'''
		num_voxels_total = variable_shape[self.dim]

		layer_idxs = []

		num_voxels_per_layer = int(num_voxels_total / self.num_layers)
		num_voxels_last_layer = num_voxels_total - (self.num_layers - 1) * num_voxels_per_layer

		for layer_idx in range(0, self.num_layers - 1):
			layer_start = layer_idx * num_voxels_per_layer
			layer_idxs.append( layer_start )

		last_layer_start = (self.num_layers - 1) * num_voxels_per_layer
		layer_idxs.append( last_layer_start )

		return layer_idxs

	def do_layering(self, data, dim, output):
		data_shape = data.shape

		# Takes the average of the data along the dimension declared to the Layering object
		average = np.take(data, 0, axis=dim)
		for idx in range(1, data_shape[dim]):
			average += np.take(data, idx, axis=dim)
		average /= data_shape[dim]

		for idx in range(0, data_shape[dim]):
			# Create a tuple of (slice(None), slice(None), ..., idx) for each (:, :, idx) of the data_shape
			idx_bunch = [slice(None)] * data.ndim
			idx_bunch[dim] = np.array(idx)
			# Populates all entries of this slice with the average value from above
			output[tuple(idx_bunch)] = average

		return output

	def layer_averaging(self, variable, spacer_value):
		'''Averages out all the voxels within one layer in the vertical direction, such that they all have the same 2D profile within that layer.'''
		variable_shape = variable.shape
		num_voxels_total = variable_shape[self.dim]

		num_voxels_per_layer = int(num_voxels_total / self.num_layers)
		num_voxels_last_layer = num_voxels_total - (self.num_layers - 1) * num_voxels_per_layer

		variable_out = np.zeros(variable_shape)

		for layer_idx in range(0, self.num_layers - 1):
			layer_start = layer_idx * num_voxels_per_layer

			idx_bunch = [slice(None)] * variable.ndim
			idx_bunch[self.dim] = np.arange(layer_start, layer_start + num_voxels_per_layer - self.spacer_height_voxels)

			idx_bunch_spacer = [slice(None)] * variable.ndim
			idx_bunch_spacer[self.dim] = np.arange(layer_start + num_voxels_per_layer - self.spacer_height_voxels, layer_start + num_voxels_per_layer)

			variable_out[tuple(idx_bunch)] = self.do_layering(variable[tuple(idx_bunch)], self.dim, variable_out[tuple(idx_bunch)])
			variable_out[tuple(idx_bunch_spacer)] = spacer_value

		last_layer_start = (self.num_layers - 1) * num_voxels_per_layer
		idx_bunch = [slice(None)] * variable.ndim
		idx_bunch[self.dim] = np.arange(last_layer_start, num_voxels_total - self.spacer_height_voxels)

		self.last_layer_start = last_layer_start
		self.last_layer_end = num_voxels_total - self.spacer_height_voxels

		idx_bunch_spacer = [slice(None)] * variable.ndim
		idx_bunch_spacer[self.dim] = np.arange(num_voxels_total - self.spacer_height_voxels, num_voxels_total)

		variable_out[tuple(idx_bunch)] = self.do_layering(variable[tuple(idx_bunch)], self.dim, variable_out[tuple(idx_bunch)])
		variable_out[tuple(idx_bunch_spacer)] = spacer_value

		return variable_out

	def forward(self, variable_in):
		'''Performs normal layer averaging.'''
		return self.layer_averaging(variable_in, self.spacer_voxels_value)

	def chain_rule(self, derivative_out, variable_out_, variable_in_):
		return self.layer_averaging(derivative_out, 0)

	def fabricate(self, variable_in):
		'''Performs normal layer averaging.'''
		return self.forward(variable_in)



