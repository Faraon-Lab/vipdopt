import numpy as np

class Device:
	'''Creates an object of class Device that can be used for any type of functionality
	that can be optimized via adjoint optimization and then gradient descent.'''
 
	def __init__(self, size, permittivity_bounds, init_permittivity):
		self.size = size                                    # N-D array of device size, in voxels
		self.init_permittivity = init_permittivity
		self.permittivity_bounds = permittivity_bounds      # typically [0, 1] - later to be scaled to whichever min. and max. indices are being used
		self.num_filters	 = 0
		self.num_variables = self.num_filters + 1

		self.mid_permittivity = 0.5 * (self.permittivity_bounds[0] + self.permittivity_bounds[1])       # takes average of the permittivity range

		self.init_variables()

	def init_variables(self):
		'''Creates the w variable for the device. For each variable of the device, creates an array with all values representing permittivity.
		Each subsequent layer represents the permittivity after being passed through the corresponding number of filters.
		The first entry / array is the design variable, and the last entry / array is the updated permittivity.'''
		
		self.w = [np.zeros(self.size, dtype=np.complex) for i in range(0, self.num_variables)]
		self.w[0] = np.multiply(self.init_permittivity, np.ones(self.size, dtype=np.complex))

	def get_design_variable(self):
		return self.w[0]

	def set_design_variable(self, new_design):
		self.w[0] = new_design.copy()
		self.update_permittivity()

	def get_permittivity(self):
		return self.w[-1]

	def update_permittivity(self):
		'''Passes each layer of the permittivity through filters.'''

		for filter_idx in range(0, self.num_filters):
			variable_in = self.w[filter_idx]
			variable_out = (self.filters[filter_idx]).forward(variable_in)          # TODO: This method is defined not here, but in the classes that inherit from filter.py. Fix???
			self.w[filter_idx + 1] = variable_out

	def apply_binary_filter_pipeline(self, variable_in):
		'''Applies a direct binarized step function, replacing the sigmoid filter in the filter chain.'''
		# Hmm.. not sure how to make this explicit, but the filters run here do not need        # TODO: What
		# to do any padding and only need to produce a partial output of the same size

		# Need a copy or else the original variable in will be modified, which is not what
		# is likely expected
		variable_out = np.copy(variable_in)
		for filter_idx in range(0, self.num_filters):
			variable_out = (self.filters[filter_idx]).fabricate(variable_out)   # TODO: fabricate() is also not defined here, but in sigmoid.py

		return variable_out

	def backpropagate(self, gradient):
		'''Backpropagates the gradient so it can be applied to the pre-filtered design variable.'''
		for f_idx in range(0, self.num_filters):
			get_filter = self.filters[self.num_filters - f_idx - 1]
			variable_out = self.w[self.num_filters - f_idx]
			variable_in = self.w[self.num_filters - f_idx - 1]

			gradient = get_filter.chain_rule(gradient, variable_out, variable_in)

		return gradient

	def proposed_design_step(self, gradient, step_size):
		'''Takes a gradient and updates refractive index (design variable) accordingly.'''
		gradient = self.backpropagate(gradient)

		# Eq. S2 of Optica Paper Supplement, https://doi.org/10.1364/OPTICA.384228
		# minus sign here because it was fed the negative of the gradient so (-1)*(-1) = +1
		proposed_design_variable = self.w[0] - np.multiply(step_size, gradient)
		proposed_design_variable = np.maximum(                             # Has to be between minimum_design_value and maximum_design_value
									np.minimum(
										proposed_design_variable,
										self.maximum_design_value),
									self.minimum_design_value)

		return proposed_design_variable

	def fabrication_version(self):
		 # TODO: pipeline_half_width isn't defined anywhere???
		padded_design = np.pad(
			self.w[0],
			(
				(self.pipeline_half_width[0], self.pipeline_half_width[0]),
				(self.pipeline_half_width[1], self.pipeline_half_width[1]),
				(self.pipeline_half_width[2], self.pipeline_half_width[2])
			),
			'constant'
		)

		# Step 4: We need here the current fabrication target before we start the stepping
		design_shape = (self.w[0]).shape

		current_device = self.convert_to_binary_map(self.apply_binary_filter_pipeline(padded_design))

		return current_device[
				self.pipeline_half_width[0] : (self.pipeline_half_width[0] + design_shape[0]),
				self.pipeline_half_width[1] : (self.pipeline_half_width[1] + design_shape[1]),
				self.pipeline_half_width[2] : (self.pipeline_half_width[2] + design_shape[2])]

	def step(self, gradient, step_size):
		# In the step function, we should update the permittivity with update_permittivity
		self.w[0] = self.proposed_design_step(gradient, step_size)
		# Update the variable stack including getting the permittivity at the w[-1] position
		self.update_permittivity()






