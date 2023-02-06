import filter as filter

import numpy as np

class Scale(filter.Filter):
	'''Assuming that there is an input between 0 and 1, scales it to the range, min, and max, that are declared during initialization.
	See OPTICA paper supplement Section IIA, https://doi.org/10.1364/OPTICA.384228,  for details. 
	This class is used directly after the sigmoid filter is applied.'''
	
	def __init__(self, variable_bounds):
		super(Scale, self).__init__(variable_bounds)
  
		#
		# variable_bounds specifies the minimum and maximum value of the scaled variable.
		# We assume we are scaling an input that is between 0 and 1
		#

		self.min_value = variable_bounds[0]
		self.max_value = variable_bounds[1]
		self.range = self.max_value - self.min_value


	def forward(self, variable_in):
		'''Performs scaling according to min and max values declared. Assumed that input is between 0 and 1.'''
		return np.add(self.min_value, np.multiply(self.range, variable_in))


	def chain_rule(self, derivative_out, variable_out, variable_in):
		return np.multiply(self.range, derivative_out)


	def fabricate(self, variable_in):
		'''Calls forward()'''
		return self.forward(variable_in)
