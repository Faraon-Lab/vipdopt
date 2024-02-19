# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

SIGMOID_BOUNDS = (0.0, 1.0)

def sech(z):
	"""Hyperbolic Secant."""
	return 1.0 / np.cosh(np.asanyarray(z))

class Filter:

	def __init__(self, variable_bounds=SIGMOID_BOUNDS):
		self.variable_bounds = variable_bounds

	def verify_bounds(self, variable):
		'''Checks if the variable passed as argument is within the pre-defined bounds of this filter. Variable can be a single float, or a list / array.'''
		
		return bool(
			(np.min(np.array(variable)) >= self.variable_bounds[0])
			and (np.max(np.array(variable)) <= self.variable_bounds[1])
		)

	def forward(self, variable_in):
		'''Propagate the variable_in through the filter and return the result. NOTE: NOT IMPLEMENTED.'''
		raise NotImplementedError()

	def chain_rule(self, derivative_out, variable_out, variable_in):
		'''Apply the chain rule to this filter to propagate the derivative back one step. NOTE: NOT IMPLEMENTED.'''
		raise NotImplementedError()

class Identity(Filter):
	'''Applies an identity filter which should not perturb the density at all. For testing purposes'''
	
	def forward(self, variable_in):
		return variable_in
	
	def chain_rule(self, derivative_out, variable_out, variable_in):
		return derivative_out

class Sigmoid(Filter):
	"""Applies a sigmoidal projection filter to binarize an input.

	Takes an input auxiliary density p(x) ranging from 0 to 1 and applies a sigmoidal
	projection filter to binarize / push it to either extreme. This depends on the
	strength of the filter. See OPTICA paper supplement Section IIA,
	https://doi.org/10.1364/OPTICA.384228,  for details. See also Eq. (9) of
	https://doi.org/10.1007/s00158-010-0602-y.

	Attributes:
		eta (Number): The center point of the sigmoid. Must be in range [0, 1].
		beta (Number): The strength of the sigmoid
		_bounds (tuple[Number]): The bounds of the filter. Always equal to (0, 1)
		_denominator (Number | npt.NDArray): The denominator used in various methods;
			for reducing re-computation.
	"""

	def __init__(self, beta, eta, variable_bounds=SIGMOID_BOUNDS):
		"""Initialize a sigmoid filter based on eta and beta values."""
		self.variable_bounds = variable_bounds

		if not self.verify_bounds(eta):
			raise ValueError('Eta must be in the range [0, 1]')

		self.eta  = eta
		self.beta = beta

		# Calculate denominator for use in methods
		self._denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))


	def __repr__(self):
		"""Return a string representation of the filter."""
		return f'Sigmoid filter with eta={self.eta:0.3f} and beta={self.beta:0.3f}'

	# @override
	def forward(self, x):
		"""Propagate x through the filter and return the result.
		All input values of x above the threshold eta, are projected to 1, and the
		values below, projected to 0. This is Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y.
		"""
		numerator = np.tanh(self.beta * self.eta) + \
			np.tanh(self.beta * (np.copy(np.array(x)) - self.eta))
		return numerator / self._denominator

	# @override  # type: ignore
	def chain_rule(
		self,
		deriv_out,
		var_out,
		var_in	):
		"""Apply the chain rule and propagate the derivative back one step.

		Returns the first argument, multiplied by the direct derivative of forward()
		i.e. Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y, with respect to
		\\tilde{p_i}.
		"""
		del  var_out  # not needed for sigmoid filter

		numerator = self.beta * \
			np.power(sech(self.beta * (var_in - self.eta)), 2) # type: ignore
		return deriv_out * numerator / self._denominator

	# @override
	def fabricate(self, x):
		"""Apply filter to input as a hard step-function instead of sigmoid.

		Returns:
			_bounds[0] where x <= eta, and _bounds[1] otherwise
		"""
		fab = np.array(x)
		fab[fab <= self.eta] = self._bounds[0]
		fab[fab > self.eta] = self._bounds[1]
		if isinstance(x, Number):  # type: ignore
			return fab.item()
		return fab
	
class Scale(Filter):
	'''Assuming that there is an input between 0 and 1, scales it to the range, min, and max, that are declared during initialization.
	See OPTICA paper supplement Section IIA, https://doi.org/10.1364/OPTICA.384228,  for details. 
	This class is used directly after the sigmoid filter is applied.

	Attributes:
		variable_bounds: here they will denote the permittivity bounds
	'''

	def __init__(self, variable_bounds):
		self.variable_bounds = variable_bounds
  
		#
		# variable_bounds specifies the minimum and maximum value of the scaled variable.
		# We assume we are scaling an input that is between 0 and 1 -> between min_permittivity and max_permittivity
		#

		self.min_value = variable_bounds[0]
		self.max_value = variable_bounds[1]
		self.range = self.max_value - self.min_value
	
	def __repr__(self):
		"""Return a string representation of the filter."""
		return f'Scale filter with minimum={self.min_value:0.3f} and max={self.max_value:0.3f}'

	# @override
	def forward(self, x):
		'''Performs scaling according to min and max values declared. Assumed that input is between 0 and 1.'''
		return np.add(self.min_value, np.multiply(self.range, x))

	# @override  # type: ignore
	def chain_rule(
		self,
		deriv_out,
		var_out,
		var_in	):
		"""Apply the chain rule and propagate the derivative back one step.
		"""
		return np.multiply(self.range, deriv_out)

	# @override
	def fabricate(self, x):
		'''Calls forward()'''
		return self.forward(x)
	
	
