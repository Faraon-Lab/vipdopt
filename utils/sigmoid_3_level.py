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

def sech(input):
	return np.divide(1, np.cosh(input))


# Needs more testing and figuring out.. quick version for testing

class Sigmoid3Level(filter.Filter):
	'''Takes an input auxiliary density p(x) ranging from 0 to 1 and applies a sigmoidal projection filter to binarize / push it to either extreme.
	This depends on the strength of the filter. See OPTICA paper supplement Section IIA, https://doi.org/10.1364/OPTICA.384228,  for details.
	See also Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y.'''


	#
	# beta: strength of sigmoid3Level filter
	# eta: center point of sigmoid3Level filter
	#
	def __init__(self, beta):
		variable_bounds = [0.0, 1.0]
		super(Sigmoid3Level, self).__init__(variable_bounds)

		self.beta = beta
		self.eta = 0.5
		self.denominator = np.tanh(beta * self.eta) + np.tanh(beta * (1 - self.eta))


	def forward(self, variable_in):
		# numerator0 = np.add(np.tanh(self.beta * self.eta), np.tanh(np.multiply(self.beta, np.subtract(variable_in, self.eta))))
		# numerator1 = np.add(np.tanh(self.beta * self.eta), np.tanh(np.multiply(self.beta, np.subtract(variable_in, self.eta))))


		numerator0 = 0.5 * ( np.tanh( self.beta * self.eta ) + np.tanh( self.beta * ( 2 * variable_in - self.eta ) ) )
		numerator1 = 0.5 * ( ( np.tanh( self.beta * self.eta ) + np.tanh( self.beta * ( 2 * ( variable_in - self.eta ) - self.eta ) ) ) )


		sig0 = numerator0 / self.denominator
		sig1 = 0.5 + ( numerator1 / self.denominator )

		return sig0 * np.less_equal( variable_in, 0.5 ) + sig1 * np.greater( variable_in, 0.5 )
		# return sig1 * np.greater( variable_in, 0.5 )

		# return ( sig0 + sig1 )

	def chain_rule(self, derivative_out, variable_out, variable_in):

		deriv0 = np.less_equal( variable_in, 0.5 ) * np.multiply(self.beta, np.power(sech(np.multiply(self.beta, np.subtract(2 * variable_in, self.eta))), 2)) / self.denominator
		deriv1 = np.greater( variable_in, 0.5 ) * np.multiply(self.beta, np.power(sech(np.multiply(self.beta, np.subtract(2 * variable_in, 3 * self.eta))), 2)) / self.denominator

		return 0.5 * ( derivative_out * ( deriv0 + deriv1 ) )

		# numerator = np.multiply(self.beta, np.power(sech(np.multiply(self.beta, np.subtract(variable_in, self.eta))), 2))
		# return np.divide(np.multiply(derivative_out, numerator), self.denominator)

	# TODO: Make this three-level or even n-level
	def fabricate(self, variable_in):
		result = np.add(
			np.multiply(self.variable_bounds[1], np.greater(variable_in, self.eta)),
			np.multiply(self.variable_bounds[0], np.less_equal(variable_in, self.eta))
			)
		return result 



