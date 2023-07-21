import filter as filter

import numpy as np

def sech(input):
	return np.divide(1, np.cosh(input))


#
# Needs more testing and figuring out.. quick version for testing
#
class Sigmoid3Level(filter.Filter):

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

	def fabricate(self, variable_in):
		result = np.add(
			np.multiply(self.variable_bounds[1], np.greater(variable_in, self.eta)),
			np.multiply(self.variable_bounds[0], np.less_equal(variable_in, self.eta))
			)
		return result 



