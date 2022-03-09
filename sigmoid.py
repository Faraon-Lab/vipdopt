import filter as filter

import numpy as np

def sech(x):
    return np.divide(1, np.cosh(x))

class Sigmoid(filter.Filter):
    '''Takes an input auxiliary density p(x) ranging from 0 to 1 and applies a sigmoidal projection filter to binarize / push it to either extreme.
    This depends on the strength of the filter. See OPTICA paper supplement Section IIA, https://doi.org/10.1364/OPTICA.384228,  for details.
    See also Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y.'''

    
    def __init__(self, beta, eta):
        # beta: strength of sigmoid filter
        # eta: center point of sigmoid filter
        # All input values above the threshold eta, are projected to 1, and the values below, projected to 0.
        
        variable_bounds = [0.0, 1.0]
        super(Sigmoid, self).__init__(variable_bounds)

        self.beta = beta
        self.eta = eta
        self.denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))


    def forward(self, variable_in):
        '''# All input values of variable_in above the threshold eta (defined in initialization), are projected to 1, and the values below, projected to 0.
        This is Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y.'''
        
        numerator = np.add(np.tanh(self.beta * self.eta), np.tanh(np.multiply(self.beta, np.subtract(variable_in, self.eta))))
        return np.divide(numerator, self.denominator)

    def chain_rule(self, derivative_out, variable_out, variable_in):
        '''Returns the first argument, multiplied by the direct derivative of forward() i.e. Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y,
        with respect to \tilde{p_i}.'''
        numerator = np.multiply(self.beta, np.power(sech(np.multiply(self.beta, np.subtract(variable_in, self.eta))), 2))
        return np.divide(np.multiply(derivative_out, numerator), self.denominator)

    def fabricate(self, variable_in):
        '''Same as forward(), but applies a hard step function filter instead of a sigmoid filter.
        Returns variable_bounds[1] if variable_in > self.eta
        Returns variable_bounds[0] if variable_in <= self.eta'''
        result = np.add(
            np.multiply(self.variable_bounds[1], np.greater(variable_in, self.eta)),
            np.multiply(self.variable_bounds[0], np.less_equal(variable_in, self.eta))
            )
        return result 



