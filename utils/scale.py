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
