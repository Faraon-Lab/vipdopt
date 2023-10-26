# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

class Filter:

	def __init__(self, variable_bounds):
		self.variable_bounds = variable_bounds

	def verify_bounds(self, variable):
		'''Checks if the variable passed as argument is within the pre-defined bounds of this filter. Variable can be a single float, or a list / array.'''
		min_value = np.minimum(variable)
		max_value = np.maximum(variable)

		if ((min_value < self.variable_bounds[0]) or (max_value > self.variable_bounds[1])):
			return False

		return True


	def forward(self, variable_in):
		'''Propagate the variable_in through the filter and return the result. NOTE: NOT IMPLEMENTED.'''
		raise NotImplementedError()

	def chain_rule(self, derivative_out, variable_out, variable_in):
		'''Apply the chain rule to this filter to propagate the derivative back one step. NOTE: NOT IMPLEMENTED.'''
		raise NotImplementedError()
