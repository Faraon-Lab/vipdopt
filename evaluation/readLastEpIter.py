# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname, join
current_dir = dirname(__file__)
file_path = join(current_dir, "figure_of_merit.npy")

f = np.load(file_path)
lastEpochReached = False

idx = []
for i in range(len(f)):
	if f[i][1] == 0:
		lastEpochReached = True
		break
	else:
		idx.append(np.max(np.nonzero(f[i])))

lastEpoch = len(idx)-1
lastIter = idx[lastEpoch]

with open('lastEpIter.txt','a') as outFile:
	outFile.write("\nLast Epoch: %d, Last Iteration: %d" % (lastEpoch, lastIter))
	outFile.write("\nTime Elapsed:")
print("Last Epoch: %d, Last Iteration: %d" % (lastEpoch, lastIter))

# Plot FOM trace
trace = []
numEpochs = len(f); numIter = len(f[0])
upperRange = np.ceil(np.max(f))
plt.figure
vlinestyle = {'color': 'gray', 'linestyle': '--', 'linewidth': 1}
for i in range(len(f)-1):
	trace = np.concatenate((trace,f[i]),axis=0)
	plt.vlines((i+1)*numIter,0,upperRange,**vlinestyle)
# plt.plot(np.linspace(0,10,len(trace)),trace)
plt.plot(trace, '.-', color='orange')
plt.vlines(0*numIter,0,upperRange,**vlinestyle)
plt.vlines(numEpochs*numIter,0,upperRange,**vlinestyle)
plt.title('Figure of Merit - Trace')
plt.show()
