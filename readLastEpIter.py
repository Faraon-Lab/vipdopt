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

with open('lastEpIter.txt','w') as outFile:
    outFile.write("Last Epoch: %d, Last Iteration: %d" % (lastEpoch, lastIter))
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
plt.plot(trace)
plt.vlines(0*numIter,0,upperRange,**vlinestyle)
plt.vlines(numEpochs*numIter,0,upperRange,**vlinestyle)
plt.title('Figure of Merit - Trace')
plt.show()
