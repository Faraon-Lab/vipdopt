import os
import numpy as np
from matplotlib import pyplot as plt
marker_style = dict(linestyle='-', linewidth=2.2, marker='o', markersize=4.5)

plt.rcParams.update({'font.sans-serif':'Helvetica Neue',            # Change this based on whatever custom font you have installed
                     'font.weight': 'normal', 'font.size':20})              
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'

RESULTS_PATH = os.path.join(os.getcwd(),'results')

hidden_layer_neurons = np.load('num_dense_neurons.npy')
# hidden_layer_neurons = np.append(np.array([1,2]), hidden_layer_neurons)
# np.save('num_dense_neurons.npy', hidden_layer_neurons)
flops_dense = 2*(784*hidden_layer_neurons+hidden_layer_neurons*10)

np_accuracies = np.load('accuracies.npy')
# np_accuracies_1st_part = np.load(r"C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Image Classifier\neural_network\[v0] accuracy_diagram\results"
#                                  + r"\accuracies.npy")
# z = np.insert(np_accuracies, 0, np_accuracies_1st_part,0)
# np.save('accuracies.npy', z)


fig, ax = plt.subplots()
plt.plot(flops_dense, np_accuracies[:,3], color='blue', **marker_style, label='Dense NN')
plt.xlim(((1e3,2e5)))
ax.set_xscale('log')
plt.title('Performance Diagram')
plt.xlabel('FLOPs')
plt.ylabel('Accuracy (%)')

# Creates legend    
ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1,.5))

# Figure size
fig_width = 8.0
fig_height = fig_width*4/5
l = ax.figure.subplotpars.left
r = ax.figure.subplotpars.right
t = ax.figure.subplotpars.top
b = ax.figure.subplotpars.bottom
figw = float(fig_width)/(r-l)
figh = float(fig_height)/(t-b)
ax.figure.set_size_inches(figw, figh, forward=True)

#plt.show()
plt.savefig('performancediagram_densenn.png', bbox_inches='tight')
print(3)